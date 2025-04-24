# Imports

# Data imports
import fastf1

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Processing imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Local Model Definition
from SimpleNN import SimpleNN

# Other
import os


# Globals


# Functions
""" 
Name: getQualiInfo
Param: Race:str year:int
Return: dataframe of the session's information
Description: Will take in a race name and year then get the qualifying session for that year
"""
def getQualiInfo(race:str, year:int):
    session = fastf1.get_session(year, race, "Q")
    session.load(laps=True, weather=True)

    laps = pd.DataFrame()

    # Relevant Columns
    laps["Driver"] = np.nan
    laps["PreviousLap"] = np.nan
    laps["TireAge"] = np.nan
    laps["Weather"] = np.nan
    laps["LapTime"] = np.nan
    laps["Race"] = np.nan

    # weather = session.get_weather_data()
    drivers = session.drivers
    
    for driver_num in drivers:
        driver = session.get_driver(driver_num)["Abbreviation"]
        driver_laps = session.laps.pick_drivers(driver).pick_wo_box()

        previousLap = 0
        lastDriver = None

        for i, lap in driver_laps.iterrows():
            # Set up new Lap

            # Load the weather condition for that lap using python "ternary"
            weather_condition = lap.get_weather_data()
            weather_condition = "Wet" if weather_condition["Rainfall"] == True else "Dry"

            # Checking so you have previous lap data to train off of
            lap_time = lap["LapTime"]

            if(previousLap != 0 and lastDriver == driver):
                new_lap = pd.DataFrame({
                    "Driver" : [driver],
                    "PreviousLap" : [pd.to_timedelta(previousLap)/pd.Timedelta(seconds=1)],
                    "TireAge" : [lap["TyreLife"]],
                    "Weather" : [weather_condition],
                    "LapTime" : [pd.to_timedelta(lap["LapTime"])/pd.Timedelta(seconds=1)],
                    "Race":[race]
                })


                laps = pd.concat([laps, new_lap], ignore_index=True)
                

            previousLap = pd.to_timedelta(lap["LapTime"])
            lastDriver = driver

    return laps

"""
Name: scale_data
Param: data:dataframe, cols_to_scale:list
Return: dataframe
Desc: Scale the data to 0->1
"""
def scale_data(data, cols_to_scale):
    scalar = StandardScaler()
    scalar.fit(data[cols_to_scale])
    scaled = scalar.transform(data[cols_to_scale])
    joblib.dump(scalar, "model.gz")
    return pd.DataFrame(scaled, index=data.index, columns=cols_to_scale)

def clean_data(data, headers, scalar):
    # Convert to correct types
    data["TireAge"] = data["TireAge"].astype("double")
    data["PreviousLap"] = data["PreviousLap"].astype("double")

    # Seperate
    y_train = data["LapTime"]
    x_train = data.drop(columns=["LapTime"])
    x_train = pd.get_dummies(x_train, columns=['Driver', 'Weather', 'Race'], prefix=['Driver', 'Weather', 'Race'], dtype=int)
    print(x_train)

    
    if(headers != None):
        x_train = x_train.reindex(columns=headers, fill_value=0)
    print(x_train)
    

    cols_to_scale = x_train.select_dtypes(include=np.double).columns.to_list()
    # print(cols_to_scale)
    # x_train[cols_to_scale] = scale_data(x_train, cols_to_scale)
    # print(x_train)

    
    scaled = scalar.transform(data[cols_to_scale])
    
    x_train[cols_to_scale] = pd.DataFrame(scaled, index=data.index, columns=cols_to_scale)

    return x_train, y_train

def to_tensor(x_train, y_train):
    x_train = x_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    return x_train_tensor, y_train_tensor

"""
Name: trainModel
Param: race: str
Return: None, exits on erro
Description:
        1. Takes in a race
        2. Loads all required sessions
        3. Merges all but the last session
        4. Convert all relevant data to correct types
        5. One-Hot encode the data
        6. Create a simmpleNN model from import
        7. Train model with data
        8. Save the trained model
"""
def train_model(training_data, test_data):
    cols_to_scale = training_data.select_dtypes(include=np.double).columns.to_list()
    scalar = StandardScaler()
    scalar.fit(training_data[cols_to_scale])
    x_train, y_train = clean_data(training_data, None, scalar)
    x_test, y_test = clean_data(test_data, None, scalar)
    joblib.dump(scalar, "model.gz")

    # Add this line to align test columns with train columns:
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    file = open("quali_data.txt", "w+")
    file.write(x_train.to_markdown(index=False))
    file.write('\n')
    # file.write(X_2_train.to_markdown(index=False))
    # file.write('\n')
    file.write(x_test.to_markdown(index=False))

    num_input = len(x_train.columns)
    headers = x_train.columns

    # Convert to Tensor
    x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
    x_test_tensor, y_test_tensor = to_tensor(x_test, y_test)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Model Parameters
    model = SimpleNN(num_input, 20, 20, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 1000
    batch_size = 64

    # Loading
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Training Loop
    print(f"Started training the model")
    best_val_loss = float('inf') # For early stopping
    patience = 100 # How many epochs to wait for improvement
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch_in, batch_targ in train_dataloader:

            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_targ)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_in.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- Validation Evaluation ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in test_dataloader:
                val_outputs = model(val_inputs)
                batch_val_loss = criterion(val_outputs, val_targets)
                running_val_loss += batch_val_loss.item() * val_inputs.size(0)

        epoch_val_loss = running_val_loss / len(test_data)

        # Print epoch statistics
        if (epoch + 1) % 100 == 0 or epoch == 0: # Print every 100 epochs and the first one
             print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

        # --- Early Stopping Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Optional: Save the best model state
            # torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break # Exit the training loop



    print(f"Finished training the model")
    pathname = f"Models/model.pth"
    torch.save(model.state_dict(), pathname)

    pathname = f"Models/model.txt"
    with open(pathname, 'w+') as f:
        for header in headers:
            f.write(f"{header}\n")

        f.write(str(num_input))
    



### MAIN ###

# Get all the races of the season
season = fastf1.get_event_schedule(2025)

# Create a directory to store all sessions
directory_name = "Models"

# Create the directory
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")


# Loop through all the seasons
print(season)
all_training_races = None
all_val_races = None
for i, race in season.iterrows():
    # Print the race
    print(race["Country"])
    location = race["Location"]
    if(' ' in race["Location"]):
        location = location.replace(" ", "_")
        print(race)

    year_1 = getQualiInfo(race["Location"], 2022)
    if(year_1.empty):
        print("Error: No race")
        break
    all_training_races = pd.concat([all_training_races, year_1])
    

    year_2 = getQualiInfo(race["Location"], 2023)
    if(year_2.empty):
        print("Error: No race")
        break

    all_training_races = pd.concat([all_training_races, year_2])

    year_3 = getQualiInfo(race["Location"], 2024)
    if(year_3.empty):
        print("Error: No race")
        break
    all_val_races = pd.concat([all_val_races, year_3])

    
    file = open("quali_data.txt", "w+")
    file.write(all_training_races.to_markdown(index=False))
    file.write('\n')
    file = open("val.txt", "w+")
    file.write(all_val_races.to_markdown(index=False))


if(train_model(all_training_races, all_val_races) == -1):
    print("Training error")

# Take in Race, Driver, Condition, Previous Lap, Tire Age and give a prediction
# race = input("Enter race: ")
# if(' ' in race):
#     race = race.replace(" ", "_")
#     print(race)



driver = input("Enter driver: ")
previousLap = float(input("Enter previous lap: "))
tireAge = int(input("Enter tire age: "))
weather = input("Enter weather conditions: ")



# Print the race
# print(race["Location"])
headers = []
# Train a model -> make sure it is saves
filename = f"Models/model.txt"
with open(filename, 'r') as f:
    for line in f:
        headers.append(line.strip())

input_layer = int(headers.pop(len(headers)-1))

file = open("other.txt", "w+")
path = f"Models/model.pth"
model = SimpleNN(input_layer, 20, 20, 1)
model.load_state_dict(torch.load(path, weights_only=True))
for i, race in season.iterrows():
    race = race["Location"]
    print(race)
    

    data = pd.DataFrame({
        "Driver":[driver],
        "PreviousLap":[previousLap],
        "TireAge" : [tireAge],
        "Weather" : [weather],
        "LapTime" : [np.nan],
        "Race": [race]
    })

    # 
    # data, etc = clean_data(data, headers)
    # print(data)

    data["TireAge"] = data["TireAge"].astype("double")
    data["PreviousLap"] = data["PreviousLap"].astype("double")

    # Seperate
    y_train = data["LapTime"]
    x_train = data.drop(columns=["LapTime"])
    x_train = pd.get_dummies(x_train, columns=['Driver', 'Weather', 'Race'], prefix=['Driver', 'Weather', 'Race'], dtype=int)

    
    if(headers != None):
        x_train = x_train.reindex(columns=headers, fill_value=0)
    

    cols_to_scale = x_train.select_dtypes(include=np.double).columns.to_list()
    loaded_scalar = joblib.load("model.gz")
    scaled = loaded_scalar.transform(x_train[cols_to_scale])
    x_train[cols_to_scale] = pd.DataFrame(scaled, index=x_train.index, columns=cols_to_scale)
    data = x_train



    file.write(data.to_markdown(index=False))
    file.write('\n')
    

    data = torch.from_numpy(data.to_numpy(dtype=np.float32))

    print(data)


    

    model.eval()

    with torch.no_grad():
        predicted = model(data)

        print(f"Predicted lap time: {predicted}")