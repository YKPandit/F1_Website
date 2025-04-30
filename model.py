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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def to_tensor(x_train, y_train):
    x_train = x_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    return x_train_tensor, y_train_tensor


### MAIN ###
season = fastf1.get_event_schedule(2025)
sessions = pd.DataFrame()
latest_season = pd.DataFrame()

# Gather all the data
counter = 0
for i, race in season.iterrows():
    # Print country
    print(race["Location"])
    counter += 1

    # Get the session
    y1 = getQualiInfo(race["Location"], 2022)
    y2 = getQualiInfo(race["Location"], 2023)
    y3 = getQualiInfo(race["Location"], 2024)

    # Merge
    merged = pd.concat([y1, y2])

    # Print
    # print(merged)

    sessions = pd.concat([sessions, merged])
    latest_season = pd.concat([latest_season, y3])
    
    # if(counter == 2):
    #     break
    

# print(sessions)

## Clean and seperate the data into x and y sets ##
# Re-rep data
sessions["TireAge"] = sessions["TireAge"].astype("double")
sessions["PreviousLap"] = sessions["PreviousLap"].astype("double")

latest_season["TireAge"] = latest_season["TireAge"].astype("double")
latest_season["PreviousLap"] = latest_season["PreviousLap"].astype("double")

# Seperate
y_train = sessions["LapTime"]
x_train = sessions.drop(columns=["LapTime"])

y_val = latest_season["LapTime"]
x_val = latest_season.drop(columns=["LapTime"])


## Encoding the Data ##

# Data to scale and data to encode
cols_to_encode = ['Driver', 'Weather', 'Race']
other_cols = ['TireAge', 'PreviousLap']

# Encode
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# Fit the encoder
x_train_encoded = encoder.fit_transform(x_train[cols_to_encode])
# print(f"{x_train.index} vs {x_train.columns}")

# One hot Encoding
x_train_encoded = pd.DataFrame(x_train_encoded, index=x_train.index, columns=encoder.get_feature_names_out(cols_to_encode))
# x_train = x_train[cols_to_encode + other_cols].join(x_train_encoded)
# x_train = x_train.drop(columns=cols_to_encode)
x_train = pd.concat([x_train[other_cols], x_train_encoded], axis=1)


x_val_encoded = encoder.transform(x_val[cols_to_encode])
x_val_encoded = pd.DataFrame(x_val_encoded, index=x_val.index, columns=encoder.get_feature_names_out(cols_to_encode))
# x_val = x_val[cols_to_encode + other_cols].join(x_val_encoded)
# x_val = x_val.drop(columns=cols_to_encode)
x_val = pd.concat([x_val[other_cols], x_val_encoded], axis=1)




# Scale
scaler = StandardScaler()
x_train[other_cols] = scaler.fit_transform(x_train[other_cols])

x_val[other_cols] = scaler.transform(x_val[other_cols])
if(len(x_train) != len(y_train)):
    exit()

x_train = x_train.fillna(0)
y_train = y_train.fillna(0)
x_val = x_val.fillna(0)
y_val = y_val.fillna(0)


# Convert everything to a tensor
x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
x_val_tensor, y_val_tensor = to_tensor(x_val, y_val)

# print(x_train_tensor)
# print(y_train_tensor)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Model Parameters
num_input = len(x_train.columns)
model = SimpleNN(num_input, 20, 20, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
epochs = 7000
batch_size = 32

# Loading
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
print(x_train_tensor)

# Training Loop
print(f"Started training the model")
best_val_loss = float('inf') # For early stopping
patience = 100 # How many epochs to wait for improvement
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    # print(f"First one: {running_train_loss}")

    for batch_in, batch_targ in train_dataloader:
        # print(batch_in)
        # print(batch_targ)
        optimizer.zero_grad()
        outputs = model(batch_in)
        loss = criterion(outputs, batch_targ)
        
        loss.backward()
        # print(loss.item())
        optimizer.step()

        running_train_loss += loss.item() * batch_in.size(0)
    


    # print(len(train_dataset))
    epoch_train_loss = running_train_loss / len(train_dataset)

    # --- Validation Evaluation ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in test_dataloader:
            val_outputs = model(val_inputs)
            batch_val_loss = criterion(val_outputs, val_targets)
            running_val_loss += batch_val_loss.item() * val_inputs.size(0)

    epoch_val_loss = running_val_loss / len(x_val)

    # Print epoch statistics
    if (epoch + 1) % 100 == 0 or epoch == 0: # Print every 100 epochs and the first one
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    # --- Early Stopping Check ---
    # if epoch_val_loss < best_val_loss:
    #     best_val_loss = epoch_val_loss
    #     epochs_no_improve = 0
    #     # Optional: Save the best model state
    #     # torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve == patience:
    #         print(f"Early stopping triggered after {epoch+1} epochs.")
    #         break # Exit the training loop




data = pd.DataFrame({
        "Driver":["VER"],
        "PreviousLap":[82],
        "TireAge" : [2.0],
        "Weather" : ["Dry"],
        "Race": ["Melbourne"]
    })

data_encoded = encoder.transform(data[cols_to_encode])
data_encoded = pd.DataFrame(data_encoded, index=data.index, columns=encoder.get_feature_names_out(cols_to_encode))
data = data[cols_to_encode + other_cols].join(data_encoded)
data = data.drop(columns=cols_to_encode)
data[other_cols] = scaler.transform(data[other_cols])
data = data.reindex(columns=x_train.columns, fill_value=0)
# file = open("quali_data.txt", "w+")
# file.write(data.to_markdown(index=False))
# file.write('\n')
data_num = data.to_numpy(dtype=np.float32)
data = torch.from_numpy(data_num)


print(data)
model.eval()

with torch.no_grad():
    predicted = model(data)

    print(f"Predicted lap time: {predicted}")


joblib.dump(scaler, "./Models/scalar.pkl")
joblib.dump(encoder, "./Models/encoder.pkl")

file = open("Models/columns.txt", "w+")

for name in x_train.columns.to_list():
    file.write(name)
    file.write('\n')

file.write(str(len(x_train.columns.to_list())))
file.write('\n')
for col in cols_to_encode:
    file.write(col)
    file.write('\n')

for col in other_cols:
    file.write(col)
    file.write('\n')

torch.save(model.state_dict(), "Models/model.pth")