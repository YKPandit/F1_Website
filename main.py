import fastf1
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import math
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset


from SimpleNN import SimpleNN



"""
    Steps:
        1. Get the qualifying data of a given driver, example Leclerc
        2. Train a model based on all their laptimes and predict what their qualifying lap will be
            
        
    Setup:
        1. Get a given session's data
        2. Get all driver lap times for Q3 and Quali
            -> get it as driver track lap # tireCompund laps on Compound etc
        3. Create a new pandas dataframe that is set up as follows:
            -> Same as before but each non-numerical thing is set up as a binary value(one hot encoding)
        4. Delete useless data and set it up to be trained

    Training:
        1. See MLP model trained before -> use 2022->2023 to train model and 2024 to test the model
"""


"""
    Data to keep in the new pandas df(sort the df by fastest laptime):
        driver abbrv
        laptimes
        tire age
        weather conditions
"""
scalar = StandardScaler()
# Functions
def get_qualifying_laps(year: int, gp_name: str) -> dict:
    """
    Retrieves all quick qualifying laps for each driver in a specific F1 qualifying session.

    Args:
        year: The year of the session.
        gp_name: The name of the Grand Prix (e.g., 'Monaco Grand Prix').

    Returns:
        A dictionary where keys are driver abbreviations and values are
        fastf1 Laps objects containing their quick laps from qualifying.
        Returns an empty dictionary if the session data cannot be loaded.
    """
    try:
        session = fastf1.get_session(year, gp_name, 'Q')
        session.load(laps=True, telemetry=False, weather=True, messages=False) # Load only necessary data
    except Exception as e:
        print(f"Error loading session {year} {gp_name} Qualifying: {e}")
        return {}

    driver_laps = {}
    drivers = session.drivers # Get list of driver numbers
    for driver_number in drivers:
        try:
            driver_info = session.get_driver(driver_number)
            driver_abbreviation = driver_info['Abbreviation']
            laps = session.laps.pick_driver(driver_number).pick_quicklaps()
            # Merge weather data onto the laps DataFrame
            temp = laps.get_weather_data()

            # Create a simplified weather condition column
            if 'Rainfall' in temp.columns:
                laps['WeatherCondition'] = np.where(temp['Rainfall'] == True, 'Wet', 'Dry')
            else:
                laps['WeatherCondition'] = 'Unknown' # Or handle as appropriate if weather data missing

            if not laps.empty:
                driver_laps[driver_abbreviation] = laps
        except Exception as e:
            # Handle cases where a driver might not have set a quick lap or other data issues
            print(f"Could not process laps for driver {driver_number} in {year} {gp_name} Qualifying: {e}")
            continue # Continue to the next driver

    return driver_laps

def clean_data(q1):
    newInfo = pd.DataFrame()
    newInfo['Driver'] = np.nan
    newInfo['LapTime'] = np.nan
    newInfo['TireAge'] = np.nan
    newInfo['Weather'] = np.nan
    newInfo['PreviousLap'] = np.nan


    for i in q1: # i is the driver abbreviation (e.g., 'LEC')
        lap_data_lines = [] # Store lines for this driver
        # laps_df is the DataFrame for the current driver 'i'
        laps_df = q1[i]

        previous_lap_time = None
        previous_lap_driver = None
        # Iterate over each lap (row) in the DataFrame
        for index, lap_row in laps_df.iterrows():
            # # lap_row is a pandas Series containing all data for one lap
            driver_abbr = i # We already have the abbreviation from the outer loop
            weather = lap_row['WeatherCondition']

            lap_time = lap_row['LapTime']
            lap_time = pd.to_timedelta(lap_time)
            lap_time = lap_time/pd.Timedelta(seconds=1)

            tire_age = lap_row['TyreLife']

            if(previous_lap_time != None and previous_lap_driver == driver_abbr):
                temp = pd.DataFrame({
                    'Driver': [driver_abbr],  # Value needs to be in a list/iterable
                    'LapTime': [lap_time],
                    'Weather': [weather],
                    'PreviousLap': [previous_lap_time],
                    'TireAge':[tire_age]
                    # Add other columns if needed
                })

                newInfo = pd.concat([newInfo, temp], ignore_index=True)

            previous_lap_time = lap_time
            previous_lap_driver = driver_abbr


    return newInfo

def scale_data(X_1_train):
    cols_to_scale = X_1_train.select_dtypes(include=np.double).columns.to_list()
    
    scalar.fit(X_1_train[cols_to_scale])
    x_1_scaled = scalar.transform(X_1_train[cols_to_scale])
    return pd.DataFrame(x_1_scaled, index=X_1_train.index, columns=cols_to_scale)


### Main ###
num_features = 5
# race = input("Enter race name: ")
def getPrediction(info):
    # Training set 1
    q1 = get_qualifying_laps(2022, info.race)
    # Training set 2
    q2 = get_qualifying_laps(2023, info.race)

    # Testing set
    qt = get_qualifying_laps(2024, info.race)


    q_1 = clean_data(q1)
    q_2 = clean_data(q2)
    # q_2.sort_values(by=['LapTime'], inplace=True)
    q_t = clean_data(qt)
    # q_t.sort_values(by=['LapTime'], inplace=True)

    # Print the final DataFrame as a markdown table
    print("\n--- Final newInfo DataFrame ---")



    print(f"Len of df: {len(q_1)+ len(q_2)+ len(q_t)}")





    batch_1 = len(q_1)
    batch_2 = len(q_2)
    batch_test = len(q_t)


    # Spliting into data and comparison
    X_1_train = q_1.drop(columns=['LapTime'])
    y_1_train = q_1['LapTime']

    X_2_train = q_2.drop(columns=['LapTime'])
    y_2_train = q_2['LapTime']


    X_test = q_t.drop(columns=['LapTime'])
    y_test = q_t['LapTime']



    X_1_train = pd.get_dummies(X_1_train, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'], dtype=int)
    X_2_train = pd.get_dummies(X_2_train, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'], dtype=int)
    X_test = pd.get_dummies(X_test, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'], dtype=int)


    X_1_train['TireAge'] = X_1_train['TireAge'].astype(np.double)
    X_2_train['TireAge'] = X_2_train['TireAge'].astype(np.double)
    X_test['TireAge'] = X_test['TireAge'].astype(np.double)

    cols_to_scale = X_1_train.select_dtypes(include=np.double).columns.to_list()


    X_1_train[cols_to_scale] = scale_data(X_1_train)
    X_2_train[cols_to_scale] = scale_data(X_2_train)
    X_test[cols_to_scale] = scale_data(X_test)


    # Writing so I can understand what the data looks like
    file = open("quali_data.txt", "w+")
    file.write(X_1_train.to_markdown(index=False))
    file.write('\n')
    file.write(X_2_train.to_markdown(index=False))
    file.write('\n')
    file.write(X_test.to_markdown(index=False))

    num_features = len(X_1_train.columns)


    # Training the model
    model = SimpleNN(num_features, 5, 5, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    cols_to_train = X_1_train.columns.to_list()

    # Convert features to numpy float32 first
    X_1_train_np = X_1_train.to_numpy(dtype=np.float32)
    X_2_train_np = X_2_train.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)

    # Convert targets to numpy float32 AND reshape for loss function (usually needs [N, 1])
    y_1_train_np = y_1_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_2_train_np = y_2_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    # Convert to tensors from float32 numpy arrays
    X_1_train = torch.from_numpy(X_1_train_np)
    y_1_train = torch.from_numpy(y_1_train_np)
    X_2_train = torch.from_numpy(X_2_train_np)
    y_2_train = torch.from_numpy(y_2_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    # --- 1. Combine Training Data ---
    X_train_combined = torch.cat((X_1_train, X_2_train), dim=0)
    y_train_combined = torch.cat((y_1_train, y_2_train), dim=0)

    # --- 2. Create Dataset and DataLoader for Combined Training Data ---
    train_dataset = TensorDataset(X_train_combined, y_train_combined)
    batch_size = 32 # Choose a reasonable batch size (e.g., 32, 64, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle training data is important!

    # --- 3. Create DataLoader for Validation Data ---
    val_dataset = TensorDataset(X_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size) # No need to shuffle validation data



    # --- 5. The Correct Training Loop with Batches and Validation ---
    epochs = 4000 # Total epochs over the combined data

    print("Starting training loop...")

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        train_loss_sum = 0.0 # Accumulate loss for the epoch

        # Iterate over batches from the *combined* training dataloader
        for batch_inputs, batch_targets in train_dataloader:
            

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(batch_inputs)

            # Calculate Loss for the CURRENT BATCH
            # Ensure targets have the same shape as outputs if necessary (e.g., both [batch_size, 1])
            loss = criterion(outputs, batch_targets)

            # Backward Pass (Backpropagation)
            loss.backward()

            # Optimizer Step (Update weights and biases)
            optimizer.step()

            train_loss_sum += loss.item() * batch_inputs.size(0) # Accumulate loss for the epoch

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss_sum / len(train_dataset)

        # --- Evaluation on Validation Set ---
        model.eval() # Set the model to evaluation mode
        val_loss_sum = 0.0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for val_inputs, val_targets in val_dataloader: # Use your actual val_dataloader here
                val_outputs = model(val_inputs)
                batch_val_loss = criterion(val_outputs, val_targets)
                val_loss_sum += batch_val_loss.item() * val_inputs.size(0) # Accumulate loss

        avg_val_loss = val_loss_sum / len(val_dataset)

        # Print epoch statistics (shows training progress)
        if (epoch + 1) % 100 == 0: # Print every 10 epochs, or more/less often
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    print("Training loop finished.")

    # --- Final Evaluation on Test Data (after the training loop is completely done) ---
    # You would do this using X_test, y_test (or your test_dataloader) once you've
    # finished training and potentially selected the best model based on validation performance.
    # This final step is separate from the training loop above.


    # abbr = input("Enter driver abbr: ")
    # tireAge = int(input("Enter tire age: "))
    # previousLap = float(input("Enter previous laptime: "))
    # weather_cond = input("Enter weather conditions: ")

    prediciton = {
        "Driver":[info.driver],
        "LapTime":[np.nan],
        "TireAge":[int(info.tire_age)],
        "Weather":[info.weather],
        "PreviousLap":[float(info.previous_lap)]
    }

    df = pd.DataFrame(prediciton)

    df = pd.get_dummies(df, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'], dtype=int)
    df = df.reindex(columns=cols_to_train, fill_value=0)

    # scalar.fit(df)
    # df = scalar.transform(df)

    df['TireAge'] = df['TireAge'].astype(np.double)
    df['PreviousLap'] = df['PreviousLap'].astype(np.double)

    cols_to_scale = df.select_dtypes(include=np.double).columns.to_list()
    df[cols_to_scale] = scale_data(df)

    input_val = torch.from_numpy(df.to_numpy(dtype=np.float32))



    model.eval()

    with torch.no_grad():
        predicted = model(input_val)

        print(f"Predicted lap time: {predicted}")

    return predicted