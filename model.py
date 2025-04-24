# Imports

# Data imports
import fastf1

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Processing imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Local Model Definition
from SimpleNN import SimpleNN

# Other
import os


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
    return pd.DataFrame(scaled, index=data.index, columns=cols_to_scale)

def clean_data(data):
    # Convert to correct types
    data["TireAge"] = data["TireAge"].astype("double")
    data["PreviousLap"] = data["PreviousLap"].astype("double")

    # Seperate
    y_train = data["LapTime"]
    x_train = data.drop(columns=["LapTime"])

    x_train = pd.get_dummies(x_train, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'], dtype=int)
    cols_to_scale = x_train.select_dtypes(include=np.double).columns.to_list()
    x_train[cols_to_scale] = scale_data(x_train, cols_to_scale)

    return x_train, y_train

def to_tensor(x_train, y_train):
    x_train = x_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_train = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train_tensor = torch.from_numpy(x_train.values)
    y_train_tensor = torch.from_numpy(y_train.values)

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
def train_model(race:str):
    # Get the sessions
    year_1 = getQualiInfo(race, 2022)
    year_2 = getQualiInfo(race, 2023)
    year_3 = getQualiInfo(race, 2024)
    
    # Merge 1 & 2
    training_data = pd.concat([year_1, year_2])
    
    x_train, y_train = clean_data(training_data)
    x_test, y_test = clean_data(year_3)


    # Writing so I can understand the data
    file = open("quali_data.txt", "w+")
    file.write(x_train.to_markdown(index=False))
    file.write("\n")
    file.write(x_test.to_markdown(index=False))

    # Convert to Tensor
    x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
    x_test_tensor, y_test_tensor = to_tensor(x_test, y_test)

    
    


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
row = 0
for i, race in season.iterrows():
    # Print the race
    print(race["Country"])

    # Train a model -> make sure it is saves
    train_model(race["Country"])

    break
    

# Take in Race, Driver, Condition, Previous Lap, Tire Age and give a prediction