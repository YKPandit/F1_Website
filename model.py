# Imports

# Data imports
import fastf1

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Processing imports
import pandas
import numpy
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
    session = None
    return session


"""
Name: mergeSessions
Param: numSess:int, all dataframes with quali info
Return: A new datframe that contains all the sessions merged
"""


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
def train_mode(race:str):
    print("Starting Data Collection")
    year_1 = getQualiInfo(race, 2022)
    year_2 = getQualiInfo(race, 2023)
    year_3 = getQualiInfo(race, 2024)
    print("Finished Data Collection")


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
for race in season['Location']:
    # Print the race
    print(race)
    # Train a model -> make sure it is saves
    train_model(race)

# Take in Race, Driver, Condition, Previous Lap, Tire Age and give a prediction