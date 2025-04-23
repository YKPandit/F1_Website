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
from sklearn.preprocessing import StandardScalar

# Local Model Definition
from SimpleNN import SimpleNN

# Functions

""" 
Name: getQualiInfo
Param: Race:str
Return: dataframe of the session's information
Description: Will take in a race name and year then get the qualifying session for that year
"""

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



### MAIN ###

# Get all the races of the season

# Create a directory to store all sessions

# Loop through all the seasons

    # Print the race

    # Train a model -> make sure it is saves

# Take in Race, Driver, Condition, Previous Lap, Tire Age and give a prediction