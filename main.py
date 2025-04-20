import fastf1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

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


### Main ###
num_features = 5
race = input("Enter race name: ")

# Training set 1
q1 = get_qualifying_laps(2022, race)
# Training set 2
q2 = get_qualifying_laps(2023, race)

# Testing set
qt = get_qualifying_laps(2024, race)


q_1 = clean_data(q1)
q_1.sort_values(by=['LapTime'], inplace=True)
q_2 = clean_data(q2)
q_2.sort_values(by=['LapTime'], inplace=True)
q_t = clean_data(qt)
q_t.sort_values(by=['LapTime'], inplace=True)

q_1 = pd.get_dummies(q_1, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'])
q_2 = pd.get_dummies(q_2, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'])
q_t = pd.get_dummies(q_t, columns=['Driver', 'Weather'], prefix=['Driver', 'Weather'])


# Print the final DataFrame as a markdown table
print("\n--- Final newInfo DataFrame ---")

# Writing so I can understand what the data looks like
file = open("quali_data.txt", "w+")
file.write(q_1.to_markdown(index=False))
file.write(q_2.to_markdown(index=False))
file.write(q_t.to_markdown(index=False))

print(f"Len of df: {len(q_1)+ len(q_2)+ len(q_t)}")

num_features = len(q_1.columns)


# Training the model
model = SimpleNN(num_features, 10, 10, 1)

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


