import fastf1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader


# AI Written functions
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

def writeInfo(q1, q2, qt):
    with open("quali_data.txt", "w") as f:
        # Set pandas display options for wider, non-wrapping output
        pd.set_option('display.width', 10000)  # Adjust width as needed
        pd.set_option('display.max_columns', None) # Ensure all columns are shown
        pd.set_option('display.max_rows', None) # Ensure all rows are shown

        f.write(f"Qualifying Data for {race}\\n")
        f.write("=============================\\n")

        f.write("\n--- 2022 ---\\n\n")
        for driver_abbr, laps_df in q1.items():
                #f.write(f"\n----- {driver_abbr} -----\\n")
                f.write(laps_df.to_markdown() + "\\n\\n")

        f.write("\n--- 2023 ---\\n\n")
        for driver_abbr, laps_df in q2.items():
                #f.write(f"\n----- {driver_abbr} -----\\n")
                f.write(laps_df.to_markdown() + "\\n\\n")

        # Note: qt is currently the same as q2 (both 2023), you might want to adjust this
        f.write("\n--- 2024 (Test Set?) ---\\n\n")
        for driver_abbr, laps_df in qt.items():
                #f.write(f"\n----- {driver_abbr} -----\\n")
                f.write(laps_df.to_markdown() + "\\n\\n")

    print("Data written to quali_data.txt")

    # Reset pandas options to default (optional, good practice)
    pd.reset_option('display.width')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')


# Functions

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

race = input("Enter race name: ")

q1 = get_qualifying_laps(2022, race)
q2 = get_qualifying_laps(2023, race)
qt = get_qualifying_laps(2023, race)


writeInfo(q1, q2, qt)

"""
    Data to keep in the new pandas df(sort the df by fastest laptime):
        driver abbrv
        laptimes
        tire age
        weather conditions
"""

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


# Print the final DataFrame as a markdown table
print("\n--- Final newInfo DataFrame ---")
# index=False prevents printing the default DataFrame index column
print(newInfo.to_markdown(index=False))
print(f"Len of df: {len(newInfo)}")
