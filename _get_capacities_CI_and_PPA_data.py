from global_variables import *
import ast
from datetime import datetime
import pytz
from PPA_model import LCOE_dataset
import pandas as pd

def dataframes_to_excel(dfs, file_name):
    """
    Convert multiple pandas DataFrames into one Excel file with multiple sheets.

    Parameters:
    dfs (dict): Dictionary where keys are sheet names and values are corresponding DataFrames
    file_name (str): Name of the Excel file

    Returns:
    None
    """
    try:
        with pd.ExcelWriter(file_name) as writer:
            for sheet_name, df in dfs.items():
                if isinstance(df, dict):  # check if value is another dict (specifically for 'CI')
                    for sub_sheet_name, sub_df in df.items():
                        # use a combination of main key and sub key as the sheet name
                        sub_sheet_name = f"{sheet_name}_{sub_sheet_name}"
                        sub_df.to_excel(writer, sheet_name=sub_sheet_name, index=False)
                else:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Successfully written to {file_name}')
    except Exception as e:
        print(f'An error occurred: {e}')
def get_current_est_time():
    est_timezone = pytz.timezone('US/Eastern')

    current_time_est = datetime.now(est_timezone)

    return current_time_est.strftime('%Y-%m-%d %H:%M:%S')

#Load the AEO22 and AEO23 data from the Excel file
aeo22_data = pd.read_excel(
    electric_grid_carbon_intensity_data_file_path, sheet_name='AEO22',
    index_col=0)
aeo23_data = pd.read_excel(
    electric_grid_carbon_intensity_data_file_path, sheet_name='AEO23',
    index_col=0)
wind_and_battery_excel = pd.read_excel(
    optimization_results_file_path, sheet_name='Sheet1')

PPA_data = LCOE_dataset

#capacity for wind and battery value ranges
def extract_values(time, matching, results_df=wind_and_battery_excel):
    # Filter for given time and matching, and capex_description 'low'
    mask_high = (results_df['time'] == time) & (results_df['matching'] == matching) & (
                results_df['capex_description'] == 'high') & (results_df['location'] == 'high')
    mask_low = (results_df['time'] == time) & (results_df['matching'] == matching) & (
                results_df['capex_description'] == 'low') & (results_df['location'] == 'low')

    # Extracting values for each technology
    extracted_values = {}
    for mask in [mask_high, mask_low]:
        filtered_df = results_df[mask]
        for _, row in filtered_df.iterrows():
            technology = row['technology']
            wind_capacity = row['wind_capacity']
            battery_capacity = row['battery_capacity']
            curtailment = ast.literal_eval(row['curtailment'])
            discharge = ast.literal_eval(row['discharge'])
            extracted_values[technology] = {
                'wind_capacity': wind_capacity,
                'battery_capacity': battery_capacity,
                'curtailment': curtailment,
                'discharge': discharge
            }

    return extracted_values
wind_and_battery_data = {time:{matching:extract_values(time, matching) for matching in matching_type} for time in times}
