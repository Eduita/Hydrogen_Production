import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus
import os
import matplotlib.pyplot as plt


def optimize_energy_system(COST_wind, COST_battery, EFF, D, num_time_periods, CF_t, monthly_aggregation=False, yearly_aggregation=False):

    print(CF_t)

    problem = LpProblem(str(np.random.uniform(0,1)), LpMinimize)

    # Create decision variables
    w = LpVariable("w", lowBound=0.01)
    b = LpVariable("b", lowBound=0)
    w_t = [LpVariable(f"w_{t}", lowBound=0) for t in range(num_time_periods)]
    c_t = [LpVariable(f"c_{t}", lowBound=0) for t in range(num_time_periods)]
    d_t = [LpVariable(f"d_{t}", lowBound=0) for t in range(num_time_periods)]
    b_t = [LpVariable(f"b_{t}", lowBound=0) for t in range(num_time_periods)]

    # Objective function
    objective = COST_wind * w + COST_battery * b
    problem += objective

    # Adding constraints as per the given code

    # Power balance constraints
    for t in range(num_time_periods):
        problem += w_t[t] - c_t[t] + d_t[t] == D

    # Wind generation constraints
    for t in range(num_time_periods):
        problem += w_t[t] <= CF_t[t] * w

    # Battery charge constraints
    for t in range(num_time_periods):
        problem += c_t[t] <= b

    # Battery discharge constraints
    for t in range(num_time_periods):
        problem += d_t[t] <= b

    # Battery state of charge constraints
    for t in range(1, num_time_periods):
        problem += b_t[t] == b_t[t - 1] + EFF * c_t[t] - d_t[t]
        problem += b_t[t] <= b
        problem += b_t[0] == EFF * c_t[0] - d_t[0]  # Initial state of charge

    # Solve the optimization problem
    problem.solve()

    # Get the optimal values for w_t
    optimal_w_t = [w_t_var.value() for w_t_var in w_t]

    # Calculate wind supply curtailment
    wind_supply_curtailment = [CF_t[t] * w.value() - optimal_w_t[t] for t in range(num_time_periods)]
    d_t = [d_t[t].value() for t in range(num_time_periods)]

    if monthly_aggregation:
        wind_supply_curtailment = [sum(wind_supply_curtailment[month_start:month_start + hours]) for month_start, hours
                                   in zip(range(0, num_time_periods, 730),
                                          [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])]
        d_t = [sum(d_t[month_start:month_start + hours]) for month_start, hours
                                   in zip(range(0, num_time_periods, 730),
                                          [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])]

        CF_t = [sum(CF_t[month_start:month_start + hours]) for month_start, hours
                                   in zip(range(0, num_time_periods, 730),
                                          [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])]

    elif yearly_aggregation:
        total_yearly_curtailment = sum(wind_supply_curtailment)
        average_monthly_curtailment = total_yearly_curtailment / 12
        wind_supply_curtailment = [average_monthly_curtailment for _ in range(12)]

        total_yearly_discharge = sum(d_t)
        average_monthly_discharge = total_yearly_discharge / 12
        d_t = [ average_monthly_discharge for _ in range(12)]

        total_yearly_capacity = sum(CF_t)
        average_monthly_discharge = total_yearly_capacity / 12
        CF_t = [average_monthly_discharge for _ in range(12)]



    return w.value(), b.value(), wind_supply_curtailment, d_t, [w.value()*i for i in CF_t]


def cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Function to clean the given data file by retaining only 'time' and 'electricity' columns.
    The 'time' column is converted to a proper datetime format.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'time' and 'electricity' columns.
    """
    # Reading the CSV file while skipping the header lines
    data = pd.read_csv(file_path, skiprows=3)

    # Keeping only 'time' and 'electricity' columns
    cleaned_data = data[['time', 'electricity']].copy()

    # Converting 'time' to datetime format
    cleaned_data['time'] = pd.to_datetime(cleaned_data['time'])

    return cleaned_data


def aggregate_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates hourly capacity factor data into monthly data.

    Args:
    - data (pd.DataFrame): DataFrame containing 'time' and 'electricity' columns.

    Returns:
    - pd.DataFrame: Aggregated monthly data with 'time' (first day of the month) and mean 'electricity' columns.
    """
    # Convert 'time' to datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Set 'time' as the index for resampling
    data.set_index('time', inplace=True)

    # Resample and calculate the monthly mean
    monthly_data = data['electricity'].resample('M').sum().reset_index()

    return monthly_data

def aggregate_yearly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates hourly capacity factor data into yearly data.

    Args:
    - data (pd.DataFrame): DataFrame containing 'time' and 'electricity' columns.

    Returns:
    - pd.DataFrame: Aggregated yearly data with 'time' (last day of the year) and sum 'electricity' columns.
    """
    # Convert 'time' to datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Set 'time' as the index for resampling
    data.set_index('time', inplace=True)

    # Resample and calculate the yearly sum
    yearly_data = data['electricity'].resample('Y').sum().reset_index()

    return yearly_data


def open_files(folder_path):
    file_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            file_paths.append(file_path)

    store_dict = {}

    for file in file_paths:
        data = cleaned_data(file)
        store_dict[file] = data

    return store_dict


def find_key(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None


def write_to_csv(df, file_path):
    """
    Write a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to be written to CSV.
        file_path (str): Path to the CSV file where the data will be saved.

    Returns:
        None
    """
    df.to_excel(file_path, index=False)


years = [2023, 2030]

matchings = ['hourly', 'monthly', 'yearly']

electricity_demand = { #MW
    'AP AEC low': [913],
    'AP AEC high': [1007],
    'AP CCS': [41],
    'AP BH2S': [76.6]
}

F_a = 0.9

CAPEX_iter = ['high', 'low']

capex_scenarios_data = {
    '2023': {
        'high': {
            'wind': 1400,
            'battery': 1500
        },
        'low': {
            'wind': 1200,
            'battery': 800
        }},
    '2030': {
        'high': {
            'wind': 1200,
            'battery': 1200
        },
        'low': {
            'wind': 750,
            'battery': 450
        }}
}

# capacity_iterator = np.linspace(0,3,10)

opex_data = {
    'wind': 52.2,
    'battery_var': 8.63 / 2,
    'battery_fix': 27.745
}

EFF = 0.85  # round-trip efficiency battery

simulation_data = pd.DataFrame(columns=['time', 'matching', 'location', 'capex_description', 'technology',
                                        'wind_capacity', 'battery_capacity', 'curtailment', 'discharge', 'total_gen'])

y_values = []

for year in years:
    for matching in matchings:
        # Aggregate monthly if the matching is monthly
        folder_path = r'Wind data/Capacity factor data/{}'.format(year)
        open_data = open_files(folder_path)

        if matching == 'monthly':
            location_data = {k: aggregate_monthly(v) for k, v in
                         zip(open_data.keys(), open_data.values())}
        elif matching == 'yearly':
            location_data = {k: aggregate_yearly(v) for k, v in
                             zip(open_data.keys(), open_data.values())}
        else:
            location_data = open_data

        # Find the two locations with the best and worst performance
        location_data_sum = {k: i['electricity'].sum() for k, i in zip(location_data.keys(), location_data.values())}
        locations = {'high': find_key(location_data_sum, max(location_data_sum.values())),
                     'low': find_key(location_data_sum, min(location_data_sum.values()))}

        for location in locations.values():
            for capex_desc in CAPEX_iter:
                for electricity in electricity_demand.values():
                    technology = find_key(electricity_demand, electricity)

                    # print(location)
                    COST_wind = capex_scenarios_data[str(year)][capex_desc]['wind'] * 1000
                    COST_battery = capex_scenarios_data[str(year)][capex_desc]['battery'] * 1000 / 4
                    T = location_data[location].index
                    CF_t = list(location_data[location]['electricity'])
                    num_time_periods = len(T)
                    num_variables = 4 * num_time_periods + 2
                    x0 = np.ones(num_variables)
                    print("HEREEEEEEEEEEEEE", location, sum(CF_t), year)
                    if matching == 'hourly':
                        D = electricity
                        wind_capacity, battery_capacity, curtailment, discharge, total_gen = optimize_energy_system(COST_wind, COST_battery,
                                                                                          EFF,
                                                                                          D, num_time_periods, CF_t,
                                                                                          monthly_aggregation=True)
                    elif matching == 'monthly':
                        D = electricity[0] * 365 * 24 / 12 * F_a
                        wind_capacity, battery_capacity, curtailment, discharge, total_gen = optimize_energy_system(COST_wind, COST_battery,
                                                                                              EFF,
                                                                                              D, num_time_periods, CF_t)
                    else:
                        D = electricity[0] * 365 * 24 * F_a
                        wind_capacity, battery_capacity, curtailment, discharge, total_gen = optimize_energy_system(COST_wind, COST_battery,
                                                                                              EFF,
                                                                                              D, num_time_periods, CF_t,
                                                                                              yearly_aggregation=True)


                    #
                    # if year == 2023 and matching == 'hourly' and location == list(location_data_sum.keys())[0] and capex_desc == 'low' and electricity == electricity_demand['AP AEC high']:
                    #     for val in capacity_iterator:
                    # #time period MW, MWh, MWh/unit
                    #         wind_capacity, battery_capacity, curtailment = optimize_energy_system(COST_wind, COST_battery*val, EFF,
                    #                                                                       D, num_time_periods, CF_t,
                    #                                                                       monthly_aggregation=True if matching == 'hourly' else False)
                    #         print(year, matching, find_key(locations, location), capex_desc, technology, battery_capacity)
                    #         y_values.append(wind_capacity)


                    data_row = [year, matching, find_key(locations, location), capex_desc, technology, wind_capacity,
                                battery_capacity, str(curtailment), str(discharge), str(total_gen)]

                    simulation_data.loc[len(simulation_data)] = data_row

file_return = 'Wind data/optimization_results.xlsx'
write_to_csv(simulation_data, file_return)
#
# plt.plot(capacity_iterator, y_values)
# plt.show()