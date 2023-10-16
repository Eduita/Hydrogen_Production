import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast

def load_excel_sheets(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Dictionary to store the DataFrames for each sheet
    dataframes = {}

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Store the DataFrame in the dictionary
        dataframes[sheet_name] = df

    return dataframes


root_path = 'TC_FINAL_alldata_v10'
matchings = ['yearly', 'monthly', 'hourly']
CBAM = ['', '_CBAM']
times = [2023, 2030]
scenarios = ['A', 'B', 'C', 'D']
technologies = ['AP_SMR', 'AP_CCS', 'AP_BH2S', 'AP_AEC']
CI_technologies = ['AP SMR', 'AP CCS', 'AP BH2S', 'AP AEC']
metrics = ['_NPV', '_CAC', '_Potential', '_CE', '_CAPEX', '_OPEX', '_support_45V', '_support_45Q', '_support_45Y',
           '_support_48E']


datafiles = {}

for matching in matchings:
    for C in CBAM:
        temp = C if matching == 'monthly' else ''
        baselineData = load_excel_sheets(root_path + temp + '_' + matching + '.xlsx')
        datafiles[matching + temp] = baselineData

# Clean the data into scenarios and timeframe
cleaned_datafiles_with_matching = {time: {} for time in times}

# #Manually get CI Files
# CI_data = {
#     'AP CCS': datafiles[list(datafiles.keys())[0]]['CI_AP CCS'],
#     'AP BH2S': datafiles[list(datafiles.keys())[0]]['CI_AP BH2S'],
#     'AP AEC': datafiles[list(datafiles.keys())[0]]['CI_AP AEC'],
#     'AP SMR': datafiles[list(datafiles.keys())[0]]['CI_AP SMR']
# }

# Process each file and clean the data with additional checks for columns
for key, file_data in datafiles.items():
    for sheet_name, df in file_data.items():
        matching_type = key.split("_")[-1]  # Extract the matching type from the key

        # Check if the sheet name contains "CI_"
        if sheet_name.startswith("CI_"):
            # Determine the technology based on the sheet name
            technology = sheet_name.replace("CI_", "").replace(" ", "_")

            # For each time in the dataframe
            for time in df['time'].unique():
                if matching_type not in cleaned_datafiles_with_matching[time]:
                    cleaned_datafiles_with_matching[time][matching_type] = {}
                if "CI" not in cleaned_datafiles_with_matching[time][matching_type]:
                    cleaned_datafiles_with_matching[time][matching_type]["CI"] = {}
                cleaned_datafiles_with_matching[time][matching_type]["CI"][technology] = df[df['time'] == time].copy()

        elif 'time' in df.columns and 'scenario' in df.columns:  # Check if the columns exist before grouping
            # Group by time and scenario
            grouped = df.groupby(['time', 'scenario'])

            for (time, scenario), group_df in grouped:
                if matching_type not in cleaned_datafiles_with_matching[time]:
                    cleaned_datafiles_with_matching[time][matching_type] = {}
                if scenario not in cleaned_datafiles_with_matching[time][matching_type]:
                    cleaned_datafiles_with_matching[time][matching_type][scenario] = {}
                cleaned_datafiles_with_matching[time][matching_type][scenario][sheet_name] = group_df.copy()

for time_key, matching_data in cleaned_datafiles_with_matching.items():
    for matching_type, scenario_data in matching_data.items():
        if "CI" in scenario_data:
            CI_data = scenario_data["CI"]
            for tech, tech_df in CI_data.items():
                grouped = tech_df.groupby(['time', 'scenario'])
                for (time, scenario), group_df in grouped:
                    if scenario not in scenario_data:
                        scenario_data[scenario] = {}
                    if f"CI_{tech}" not in scenario_data[scenario]:
                        scenario_data[scenario][f"CI_{tech}"] = {}
                    scenario_data[scenario][f"CI_{tech}"] = group_df.copy()
            del scenario_data["CI"]  # Remove the original ungrouped CI data


def adapted_box_plot_on_ax(data_dict, time, scenario, matching, ax,
                           title='', show_legend=True, bottom=True, left=True, top=False, colors=None):
    """
    Creates a box plot on the provided axis object based on the specified data, time, scenario, and matching.

    Parameters:
    - data_dict: Dictionary containing the cleaned data.
    - time: Year for which the box plot needs to be generated.
    - scenario: Scenario (e.g., 'A', 'B') for which the box plot needs to be generated.
    - matching: Matching type (e.g., 'yearly', 'monthly') for which the box plot needs to be generated.
    - ax: Matplotlib axis object on which the box plot will be created.
    - title: Title for the subplot.
    - show_legend: Whether to display the legend.
    - bottom: Whether to display the x-axis labels at the bottom.
    - left: Whether to display the y-axis labels on the left.
    - top: Whether to display the title at the top.
    - colors: Dictionary for custom color mapping.

    Returns:
    - None
    """

    # Extract data for the given time, scenario, and matching
    data = data_dict[time][matching][scenario]["TC"]

    # Custom color palette
    if colors is None:
        colors = {"Cash-Equivalent": "gray", "Potential": "lightblue"}

    # Extract columns for potential and cash-equivalent
    potential_cols = [col for col in data.columns if "Potential" in col]
    cash_eq_cols = [col for col in data.columns if "_CE" in col]

    # Create a DataFrame to store data for box plotting
    box_plot_data = []

    # Add data for potential and cash-equivalent
    for ce_col, pot_col in zip(cash_eq_cols, potential_cols):
        label = pot_col.split("_")[1]  # Extract technology
        box_plot_data.append(pd.DataFrame(
            {'Value': data[ce_col], 'Group': [label] * len(data), 'Type': ["Cash-Equivalent"] * len(data)}))
        box_plot_data.append(
            pd.DataFrame({'Value': data[pot_col], 'Group': [label] * len(data), 'Type': ["Potential"] * len(data)}))


    # Concatenate the DataFrames
    box_plot_data = pd.concat(box_plot_data)

    # Create the box plot
    sns.boxplot(x='Group', y='Value', hue='Type', data=box_plot_data, ax=ax, palette=colors, width=0.5)
    ax.set_xlabel('')
    ax.set_title(title if top else '', fontweight='bold', fontsize=20)
    ax.set_xticklabels([label.get_text().replace('_', ' ') for label in ax.get_xticklabels()] if bottom else [])
    ax.set_ylabel(r'Total Policy Support [$2023\$ $]'+f',{time+3}' if left else '', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(0, 28)
    ax.legend(loc='lower right')

    for _, spine in ax.spines.items():
        spine.set_color('black')

    if not show_legend:
        ax.get_legend().remove()

time = 2023
matching = 'monthly'

colors = ["gray", "#4169E1", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 3, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.22, 'hspace': 0.15})

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'B',matching, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False, left= True, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'C',matching, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'D',matching, ax=axs[0, 2], title='Scenario C', show_legend=True, bottom=False, left= False, top=True)

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'B',matching, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True, left= True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'C',matching, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'D',matching, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False)

plt.savefig(f"FINAL final figures/TC Baseline.png", dpi=300)


time = 2023
matching = 'yearly'

colors = ["gray", "#4169E1", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 3, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.22, 'hspace': 0.15})

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'B',matching, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False, left= True, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'C',matching, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'D',matching, ax=axs[0, 2], title='Scenario C', show_legend=True, bottom=False, left= False, top=True)

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'B',matching, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True, left= True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'C',matching, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'D',matching, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False)

plt.savefig(f"FINAL final figures/TC Yearly.png", dpi=300)

time = 2023
matching = 'hourly'

colors = ["gray", "#4169E1", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 3, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.22, 'hspace': 0.15})

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'B',matching, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False, left= True, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'C',matching, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time, 'D',matching, ax=axs[0, 2], title='Scenario C', show_legend=True, bottom=False, left= False, top=True)

adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'B',matching, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True, left= True)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'C',matching, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False)
adapted_box_plot_on_ax(cleaned_datafiles_with_matching, time+7, 'D',matching, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False)

plt.savefig(f"FINAL final figures/TC Hourly.png", dpi=300)



