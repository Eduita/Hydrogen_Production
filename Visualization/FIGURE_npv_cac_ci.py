import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast
# Load the data files
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


root_path = 'TC_FINAL_alldata_v12'
matchings = ['yearly', 'monthly', 'hourly']
CBAM = ['', '_CBAM']
times = [2023, 2030]
scenarios = ['B', 'C', 'D']
technologies = ['AP_SMR', 'AP_CCS', 'AP_BH2S', 'AP_AEC']
CI_technologies = ['AP SMR', 'AP CCS', 'AP BH2S', 'AP AEC']
metrics = ['_NPV', '_CAC']


datafiles = {}

for matching in matchings:
    for C in CBAM:
        temp = C if matching == 'monthly' else ''
        baselineData = load_excel_sheets(root_path + temp + '_' + matching + '.xlsx')
        datafiles[matching + temp] = baselineData

# Clean the data into scenarios and timeframe
cleaned_datafiles_with_matching = {time: {} for time in times}

#Manually get CI Files
CI_data = {
    'AP CCS': datafiles[list(datafiles.keys())[0]]['CI_AP CCS'],
    'AP BH2S': datafiles[list(datafiles.keys())[0]]['CI_AP BH2S'],
    'AP AEC': datafiles[list(datafiles.keys())[0]]['CI_AP AEC'],
    'AP SMR': datafiles[list(datafiles.keys())[0]]['CI_AP SMR']
}

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

# {2023: {'Scenarios': ['A', 'B', 'C', 'D', 'CI'], 'Technologies under CI': ['AP_SMR', 'AP_CCS', 'AP_BH2S', 'AP_AEC']}, 2030: {'Scenarios': ['A', 'B', 'C', 'D', 'CI'], 'Technologies under CI': ['AP_SMR', 'AP_CCS', 'AP_BH2S', 'AP_AEC']}}

# Plotting of input effect


# Plotting CI charts
def plot_sheet_subplot(matching,scenario, colors, ax, alpha=0.2, time=2023, show_legend=False, left=False, top=False, right=False, annotate=False, ylim=None, annotate_CBAM=False):


    # Increase the font size of the x and y axis

    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=16)
    n=0
    for index, tech in enumerate(technologies):
        data = cleaned_datafiles_with_matching[2023][matching][scenario]['CI_'+tech]
        x_values = [int(i) for i in list(data.columns[3:])]
        data = data[data.columns[3:]]

        mean_values = data.mean()
        min_values = data.min()
        max_values = data.max()

        ax.plot(x_values, mean_values.values, color=colors[index], label=tech)
        ax.fill_between(x_values, min_values.values, max_values.values, color=colors[index], alpha=alpha)


            # ax2.text(2027, 6.95, 'EU AP Reference', ha='left', va='baseline', rotation=-10, fontweight='bold')

    # ax.yaxis.set_ticks_position('none')
    # ax.set_xlabel('Year', fontweight='bold')
    ax.set_xlim(2023, 2050)
    # ax.yaxis.set_visible(False)

    ax.axhline(y=4, color='gray', linestyle='--', linewidth=1, alpha=0.2)

    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=1, alpha=0.2)

    ax.axhline(y=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.2)

    ax.axhline(y=0.45, color='gray', linestyle='--', linewidth=1, alpha=0.2)

    if annotate:
        BAM_correct = 11 if matching == 'CBAM' else 0
        x1, y1 = 2029+BAM_correct, 10.2
        x2, y2 = 2029+BAM_correct, 1.10

        ax.annotate('45Q',
                    xy=(x2, y2),  # Arrow end point
                    xytext=(x1, y1),  # Arrow start point (also the text location)
                    arrowprops=dict(facecolor='black', arrowstyle='<->', linestyle='dashed', alpha=0.5),  # Arrow properties
                    ha='center', va='center')

    if annotate_CBAM == True and matching == 'CBAM':
        x1, y1 = 2041, 10.2
        x2, y2 = 2041, 6.5

        # ax2.annotate('CBAM',
        #              xy=(x2, y2),  # Arrow end point
        #              xytext=(x1, y1),  # Arrow start point (also the text location)
        #              arrowprops=dict(facecolor='black', arrowstyle='<->', linestyle='dashed', alpha=0.5),
        #              # Arrow properties
        #              ha='center', va='center')

    if right:
        ax.text(2024, 0.55, r'3.0 $/Kg H2', ha='left', va='center', rotation=0, fontsize=10)
        ax.text(2024, 1.5, r'1.0 $/Kg H2', ha='left', va='center', rotation=0, fontsize=10)
        ax.text(2024, 2.5, r'0.75 $/Kg H2', ha='left', va='center', rotation=0, fontsize=10)
        ax.text(2024, 4.0, r'0.60 $/Kg H2', ha='left', va='center', rotation=0, fontsize=10)

    if matching == 'CBAM' and n == 0:
        calculation = [8.82 * (1 - 0.014) ** (i - x_values[0]) for i in x_values]
        ax.plot(x_values, calculation, color='gray',
                 linestyle=':', linewidth=1.5, label="EU AP SMR Reference")
        n += 1

    if top:
        ax.set_ylim(0, 33)
    else:
        ax.set_ylim(0, 33)

    if left:
        ax.set_ylabel(r'Carbon Intensity [$\frac{kgCO_2 \ eq}{Kg \ H_2}$]', fontweight='bold', fontsize=14, labelpad=1)

    if show_legend:

        legend = ax.legend()
        for label in legend.get_texts():
            label.set_text(label.get_text().replace("_", " "))




    ax.set_xticks(np.arange(2023, 2050, 6))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=16)

# fig, axs = plt.subplots(1, 2, figsize=(10, 5),gridspec_kw={'width_ratios': [1.1, 1.1], 'height_ratios': [1], 'wspace': 0.22, 'hspace': 0.15})

# colors = ["gray", "#4169E1", "#FF4500", "#008000"]
#
# plot_sheet_subplot('monthly','B', colors, axs[0], top=True)
# axs[0].set_title('Scenario A', fontweight='bold')
# plot_sheet_subplot('monthly','C', colors, axs[1], show_legend=True, top=True, right=True, left=True)
# axs[1].set_title('Scenarios B & C', fontweight='bold')
#
# plt.savefig("FINAL final figures/CI_base.png", dpi = 300)
# plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 5),gridspec_kw={'width_ratios': [1.1, 1.1], 'height_ratios': [1], 'wspace': 0.22, 'hspace': 0.15})

colors = ["gray", "#4169E1", "#FF4500", "#008000"]

plot_sheet_subplot('CBAM','B', colors, axs[0], top=True, annotate_CBAM=True, left=True)
axs[0].set_title('Scenario A', fontweight='bold', fontsize=18)
plot_sheet_subplot('CBAM','C', colors, axs[1], show_legend=True, top=True, right=True, left=False)
axs[1].set_title('Scenarios B & C', fontweight='bold', fontsize=18)

plt.savefig("FINAL final figures/CI_CBAM v2.png", dpi = 500)
plt.close()

# Plotting NPVs
def compare_policy_scenarios_subplot(time, matching, scenario, colors, ax, title, show_legend=True, bottom=True,
                                     left=True, top=False):
    """
    Function to compare the policy scenarios and plot them.
    """
    # Fetch the data based on the provided time, matching, and scenario
    scenario_data = cleaned_datafiles_with_matching[time][matching][scenario]
    with_policy_df = scenario_data.get("NPV")
    no_policy_df = scenario_data.get("NP_NPV")

    if with_policy_df is None or no_policy_df is None:
        raise ValueError(
            f"Data for the given scenario: {scenario} and matching: {matching} at time: {time} is missing.")

    # Create a copy of the dataframes to avoid the SettingWithCopyWarning
    no_policy_df = no_policy_df.copy()
    with_policy_df = with_policy_df.copy()



    # Add a 'Scenario' column to each DataFrame
    no_policy_df['Scenario'] = 'No Policy'
    with_policy_df['Scenario'] = 'IRA'

    # Combine the two DataFrames
    combined_df = pd.concat([no_policy_df, with_policy_df])

    # Melt the dataframe to long format for seaborn boxplot
    column_names = ['AP_CCS'+metrics[0], 'AP_BH2S'+metrics[0], 'AP_AEC'+metrics[0]]

    #'AP_SMR'+metrics[0]
    # Calculate the quartiles and Interquartile Range (IQR)
    reference = no_policy_df['AP_SMR'+metrics[0]]

    q1 = reference.quantile(0.25)
    q2 = reference.quantile(0.5)
    q3 = reference.quantile(0.75)
    iqr = q3-q1
    q0 = q1-1.5*iqr
    q5 = q3+1.5*iqr

    ax.axhline(y=q1, color='gray', linestyle='--', linewidth=2, alpha=0.25)
    ax.axhline(y=q2, color='gray', linestyle='--', linewidth=2, alpha=0.25)
    ax.axhline(y=q3, color='gray', linestyle='--', linewidth=2, alpha=0.25)
    ax.axhline(y=q0, color='gray', linestyle='--', linewidth=2, alpha=0.25)
    ax.axhline(y=q5, color='gray', linestyle='--', linewidth=2, alpha=0.25)
    ax.annotate('AP SMR', xy=(0.5, 3), xytext=(-.5 * 0.95, 1.05 * q5), color='gray')

    # Add filled areas between quartiles
    x = np.linspace(-10, 10, 100)
    y1 = np.full_like(x, q2)
    y2 = np.full_like(x, q3)
    ax.fill_between(x, y1, y2, color='gray', alpha=0.1, edgecolor='black', hatch='////')
    y1 = np.full_like(x, q1)
    ax.fill_between(x, y1, y2, color='gray', alpha=0.1, edgecolor='black', hatch='////')

    melted_df = pd.melt(combined_df, id_vars='Scenario', value_vars=column_names,
                        var_name='Column', value_name='Value')

    # Create the boxplot
    sns.boxplot(data=melted_df, x='Column', y='Value', hue='Scenario', palette=colors, showfliers=False, width=0.5,
                ax=ax)

    if matching == 'monthly' or matching == 'CBAM':
        ranges = {
            'A': [-300, 300],
            'B': [-300, 200],
            'C': [-800, 300],
            'D': [-300, 300]
        }
        ax.set_ylim(*ranges[scenario])
    elif matching in ['hourly', 'yearly']:
        ranges = {
            'A': [-200, 400],
            'B': [-200, 400],
            'C': [-3500, 400],
            'D': [-1000, 400]
        }
        ax.set_ylim(*ranges[scenario])

    ax.set_xlim(-0.5,2.5)
    if not show_legend:
        ax.get_legend().remove()

    # Set labels and title
    ax.set_ylabel('Value' if left else '', fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=16)
    if matching in ['hourly', 'yearly']:
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.tick_params(axis='x', which='major', labelsize=16)

    if top:
        ax.set_title(title, fontweight='bold', fontsize=18)

    # Rotate x-axis labels if needed
    updated_labels = [label.replace('_', ' ').replace('AP', '').replace('NPV','') for label in column_names]
    if not bottom:
        ax.set_xticklabels([''] * len(updated_labels))
    else:
        ax.set_xticklabels(updated_labels, fontweight='bold')
    if left:
        ax.set_ylabel(r'NPV [$\frac{2023\$}{Tonne \ NH_3}$], '+ str(time+3), fontweight='bold', fontsize=14)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], loc='lower left', facecolor='white', edgecolor='black')

    return ax

    # Test the function with a sample data


colors = ["gray", "#b8f4ff", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 3, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.25, 'hspace': 0.15})

compare_policy_scenarios_subplot(2023, 'monthly', 'B', colors, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False, left= True, top=True)
compare_policy_scenarios_subplot(2023, 'monthly', 'C', colors, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
compare_policy_scenarios_subplot(2023, 'monthly', 'D', colors, ax=axs[0, 2], title='Scenario C', show_legend=True, bottom=False, left= False, top=True)

compare_policy_scenarios_subplot(2030, 'monthly', 'B', colors, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True, left= True)
compare_policy_scenarios_subplot(2030, 'monthly', 'C', colors, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False)
compare_policy_scenarios_subplot(2030, 'monthly', 'D', colors, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False)


plt.savefig("FINAL final figures/NPV Baseline v2.png", dpi=300)
plt.close()

colors = ["gray", "#b8f4ff", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 4, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.25, 'hspace': 0.15})

#Row 1
compare_policy_scenarios_subplot(2023, 'hourly', 'C', colors, ax=axs[0, 0], title='HOURLY Scenario B', show_legend=False, bottom=False, left= True, top=True)
compare_policy_scenarios_subplot(2023, 'yearly', 'C', colors, ax=axs[0, 1], title='YEARLY Scenario B', show_legend=False, bottom=False, left= False, top=True)
compare_policy_scenarios_subplot(2023, 'hourly', 'D', colors, ax=axs[0, 2], title='HOURLY Scenario C', show_legend=False, bottom=False, left= False, top=True)
compare_policy_scenarios_subplot(2023, 'yearly', 'D', colors, ax=axs[0, 3], title='YEARLY Scenario C', show_legend=True, bottom=False, left= False, top=True)

#Row 2
compare_policy_scenarios_subplot(2030, 'hourly', 'C', colors, ax=axs[1, 0], title='Scenario B', show_legend=False, bottom=True,left= True)
compare_policy_scenarios_subplot(2030, 'yearly', 'C', colors, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False)
compare_policy_scenarios_subplot(2030, 'hourly', 'D', colors, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False)
compare_policy_scenarios_subplot(2030, 'yearly', 'D', colors, ax=axs[1, 3], title='Scenario C', show_legend=False, bottom=True, left= False)


plt.savefig("FINAL final figures/NPV Matching v2.png", dpi=300)
plt.close()

colors = ["gray", "#b8f4ff", "#FF4500", "#008000"]
fig, axs = plt.subplots(2, 3, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.25, 'hspace': 0.15})

compare_policy_scenarios_subplot(2023, 'CBAM', 'B', colors, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False, left=True , top=True)
compare_policy_scenarios_subplot(2023, 'CBAM', 'C', colors, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
compare_policy_scenarios_subplot(2023, 'CBAM', 'D', colors, ax=axs[0, 2], title='Scenario C', show_legend=True, bottom=False, left= False, top=True)

compare_policy_scenarios_subplot(2030, 'CBAM', 'B', colors, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True, left= True, top=True)
compare_policy_scenarios_subplot(2030, 'CBAM', 'C', colors, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False, top=True)
compare_policy_scenarios_subplot(2030, 'CBAM', 'D', colors, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False, top=True)


plt.savefig("FINAL final figures/NPV CBAM v2.png", dpi=300)
plt.close()

# colors = ["gray", "#4169E1", "#FF4500", "#008000"]
# fig, axs = plt.subplots(2, 4, figsize=(14, 11),gridspec_kw={'width_ratios': [1.1, 1.1, 1.1, 1.1], 'height_ratios': [1, 1], 'wspace': 0.22, 'hspace': 0.15})
#
# compare_policy_scenarios_subplot(2023, 'yearly', 'A', colors, ax=axs[0, 0], title='Scenario A', show_legend=False, bottom=False,left= True, top=True)
# compare_policy_scenarios_subplot(2023, 'yearly', 'B', colors, ax=axs[0, 1], title='Scenario B', show_legend=False, bottom=False, left= False, top=True)
# compare_policy_scenarios_subplot(2023, 'yearly', 'C', colors, ax=axs[0, 2], title='Scenario C', show_legend=False, bottom=False, left= False, top=True)
# compare_policy_scenarios_subplot(2023, 'yearly', 'D', colors, ax=axs[0, 3], title='Scenario D', show_legend=True, bottom=False, left= False, top=True)
#
# compare_policy_scenarios_subplot(2030, 'yearly', 'A', colors, ax=axs[1, 0], title='Scenario A', show_legend=False, bottom=True,left= True, top=True)
# compare_policy_scenarios_subplot(2030, 'yearly', 'B', colors, ax=axs[1, 1], title='Scenario B', show_legend=False, bottom=True, left= False, top=True)
# compare_policy_scenarios_subplot(2030, 'yearly', 'C', colors, ax=axs[1, 2], title='Scenario C', show_legend=False, bottom=True, left= False, top=True)
# compare_policy_scenarios_subplot(2030, 'yearly', 'D', colors, ax=axs[1, 3], title='Scenario D', show_legend=False, bottom=True, left= False, top=True)
#
#
# plt.savefig("Final figures/V10_2pecennt_NPVs figure yearly.png", dpi=300)
# plt.close()

# Plotting CACs
def compare_cac_scenarios_subplot(time, matching, scenario, colors, ax, title, show_legend=True, bottom=True,
                                  left=True):
    """
    Function to compare the CAC of different technologies for given policy and no policy scenarios.
    """
    # Fetch the data based on the provided time, matching, and scenario
    scenario_data = cleaned_datafiles_with_matching[time][matching][scenario]
    cac_data = scenario_data.get("CAC")
    npv_data = scenario_data.get("NPV")

    if cac_data is None:
        raise ValueError(
            f"Data for the given scenario: {scenario} and matching: {matching} at time: {time} is missing.")

    # Create a copy of the dataframe to avoid the SettingWithCopyWarning
    cac_data = cac_data.copy()
    npv_data = npv_data.copy()

    # Define the technologies to plot
    technologies = ['AP_CCS'+metrics[1], 'AP_BH2S'+metrics[1], 'AP_AEC'+metrics[1]]
    tech_npvs = ['AP_CCS' + metrics[0], 'AP_BH2S' + metrics[0], 'AP_AEC' + metrics[0]]
    ax.set_ylim(-800, 450) #-600
    if matching == 'hourly' and scenario == 'C':
        ax.set_ylim(-800, 450) #-2200

    midpoint = np.mean(ax.get_ylim())

    ax.fill_betweenx(ax.get_ylim(), 18, 20, color='gray', alpha=0.2)
    if matching == 'hourly' and scenario == 'C':
        ax.text(15, midpoint, 'California 2020-2022', ha='center', va='center', rotation=90, fontweight='bold', fontsize=8)
    else:
        ax.text(15, midpoint, 'California 2020-2022', ha='center', va='center', rotation=90, fontweight='bold', fontsize=8)

    if matching == 'hourly' and scenario == 'C':
        ax.text(215, midpoint, '2020-2030 SC of Carbon (2.0% Discount)', ha='center', va='center', rotation=90,
                fontweight='bold', fontsize=8)
    else:
        ax.text(215, midpoint, '2020-2030 SC of Carbon (2.0% Discount)', ha='center', va='center', rotation=90,
                fontweight='bold', fontsize=8)
    ax.fill_betweenx(ax.get_ylim(), 190, 230, color='gray', alpha=0.2)

    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    if matching == 'hourly' and scenario == 'C':
        ax.text(60, midpoint, 'EU 2020-2022', ha='center', va='center', rotation=90, fontweight='bold', fontsize=8)
    else:
        ax.text(60, midpoint, 'EU 2020-2022', ha='center', va='center', rotation=90, fontweight='bold', fontsize=8)
    ax.fill_betweenx(ax.get_ylim(), 30, 89, color='gray', alpha=0.2)

    markers = ['o', '^', 'X']

    ax.grid(False)
    # For each technology
    for tech, color, tech_npv, marker in zip(technologies, colors, tech_npvs, markers):
        # Get the CAC data for the technology
        tech_data = cac_data[tech]
        tech_data_npv = npv_data[tech_npv]

        # Create a scatter plot for the technology CAC data
        ax.scatter(tech_data, tech_data_npv, label=tech.replace('_', ' '), color=color, s=30, alpha=0.05, edgecolor=color, linewidths=0)

    # Set labels and title

    ax.set_title(title, fontweight='bold', fontsize=18, pad=1.5)
    ax.set_xlim(0, 450) #upper bound: 250
    if matching == 'hourly' and scenario == 'C':
        ax.set_xlim(0, 450)
    ax.set_xticks(np.arange(0, 450, 100))

    if bottom:
        ax.set_xlabel(r'CAC [$\mathbf{\frac{2023\$}{Tonne \ CO_2}}$]',
                      fontweight='bold', fontsize=11)

    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=16)

    if matching in ['hourly', 'yearly']:
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.tick_params(axis='x', which='major', labelsize=16)


    ax.tick_params(axis='y', labelleft=False, left=True, labelsize=13)
    if left:
        ax.set_ylabel(r'NPV [$\mathbf{\frac{2023\$}{Tonne \ NH_3}}$], '+ str(time+3), fontweight='bold', fontsize=14)
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=13)

    # Display the legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        labels = [i.replace(" CAC","") for i in labels]
        legend = ax.legend(handles=handles, labels=labels, loc='lower left', facecolor='white', edgecolor='black', bbox_to_anchor=(1.02, 0))
        for i in range(len(legend.legendHandles)):
            legend.legendHandles[i]._sizes = [75]
            legend.legendHandles[i].set_alpha(1)
        ax.set_yticks(np.arange(0, 450, 100))

    return ax


# Test the function with a sample data
fig, axs = plt.subplots(2, 3,figsize=(14, 10))
colors = ['blue', 'orange', 'green', '#008000']

compare_cac_scenarios_subplot(2023, 'monthly', 'B', colors, axs[0,0], 'Scenario A', show_legend=False, bottom=False, left=True)
compare_cac_scenarios_subplot(2023, 'monthly', 'C', colors, axs[0,1], 'Scenario B', show_legend=False, bottom=False, left=False)
compare_cac_scenarios_subplot(2023, 'monthly', 'D', colors, axs[0,2], 'Scenario C', show_legend=True, bottom=False, left=False)

compare_cac_scenarios_subplot(2030, 'monthly', 'B', colors, axs[1,0], '', show_legend=False, bottom=True, left=True)
compare_cac_scenarios_subplot(2030, 'monthly', 'C', colors, axs[1,1], '', show_legend=False, bottom=True, left=False)
compare_cac_scenarios_subplot(2030, 'monthly', 'D', colors, axs[1,2], '', show_legend=False, bottom=True, left=False)

plt.savefig("FINAL final figures/CAC Baseline v2.png", dpi=300)
plt.close()


# Test the function with a sample data
fig, axs = plt.subplots(2, 4,figsize=(14, 10))
colors = ['blue', 'orange', 'green', '#008000']

compare_cac_scenarios_subplot(2023, 'hourly', 'C', colors, axs[0,0], 'HOURLY Scenario B', show_legend=False, bottom=False, left=True)
compare_cac_scenarios_subplot(2023, 'yearly', 'C', colors, axs[0,1], 'YEARLY Scenario B', show_legend=False, bottom=False, left=False)
compare_cac_scenarios_subplot(2023, 'hourly', 'D', colors, axs[0,2], 'HOURLY Scenario C', show_legend=False, bottom=False, left=False)
compare_cac_scenarios_subplot(2023, 'yearly', 'D', colors, axs[0,3], 'YEARLY Scenario C', show_legend=True, bottom=False, left=False)

compare_cac_scenarios_subplot(2030, 'hourly', 'C', colors, axs[1,0], '', show_legend=False, bottom=True, left=True)
compare_cac_scenarios_subplot(2030, 'yearly', 'C', colors, axs[1,1], '', show_legend=False, bottom=True, left=False)
compare_cac_scenarios_subplot(2030, 'hourly', 'D', colors, axs[1,2], '', show_legend=False, bottom=True, left=False)
compare_cac_scenarios_subplot(2030, 'yearly', 'D', colors, axs[1,3], '', show_legend=False, bottom=True, left=False)

plt.savefig("FINAL final figures/CAC Matching v2.png", dpi=300)
plt.close()

