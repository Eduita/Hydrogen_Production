scenarios = ['B', 'C', 'D']
technologies = ['AP SMR', 'AP CCS', 'AP BH2S', 'AP AEC']
times = [2023, 2030]
starts = [0, (times[1] - times[0])*12]
matching_type = ['yearly', 'monthly', 'hourly']
policies = [True, False]
L = 40*12

#Revelant folder paths
optimization_results_file_path = r"Wind data\optimization_results.xlsx"
baseline_model_inputs = r"input_parameters.json"
electric_grid_carbon_intensity_data_file_path = r"AEO22_AEO23_energy_mix_fraction.xlsx"
deterministic_model_inputs =  r"input_parameters_deterministic.json"