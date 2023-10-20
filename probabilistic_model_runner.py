#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy.random import uniform as uni
from numpy.random import seed as seed
import sys
import json
from _instantiate_inputs import InstantiateInputs
from _CAPEX import CAPEX
from _GBM import brownian_motion
from _MI_OPEX import MI_OPEX
from _CI_Calculator import Carbon_Intensity_of_technology
from _DCF_Model import Stochastic_DCF
from _get_capacities_CI_and_PPA_data import *

CBAM = False # False or True
matching = 'monthly' #options: hourly, monthly, and yearly
# NPV_dataset = run_simulation(100, NPV=True, isCBAM=CBAM)
mult = 8 #Multiplies the number of simulations

def run_simulation(simulations, NPV=False, CAC=False, CI=False, Potential_TC=False, NP_NPV=False, CAPEX_OPEX=False,
                   absolute_support=False, ROI=False, El_market=False, CAPEX_component=False, Sensitivity=False, isCBAM=False,
                   quality_assurance=False, matching='hourly', CAC_quality = False):
    """
    Args:
        isCBAM: Boolean, determines if CBAM is considered. Applies CBAM to SMR too.
        matching: String, options: 'hourly', 'monthly', or 'yearly', determines the type of electricity matching.
        simulations: Integer, Number of simulations desired.

        %%NOTE: The arguments below are mutually exclusive. When performing a run_simulation call,
        %%  you can only set one of the arguments as true.
            NPV: Boolean, determines if NPV is considered. MAIN PAPER METRIC.
            CAC: Boolean, determines if CAC is considered. MAIN PAPER METRIC.
            CI: Boolean, determines if CI is considered. MAIN PAPER METRIC.
            Potential_TC: Boolean, determines if Potential and CAPEX are considered. MAIN PAPER METRIC.
            absolute_support: Boolean, determines if absolute support is considered. MAIN PAPER METRIC.
            NP_NPV: Boolean, determines if NP_NPV is considered.NP_NPV is the NPV without IRA credits.
            CAPEX_OPEX: Boolean, determines if CAPEX and OPEX are considered. Deprecated. Ignore.
            ROI: Boolean, determines if ROI is considered. Deprecated. Ignore.
            El_market: Boolean, determines if El_market is considered.
            CAPEX_component: Boolean, determines if CAPEX component is considered.
            Sensitivity:  Boolean, determines if Sensitivity is considered. Deprecated. Ignore.
            quality_assurance: Boolean, determines if quality assurance is considered. Only for QA purposes.
            CAC_quality: Boolean, determines if CAC quality is considered. Only for QA purposes.

    Returns: A pandas dataframe containing the simulation results for various business model scenarios (see global variable)
    """

    """
    Below is a set of empty dataframes corresponding to the metrics of interest. These dataframes label the dataframe columns
    so that plotting becomes easier. 
    """
    NPV_data = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_SMR_NPV', 'AP_CCS_NPV', 'AP_BH2S_NPV', 'AP_AEC_NPV'])
    NP_NPV_data = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_SMR_NPV', 'AP_CCS_NPV', 'AP_BH2S_NPV', 'AP_AEC_NPV'])
    ROI_data = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_SMR_ROI', 'AP_CCS_ROI', 'AP_BH2S_ROI', 'AP_AEC_ROI'])
    CAC_data = pd.DataFrame(columns=['time', 'scenario', 'simulation', 'AP_CCS_CAC', 'AP_BH2S_CAC', 'AP_AEC_CAC'])
    El_data = pd.DataFrame(columns=['time', 'scenario', 'simulation', 'Grid Electricity'])

    multiples_of_12_CI = [12 * i for i in range(28)]
    years_CI = [2023 + i for i in range(28)]
    CI_columns = ['time', 'scenario', 'simulation']
    for time in years_CI:
        CI_columns.append(str(time))
    CI_data = {
        'AP SMR': pd.DataFrame(columns=CI_columns),
        'AP CCS': pd.DataFrame(columns=CI_columns),
        'AP BH2S': pd.DataFrame(columns=CI_columns),
        'AP AEC': pd.DataFrame(columns=CI_columns)
    }

    Potential_and_AP_CE_TC = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_CCS_Potential', 'AP_BH2S_Potential', 'AP_AEC_Potential',
                 'AP_CCS_CE', 'AP_BH2S_CE', 'AP_AEC_CE', 'AP_CCS_CE_social', 'AP_BH2S_CE_social', 'AP_AEC_CE_social'])

    OPEX_and_CAPEX = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_SMR_CAPEX', 'AP_CCS_CAPEX', 'AP_BH2S_CAPEX', 'AP_AEC_CAPEX',
                 'AP_SMR_OPEX', 'AP_CCS_OPEX', 'AP_BH2S_OPEX', 'AP_AEC_OPEX'])

    absolute_support_data = pd.DataFrame(
        columns=['time', 'scenario', 'simulation', 'AP_CCS_support_45V', 'AP_BH2S_support_45V', 'AP_AEC_support_45V',
                 'AP_CCS_support_45Q', 'AP_BH2S_support_45Q', 'AP_AEC_support_45Q', 'AP_CCS_support_45Y',
                 'AP_BH2S_support_45Y', 'AP_AEC_support_45Y', 'AP_CCS_support_48C', 'AP_BH2S_support_48C',
                 'AP_AEC_support_48C', 'AP_CCS_support_48E', 'AP_BH2S_support_48E',
                 'AP_AEC_support_48E'])

    CAPEX_component_data = pd.DataFrame(columns=[
        'time', 'scenario', 'sim', 'AP_SMR_NPV', 'AP_CCS_NPV', 'AP_BH2S_NPV', 'AP_AEC_NPV'
    ])

    times =  [2023, 2030]

    column_names = ['time', 'scenario', 'simulation', 'CAPEX', 'Electricity Cost', 'Plant-based OPEX', 'Feedstock Price','Ammonia Price',
                    'Carbon Intensity', 'Battery Variable O&M', 'Battery Fixed O&M', 'Battery CAPEX', 'Wind Turbine CAPEX',
                    '48C Credits', 'Tax Credit Market Value', 'PPA price', 'Curtailment', 'AEC Stack CAPEX', 'AEC Efficiency', 'CBAM Certificates']

    sensitivity_data = {tech:pd.DataFrame(columns=column_names) for tech in technologies}

    print("Current time:", get_current_est_time())

    for start in starts:
        if start == 0:
            time = 2023
        elif start == 84:
            time = 2030

        storage_dictionary = {}

        for scenario in scenarios:
            for sim in range(simulations):

                inter = int(sim)
                seed(inter)

                INPUT_PARAMETERS_PATH = baseline_model_inputs
                with open(INPUT_PARAMETERS_PATH, 'r') as json_file:
                    INPUT_PARAMETERS = json.load(json_file)
                INPUT_PARAMETERS_copy = INPUT_PARAMETERS.copy()
                #Instantiate inputs from JSON file
                instantiated_input = InstantiateInputs(inter)
                INSTANTIATED_MODEL_INPUTS = instantiated_input.randomness_from_JSON_inputs(INPUT_PARAMETERS_copy)

                # print(inter,  INSTANTIATED_MODEL_INPUTS)
                CAPEX_inputs = INSTANTIATED_MODEL_INPUTS['CAPEX_inputs']
                basic_equipment_costs = INSTANTIATED_MODEL_INPUTS['basic_equipment_costs']
                financial_inputs = INSTANTIATED_MODEL_INPUTS['financial_inputs']
                engineering_inputs = INSTANTIATED_MODEL_INPUTS['engineering_inputs']
                electricity_requirements = INSTANTIATED_MODEL_INPUTS['electricity_requirements']
                natural_gas_requirements = INSTANTIATED_MODEL_INPUTS['natural_gas_requirements']
                BFW_requirements = INSTANTIATED_MODEL_INPUTS['BFW_requirements']
                HP_steam_requirements = INSTANTIATED_MODEL_INPUTS['HP_steam_requirements']
                MI_OPEX_inputs = INSTANTIATED_MODEL_INPUTS['MI_OPEX_inputs']
                Market_inputs = INSTANTIATED_MODEL_INPUTS['Market_inputs']
                carbon_intensity = INSTANTIATED_MODEL_INPUTS['carbon_intensity']
                IRA_credits = INSTANTIATED_MODEL_INPUTS['IRA_credits']

                #Calculate additional inputs from datasets
                CAPEX_inputs['wind_capacity'] = {}
                CAPEX_inputs['battery_capacity'] = {}

                for tech in technologies[1:]:
                    if tech != 'AP AEC':
                        CAPEX_inputs['wind_capacity'][tech] = wind_and_battery_data[time][matching][tech]['wind_capacity']
                        CAPEX_inputs['battery_capacity'][tech] = wind_and_battery_data[time][matching][tech]['wind_capacity']
                    else:
                        CAPEX_inputs['wind_capacity'][tech] = uni(
                            wind_and_battery_data[time][matching][tech +" low"]['wind_capacity'],
                            wind_and_battery_data[time][matching][tech + " high"]['wind_capacity']
                        )
                        CAPEX_inputs['battery_capacity'][tech] = uni(
                            wind_and_battery_data[time][matching][tech + " low"]['battery_capacity'],
                            wind_and_battery_data[time][matching][tech + " high"]['battery_capacity']
                        )

                def back_calculate_depreciable_capital_factor(capex_inputs, excluded_factors):
                    keys = list(capex_inputs.keys())[1:12]
                    depreciable_capital_factor = 1
                    for key in keys:
                        if key not in excluded_factors:
                            depreciable_capital_factor += capex_inputs[key] * (1 / ((2717 / 100) ** 0.6))

                    capex_inputs['CAPEX_from_installed_cost'] = depreciable_capital_factor

                exclude_from_depreciation = ["Engineering and Supervision Cost", "Legal Expenses Cost",
                                             "Construction Expense and Contractors Fee Cost",
                                             "Working Capital", "Contingency Cost", "Land Cost"]

                back_calculate_depreciable_capital_factor(CAPEX_inputs, exclude_from_depreciation)

                financial_inputs['operating_hours_per_year'] = 365 * 24 * financial_inputs['availability']

                engineering_inputs['AP AEC H2_req'] = engineering_inputs['H2'] * engineering_inputs['H2 LHV'] / \
                                                      engineering_inputs['Eff_electrolysis'] / 24

                AP_AEC_curtailment_correlative_prob = uni(0,1)

                AP_AEC_curtailment_list = [low + AP_AEC_curtailment_correlative_prob*(high-low) for low,high in zip(wind_and_battery_data[time][matching]['AP AEC low']['curtailment'],
                                wind_and_battery_data[time][matching]['AP AEC high']['curtailment'])]

                AP_AEC_discharge_list = [low + AP_AEC_curtailment_correlative_prob * (high - low) for low, high in
                                           zip(wind_and_battery_data[time][matching]['AP AEC low']['discharge'],
                                               wind_and_battery_data[time][matching]['AP AEC high']['discharge'])]

                electricity_requirements['AP AEC'] = (engineering_inputs['AP AEC H2_req'],
                               engineering_inputs['AP AEC H2_req'] + (408305790 + 3900560.592) / (
                                       365 * 24 * financial_inputs['availability']) / 1000)  # MW

                electricity_requirements['curtailment'] = { #MWh/month
                        'AP CCS': wind_and_battery_data[time][matching]['AP CCS']['curtailment'],
                        'AP BH2S': wind_and_battery_data[time][matching]['AP CCS']['curtailment'],
                        'AP AEC': AP_AEC_curtailment_list
                    }

                electricity_requirements['discharge'] = {  # MWh/month
                        'AP CCS': wind_and_battery_data[time][matching]['AP CCS']['discharge'],
                        'AP BH2S': wind_and_battery_data[time][matching]['AP BH2S']['discharge'],
                        'AP AEC': AP_AEC_discharge_list
                        }

                biomass_requirement = (engineering_inputs['H2']*1000/70.4) * (365 * financial_inputs['availability'])   # tonnes/year
                biomass_price = uni(50.68, 118.25)  # $/ton

                market_correlator = uni(0,1)
                def _corr(val1,val2,correlator=market_correlator):
                    return val1+(val2-val1)*correlator

                Market_inputs['PPA_pricing'] = {time: {matching: {policy: _corr(
                        # Input the range of PPA values
                        PPA_data[(PPA_data['time'] == time) & (PPA_data['policy'] == policy) & (
                                    PPA_data['matching'] == matching) & (
                                         PPA_data['CAPEX_desc'] == 'high')]['LCOE'].iloc[0] / 1000,
                        PPA_data[(PPA_data['time'] == time) & (PPA_data['policy'] == policy) & (
                                    PPA_data['matching'] == matching) & (
                                         PPA_data['CAPEX_desc'] == 'low')]['LCOE'].iloc[0] / 1000
                    ) for policy in policies} for matching in matching_type} for time in times}

                #Defining that the one of the prices for selling curtailment in scenario C are the yearly matching PPA prices
                Market_inputs['PPA_pricing_for_C'] = {
                    time: {policy: Market_inputs['PPA_pricing'][time]['yearly'][policy] for policy in policies} for time
                    in times}

                #Adjusting direct emissions of AP CCS with relation to AP SMR
                AP_SMR_lower_bound = 0.243*carbon_intensity['NGCC thermal efficiency']*13.4/0.28 # (Kg CO2/kWh_e)*(kWh_e/kWh_t_NG)*(kWh_t_NG/Kg NG)/(Kg H2/Kg NG) = Kg CO2/Kg H2
                AP_SMR_upper_bound = 0.527*carbon_intensity['NGCC thermal efficiency']*13.4/0.28 # (Kg CO2/kWh_e)*(kWh_e/kWh_t_NG)*(kWh_t_NG/Kg NG)/(Kg H2/Kg NG) = Kg CO2/Kg H2

                carbon_intensity['stack']['AP SMR'] = uni(AP_SMR_lower_bound, AP_SMR_upper_bound)
                capture_rate_CCS = 0.956
                carbon_intensity['stack']['AP CCS'] = carbon_intensity['stack']['AP SMR']*(1-capture_rate_CCS)

                carbon_intensity['natural gas'] = (uni(0.01, 7.9) / 1000) * carbon_intensity['NGCC thermal efficiency']   # ((g CO2 / kWh_e) / (1000 g CO2 / kg CO2)) * (kWh_e/kWh_t)

                Policy45V_sensitivity_parameter = 1 if start == 0 else 1 #no units

                IRA_credits['45V'] = {float(key): value for key, value in IRA_credits['45V'].items()}

                def calculate_battery_and_turbine_cost(technology, time, CAPEX_inputs):

                    if technology == 'AP SMR':
                        return 0

                    electricity_requirement_for_wind = CAPEX_inputs['wind_capacity'][technology] * 1000 #kW
                    electricity_requirement_for_battery = CAPEX_inputs['battery_capacity'][technology] * 1000 #kW

                    return electricity_requirement_for_wind * CAPEX_inputs[f'Wind turbine CAPEX {time}'] + \
                        electricity_requirement_for_battery * CAPEX_inputs[f'Battery Storage CAPEX {time}']/CAPEX_inputs['Battery roundtrip eff']/4 #$/kW * kW = kW

                def calculate_electrode_cost(time, electricity_requirements, CAPEX_inputs):
                    stack_cost_key = 'Stack cost 2023' if time == 2023 else 'Stack cost 2030'
                    return electricity_requirements['AP AEC'][1] * CAPEX_inputs[stack_cost_key] * 1000

                def calculate_final_CAPEX(technology, scenario, battery_and_turbine, electrode_cost,
                                          basic_equipment_costs, CAPEX_inputs):
                    capex_obj2 = CAPEX(CAPEX_inputs)
                    capex_obj2.get_installed_and_uninstalled_cost(basic_equipment_costs, technology)
                    capex_obj2.calculate_FCI_CAPEX_and_WC()

                    # Calculating the final CAPEX based on the scenario and technology
                    final_cost = 0
                    if scenario == 'C' and technology != 'AP SMR':
                        additional_cost = battery_and_turbine + (electrode_cost if technology == 'AP AEC' else 0)
                        final_cost = capex_obj2.add_cost_outside_of_equipment_list(additional_cost)
                    else:
                        if technology == 'AP AEC':
                            final_cost = capex_obj2.add_cost_outside_of_equipment_list(electrode_cost)
                        else:
                            final_cost = capex_obj2.calculate_FCI_CAPEX_and_WC()

                    # Collecting all the required variables into a dictionary
                    capex_details = {
                        'UC': capex_obj2.UC,  # Assuming that UC is a property of capex_obj2
                        'FCI': capex_obj2.FCI,  # Assuming that FCI is a property of capex_obj2
                        'WC': capex_obj2.WC,  # Assuming that WC is a property of capex_obj2
                        'CAPEX': capex_obj2.CAPEX,  # Assuming that CAPEX is a property of capex_obj2
                        'installation': capex_obj2.installation,
                        'instrumentation_and_controls': capex_obj2.instrumentation_and_controls,
                        'piping': capex_obj2.piping,
                        'electrical': capex_obj2.electrical,
                        'building_process_auxiliary': capex_obj2.building_process_auxiliary,
                        'service_facilities_and_yard_improvements': capex_obj2.service_facilities_and_yard_improvements,
                        'land': capex_obj2.land,
                        'engineering_supervision': capex_obj2.engineering_supervision,
                        'legal_expenses': capex_obj2.legal_expenses,
                        'construction_expense_and_contractors_fee': capex_obj2.construction_expense_and_contractors_fee,
                        'contingency': capex_obj2.contingency
                    }

                    return capex_details

                final_CAPEX = {}
                battery_and_turbine_data_final = {}

                for technology in technologies:
                    # Calculate the cost of battery and turbine for the technology
                    battery_and_turbine = calculate_battery_and_turbine_cost(technology, time,
                                                                             CAPEX_inputs)
                    battery_and_turbine_data_final[technology] = battery_and_turbine

                    # Calculate the electrode cost if applicable
                    electrode_cost = 0 if technology != 'AP AEC' else calculate_electrode_cost(time,
                                                                                               electricity_requirements,
                                                                                               CAPEX_inputs)

                    # Calculate the final CAPEX for the technology
                    final_CAPEX[technology] = calculate_final_CAPEX(technology, scenario, battery_and_turbine,
                                                                    electrode_cost, basic_equipment_costs, CAPEX_inputs)

                final_MI_OPEX = {}

                for tecnology in technologies:
                    opex_calculator = MI_OPEX(MI_OPEX_inputs['processing_steps'][tecnology],
                                              MI_OPEX_inputs['operator_pay'], MI_OPEX_inputs['heuristics_factors'],
                                              tecnology,
                                              financial_inputs, MI_OPEX_inputs, final_CAPEX, HP_steam_requirements,
                                              BFW_requirements, biomass_price, biomass_requirement, CAPEX_inputs)
                    total_labor_costs = opex_calculator.calculate_labor_costs()
                    total_fixed_charges = opex_calculator.fixed_charges()
                    misc_costs = opex_calculator.misc_up_costs()
                    start_costs = opex_calculator.get_start_up_costs()
                    utilities = opex_calculator.utilities_costs()

                    if scenario == 'C' and tecnology != 'AP SMR':
                        electricity_requirement_for_wind = CAPEX_inputs['wind_capacity'][tecnology] * 1000  # kW
                        electricity_requirement_for_battery = CAPEX_inputs['battery_capacity'][tecnology] * 1000  # kW

                        battery_OPEX_fixed = (MI_OPEX_inputs['Battery OPEX']) * electricity_requirement_for_battery \
                                       / CAPEX_inputs['Battery roundtrip eff'] / 12 / 4 #VOM BATTERY COSTS ADDED IN THE market-dependent OPEX

                        wind_OPEX = (MI_OPEX_inputs['Wind OPEX']) * electricity_requirement_for_wind / 12  # ($/kW-year) * (kW) * (1 year/ 12 months) = $/month

                        final_MI_OPEX[tecnology] = {
                            'MI_OPEX': total_labor_costs + total_fixed_charges + misc_costs + utilities + wind_OPEX + battery_OPEX_fixed,
                            'MI_OPEX_start': start_costs}
                    else:
                        final_MI_OPEX[tecnology] = {
                            'MI_OPEX': total_labor_costs + total_fixed_charges + misc_costs + utilities,
                            'MI_OPEX_start': start_costs}








                """
                Above this point are the calculations of inputs and quantities needed for the Stochastic DCF class
                to be able to run simulations. 

                Below this point, the metrics are calculated based on the boolean value set in the run_simulation call. 
                """
                if NPV:
                    NPV_metrics = (time, scenario, sim,
                                   Stochastic_DCF('AP SMR', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                   Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                   Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                   Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV())
                    NPV_data.loc[len(NPV_data)] = NPV_metrics
                    # print(Stochastic_DCF('AP CCS', start, L, sim, True, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final).TCvalue,
                    #       Stochastic_DCF('AP CCS', start, L, sim, True, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final).inflation_rate)


                    message = 'Calculating NPV...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if NP_NPV:
                    NP_NPV_metrics = (time, scenario, sim,
                                      Stochastic_DCF('AP SMR', start, L, sim, False, scenario, financial_inputs,
                                                     final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                     IRA_credits, carbon_intensity, natural_gas_requirements,
                                                     final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                     electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                     battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                      Stochastic_DCF('AP CCS', start, L, sim, False, scenario, financial_inputs,
                                                     final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                     IRA_credits, carbon_intensity, natural_gas_requirements,
                                                     final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                     electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                     battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                      Stochastic_DCF('AP BH2S', start, L, sim, False, scenario, financial_inputs,
                                                     final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                     IRA_credits, carbon_intensity, natural_gas_requirements,
                                                     final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                     electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                     battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(),
                                      Stochastic_DCF('AP AEC', start, L, sim, False, scenario, financial_inputs,
                                                     final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                     IRA_credits, carbon_intensity, natural_gas_requirements,
                                                     final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                     electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                     battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV())
                    NP_NPV_data.loc[len(NP_NPV_data)] = NP_NPV_metrics

                    message = 'Calculating NP NPV...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if CAC:
                    CAC_metrics = [time, scenario, sim,
                                   Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).CAC_updated(),
                                   Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).CAC_updated(),
                                   Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).CAC_updated()]

                    CAC_data.loc[len(CAC_data)] = CAC_metrics


                    message = 'Calculating CAC...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if Potential_TC:
                    Potential_metrics = [time, scenario, sim,
                                         Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_support(),
                                         Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_support(),
                                         Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_support(),
                                         Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support(),
                                         Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support(),
                                         Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support(),
                                         Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support_social(),
                                         Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support_social(),
                                         Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                        final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                        IRA_credits, carbon_intensity, natural_gas_requirements,
                                                        final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                        electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                        battery_and_turbine_data_final,
                                                        isCBAM=isCBAM, matching=matching).total_CE_support_social()]


                    Potential_and_AP_CE_TC.loc[len(Potential_and_AP_CE_TC)] = Potential_metrics

                    message = 'Calculating Potential TC...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if CI:
                    CI_time_SMR = [time, scenario, sim]
                    CI_time_CCS = [time, scenario, sim]
                    CI_time_BH2S = [time, scenario, sim]
                    CI_time_AEC = [time, scenario, sim]

                    for months in multiples_of_12_CI:
                        scenario = scenario
                        CI_time_SMR_vals = Carbon_Intensity_of_technology('AP SMR', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).total_emissions(
                            months)
                        CI_time_CCS_vals = Carbon_Intensity_of_technology('AP CCS', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).total_emissions(
                            months)

                        CI_time_BH2S_vals = Carbon_Intensity_of_technology('AP BH2S', carbon_intensity, scenario,
                                                                           financial_inputs, engineering_inputs,
                                                                           biomass_requirement,
                                                                           natural_gas_requirements,
                                                                           electricity_requirements, aeo22_data,
                                                                           aeo23_data).total_emissions(months)
                        CI_time_AEC_vals = Carbon_Intensity_of_technology('AP AEC', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).total_emissions(
                            months)
                        CI_time_SMR.append(CI_time_SMR_vals)
                        CI_time_CCS.append(CI_time_CCS_vals)
                        CI_time_BH2S.append(CI_time_BH2S_vals)
                        CI_time_AEC.append(CI_time_AEC_vals)

                    list_of_tech_CI = [CI_time_SMR, CI_time_CCS, CI_time_BH2S, CI_time_AEC]

                    for data, technology in zip(list_of_tech_CI, technologies):
                        CI_data[technology].loc[len(CI_data[technology])] = data

                    message = 'Calculating CI...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if CAPEX_OPEX:
                    OPEX_SMR = []
                    OPEX_CCS = []
                    OPEX_BH2S = []
                    OPEX_AEC = []

                    for T in range(601):
                        Point_SMR = Stochastic_DCF('AP SMR', start, L, sim, True, scenario, financial_inputs,
                                                   final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                   IRA_credits, carbon_intensity, natural_gas_requirements,
                                                   final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                   electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                   battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_OPEX(T) / (
                                            engineering_inputs['NH3'] * 365 / 12 * financial_inputs[
                                        'availability']),
                        Point_CCS = Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                   final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                   IRA_credits, carbon_intensity, natural_gas_requirements,
                                                   final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                   electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                   battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_OPEX(T) / (
                                            engineering_inputs['NH3'] * 365 / 12 * financial_inputs[
                                        'availability']),
                        Point_BH2S = Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                    final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                    IRA_credits, carbon_intensity, natural_gas_requirements,
                                                    final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                    electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                    battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_OPEX(T) / (
                                             engineering_inputs['NH3'] * 365 / 12 * financial_inputs[
                                         'availability']),
                        Point_AEC = Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                   final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                   IRA_credits, carbon_intensity, natural_gas_requirements,
                                                   final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                   electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                   battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_OPEX(T) / (
                                            engineering_inputs['NH3'] * 365 / 12 * financial_inputs['availability'])
                        OPEX_SMR.append(Point_SMR)
                        OPEX_CCS.append(Point_CCS)
                        OPEX_BH2S.append(Point_BH2S)
                        OPEX_AEC.append(Point_AEC)

                    CAPEX_OPEX_metric = [time, scenario, sim,
                                         final_CAPEX['AP SMR']['CAPEX'] / engineering_inputs['NH3'] / 1000,
                                         final_CAPEX['AP CCS']['CAPEX'] / engineering_inputs['NH3'] / 1000,
                                         final_CAPEX['AP BH2S']['CAPEX'] / engineering_inputs['NH3'] / 1000,
                                         final_CAPEX['AP AEC']['CAPEX'] / engineering_inputs['NH3'] / 1000,
                                         OPEX_SMR,
                                         OPEX_CCS,
                                         OPEX_BH2S,
                                         OPEX_AEC]

                    OPEX_and_CAPEX.loc[len(OPEX_and_CAPEX)] = CAPEX_OPEX_metric

                    message = 'Calculating CAPEX OPEX...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if absolute_support:
                    AP_CCS = Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs, final_CAPEX,
                                            CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits,
                                            carbon_intensity, natural_gas_requirements, final_MI_OPEX,
                                            electricity_requirements, MI_OPEX_inputs, electrode_cost,
                                            biomass_requirement, aeo22_data, aeo23_data,
                                            battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).separate_total_support()
                    AP_BH2S = Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs, final_CAPEX,
                                             CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits,
                                             carbon_intensity, natural_gas_requirements, final_MI_OPEX,
                                             electricity_requirements, MI_OPEX_inputs, electrode_cost,
                                             biomass_requirement, aeo22_data, aeo23_data,
                                             battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).separate_total_support()
                    AP_AEC = Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs, final_CAPEX,
                                            CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits,
                                            carbon_intensity, natural_gas_requirements, final_MI_OPEX,
                                            electricity_requirements, MI_OPEX_inputs, electrode_cost,
                                            biomass_requirement, aeo22_data, aeo23_data,
                                            battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).separate_total_support()

                    absolute_support_metric = [time, scenario, sim,
                                               AP_CCS['45V'], AP_BH2S['45V'], AP_AEC['45V'],
                                               AP_CCS['45Q'], AP_BH2S['45Q'], AP_AEC['45Q'],
                                               AP_CCS['45Y'], AP_BH2S['45Y'], AP_AEC['45Y'],
                                               AP_CCS['45Y'], AP_BH2S['45Y'], AP_AEC['45Y'], #This is a placeholder for 48C. Ignore.
                                               AP_CCS['48E'], AP_BH2S['48E'], AP_AEC['48E']]

                    absolute_support_data.loc[len(absolute_support_data)] = absolute_support_metric

                    message = 'Calculating absolute support...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                # for T in range(601):
                #     obj1 = Stochastic_DCF('AP BH2S', 0, L, 0, True, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final)
                #     obj2 = Stochastic_DCF('AP BH2S', 0, L, 0, False, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final)
                #     print(time,scenario,T, obj1.calculate_NPV(), obj2.calculate_NPV())

                if ROI:
                    ROI_metrics = (time, scenario, sim,
                                   Stochastic_DCF('AP SMR', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(
                                       ROI=True),
                                   Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(
                                       ROI=True),
                                   Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(
                                       ROI=True),
                                   Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).calculate_NPV(
                                       ROI=True))
                    ROI_data.loc[len(ROI_data)] = ROI_metrics

                    # print(Stochastic_DCF('AP CCS', start, L, sim, True, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final).TCvalue,
                    #       Stochastic_DCF('AP CCS', start, L, sim, True, scenario,financial_inputs, final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements, final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost, biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final).inflation_rate)

                    message = 'Calculating ROI...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if El_market and (scenario == 'A' or scenario == 'B'):
                    El_metric = [time, scenario, sim,
                                 Stochastic_DCF('AP SMR', start, L, sim, True, scenario, financial_inputs, final_CAPEX,
                                                CAPEX_inputs, engineering_inputs, Market_inputs, IRA_credits,
                                                carbon_intensity, natural_gas_requirements, final_MI_OPEX,
                                                electricity_requirements, MI_OPEX_inputs, electrode_cost,
                                                biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).El_market]

                    # print(scenario, El_metric[3])

                    El_data.loc[len(El_data)] = El_metric

                    message = 'Calculating Electricity Market...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if CAPEX_component:
                    capex_metrics = [
                        time, scenario, sim,
                        final_CAPEX['AP SMR'],
                        final_CAPEX['AP CCS'],
                        final_CAPEX['AP BH2S'],
                        final_CAPEX['AP AEC']
                    ]

                    CAPEX_component_data.loc[len(CAPEX_component_data)] = capex_metrics
                    message = 'Calculating CAPEX_component...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if Sensitivity:
                    A_elec_cost = np.mean(brownian_motion(drift=Market_inputs['El_drift'][1],
                                             std_dev=Market_inputs['El_STD'],
                                             n_steps=601,
                                             seed=sim).uncorrelated_GBM(Market_inputs['El_initial_price']))

                    B_elec_cost = np.mean(brownian_motion(drift=Market_inputs['El_drift'][0],
                                             std_dev=Market_inputs['El_STD'],
                                             n_steps=601,
                                             seed=sim).uncorrelated_GBM(Market_inputs['El_initial_price']))

                    electricityCost = 0
                    if scenario == 'A':
                        electricityCost = A_elec_cost
                    elif scenario == 'B':
                        electricityCost = B_elec_cost

                    NH3_average, NG_average = brownian_motion(drift=Market_inputs['NG_drift'],
                                                          std_dev=Market_inputs['NG_std'],
                                                          correlation=0.8,
                                                          n_steps=601,
                                                          seed=sim).correlated_GBM([Market_inputs['NH3_initial_price'], Market_inputs['NG_initial_price']])

                    NH3_average, NG_average = np.mean(NH3_average), np.mean(NG_average)

                    CI_time_SMR = []
                    CI_time_CCS = []
                    CI_time_BH2S = []
                    CI_time_AEC = []

                    for months in multiples_of_12_CI:
                        scenario = scenario
                        CI_time_SMR_vals = Carbon_Intensity_of_technology('AP SMR', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).electricity_emissions(
                            months)
                        CI_time_CCS_vals = Carbon_Intensity_of_technology('AP CCS', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).electricity_emissions(
                            months)
                        CI_time_BH2S_vals = Carbon_Intensity_of_technology('AP BH2S', carbon_intensity, scenario,
                                                                           financial_inputs, engineering_inputs,
                                                                           biomass_requirement,
                                                                           natural_gas_requirements,
                                                                           electricity_requirements, aeo22_data,
                                                                           aeo23_data).electricity_emissions(months)
                        CI_time_AEC_vals = Carbon_Intensity_of_technology('AP AEC', carbon_intensity, scenario,
                                                                          financial_inputs, engineering_inputs,
                                                                          biomass_requirement, natural_gas_requirements,
                                                                          electricity_requirements, aeo22_data,
                                                                          aeo23_data).electricity_emissions(
                            months)
                        CI_time_SMR.append(CI_time_SMR_vals)
                        CI_time_CCS.append(CI_time_CCS_vals)
                        CI_time_BH2S.append(CI_time_BH2S_vals)
                        CI_time_AEC.append(CI_time_AEC_vals)

                    CI_sens = {
                        'AP SMR': np.mean(CI_time_SMR),
                        'AP CCS': np.mean(CI_time_CCS),
                        'AP BH2S': np.mean(CI_time_BH2S),
                        'AP AEC': np.mean(CI_time_AEC)
                    }

                    CAPEX_sensitivity = {
                        'AP SMR': final_CAPEX['AP SMR']['CAPEX'],
                        'AP CCS': final_CAPEX['AP CCS']['CAPEX'],
                        'AP BH2S': final_CAPEX['AP BH2S']['CAPEX'],
                        'AP AEC': final_CAPEX['AP AEC']['CAPEX']
                    }

                    Feedstock_price = {
                        'AP SMR': NG_average,
                        'AP CCS': NG_average,
                        'AP BH2S': biomass_price,
                        'AP AEC': 0
                    }

                    MI_OPEX_sensitivity = {
                        'AP SMR': final_MI_OPEX['AP SMR']['MI_OPEX'],
                        'AP CCS': final_MI_OPEX['AP CCS']['MI_OPEX'],
                        'AP BH2S': final_MI_OPEX['AP BH2S']['MI_OPEX'],
                        'AP AEC': final_MI_OPEX['AP AEC']['MI_OPEX']
                    }

                    Credits_48C = IRA_credits['48E'] if scenario == 'C' else 0
                    IRA_Market_value =  np.mean([IRA_credits['TCvalue'][i] for i in IRA_credits['TCvalue'].keys()])
                    PPA_pricing = Market_inputs['PPA_pricing'][time][matching][True] if scenario == 'D' else 0
                    Curtailment = Market_inputs['PPA_pricing_for_C'][time][True] if scenario == 'C' else 0
                    Battery_CAPEX = CAPEX_inputs[f'Battery Storage CAPEX {time}'] if scenario == 'C' else 0
                    Wind_CAPEX = CAPEX_inputs[f'Wind turbine CAPEX {time}'] if scenario == 'C' else 0
                    Battery_var_OPEX = MI_OPEX_inputs['Battery Var OPEX'] if scenario == 'C' else 0
                    Battery_OPEX = MI_OPEX_inputs['Battery OPEX'] if scenario == 'C' else 0
                    stackCAPEX = CAPEX_inputs[f'Stack cost {time}']

                    for tech in technologies:
                        metric_Sense = [time, scenario, sim, CAPEX_sensitivity[tech], electricityCost, MI_OPEX_sensitivity[tech], Feedstock_price[tech], NH3_average,
                                  CI_sens[tech], Battery_var_OPEX, Battery_OPEX, Battery_CAPEX, Wind_CAPEX,
                                  Credits_48C, IRA_Market_value, PPA_pricing, Curtailment,
                                  stackCAPEX if tech == 'AP AEC' else 0,
                                  engineering_inputs['Eff_electrolysis'] if tech == 'AP AEC' else 0,
                                  IRA_credits['EU_CO2_price'] if isCBAM else 0]

                        sensitivity_data[tech].loc[len(sensitivity_data[tech])] = metric_Sense


                    message = 'Calculating sensitivity...' + str(time) + ' ' + str(scenario) + ' ' + str(sim)
                    sys.stdout.write('\r' + message)
                    sys.stdout.flush()

                if quality_assurance:
                    AP_SMR_CFs = Stochastic_DCF('AP SMR', start, L, sim, True, scenario, financial_inputs,
                                                  final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                  IRA_credits, carbon_intensity, natural_gas_requirements,
                                                  final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                  electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                  battery_and_turbine_data_final, isCBAM=isCBAM, matching=matching).quality_assure()
                    AP_CCS_CFs = Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                IRA_credits, carbon_intensity, natural_gas_requirements,
                                                final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM,
                                                matching=matching).quality_assure()
                    AP_BH2S_CFs = Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                IRA_credits, carbon_intensity, natural_gas_requirements,
                                                final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM,
                                                matching=matching).quality_assure()
                    AP_AEC_CFs = Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                IRA_credits, carbon_intensity, natural_gas_requirements,
                                                final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM,
                                                matching=matching).quality_assure()
                    dfs = {
                        'AP SMR': AP_SMR_CFs,
                        'AP CCS': AP_CCS_CFs,
                        'AP BH2S': AP_BH2S_CFs,
                        'AP AEC': AP_AEC_CFs
                    }

                    dataframes_to_excel(dfs,f'quality_assurance/quality_{matching}_{time}_{scenario}_{sim}.xlsx')

                if CAC_quality:

                    AP_CCS_CFs = Stochastic_DCF('AP CCS', start, L, sim, True, scenario, financial_inputs,
                                                final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                IRA_credits, carbon_intensity, natural_gas_requirements,
                                                final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM,
                                                matching=matching).quality_assure_CAC()

                    AP_BH2S_CFs = Stochastic_DCF('AP BH2S', start, L, sim, True, scenario, financial_inputs,
                                                 final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                 IRA_credits, carbon_intensity, natural_gas_requirements,
                                                 final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                 electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                 battery_and_turbine_data_final, isCBAM=isCBAM,
                                                 matching=matching).quality_assure_CAC()

                    AP_AEC_CFs = Stochastic_DCF('AP AEC', start, L, sim, True, scenario, financial_inputs,
                                                final_CAPEX, CAPEX_inputs, engineering_inputs, Market_inputs,
                                                IRA_credits, carbon_intensity, natural_gas_requirements,
                                                final_MI_OPEX, electricity_requirements, MI_OPEX_inputs,
                                                electrode_cost, biomass_requirement, aeo22_data, aeo23_data,
                                                battery_and_turbine_data_final, isCBAM=isCBAM,
                                                matching=matching).quality_assure_CAC()





    print("Current time:", get_current_est_time())

    if CAC:
        return CAC_data
    if NPV:
        return NPV_data
    if Potential_TC:
        return Potential_and_AP_CE_TC
    if CI:
        return CI_data
    if NP_NPV:
        return NP_NPV_data
    if CAPEX_OPEX:
        return OPEX_and_CAPEX
    if absolute_support:
        return absolute_support_data
    if ROI:
        return ROI_data
    if El_market:
        return El_data
    if CAPEX_component:
        return CAPEX_component_data
    if Sensitivity:
        return sensitivity_data


# TC_dataset = run_simulation(2, Potential_TC=True)
# run_simulation(2)

# run_simulation(1, quality_assurance=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
#
CAPEX_component_dataset = run_simulation(500*mult, CAPEX_component=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(CAPEX_component_dataset.describe())
NPV_dataset = run_simulation(500*mult, NPV=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(NPV_dataset.describe())
NP_NPV_dataset = run_simulation(500*mult, NP_NPV=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(NP_NPV_dataset.describe())
CAC_dataset = run_simulation(500*mult, CAC=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(CAC_dataset.describe())
CI_dataset = run_simulation(500*mult, CI=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
# #
TC_dataset = run_simulation(500*mult, Potential_TC=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(TC_dataset.describe())
# CAPEX_OPEX_dataset = run_simulation(500*mult, CAPEX_OPEX=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
# print(CAPEX_OPEX_dataset.describe())
absolute_support_dataset = run_simulation(500*mult, absolute_support=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(absolute_support_dataset.describe())
# ROI_dataset = run_simulation(500*mult, ROI=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
# print(ROI_dataset.describe())
El_dataset = run_simulation(500*mult, El_market=True, isCBAM=CBAM, matching=matching)  # checked all scenarios v5
print(El_dataset.describe())
# sensitivity_dataset = run_simulation(500*mult, Sensitivity=True, isCBAM=CBAM, matching=matching)
# print(sensitivity_dataset.describe())

# NPV_dataset = run_simulation(50, NPV=True)
# NP_NPV_dataset = run_simulation(50, NP_NPV=True)
# CAC_dataset = run_simulation(25, CAC=True)
# CI_dataset = run_simulation(50, CI=True)
# TC_dataset = run_simulation(25, Potential_TC=True)


dfs = {
    'NPV': NPV_dataset,
    'NP_NPV': NP_NPV_dataset,
    'CAC': CAC_dataset,
    'CI': CI_dataset,
    'TC': TC_dataset,
    # 'CAPEX_OPEX': CAPEX_OPEX_dataset,
    'absolute_support': absolute_support_dataset,
    # 'ROI': ROI_dataset,
    'El': El_dataset,
    # 'CAPEX_component': CAPEX_component_dataset,
    # # 'sensitivity': sensitivity_dataset
}

# df_sens = sensitivity_dataset

dataframes_to_excel(dfs, f"TC_FINAL_alldata_v12{'_CBAM' if CBAM else ''}_{matching}.xlsx")
# dataframes_to_excel(dfs, "calibrate_electricity.xlsx")
# dataframes_to_excel(df_sens, f"FINAL_sensitivities{'_CBAM' if CBAM else ''}_{matching}.xlsx")
# dataframes_to_excel(dfs, "test.xlsx")
# dataframes_to_excel(dfs, "test.xlsx")


# In[ ]:


# In[ ]:
