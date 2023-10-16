import pandas as pd
import ast
from scipy.optimize import root_scalar
import json
from BASELINE_global_variables import *
from _instantiate_inputs import InstantiateInputs

#Obtain optimization results
OPTIMIZATION_RESULTS_FILE_PATH = r"C:\Users\Work\PycharmProjects\pythonProject1\Wind data\optimization_results.xlsx"
wind_and_battery_excel = pd.read_excel(OPTIMIZATION_RESULTS_FILE_PATH, sheet_name='Sheet1')

CAPEX_range = ['high','low']
def extract_values(time, matching, results_df= wind_and_battery_excel):
    # Filter for given time and matching, and capex_description 'low'
    mask_high = (results_df['time'] == time) & (results_df['matching'] == matching) & (
                results_df['capex_description'] == 'high') & (results_df['location'] == 'high') & (results_df['technology'] == 'AP AEC high')
    mask_low = (results_df['time'] == time) & (results_df['matching'] == matching) & (
                results_df['capex_description'] == 'low') & (results_df['location'] == 'low') & (results_df['technology'] == 'AP AEC high')

    # Extracting values for each technology
    extracted_values = {}
    for i, mask in enumerate([mask_high, mask_low]):
        filtered_df = results_df[mask]
        temp = {}
        for _, row in filtered_df.iterrows():
            technology = row['technology']
            wind_capacity = row['wind_capacity']
            battery_capacity = row['battery_capacity']
            curtailment = ast.literal_eval(row['curtailment'])
            discharge = ast.literal_eval(row['discharge'])
            total_gen = ast.literal_eval(row['total_gen'])

            temp[technology] = {
                'wind_capacity': wind_capacity,
                'battery_capacity': battery_capacity,
                'curtailment': curtailment,
                'discharge': discharge,
                'total_generation': total_gen
            }
        extracted_values[CAPEX_range[i]] = temp

    return extracted_values
wind_and_battery_data = {time:{matching:extract_values(time, matching) for matching in matching_type} for time in times}

#Define PPA specific variables
scenarios = ['C', 'D']
technology = 'AP AEC high'

#Import average inputs from JSON file
INPUT_PARAMETERS_PATH = r"C:\Users\Work\PycharmProjects\pythonProject1\APPLICATION\input_parameters.json"
with open(INPUT_PARAMETERS_PATH, 'r') as json_file:
    INPUT_PARAMETERS = json.load(json_file)
PPA_INPUTS = InstantiateInputs(0).average_values_from_JSON_inputs(INPUT_PARAMETERS)

#recalculate CAPEX_inputs peters et al heuristics in terms of CAPEX
def redefine_capex_heuristics(capex_inputs: dict) -> dict:
    values = list(capex_inputs.values())[:12]
    keys = list(capex_inputs.keys())[:12]
    sum_values = sum(values)    # = CAPEX in percentage form
    values = [i/sum_values for i in values]
    for key, value in zip(keys, values):
        capex_inputs[key] = value
CAPEX_inputs = PPA_INPUTS['CAPEX_inputs']
CAPEX_inputs['Land Cost'] = 0.04
redefine_capex_heuristics(CAPEX_inputs)

basic_equipment_costs = PPA_INPUTS['basic_equipment_costs']
financial_inputs = PPA_INPUTS['financial_inputs']
engineering_inputs = PPA_INPUTS['engineering_inputs']

#Change the electricity requirements to have the fixed demand electricity
electricity_requirements = PPA_INPUTS['electricity_requirements']
electricity_requirements['fixed_demand'] = 1007 #MW

natural_gas_requirements = PPA_INPUTS['natural_gas_requirements']
BFW_requirements = PPA_INPUTS['BFW_requirements']
HP_steam_requirements = PPA_INPUTS['HP_steam_requirements']
MI_OPEX_inputs = PPA_INPUTS['MI_OPEX_inputs']
Market_inputs = PPA_INPUTS['Market_inputs']
carbon_intensity = PPA_INPUTS['carbon_intensity']
IRA_credits = PPA_INPUTS['IRA_credits']

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

class CAPEX:
    #This function needs to infer the components of the CAPEX based on the heuristic provided by the literature
    def __init__(self, inputs):
        self.inputs = inputs

    def capex_components(self, CAPEX):
        CAPEX_final = {
            'CAPEX': CAPEX
        }
        for capex_component in list(self.inputs.keys())[:12]:
            CAPEX_final[capex_component] = CAPEX*self.inputs[capex_component]

        CAPEX_final['FCI'] = CAPEX_final['CAPEX'] - CAPEX_final['Working Capital']
        CAPEX_final['WC'] = CAPEX_final['Working Capital']
        CAPEX_final['UC'] = CAPEX_final['Purchased Equipment Cost']

        return CAPEX_final

class TaxCreditCalculator:
    def __init__(self, start_month, scenario, financial_inputs,
                 IRA_credits, final_CAPEX, battery_and_turbine_data_final, electricity_requirements, CAPEX_inputs, location):

        self.start_month = start_month
        self.scenario = scenario
        self.IRA_credits = IRA_credits
        self.time = 2023 if self.start_month == 0 else 2030
        self.battery_and_turbine_data_final = battery_and_turbine_data_final
        self.financial_inputs = financial_inputs
        self.final_CAPEX = final_CAPEX

        self.electricity_requirements = electricity_requirements
        self.CAPEX_inputs = CAPEX_inputs
        self.location = location


        # Pre-calculate constants
        self.start_operation_month = self.start_month + 36
        self.end_operation_month = self.start_operation_month + 12 * 40

    # Helper method to check if a month is within the operation period
    def _is_operating(self, month):
        start_operation_month = self.start_month + 36  # Operation starts after 3 years of construction
        end_operation_month = start_operation_month + 12 * 40  # Operation lasts for 30 years
        return start_operation_month <= month < end_operation_month

    # Helper method to check if a month is within the credit lifetime
    def _is_within_lifetime(self, month, credit_lifetime_years):
        start_credit_month = self.start_month + 36  # Credit starts when operation starts
        end_credit_month = start_credit_month + 12 * credit_lifetime_years
        return start_credit_month <= month < end_credit_month

    # Method to calculate 45Y tax credit
    def calculate_45Y(self, month):
        if self._is_operating(month) and month < self.IRA_credits['45Y expiry']:
            monthly_fixed_electricity = self.electricity_requirements[
                                            'fixed_demand'] * 365 / 12 * 24 * self.financial_inputs['availability']  # MWh/month

            index = month % len(
                self.electricity_requirements['curtailment']['AP AEC'][self.location])  # select a month of the year

            curtailment = self.electricity_requirements['curtailment']['AP AEC'][self.location][index]  # MWh/month

            self.electricity_demand_monthly = (monthly_fixed_electricity + curtailment) * 1000  # KWh/month

            return self.IRA_credits[
                '45Y'] * self.electricity_demand_monthly / 100  # Convert from cents/KWh to $/KWh
        else:
            return 0

    def calculate_48E(self, month):
        if not self._is_operating(month) or month >= self.IRA_credits['48E expiry']:
            return 0

        wind_nameplate_capacity = self.CAPEX_inputs['wind_capacity']['AP AEC'][self.location] * 1000  # kW
        battery_nameplate_capacity = self.CAPEX_inputs['battery_capacity']['AP AEC'][self.location] * 1000  # kW

        return (wind_nameplate_capacity * self.CAPEX_inputs[f'Wind turbine CAPEX {self.time}'] + \
                battery_nameplate_capacity * self.CAPEX_inputs[f'Battery Storage CAPEX {self.time}']/4)*self.IRA_credits['48E'] #$/kW * kW = kW

class Stochastic_DCF:
    def __init__(self, start, L, policy, scenario, financial_inputs, final_CAPEX, CAPEX_inputs, IRA_credits, electricity_requirements, MI_OPEX_inputs, battery_and_turbine_data_final, location, LCOE, matching = 'hourly'):

        self.location = location
        self.policy = policy
        self.start = start
        self.scenario = scenario
        self.matching = matching
        self.LCOE = LCOE

        self.is48E = True

        self.financial_inputs = financial_inputs
        self.final_CAPEX = final_CAPEX
        self.CAPEX_inputs = CAPEX_inputs
        self.IRA_credits = IRA_credits

        self.electricity_requirements = electricity_requirements
        self.MI_OPEX_inputs = MI_OPEX_inputs


        self.battery_and_turbine_data_final = battery_and_turbine_data_final

        self.construction = self.financial_inputs['construction_time']
        self.L = L
        self.inflation_rate = self.financial_inputs['inflation']
        self.Y = self.financial_inputs['Y']
        self.FCI = self.final_CAPEX['FCI']
        self.land_cost = self.final_CAPEX['Land Cost']
        self.WC = self.final_CAPEX['WC']
        self.r = self.financial_inputs['cost_of_debt']
        self.e = self.financial_inputs['equity']

        self.F_A = self.financial_inputs['availability']
        self.H_operating = self.financial_inputs['operating_hours_per_year']
        self.phi_state = self.financial_inputs['state_tax']
        self.phi_federal = self.financial_inputs['federal_tax']
        self.L_loan = self.financial_inputs['loan_lifetime']
        self.L_equipment = self.financial_inputs['equipment_lifetime_depreciation']

        self.C_equipment = self.final_CAPEX['CAPEX']/self.CAPEX_inputs['CAPEX_from_installed_cost']

        self.discount_rate = self.e * self.financial_inputs['return_on_equity'] + (1 - self.e) * self.r * (
                1 - (self.phi_federal + self.phi_state))


        if self.start == 0:
            self.time = 2023
        elif self.start == 84:
            self.time = 2030

        # Market costs definitions.

        self.TCvalue = self.IRA_credits['TCvalue']
        self.income_tax = 0
        self.tax_credit_calculator = TaxCreditCalculator(self.start, self.scenario, self.financial_inputs,
                 self.IRA_credits, self.final_CAPEX, self.battery_and_turbine_data_final, self.electricity_requirements, self.CAPEX_inputs
                                                         , self.location)

        self.discount_factor = ((1 + self.discount_rate / 12))
        self.inflation_correction = 1  # (1 + self.inflation_rate / 12) ** (self.start)

    def calculate_FCI(self, T):
        FCI_T = 0

        T_adjusted = T - self.start
        if 0 < T_adjusted <= 12:
            FCI_T += -self.Y[0] * self.FCI / 12
        elif 12 < T_adjusted <= 24:
            FCI_T += -self.Y[1] * self.FCI / 12
        elif 24 < T_adjusted <= 36:
            FCI_T += -self.Y[2] * self.FCI / 12
        else:
            FCI_T += 0

        return FCI_T

    def calculate_land(self, T):
        if T == 0 + self.start:
            Land_T = -self.land_cost
        elif T == self.start + self.construction + self.L:
            Land_T = self.land_cost
        else:
            Land_T = 0
        return Land_T

    def calculate_WC(self, T):
        WC_T = 0
        if T == self.construction + self.start:
            WC_T += -self.WC
        elif T == self.start + self.construction + self.L:
            WC_T += self.WC
        else:
            WC_T += 0
        return WC_T

    def calculate_PMT(self, T):
        if 0 + self.start < T <= 12 + self.start:
            PMT_T = -(self.r / 12) * (T - self.start) * (1 - self.e) * self.Y[0] * self.FCI / 12
        elif 12 + self.start < T <= 24 + self.start:
            PMT_T = -(self.r / 12) * (1 - self.e) * (
                    (T - 12 - self.start) * self.Y[1] * self.FCI / 12 + self.Y[0] * self.FCI)
        elif 24 + self.start < T <= 36 + self.start:
            PMT_T = -(self.r / 12) * (1 - self.e) * (
                    (T - 24 - self.start) * self.Y[2] * self.FCI / 12 + (self.Y[0] + self.Y[1]) * self.FCI)
        elif 36 + self.start < T <= 36 + self.L_loan + self.start:
            PMT_T = -self.FCI * (1 - self.e) * self.r / 12 / (1 - (1 + self.r / 12) ** (-12 * self.L_loan))
        else:
            PMT_T = 0
        return PMT_T

    def calculate_Sales(self, T):
        #The sales should look to sell at the LCOE price no matter what. So, sell the AP AEC high electricity demand
        #Sales = (AP AEC high monthly demand + monthly curtailment)*LCOE
        monthly_fixed_electricity = self.electricity_requirements['fixed_demand']*365/12*24*self.F_A #MWh/month
        index = T % len(self.electricity_requirements['curtailment']['AP AEC'][self.location]) #select a month of the year
        curtailment = self.electricity_requirements['curtailment']['AP AEC'][self.location][index] #MWh/month


        if self.start + 36 < T <= self.start + self.construction + self.L:
            Sales_T = (monthly_fixed_electricity+curtailment) # (MWh/month) * ($/MWh) = $/month

            # if self.matching != 'yearly':
            #     # Sales_T = (self.electricity_requirements['total_generation']['AP AEC'][self.location][index])
            #     Sales_T = (capacity_factor_data[self.time][self.location]['electricity'][index] *
            #                self.CAPEX_inputs['wind_capacity']['AP AEC'][self.location])
            # else:
            #     Sales_T = ((capacity_factor_data[self.time][self.location]['electricity'][index])*self.CAPEX_inputs['wind_capacity']['AP AEC'][self.location])
        else:
            Sales_T = 0
        return Sales_T

    def calculate_OPEX(self, T):
        if not (self.start + 36 < T <= self.start + self.construction + self.L):
            return 0

        electricity_requirement_for_wind = self.CAPEX_inputs['wind_capacity']['AP AEC'][self.location] * 1000  # kW
        electricity_requirement_for_battery = self.CAPEX_inputs['battery_capacity']['AP AEC'][self.location] * 1000  # kW # CAPACITY FIXED

        index = T % len(self.electricity_requirements['discharge']['AP AEC'][self.location])
        variable_battery_discharge = self.electricity_requirements['discharge']['AP AEC'][self.location][index] # MWh/month

        wind_OPEX = (self.MI_OPEX_inputs['Wind OPEX']) * electricity_requirement_for_wind / 12  # ($/kW-year) * (kW) * (1 year/ 12 months) = $/month

        battery_OPEX = (self.MI_OPEX_inputs['Battery OPEX']) * electricity_requirement_for_battery \
                       / 12 / 4 \
                       + \
                       self.MI_OPEX_inputs['Battery Var OPEX'] * variable_battery_discharge / 4  # $/(kW-year) * (kW) * (1 year/ 12 months)

        MD_OPEX = wind_OPEX + battery_OPEX


        return -MD_OPEX

    def calculate_Depreciation(self, T):
        if self.start + self.construction <= T <= 36 + self.L_equipment:
            return -(1 / self.L_equipment) * self.C_equipment
        return 0

    def calculate_replacement_cost(self, time_difference, lifetime, capex_key):
        if int(time_difference) % (lifetime * 12) == 0:

            wind_nameplate_capacity = self.CAPEX_inputs['wind_capacity']['AP AEC'][self.location] # #MW
            battery_nameplate_capacity = self.CAPEX_inputs['battery_capacity']['AP AEC'][self.location]  #MW for 1h

            other_cost_factor = 0

            if capex_key == 'Battery Storage CAPEX':
                other_cost_factor += self.CAPEX_inputs[capex_key + f' {self.time}'] * 1000 / 4 #($/kW * 1000kW/MW) = 1000 * $/MW for 1h battery
                return battery_nameplate_capacity * other_cost_factor / self.CAPEX_inputs['CAPEX_from_installed_cost'] #MW * $/MW

            elif capex_key == 'Wind turbine CAPEX':
                other_cost_factor += (self.CAPEX_inputs[capex_key + f' {self.time}']) * 1000  # ($/kW * 1000kW/MW) = 1000 * $/MW
                return wind_nameplate_capacity * other_cost_factor / self.CAPEX_inputs['CAPEX_from_installed_cost']

        return 0

    def stack_replacement_costs(self, T):
        replacement_cost = 0
        time_difference = T - self.start - self.construction
        if time_difference > 0:
            # Battery replacement
            replacement_cost += self.calculate_replacement_cost(time_difference,
                                                            self.CAPEX_inputs['Battery lifetime'],
                                                            'Battery Storage CAPEX')

            # wind farm replacement
            replacement_cost += self.calculate_replacement_cost(time_difference,
                                                            self.CAPEX_inputs['Wind farm lifetime'],
                                                            'Wind turbine CAPEX')

        return -replacement_cost

    def calculate_Tax(self, T):
        self.income_tax = 0
        if self.start + self.construction < T < self.start + self.construction + self.L:
            Net_revenue_T = (
                    self.calculate_Depreciation(T)
                    + self.calculate_OPEX(T)
                    + self.calculate_Sales(T)
                    + self.calculate_PMT(T)
                    + self.stack_replacement_costs(T)
            )
            if Net_revenue_T > 0:
                self.income_tax = -Net_revenue_T * (self.phi_state + self.phi_federal)
            else:
                self.income_tax = 0

        return self.income_tax

    def _tax_credit_value_HELPER(self, T, credit_45Y=False):
        construction_end_time = self.start + self.construction
        relative_time = T - construction_end_time

        if relative_time <= 0:
            return 0
        if relative_time <= 5 * 12:
            return self.TCvalue['Year 6'] if credit_45Y else self.TCvalue['Year 1-5']
        if relative_time <= 6 * 12:
            return self.TCvalue['Year 6']
        if relative_time <= 7 * 12:
            return self.TCvalue['Year 7']
        if relative_time <= 8 * 12:
            return self.TCvalue['Year 8']
        if relative_time <= 9 * 12:
            return self.TCvalue['Year 9']
        return self.TCvalue['Year 10 and after']

    def _tax_credit_converter_for_program(self, income_tax, tax_credit, T, is45Y = False, set_value=False, value=None): #Let the income tax be positive
        TC_market_value = value if set_value else self._tax_credit_value_HELPER(T, credit_45Y=is45Y)

        if tax_credit - income_tax < 0:
            income_tax = income_tax - tax_credit
            return tax_credit, income_tax
        elif tax_credit - income_tax >= 0:
            cash_equivalent_tax_credit = tax_credit + (tax_credit-income_tax)*TC_market_value
            income_tax = 0
            return cash_equivalent_tax_credit, income_tax

    def _choose_policy(self, compare45V_45Q = False, compare45Y_48E = False, set_value = False, value = None):
        counter_1 = 0
        counter_2 = 0

        # Iterates through the timepoints
        for T in range(self.start+self.construction+self.L):

            if compare45Y_48E: # compare45Y_48E is boolean. If true, performs the comparison of 45V and 45Q
                income_tax = abs(self.calculate_Tax(T)) #Need positive IT for comparison with tax credits

                if T == self.start+self.construction+1: #Just calculates the one ITC timepoint
                    counter_1 += self.tax_credit_calculator.calculate_48E(T) / self.discount_factor ** (T - self.start)

                # Calculate cash-equivalent 45Y
                cash_equivalent_45Y, _ = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                             tax_credit=self.tax_credit_calculator.calculate_45Y(T),
                                                                                T=T,
                                                                                set_value=set_value,
                                                                                value=value)
                # Add CE PTCs to comparator 2
                counter_2 += cash_equivalent_45Y / self.discount_factor ** (T - self.start)

        if counter_1 > counter_2:
            return True
        else:
            return False

    def cash_equivalent_credits(self, T):
        if not self.policy:
            return 0

        # Checking which program is better. 45Y or 48
        if T == 0:
            self.is48E = self._choose_policy(compare45Y_48E=True) #Compare the NPV of 45Y and 48E. Don't ignore transaction costs.

        if self.is48E:
            total_credits_48E = self.tax_credit_calculator.calculate_48E(
                T) if T == self.start + self.construction + 1 else 0
            total_credits_45Y = 0
        else:
            total_credits_48E = 0
            total_credits_45Y = self.tax_credit_calculator.calculate_45Y(T)


        income_tax = abs(self.calculate_Tax(T)) #positive income tax
        cash_equivalent = 0  # initialize cash equivalent variable

        # Use 48E credits
        total_credits_48E, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                   tax_credit=total_credits_48E, T=T)

        # Use 45Y credits
        total_credits_45Y, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                               tax_credit=total_credits_45Y,
                                                                               is45Y=True, T=T) #Direct pay set false


        cash_equivalent += total_credits_45Y + total_credits_48E

        return cash_equivalent

    def calculate_CF(self, T, LCOE):
        CF_T = (self.calculate_FCI(T)
                + self.calculate_land(T)
                + self.calculate_WC(T)
                + self.calculate_PMT(T)
                + self.calculate_Sales(T)*LCOE
                + self.calculate_OPEX(T)
                + self.calculate_Tax(T)
                + self.cash_equivalent_credits(T)
                + self.stack_replacement_costs(T))

        return CF_T

    def calculate_NPV(self, LCOE):
        NPV = 0
        for T in range(self.start + self.construction + self.L + 1):

            index = T % len(self.electricity_requirements['curtailment']['AP AEC'][self.location])

            electricity = self.electricity_requirements['fixed_demand']*365*24/12*self.F_A + \
                self.electricity_requirements['curtailment']['AP AEC'][self.location][index] #MWh/month

            CF_T = self.calculate_CF(T, LCOE)

            NPV += CF_T / self.discount_factor ** (T - self.start)

        return NPV

    def quality_assure(self):

        data = pd.DataFrame(columns=['time','FCI','Land','WC','PMT','Sales','OPEX','Tax','Credits','Stack_replacement','Cash Flow'])
        for T in range(self.start + self.construction + self.L + 1):
            terms = [
                T,
                self.calculate_FCI(T),
                self.calculate_land(T),
                self.calculate_WC(T),
                self.calculate_PMT(T),
                self.calculate_Sales(T),
                self.calculate_OPEX(T),
                self.calculate_Tax(T),
                self.cash_equivalent_credits(T),
                self.stack_replacement_costs(T),
                self.calculate_CF(T, self.LCOE)
            ]
            data.loc[len(data)] = terms

        return data

def run_simulation():
    LCOE_i = 0
    n=0

    LCOE_data = pd.DataFrame(columns=['time', 'matching', 'CAPEX_desc', 'policy', 'LCOE'])
    for matching in matching_type:
        for start in starts:
            time = 2023 if start == 0 else 2030
            for policy in policies:
                for CAPEX_desc in CAPEX_range:

                    #TODO Check for a directional move in all the model inputs

                    #TODO Make sure to add the wind and battery capacity for the CAPEX_inputs
                    CAPEX_inputs['wind_capacity'] = {
                            'AP AEC': {
                                'high': wind_and_battery_data[time][matching]['high']['AP AEC high']['wind_capacity'],
                                'low': wind_and_battery_data[time][matching]['low']['AP AEC high']['wind_capacity']
                            }
                        }
                    CAPEX_inputs['battery_capacity'] = {
                            'AP AEC': {
                                'high': wind_and_battery_data[time][matching]['high']['AP AEC high']['battery_capacity'],
                                'low': wind_and_battery_data[time][matching]['low']['AP AEC high']['battery_capacity']
                            }
                        }

                    # TODO make sure to calculate the depreciation amount correctly
                    def back_calculate_depreciable_capital_factor(capex_inputs, excluded_factors):
                        keys = list(capex_inputs.keys())[:12]
                        depreciable_capital_factor = 0
                        for key in keys:
                            if key not in excluded_factors:
                                depreciable_capital_factor += capex_inputs[key]

                        capex_inputs['CAPEX_from_installed_cost'] = depreciable_capital_factor
                    exclude_from_depreciation = ["Engineering and Supervision Cost", "Legal Expenses Cost", "Construction Expense and Contractor\'s Fee Cost",
                                                 "Working Capital", "Contingency Cost", "Land Cost"]
                    back_calculate_depreciable_capital_factor(CAPEX_inputs, exclude_from_depreciation)

                    #TODO add operating_hours_per_year
                    financial_inputs['operating_hours_per_year'] = 365 * 24 * financial_inputs['availability']

                    #TODO make sure to calculate curtailment, total_generation, and discharge
                    electricity_requirements['curtailment'] = { #MWh/month
                            'AP AEC':{
                                'high': wind_and_battery_data[time][matching]['high']['AP AEC high']['curtailment'],
                                'low': wind_and_battery_data[time][matching]['low']['AP AEC high']['curtailment']
                            },
                        }
                    electricity_requirements['total_generation']= {
                            'AP AEC': {
                                'high': wind_and_battery_data[time][matching]['high']['AP AEC high'][
                                    'total_generation'],
                                'low': wind_and_battery_data[time][matching]['low']['AP AEC high']['total_generation']
                            }
                        }
                    electricity_requirements['discharge']= {  # MWh/month
                            'AP AEC': {
                                'high': wind_and_battery_data[time][matching]['high']['AP AEC high']['discharge'],
                                'low': wind_and_battery_data[time][matching]['low']['AP AEC high']['discharge']
                            }
                        }

                    # You can then use these values in your calculations. For example, if you have variables for CAPEX, land_cost, FCI, and operating_labor, you could calculate the total fixed charges as follows:

                    # ----------------------------------------------------------

                    def calculate_battery_and_turbine_cost(time, CAPEX_inputs, CAPEX_desc):

                        electricity_requirement_for_wind = CAPEX_inputs['wind_capacity']['AP AEC'][CAPEX_desc] * 1000 #kW
                        electricity_requirement_for_battery = CAPEX_inputs['battery_capacity']['AP AEC'][CAPEX_desc] * 1000 #kW

                        return electricity_requirement_for_wind * CAPEX_inputs[f'Wind turbine CAPEX {time}'] + \
                            electricity_requirement_for_battery * CAPEX_inputs[f'Battery Storage CAPEX {time}']/4 #$/kW * kW = kW

                    battery_and_turbine_data_final = {}
                    #CALCULATE THE DEPRECIATING VALUE BELOW

                    final_CAPEX = CAPEX(CAPEX_inputs).capex_components(calculate_battery_and_turbine_cost(time, CAPEX_inputs, CAPEX_desc))

                    NPV_object = Stochastic_DCF(start, L, policy, 'D', financial_inputs, final_CAPEX, CAPEX_inputs
                                                , IRA_credits, electricity_requirements, MI_OPEX_inputs
                                                , battery_and_turbine_data_final, CAPEX_desc, LCOE_i, matching=matching)

                    def NPV(LCOE):
                        return NPV_object.calculate_NPV(LCOE)

                    result = root_scalar(NPV, bracket=[0,2000])

                    metric = [time, matching, CAPEX_desc, policy, result.root]

                    LCOE_data.loc[len(LCOE_data)] = metric

                    NPV_object2 = Stochastic_DCF(start, L, policy, 'D', financial_inputs, final_CAPEX, CAPEX_inputs,
                                                IRA_credits, electricity_requirements, MI_OPEX_inputs,
                                                battery_and_turbine_data_final, CAPEX_desc, LCOE_i, matching=matching)


                    # if matching == 'monthly' or matching == 'yearly':
                    #
                    #     print([time, matching, CAPEX_desc, policy])
                    #     df = {
                    #         'df': NPV_object2.quality_assure()
                    #     }
                    #     dataframes_to_excel(df, f'quality_assure_wind{matching}.xlsx')

                    n+=1

                    #Insert LCOE Solver Here

    return LCOE_data

LCOE_dataset = run_simulation()

print(LCOE_dataset)

