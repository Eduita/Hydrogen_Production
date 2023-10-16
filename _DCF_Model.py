from _GBM import brownian_motion
from _TAX_CREDITS import TaxCreditCalculator
from _CI_Calculator import Carbon_Intensity_of_technology
import pandas as pd
from BASELINE_global_variables import *

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

class Stochastic_DCF:
    def __init__(self, technology, start, L, sim, policy, scenario, financial_inputs, final_CAPEX, CAPEX_inputs,
                 engineering_inputs,
                 Market_inputs, IRA_credits, carbon_intensity, natural_gas_requirements,
                 final_MI_OPEX, electricity_requirements, MI_OPEX_inputs, electrode_cost,
                 biomass_requirement, aeo22_data, aeo23_data, battery_and_turbine_data_final, isCBAM=False, matching = 'hourly', deterministic=False):
        self.technology = technology
        self.policy = policy
        self.start = start
        self.scenario = scenario
        self.matching = matching

        self.is45V = None
        self.is48E = None
        self.financial_inputs = financial_inputs
        self.final_CAPEX = final_CAPEX
        self.CAPEX_inputs = CAPEX_inputs
        self.engineering_inputs = engineering_inputs
        self.Market_inputs = Market_inputs
        self.IRA_credits = IRA_credits
        self.carbon_intensity = carbon_intensity

        self.final_MI_OPEX = final_MI_OPEX
        self.electricity_requirements = electricity_requirements
        self.MI_OPEX_inputs = MI_OPEX_inputs
        self.electrode_cost = electrode_cost
        self.natural_gas_requirements = natural_gas_requirements

        self.biomass_requirement = biomass_requirement
        self.aeo22_data = aeo22_data
        self.aeo23_data = aeo23_data
        self.battery_and_turbine_data_final = battery_and_turbine_data_final

        self.construction = self.financial_inputs['construction_time']
        self.L = L
        self.inflation_rate = self.financial_inputs['inflation']
        self.Y = self.financial_inputs['Y']
        self.FCI = self.final_CAPEX[self.technology]['FCI']
        self.land_cost = self.CAPEX_inputs['Land Cost']
        self.WC = self.final_CAPEX[self.technology]['WC']
        self.r = self.financial_inputs['cost_of_debt']
        self.e = self.financial_inputs['equity']
        self.M_NH3 = self.engineering_inputs['NH3']  # TPD
        self.F_A = self.financial_inputs['availability']
        self.H_operating = self.financial_inputs['operating_hours_per_year']
        self.phi_state = self.financial_inputs['state_tax']
        self.phi_federal = self.financial_inputs['federal_tax']
        self.L_loan = self.financial_inputs['loan_lifetime']
        self.L_equipment = self.financial_inputs['equipment_lifetime_depreciation']
        self.C_equipment = self.final_CAPEX[self.technology]['UC']
        self.discount_rate = self.e * self.financial_inputs['return_on_equity'] + (1 - self.e) * self.r * (
                1 - (self.phi_federal + self.phi_state))


        self.sim = sim

        if self.start == 0:
            self.time = 2023
        elif self.start == 84:
            self.time = 2030
            self.market_FCI = brownian_motion(drift=self.Market_inputs['SPY_drift'],
                                              std_dev=self.Market_inputs['SPY_std'],
                                              n_steps=self.L + self.start + self.construction + 1,
                                              seed=self.sim).uncorrelated_GBM(
                self.final_CAPEX[self.technology]['FCI'] * self.e)

        # Market costs definitions.
        if not deterministic:
            self.NH3_market, self.NG_market = brownian_motion(drift=self.Market_inputs['NG_drift'],
                                                          std_dev=self.Market_inputs['NG_std'],
                                                          correlation=0.8,
                                                          n_steps=self.L + self.start + self.construction + 1,
                                                          seed=self.sim).correlated_GBM(
                [self.Market_inputs['NH3_initial_price'], self.Market_inputs['NG_initial_price']])

        else:
            self.NH3_market = brownian_motion(drift=self.Market_inputs['NG_drift'],
                                                          std_dev=self.Market_inputs['NG_std'],
                                                          correlation=0.8,
                                                          n_steps=self.L + self.start + self.construction + 1,
                                                          seed=0).uncorrelated_GBM(self.Market_inputs['NH3_initial_price'])

            self.NG_market = brownian_motion(drift=self.Market_inputs['NG_drift'],
                                              std_dev=self.Market_inputs['NG_std'],
                                              correlation=0.8,
                                              n_steps=self.L + self.start + self.construction + 1,
                                              seed=0).uncorrelated_GBM(self.Market_inputs['NG_initial_price'])

        if self.scenario == 'A':
            self.El_market = brownian_motion(drift=self.Market_inputs['El_drift'][1],
                                             std_dev=self.Market_inputs['El_STD'],
                                             n_steps=self.L + self.start + self.construction + 1,
                                             seed=self.sim if not deterministic else 0).uncorrelated_GBM(self.Market_inputs['El_initial_price'])

        if self.scenario == 'B':
            self.El_market = brownian_motion(drift=self.Market_inputs['El_drift'][0],
                                             std_dev=self.Market_inputs['El_STD'],
                                             n_steps=self.L + self.start + self.construction + 1,
                                             seed=self.sim if not deterministic else 0).uncorrelated_GBM(self.Market_inputs['El_initial_price'])


        if self.scenario == 'C':
            self.El_PPA_market = 0
            self.El_market = brownian_motion(drift=self.Market_inputs['El_drift'][0],
                                             std_dev=self.Market_inputs['El_STD'],
                                             n_steps=self.L + self.start + self.construction + 1,
                                             seed=self.sim if not deterministic else 0).uncorrelated_GBM(self.Market_inputs['El_initial_price'])

        if self.scenario == 'D':
            self.El_PPA_market = self.Market_inputs['PPA_pricing'][self.time][self.matching][self.policy]



        self.TCvalue = self.IRA_credits['TCvalue']
        self.income_tax = 0
        self.TaxCreditCalculator = TaxCreditCalculator(self.technology, self.start, self.scenario,
                                                       self.engineering_inputs, self.financial_inputs,
                                                       self.carbon_intensity, self.IRA_credits, self.final_CAPEX,
                                                       self.electrode_cost, self.battery_and_turbine_data_final,
                                                       self.biomass_requirement, self.natural_gas_requirements,
                                                       self.electricity_requirements,
                                                       self.aeo22_data, self.aeo23_data, self.CAPEX_inputs)
        self.Emissions_of_technology = Carbon_Intensity_of_technology(self.technology, self.carbon_intensity,
                                                                      self.scenario, self.financial_inputs,
                                                                      self.engineering_inputs, self.biomass_requirement,
                                                                      self.natural_gas_requirements,
                                                                      self.electricity_requirements, self.aeo22_data,
                                                                      self.aeo23_data)

        self.tax_credit_calculator = TaxCreditCalculator(self.technology, self.start, self.scenario, self.engineering_inputs,
                                                    self.financial_inputs, self.carbon_intensity, self.IRA_credits,
                                                    self.final_CAPEX, self.electrode_cost,
                                                    self.battery_and_turbine_data_final, self.biomass_requirement,
                                                    self.natural_gas_requirements, self.electricity_requirements,
                                                    self.aeo22_data, self.aeo23_data, self.CAPEX_inputs)

        self.baseline_emissions = Carbon_Intensity_of_technology('AP SMR', self.carbon_intensity,
                                                                      self.scenario, self.financial_inputs,
                                                                      self.engineering_inputs, self.biomass_requirement,
                                                                      self.natural_gas_requirements,
                                                                      self.electricity_requirements, self.aeo22_data,
                                                                      self.aeo23_data)


        self.remaining_48C = 0
        self.remaining_48C_v1 = 0
        self.discount_factor = ((1 + self.discount_rate / 12))
        self.discount_factor_CAC = ((1 + self.financial_inputs['CAC discount'] / 12))
        self.inflation_correction = 1  # (1 + self.inflation_rate / 12) ** (self.start)
        self.isCBAM = isCBAM

    def calculate_FCI(self, T):
        opportunity_cost_adjustment = 0 if self.start == 0 else (self.market_FCI[self.start] - self.FCI * self.e)
        FCI_T = 0

        if self.start == 84 and T == self.start:
            tax_rate = (1 - self.phi_federal - self.phi_state) if opportunity_cost_adjustment > 0 else 1
            FCI_T += opportunity_cost_adjustment * (tax_rate)

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
        if T == self.construction + self.start:
            WC_T = -self.WC
        elif T == self.start + self.construction + self.L:
            WC_T = self.WC
        else:
            WC_T = 0
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
        if self.start + 36 < T <= self.start + self.construction + self.L:
            Sales_T = self.M_NH3 * (self.F_A * self.H_operating / 24 / 12) * self.NH3_market[T]
        else:
            Sales_T = 0
        return Sales_T

    def calculate_OPEX(self, T):
        if not (self.start + 36 < T <= self.start + self.construction + self.L):
            return 0

        T_S_cost = 0
        if self.technology == 'AP CCS':
            baseline_emissions = Carbon_Intensity_of_technology('AP SMR', self.carbon_intensity, self.scenario,
                                                                self.financial_inputs, self.engineering_inputs,
                                                                self.biomass_requirement,
                                                                self.natural_gas_requirements,
                                                                self.electricity_requirements, self.aeo22_data,
                                                                self.aeo23_data).total_emissions(T)
            T_S_cost = abs(baseline_emissions - self.Emissions_of_technology.total_emissions(T)) * \
                       self.engineering_inputs['H2'] * 365 / 12 * self.financial_inputs['availability'] * \
                       self.MI_OPEX_inputs['CCS T&S Cost']


        # CBAM CO2 Cost --
        CO2_tax = 0
        if self.isCBAM:
            EU_CI_baseline = self.carbon_intensity['EU 2023 emissions base'] * (
                        1 - self.carbon_intensity['EU_emissions_reduction']) ** (T/12)

            CO2_tax = (EU_CI_baseline - self.Emissions_of_technology.total_emissions(T)) * \
                      self.engineering_inputs['H2'] * 365 / 12 * self.financial_inputs['availability'] * \
                      self.IRA_credits['EU_CO2_price']


        MI_OPEX_temp = -(self.final_MI_OPEX[self.technology]['MI_OPEX'] + T_S_cost*12 - 12*CO2_tax)
        if T == self.start + self.construction + 1:
            MI_OPEX_temp += -self.final_MI_OPEX[self.technology]['MI_OPEX_start']

        NG_cost = (self.natural_gas_requirements[self.technology] * self.NG_market[T]) / 12
        elec_cost = 0
        if self.scenario == 'A' or self.scenario == 'B':
            elec_cost = (self.electricity_requirements[self.technology][1] * 1000 * self.H_operating / 12) * self.El_market[T]
        elif self.scenario == 'D':
            elec_cost = (self.electricity_requirements[self.technology][1] * self.El_PPA_market * 1000 * self.H_operating / 12)

        elif self.scenario == 'C' and self.technology != 'AP SMR':
            #Battery discharge costs
            index = T % len(self.electricity_requirements['discharge'][self.technology])
            variable_battery_discharge = \
                self.electricity_requirements['discharge'][self.technology][index]  # MWh/month

            battery_VOM = self.MI_OPEX_inputs['Battery Var OPEX'] * variable_battery_discharge / 4  # $/(kW-year) * (kW) * (1 year/ 12 months)

            #Electricity costs
            index = T % len(self.electricity_requirements['curtailment'][self.technology])
            elec_cost = (-self.electricity_requirements['curtailment'][self.technology][index]*max(self.El_market[T], self.Market_inputs['PPA_pricing_for_C'][self.time][self.policy])
                         - battery_VOM)
        

        MD_OPEX = (MI_OPEX_temp / 12 - (NG_cost + elec_cost))
        distribution_and_marketing_costs = self.MI_OPEX_inputs['distribution_and_marketing']
        MD_OPEX += MD_OPEX * (distribution_and_marketing_costs + distribution_and_marketing_costs * self.MI_OPEX_inputs[
            'R&D costs'])

        return MD_OPEX

    def calculate_Depreciation(self, T):
        if self.start + self.construction <= T <= 36 + self.L_equipment:
            return -(1 / self.L_equipment) * self.C_equipment
        return 0

    def calculate_replacement_cost(self, time_difference, lifetime, capex_key, cost_factor):
        if time_difference % (lifetime * 12) == 0 and self.time == 2023:
            electricity_in_MW_for_wind = self.CAPEX_inputs['wind_capacity'][self.technology] #MW
            electricity_in_MW_for_battery = self.CAPEX_inputs['battery_capacity'][self.technology] #MW for 1h

            other_cost_factor = 0

            if capex_key == 'Battery Storage CAPEX':
                other_cost_factor += (self.CAPEX_inputs[capex_key + ' 2023']/ self.CAPEX_inputs['Battery roundtrip eff'])* 1000 / 4 #($/kW * 1000kW/MW) = 1000 * $/MW for 1h battery
                return  electricity_in_MW_for_battery * other_cost_factor / self.CAPEX_inputs['CAPEX_from_installed_cost'] #MW * $/MW

            elif capex_key == 'Wind turbine CAPEX':
                other_cost_factor += (self.CAPEX_inputs[capex_key + ' 2023']) * 1000  # ($/kW * 1000kW/MW) = 1000 * $/MW
                return electricity_in_MW_for_wind * other_cost_factor / self.CAPEX_inputs['CAPEX_from_installed_cost']


        elif time_difference % (lifetime * 12) == 0 and self.time == 2030:
            electricity_in_MW_for_wind = self.CAPEX_inputs['wind_capacity'][self.technology]  # MW
            electricity_in_MW_for_battery = self.CAPEX_inputs['battery_capacity'][self.technology]  # MW for 1h

            other_cost_factor = 0

            if capex_key == 'Battery Storage CAPEX':
                other_cost_factor += (self.CAPEX_inputs[capex_key + ' 2030'] / self.CAPEX_inputs[
                    'Battery roundtrip eff']) * 1000  # ($/kW * 1000kW/MW) = 1000 * $/MW
                return electricity_in_MW_for_battery * other_cost_factor  # MW * $/MW

            elif capex_key == 'Wind turbine CAPEX':
                other_cost_factor += (self.CAPEX_inputs[capex_key + ' 2030']) * 1000  # ($/kW * 1000kW/MW) = 1000 * $/MW
                return electricity_in_MW_for_wind * other_cost_factor

        return 0

    def stack_replacement_costs(self, T):
        if self.scenario != 'C':
            replacement_cost = 0
            time_difference = T - self.start - self.construction

            if self.technology == 'AP AEC' and time_difference % (
                    self.CAPEX_inputs['AEC stack lifetime']) == 0 and T > self.start+self.construction:
                replacement_cost -= self.electrode_cost * self.CAPEX_inputs['Stack and battery replacement'] / \
                                    self.CAPEX_inputs['CAPEX_from_installed_cost']
            return replacement_cost
        else:
            replacement_cost = 0
            time_difference = T - self.start - self.construction
            if time_difference > 0:
                    # Battery replacement
                if self.technology != 'AP SMR':
                    replacement_cost -= self.calculate_replacement_cost(time_difference,
                                                                    self.CAPEX_inputs['Battery lifetime'],
                                                                    'Battery Storage CAPEX',
                                                                    self.CAPEX_inputs['Stack and battery replacement'])

                    # wind farm replacement
                    replacement_cost -= self.calculate_replacement_cost(time_difference,
                                                                    self.CAPEX_inputs['Wind farm lifetime'],
                                                                    'Wind turbine CAPEX',
                                                                    1 / self.CAPEX_inputs['wind capacity'])

                # electrode replacement
                if self.technology == 'AP AEC' and time_difference % (
                        self.CAPEX_inputs['AEC stack lifetime']) == 0 and T > self.start+self.construction:
                    replacement_cost -= self.electrode_cost * self.CAPEX_inputs['Stack and battery replacement'] / \
                                        self.CAPEX_inputs['CAPEX_from_installed_cost']

                return replacement_cost
            else:
                return 0

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
            cash_equivalent_tax_credit = income_tax + (tax_credit-income_tax)*TC_market_value
            income_tax = 0
            return cash_equivalent_tax_credit, income_tax

    def _choose_policy(self, compare45V_45Q = False, compare45Y_48E = False, set_value = False, value = None):
        counter_1 = 0
        counter_2 = 0

        # Iterates through the timepoints
        for T in range(self.start+self.construction+self.L):

            if compare45V_45Q: # compare45V_45Q is boolean. If true, performs the comparison of 45V and 45Q
                counter_1 += self.tax_credit_calculator.calculate_45V(T) / self.discount_factor ** (T - self.start)
                counter_2 += self.tax_credit_calculator.calculate_45Q(T) / self.discount_factor ** (T - self.start)

            elif compare45Y_48E: # compare45Y_48E is boolean. If true, performs the comparison of 45V and 45Q
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
                counter_2 +=  cash_equivalent_45Y / self.discount_factor ** (T - self.start)

        if counter_1 > counter_2:
            return True #TRUE means 45V > 45Q  OR  48E > 45Y
        else:
            return False #FALSE means 45V < 45Q  OR  48E < 45Y
    def _find_abated_emissions(self):
        total_abated_emissions_across_lifetime = 0
        for T in range(self.start + self.construction, self.start + self.construction + self.L):
            emissions_difference_CI = (
                        self.baseline_emissions.total_emissions(T) - self.Emissions_of_technology.total_emissions(T))
            monthly_hydrogen_flowrate = self.engineering_inputs['H2'] * 365 / 12 * self.financial_inputs['availability']

            total_abated_emissions_across_lifetime += emissions_difference_CI * monthly_hydrogen_flowrate * (
                        1 / (self.discount_factor_CAC ** (T - self.start)))

        return total_abated_emissions_across_lifetime

    def _find_hydrogen_produced(self, discount_factor):
        #the discount factor in the form: (1+r/12)
        h2_TPD = self.engineering_inputs['H2']
        h2_kgPm = h2_TPD * self.F_A * 365/12 * 1000 #(Tonne/day)*(time/time)*(day/year)/(month/year)*(1000kg/1 tonne)

        total_h2_produced = 0
        for T in range(self.start + self.construction, self.start + self.construction + 10*12):
            total_h2_produced += h2_kgPm * (1 / (discount_factor ** (T - self.start)))

        # print("hydrogen produced", total_h2_produced/1000000)
        return total_h2_produced

    def cash_equivalent_credits(self, T):
        if not self.policy or self.technology == 'AP SMR':
            return 0

        # Checking which program is better for AP CCS and for wind farm

        if self.technology == 'AP CCS' and T == 0:
            self.is45V = self._choose_policy(compare45V_45Q=True) #Compare the NPV of 45V and 45Q. Ignored transaction costs.

        if self.scenario == 'C' and T == 0:
            self.is48E = self._choose_policy(compare45Y_48E=True) #Compare the NPV of 45Y and 48E. Don't ignore transaction costs.


        # Defining policy PTCs depending on the technology
        if self.technology in ['AP BH2S', 'AP AEC']:
            total_credits_others = self.tax_credit_calculator.calculate_45V(T)
        elif self.technology == 'AP CCS':
            total_credits_others = self.tax_credit_calculator.calculate_45V(T) if self.is45V else self.tax_credit_calculator.calculate_45Q(T)

        #48C credits are set to 0
        total_credits_48C = 0

        # Defining policy for scenario C
        if self.is48E:
            total_credits_48E = self.tax_credit_calculator.calculate_48E(T) if T == self.start + self.construction + 1 else 0
            total_credits_45Y = 0
        else:
            total_credits_48E = 0
            total_credits_45Y = self.tax_credit_calculator.calculate_45Y(T)


        income_tax = abs(self.calculate_Tax(T)) #positive income tax
        cash_equivalent = 0  # initialize cash equivalent variable

        # print('income tax', income_tax)
        # Use 48C credits first
        total_credits_48C, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                   tax_credit=total_credits_48C, T=T)

        # Use 48E credits
        total_credits_48E, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                   tax_credit=total_credits_48E, T=T)
        # Use 45Y credits
        total_credits_45Y, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                               tax_credit=total_credits_45Y,
                                                                               is45Y=True, T=T) #Direct pay set false
        # Use 45V or 45Q credits
        total_credits_others, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                               tax_credit=total_credits_others, T=T)
        # print(self.is45V, self.technology, self.scenario, '45V or 45Q',total_credits_others,'48C',total_credits_48C,'45Y', total_credits_45Y,'48E', total_credits_48E)

        cash_equivalent += total_credits_others + total_credits_45Y + total_credits_48E + total_credits_48C

        # print(cash_equivalent)

        return cash_equivalent

    def _nominal_tax_credits(self, T, separate = False):
        if not self.policy or self.technology == 'AP SMR':
            return 0

        #Check which policy program is better with cash equivalency
        if self.technology == 'AP CCS' and T == 0:
            self.is45V = self._choose_policy(compare45V_45Q=True) #Compare the NPV of 45V and 45Q. Ignored transaction costs.
        elif self.technology != 'AP CCS':
            self.is45V = True

        if self.scenario == 'C' and T == 0:
            self.is48E = self._choose_policy(compare45Y_48E=True) #Compare the NPV of 45Y and 48E. Don't ignore transaction costs.

        individual_tax_credits = {
            '45V': self.tax_credit_calculator.calculate_45V(T) if self.is45V else 0,
            '45Q': self.tax_credit_calculator.calculate_45Q(T) if not self.is45V else 0,
            '48E': self.tax_credit_calculator.calculate_48E(T) if (T == self.start + self.construction + 1) and self.is48E else 0,
            '45Y': self.tax_credit_calculator.calculate_45Y(T) if not self.is48E else 0
        }

        if separate:
            return individual_tax_credits
        else:
            return sum(individual_tax_credits.values())

    def _convert_nominal_to_CE(self, T, separate = False):
        tax_credits = self._nominal_tax_credits(T, separate=True)
        income_tax = abs(self.calculate_Tax(T))

        CE_tax_credits = {}
        for key, val in tax_credits.items():
            if key == '45Y':
                CE_tax_credits[key], income_tax = self._tax_credit_converter_for_program(income_tax, val, T, is45Y=True)
            else:
                CE_tax_credits[key], income_tax = self._tax_credit_converter_for_program(income_tax, val, T)

        if separate:
            return CE_tax_credits
        else:
            return sum(CE_tax_credits.values())

    def CAC_updated(self):
        carbon_abated = self._find_abated_emissions()
        CAC = 0
        for T in range(self.start + self.construction + self.L + 1):
            CAC += (self._nominal_tax_credits(T) / carbon_abated) / ((self.discount_factor_CAC) ** (T - self.start))

        return CAC

    def total_support(self):
        total_support = 0
        hydrogen = self._find_hydrogen_produced(self.discount_factor_CAC)
        for T in range(self.start + self.construction + self.L + 1):
            total_support += (self._nominal_tax_credits(T) / hydrogen) / ((self.discount_factor_CAC) ** (T - self.start))

        return total_support

    def total_CE_support(self):
        total_CE_support = 0
        hydrogen = self._find_hydrogen_produced(self.discount_factor)
        for T in range(self.start + self.construction + self.L + 1):
            total_CE_support += (self._convert_nominal_to_CE(T) / hydrogen) / ((self.discount_factor) ** (T - self.start))

        return total_CE_support

    def total_CE_support_social(self):
        total_CE_support = 0
        hydrogen = self._find_hydrogen_produced(self.discount_factor_CAC)
        for T in range(self.start + self.construction + self.L + 1):
            total_CE_support += (self._convert_nominal_to_CE(T) / hydrogen) / ((self.discount_factor_CAC ** (T - self.start)))

        return total_CE_support

    def separate_total_support(self):
        hydrogen = self._find_hydrogen_produced(self.discount_factor_CAC)
        total_support = {
            '45V': 0,
            '45Q': 0,
            '48E': 0,
            '45Y': 0
        }
        for T in range(self.start + self.construction + self.L + 1):
            total_support_separate = self._nominal_tax_credits(T, separate=True)
            for key, val in total_support_separate.items():
                total_support[key] += (val / hydrogen) / ((self.discount_factor_CAC) ** (T - self.start))

        return total_support

    def carbon_abatement_cost(self, value=None, set_value=False, absolute=False, separate=False, quality_assure=False):


        def cash_equivalent_credits_if_no_market(T, set_value=set_value, TC_value=value, separate=separate):
            if not self.policy or self.technology == 'AP SMR':
                return 0

            # Checking which program is better for AP CCS and for wind farm
            if self.technology == 'AP CCS' and T == 0:
                self.is45V = self._choose_policy(compare45V_45Q=True)  # Compare the NPV of 45V and 45Q. Ignored transaction costs.

            if self.scenario == 'C' and T == 0:
                self.is48E = self._choose_policy(compare45Y_48E=True)  # Compare the NPV of 45Y and 48E. Don't ignore transaction costs.

            # Defining policy PTCs depending on the technology
            if self.technology in ['AP BH2S', 'AP AEC']:
                total_credits_others = self.tax_credit_calculator.calculate_45V(T)
            elif self.technology == 'AP CCS':
                total_credits_others = self.tax_credit_calculator.calculate_45V(T) if self.is45V else self.tax_credit_calculator.calculate_45Q(T)

            # 48C credits are set to 0
            total_credits_48C = 0

            # Defining policy for scenario C
            # Returns 0 if AP SMR or if not scenario C
            if self.is48E:
                total_credits_48E = self.tax_credit_calculator.calculate_48E(T) if T == self.start + self.construction + 1 else 0
                total_credits_45Y = 0
            else:
                total_credits_48E = 0
                total_credits_45Y = self.tax_credit_calculator.calculate_45Y(T)

            income_tax = abs(self.calculate_Tax(T))  # positive income tax
            cash_equivalent = 0  # initialize cash equivalent variable

            # Use 48C credits first
            total_credits_48C, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                   tax_credit=total_credits_48C,
                                                                                   T=T,
                                                                                   set_value=set_value,
                                                                                   value=TC_value)

            # Use 48E credits
            total_credits_48E, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                   tax_credit=total_credits_48E,
                                                                                   T=T,
                                                                                   set_value=set_value,
                                                                                   value=TC_value)

            # Use 45Y credits
            total_credits_45Y, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                   tax_credit=total_credits_45Y,
                                                                                   is45Y=True,
                                                                                   T=T,
                                                                                   set_value=set_value,
                                                                                   value=TC_value)  # Direct pay set false

            # Use 45V or 45Q credits
            #initialize separate variables
            total_credits_45V = 0
            total_credits_45Q = 0

            if self.is45V and self.technology =='AP CCS':
                total_credits_45V, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                      tax_credit=total_credits_others,
                                                                                      T=T,
                                                                                      set_value=set_value,
                                                                                      value=TC_value)
            elif self.is45V == False and self.technology =='AP CCS':
                total_credits_45Q, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                       tax_credit=total_credits_others,
                                                                                       T=T,
                                                                                       set_value=set_value,
                                                                                       value=TC_value)

            else:
                total_credits_45V, income_tax = self._tax_credit_converter_for_program(income_tax=income_tax,
                                                                                       tax_credit=total_credits_others,
                                                                                       T=T,
                                                                                       set_value=set_value,
                                                                                       value=TC_value)

            if not separate:
                cash_equivalent +=  (total_credits_45Y +
                                    total_credits_48E +
                                    total_credits_48C +
                                    total_credits_45V +
                                    total_credits_45Q)

                return cash_equivalent
            else:
                return total_credits_45V, total_credits_45Q, total_credits_45Y, total_credits_48C, total_credits_48E

            # END OF ENDOGENOUS FUNCTION

        # INITIALIZE NPV-like variables
        if not separate:
            CAC = 0

        CAC_45V = CAC_45Q = CAC_45Y = CAC_48C = CAC_48E = 0

        #Quality assurance storage of values
        store = {
            'T': [],
            '45V': [],
            '45Q': [],
            '45Y': [],
            '48C': [],
            '48E': []
        }

        #Calculate the present value of the carbon abatement cost
        for T in range(self.start + self.construction + self.L + 1):
            # Choose to keep the carbon abatement costs separate or together
            if separate:
                cash_equivalent_45V, cash_equivalent_45Q, cash_equivalent_45Y, cash_equivalent_48C, cash_equivalent_48E = cash_equivalent_credits_if_no_market(
                    T, TC_value=value, separate=separate)

            else:
                all_cash = cash_equivalent_credits_if_no_market(T, TC_value=value, set_value=set_value, separate=separate) / (
                    (self.discount_factor_CAC) ** (T - self.start))



            # Choose if we want to normalize the policy support by the carbon abated
            if absolute:
                denominator = 1

            else:
                denominator = self._find_abated_emissions()

            if separate:
                store['T'].append(T)
                store['45V'].append(cash_equivalent_45V / denominator)
                store['45Q'].append(cash_equivalent_45Q / denominator)
                store['45Y'].append(cash_equivalent_45Y / denominator)
                store['48C'].append(cash_equivalent_48C / denominator)
                store['48E'].append(cash_equivalent_48E / denominator)


                CAC_45V += (1 / self.discount_factor ** (T - self.start)) * cash_equivalent_45V / denominator
                CAC_45Q += (1 / self.discount_factor ** (T - self.start)) * cash_equivalent_45Q / denominator
                CAC_45Y += (1 / self.discount_factor ** (T - self.start)) * cash_equivalent_45Y / denominator
                CAC_48C += (1 / self.discount_factor ** (T - self.start)) * cash_equivalent_48C / denominator
                CAC_48E += (1 / self.discount_factor ** (T - self.start)) * cash_equivalent_48E / denominator
            else:
                # print(self.time, self.scenario, self.technology, T, all_cash / denominator, self.baseline_emissions.total_emissions(T) - self.Emissions_of_technology.total_emissions(T), total_abated_emissions_across_lifetime)

                CAC += all_cash / denominator


        if separate and quality_assure:
            return pd.DataFrame(store)
        if separate:
            return CAC_45V, CAC_45Q, CAC_45Y, CAC_48C, CAC_48E
        else:
            CAC /= self.inflation_correction
            # print(CAC)
            return CAC

    def calculate_CF(self, T):
        CF_T = self.calculate_FCI(T) + self.calculate_land(T) + self.calculate_WC(T) + self.calculate_PMT(
            T) + self.calculate_Sales(T) + self.calculate_OPEX(T) + self.calculate_Tax(
            T) + self.cash_equivalent_credits(T) + self.stack_replacement_costs(T)
        return CF_T

    def calculate_NPV(self, ROI=False):
        NPV = 0
        ROI_value = 0
        for T in range(self.start + self.construction + self.L + 1):
            CF_T = self.calculate_CF(T)
            CAPEX = self.calculate_FCI(T) + self.calculate_land(T) + self.calculate_WC(T)
            if not ROI:
                NPV += CF_T / self.discount_factor ** (T - self.start)
            if ROI:
                ROI_value += (CF_T - (CAPEX)) / self.discount_factor ** (T - self.start)

        # Correct for inflation to bring back dollars to 2023.

        NPV /= (self.M_NH3 * 365 * L / 12 * self.F_A)
        NPV /= self.inflation_correction

        ROI_value /= (self.final_CAPEX[self.technology]['CAPEX'])
        ROI_value /= self.inflation_correction
        if not ROI:
            return NPV
        else:
            return ROI_value

    def quality_assure(self):
        data = pd.DataFrame(
            columns=['time', 'FCI', 'Land', 'WC', 'PMT', 'Sales', 'OPEX', 'Tax', 'Credits', 'Stack_replacement',
                     'Cash Flow'])
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
                self.calculate_CF(T)
            ]
            data.loc[len(data)] = terms

        return data

    def quality_assure_CAC(self):
        df = {
            'CAC': self.carbon_abatement_cost(separate=True, quality_assure=True, absolute=True)
        }
        return dataframes_to_excel(df, f'CAC_quality_asssure_{self.technology}_{self.time}_{self.scenario}.xlsx')