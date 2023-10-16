from _CI_Calculator import Carbon_Intensity_of_technology

class TaxCreditCalculator:
    def __init__(self, technology, start_month, scenario, engineering_inputs, financial_inputs, carbon_intensity,
                 IRA_credits, final_CAPEX, electrode_cost, battery_and_turbine_data_final, biomass_requirement,
                 natural_gas_requirements, electricity_requirements, aeo22_data, aeo23_data, CAPEX_inputs):
        self.technology = technology
        self.start_month = start_month
        self.time = 2023 if self.start_month == 0 else 2030
        self.scenario = scenario
        self.IRA_credits = IRA_credits
        self.carbon_intensity = carbon_intensity
        self.electrode_cost = electrode_cost
        self.battery_and_turbine_data_final = battery_and_turbine_data_final
        self.engineering_inputs = engineering_inputs
        self.financial_inputs = financial_inputs
        self.final_CAPEX = final_CAPEX
        self.biomass_requirement = biomass_requirement
        self.natural_gas_requirements = natural_gas_requirements
        self.aeo22_data = aeo22_data
        self.aeo23_data = aeo23_data
        self.electricity_requirements = electricity_requirements
        self.CAPEX_inputs = CAPEX_inputs

        self.carbon_intensity_calculator = Carbon_Intensity_of_technology(self.technology, self.carbon_intensity,
                                                                          self.scenario, self.financial_inputs,
                                                                          self.engineering_inputs,
                                                                          self.biomass_requirement,
                                                                          self.natural_gas_requirements,
                                                                          self.electricity_requirements,
                                                                          self.aeo22_data, self.aeo23_data)
        self.AP_SMR_carbon_intensity_calculator = Carbon_Intensity_of_technology('AP SMR', self.carbon_intensity,
                                                                                 self.scenario, self.financial_inputs,
                                                                                 self.engineering_inputs,
                                                                                 self.biomass_requirement,
                                                                                 self.natural_gas_requirements,
                                                                                 self.electricity_requirements,
                                                                                 self.aeo22_data, self.aeo23_data)
        self.hydrogen_production_yearly = self.engineering_inputs['H2'] * 365 * self.financial_inputs[
            'availability']  # Tonne/year
        self.hydrogen_production_monthly = self.hydrogen_production_yearly / 12  # Tonne/month
        self.electricity_demand_monthly = self.carbon_intensity_calculator.electricity_requirements[self.technology][
                                              1] * 1000 * 24 * 30 * self.financial_inputs['availability']  # KWh/month

        # Pre-calculate constants
        self.start_operation_month = self.start_month + 36
        self.end_operation_month = self.start_operation_month + 12 * 40
        self.AP_SMR_carbon_intensity_calculator_start = self.AP_SMR_carbon_intensity_calculator.total_emissions(
            self.start_month)
        self.techology_emissions_start = self.carbon_intensity_calculator.total_emissions(self.start_month)
        self.difference_48C = (self.IRA_credits['48C expiry'] - self.start_operation_month)
        self.max_45V_tier = max(self.IRA_credits['45V'].values(), key=lambda x: x[1])[1]

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

    # Helper method to get the 45V credit tier for a given carbon intensity
    def _get_45V_tier(self, carbon_intensity):
        if carbon_intensity >= self.max_45V_tier:
            return 0
        for tier, (lower_bound, upper_bound) in self.IRA_credits['45V'].items():
            if lower_bound <= carbon_intensity < upper_bound:
                return tier
        raise ValueError(
            f"Invalid carbon intensity: {carbon_intensity}. Carbon intensity should be within the 45V tiers.")

    # Method to calculate 45V tax credit
    def calculate_45V(self, month):
        if self.technology == 'AP AEC' or self.technology == 'AP BH2S' or self.technology == 'AP CCS':
            if self._is_operating(month) and self._is_within_lifetime(month, self.IRA_credits['45V lifetime']):
                carbon_intensity = self.carbon_intensity_calculator.total_emissions(month - self.start_month)
                tier = self._get_45V_tier(carbon_intensity)
                return tier * self.hydrogen_production_monthly * 1000  # Convert from $/kg H2 to $/tonne H2
            else:
                return 0
        else:
            return 0

    # Method to calculate 45Q tax credit
    def calculate_45Q(self, month):
        if self.technology == 'AP CCS':
            if self._is_operating(month) and self._is_within_lifetime(month, self.IRA_credits['45Q lifetime']):
                carbon_intensity = self.carbon_intensity_calculator.stack_emissions()
                AP_SMR_carbon_intensity = self.AP_SMR_carbon_intensity_calculator.stack_emissions()
                CO2_abated = max(0, AP_SMR_carbon_intensity - carbon_intensity) * self.hydrogen_production_monthly  # CO2 abated in tonne/month
                return self.IRA_credits['45Q'] * CO2_abated  # Convert from $/tonne CO2 to $/kg CO2
            else:
                return 0
        else:
            return 0

    # Method to calculate 45Y tax credit
    def calculate_45Y(self, month):
        if self.scenario == 'C' and self.technology != 'AP SMR':
            if self._is_operating(month) and month < self.IRA_credits['45Y expiry']:
                return self.IRA_credits[
                    '45Y'] * self.electricity_demand_monthly / 100  # Convert from cents/KWh to $/KWh
            else:
                return 0
        else:
            return 0

    # Method to calculate 48C tax credit (TURNED OFF)
    def calculate_48C(self, month):
        if not self._is_operating(month) or month >= self.IRA_credits['48C expiry']:
            return 0

        AP_SMR_carbon_intensity = self.AP_SMR_carbon_intensity_calculator_start
        techology_emissions = self.techology_emissions_start

        if (AP_SMR_carbon_intensity - 0.8 * AP_SMR_carbon_intensity > AP_SMR_carbon_intensity - techology_emissions) and \
                (AP_SMR_carbon_intensity - techology_emissions < 0):
            return 0
        elif AP_SMR_carbon_intensity - techology_emissions < 0:
            return 0
        else:
            if self.scenario != 'C':
                if self.technology == 'AP SMR':
                    return 0
                elif self.technology != 'AP AEC':
                    return (self.final_CAPEX[self.technology]['CAPEX'] - self.final_CAPEX['AP SMR']['CAPEX']) * \
                        self.IRA_credits['48C']
                else:
                    return self.electrode_cost * self.IRA_credits['48C']
            else:
                if self.technology == 'AP SMR':
                    return 0
                elif self.technology != 'AP AEC':
                    return abs(((self.final_CAPEX[self.technology]['CAPEX'] - self.battery_and_turbine_data_final[
                        self.technology]) - self.final_CAPEX['AP SMR']['CAPEX']) * self.IRA_credits[
                                   '48C'])
                else:
                    return self.electrode_cost * self.IRA_credits['48C'] #Turned OFF

    def calculate_48E(self, month):
        if not self._is_operating(month) or month >= self.IRA_credits['48C expiry']:
            return 0

        if self.scenario == 'C':
            if self.technology == 'AP SMR':
                return 0
            else:
                electricity_requirement_for_wind = self.CAPEX_inputs['wind_capacity'][self.technology] * 1000  # kW
                electricity_requirement_for_battery = self.CAPEX_inputs['battery_capacity'][self.technology] * 1000  # kW

                return (electricity_requirement_for_wind * self.CAPEX_inputs[f'Wind turbine CAPEX {self.time}'] + \
                        electricity_requirement_for_battery * self.CAPEX_inputs[f'Battery Storage CAPEX {self.time}']/self.CAPEX_inputs['Battery roundtrip eff']/4)*self.IRA_credits['48E'] #$/kW * kW = kW
        else:
            return 0