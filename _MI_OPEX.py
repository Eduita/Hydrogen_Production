class MI_OPEX:
    def __init__(self, processing_steps, hourly_pay_per_operator, heuristics_factors, technology, financial_inputs,
                 MI_OPEX_inputs, final_CAPEX, HP_steam_requirements, BFW_requirements, biomass_price,
                 biomass_requirement, CAPEX_inputs):
        self.processing_steps = processing_steps
        self.hourly_pay_per_operator = hourly_pay_per_operator
        self.heuristics_factors = heuristics_factors
        self.technology = technology
        self.total_labor_costs = None
        self.operating_labor = None

        self.financial_inputs = financial_inputs
        self.MI_OPEX_inputs = MI_OPEX_inputs
        self.final_CAPEX = final_CAPEX
        self.HP_steam_requirements = HP_steam_requirements
        self.BFW_requirements = BFW_requirements
        self.biomass_price = biomass_price
        self.biomass_requirement = biomass_requirement
        self.CAPEX_inputs = CAPEX_inputs

    def calculate_labor_costs(self):

        operators_per_day = self.MI_OPEX_inputs['hours_perday_perprocessingstep'] * self.processing_steps / 8
        operator_shifts_per_week = operators_per_day * 7
        operators = operator_shifts_per_week / 5
        wage_per_week_per_operator = 40 * self.hourly_pay_per_operator
        annual_cost_of_labor = wage_per_week_per_operator * operators * 52 * self.financial_inputs['availability']

        # Apply heuristics factors
        supervision = annual_cost_of_labor * self.heuristics_factors['supervision']
        maintenance = self.final_CAPEX[self.technology]['FCI'] * self.heuristics_factors['maintenance']
        self.maintenance = maintenance
        operating_supplies = maintenance * self.heuristics_factors['operating_supplies']
        laboratory_charges = annual_cost_of_labor * self.heuristics_factors['laboratory_charges']
        patents_and_royalties = self.final_CAPEX[self.technology]['CAPEX'] * self.heuristics_factors[
            'patents_and_royalties']
        overhead_costs = (annual_cost_of_labor + supervision + maintenance) * self.heuristics_factors['overhead']
        self.operating_labor = annual_cost_of_labor
        self.total_labor_costs = annual_cost_of_labor + supervision + maintenance + operating_supplies + laboratory_charges + patents_and_royalties + overhead_costs

        return self.total_labor_costs

        # Calculate total labor costs including heuristics factors

    def fixed_charges(self):
        financing = self.MI_OPEX_inputs['fixed_charges']['Financing Costs'] * self.final_CAPEX[self.technology]['CAPEX']
        rent = self.MI_OPEX_inputs['fixed_charges']['Rent'] * self.CAPEX_inputs['Land Cost']
        insurance = self.MI_OPEX_inputs['fixed_charges']['Insurance'] * self.final_CAPEX[self.technology]['FCI']
        local_property_taxes = self.MI_OPEX_inputs['fixed_charges']['Local Property Taxes'] * \
                               self.final_CAPEX[self.technology]['FCI']
        administrative_costs = self.MI_OPEX_inputs['fixed_charges']['Administrative Costs'] * self.operating_labor
        total_fixed_charges = financing + rent + insurance + local_property_taxes + administrative_costs

        return total_fixed_charges

    def get_start_up_costs(self):
        return self.MI_OPEX_inputs['start-up costs'][self.technology]

    def misc_up_costs(self):
        return self.MI_OPEX_inputs['misc_raw_materials'][self.technology]

    def utilities_costs(self):
        if self.technology in ['AP SMR', 'AP CCS']:
            return self.BFW_requirements[self.technology] * self.MI_OPEX_inputs['BFW cost']
        elif self.technology == 'AP AEC':
            return self.BFW_requirements[self.technology] * self.MI_OPEX_inputs['BFW cost'] + \
                   self.BFW_requirements["AP AEC osmosis"] * self.MI_OPEX_inputs['AEC BFW cost reverse osmosis']
        else:
            return self.BFW_requirements[self.technology] * self.MI_OPEX_inputs['BFW cost'] + self.biomass_price * self.biomass_requirement
