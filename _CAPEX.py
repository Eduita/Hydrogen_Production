
class CAPEX:
    def __init__(self, inputs):

        self.inputs = inputs
        self.file_path = None
        self.sheet_name = None
        self.sheet_data = None
        self.installed_costs = None
        self.uninstalled_cost = None

        self.CAPEX = None
        self.WC = None
        self.FCI = None
        self.UC = None

        # Direct Cost Components
        self.installation = None
        self.instrumentation_and_controls = None
        self.piping = None
        self.electrical = None
        self.building_process_auxiliary = None
        self.service_facilities_and_yard_improvements = None
        self.land = self.inputs['Land Cost']

        # Indirect Cost Components
        self.engineering_supervision = None
        self.legal_expenses = None
        self.construction_expense_and_contractors_fee = None
        self.contingency = None

    def get_installed_and_uninstalled_cost(self, basic_equipment_costs, technology):
        self.installed_costs = basic_equipment_costs[technology][0]
        self.uninstalled_cost = basic_equipment_costs[technology][1]
        return {'installed costs': self.installed_costs, 'uninstalled cost': self.uninstalled_cost}

    def calculate_FCI_CAPEX_and_WC(self):
        # Calculate Uninstalled Cost and Installed Costs
        self.UC = self.uninstalled_cost
        self.IC = self.installed_costs
        purchased_equipment_factor = self.UC * (1 / ((2717 / 100) ** 0.6))

        # Direct Cost Components
        self.installation = self.UC + (-self.UC + self.IC)
        self.instrumentation_and_controls = self.inputs[
                                                'Instrumentation and Controls Cost'] * purchased_equipment_factor
        self.piping = self.inputs['Piping Cost'] * purchased_equipment_factor
        self.electrical = self.inputs['Electrical Cost'] * purchased_equipment_factor
        self.building_process_auxiliary = self.inputs['Buildings Cost'] * purchased_equipment_factor
        self.service_facilities_and_yard_improvements = self.inputs[
                                                            'Service Facilities and Yard Improvements Cost'] * purchased_equipment_factor
        Direct_costs = (
                self.instrumentation_and_controls + self.piping + self.electrical +
                self.building_process_auxiliary + self.service_facilities_and_yard_improvements +
                self.land + self.installation
        )

        # Indirect Cost Components
        self.engineering_supervision = self.inputs['Engineering and Supervision Cost'] * purchased_equipment_factor
        self.legal_expenses = self.inputs['Legal Expenses Cost'] * purchased_equipment_factor
        self.construction_expense_and_contractors_fee = self.inputs[
                                                            'Construction Expense and Contractors Fee Cost'] * purchased_equipment_factor
        self.contingency = self.inputs['Contingency Cost'] * purchased_equipment_factor
        Indirect_costs = (
                self.engineering_supervision + self.legal_expenses +
                self.construction_expense_and_contractors_fee + self.contingency
        )

        # Final Calculations
        self.FCI = Direct_costs + Indirect_costs
        self.WC = self.inputs['Working Capital'] * self.FCI * (1 / ((2717 / 100) ** 0.6))
        self.CAPEX = self.FCI + self.WC

        return {'UC': self.UC, 'FCI': self.FCI, 'WC': self.WC, 'CAPEX': self.CAPEX}

    def add_cost_outside_of_equipment_list(self, additional_cost):
        # Add the additional cost to the CAPEX
        copy = self.CAPEX
        self.CAPEX += additional_cost
        # Recalculate the other metrics
        ratio = 1 / (copy / additional_cost)

        self.WC += ratio * self.WC
        self.FCI += ratio * self.FCI
        self.UC += ratio * self.UC

        # Update Direct Cost Components
        self.installation += ratio * self.installation
        self.instrumentation_and_controls += ratio * self.instrumentation_and_controls
        self.piping += ratio * self.piping
        self.electrical += ratio * self.electrical
        self.building_process_auxiliary += ratio * self.building_process_auxiliary
        self.service_facilities_and_yard_improvements += ratio * self.service_facilities_and_yard_improvements

        # Update Indirect Cost Components
        self.engineering_supervision += ratio * self.engineering_supervision
        self.legal_expenses += ratio * self.legal_expenses
        self.construction_expense_and_contractors_fee += ratio * self.construction_expense_and_contractors_fee
        self.contingency += ratio * self.contingency

        return {'UC': self.UC, 'FCI': self.FCI, 'WC': self.WC, 'CAPEX': self.CAPEX}