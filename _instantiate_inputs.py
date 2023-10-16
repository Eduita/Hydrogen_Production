from numpy.random import uniform as uni
from numpy.random import seed as seed
import numpy as np
import json

class InstantiateInputs:

    def __init__(self, seed_number):
        self.constant_variables = {}
        self.random_variables = {}
        self.seed = seed_number
        self.force_correlate = uni(0,1)
        # seed(self.seed)

    def randomness_from_JSON_inputs(self, input_dictionary):
        for key, value in input_dictionary.items():
            # print(f"Seeding with {self.seed}")
            if isinstance(value, dict):
                # If the value is a dictionary, recursively search it
                self.randomness_from_JSON_inputs(value)
            elif isinstance(value, list):
                if "uniform" in value:
                    self.random_variables[key] = value[1:]
                    input_dictionary[key] = uni(*value[1:])
                    # print(f"Generated random value {self.input_dictionary[key]} for key {key}")  # Debug print
                if "correlated" in value:
                    self.random_variables[key] = value[1:]
                    input_dictionary[key] = value[1] + (value[2] - value[1]) * self.force_correlate
                if "constant" in value:
                    self.constant_variables[key] = value[1:]
                    input_dictionary[key] = value[1:]
            else:
                self.constant_variables[key] = value
                input_dictionary[key] = value

        return input_dictionary

    def average_values_from_JSON_inputs(self, input_dictionary, force_correlate=0.5):
        for key, value in input_dictionary.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively search it
                self.average_values_from_JSON_inputs(value)
            elif isinstance(value, list):
                if "uniform" in value:
                    self.random_variables[key] = value[1:]
                    input_dictionary[key] = np.mean(value[1:])
                if "correlated" in value:
                    self.random_variables[key] = value[1:]
                    input_dictionary[key] = value[1] + (value[2] - value[1]) * 0.5
                if "constant" in value:
                    self.constant_variables[key] = value[1:]
                    input_dictionary[key] = value[1:]
            else:
                self.constant_variables[key] = value

        return input_dictionary

