#!/usr/bin/env python

"""Module to parse user-provided, local parameters used for genetic algorithm/neural network.
Overrides default parameters.
"""

# Standard library
import os
import sys
import json

# 3rd party packages

# Local source
from infrastructure.config.default import DefaultSettings
from domain.selection.factory import selection_factory
from domain.adaptation.factory import adaptation_factory
from domain.cost.factory import error_calculator_factory
from domain.crossover.factory import crossover_factory
from domain.mutation.factory import mutation_factory


class UserSettings(DefaultSettings):

    PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../../../"))
    USER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")

    def __init__(self):
        super().__init__()

        try:
            with open(self.USER_CONFIG_FILE, 'r') as in_file:
                all_settings_dict = json.load(in_file)
            self.build_from(all_settings_dict)
        except (FileNotFoundError, json.JSONDecodeError):
            # No configuration file provided - using default config values
            # TODO: LOG
            pass

    def build_from(self, all_settings_dict):
        self.set_strategy_settings(all_settings_dict.get('strategy_settings', {}))
        self.set_ga_settings(all_settings_dict.get('ga_settings', {}))
        self.set_mutation_settings(all_settings_dict.get('mutation_settings', {}))
        self.set_crossover_settings(all_settings_dict.get('crossover_settings', {}))
        self.set_selection_settings(all_settings_dict.get('selection_settings', {}))
        self.set_adaptation_settings(all_settings_dict.get('adaptation_settings', {}))
        self.set_neural_net_settings(all_settings_dict.get('neural_net_settings', {}))

    def set_strategy_settings(self, strategy_settings_dict):
        for key, value in strategy_settings_dict.items():
            if key == "initialization":  # initialization and mutation share same factory
                factory_name = "mutation_factory"
            else:
                factory_name = key + "_factory"

            attribute_name = "_" + key + "_strategy"
            factory_to_call = getattr(sys.modules[__name__], factory_name)
            setattr(self._strategy_settings, attribute_name, factory_to_call(value))

    def set_ga_settings(self, ga_settings_dict):
        self.set_attributes_from_json(self._ga_settings, ga_settings_dict)

    def set_mutation_settings(self, mutation_settings_dict):
        self.set_attributes_from_json(self._mutation_settings, mutation_settings_dict)

    def set_crossover_settings(self, crossover_settings_dict):
        self.set_attributes_from_json(self._crossover_settings, crossover_settings_dict)

    def set_selection_settings(self, selection_settings_dict):
        self.set_attributes_from_json(self._selection_settings, selection_settings_dict)

    def set_adaptation_settings(self, adaptation_settings_dict):
        self.set_attributes_from_json(self._adaptation_settings, adaptation_settings_dict)

    def set_neural_net_settings(self, neural_net_settings_dict):
        self.set_attributes_from_json(self._neural_net_settings, neural_net_settings_dict)

    @staticmethod
    def set_attributes_from_json(config_object, json_dict):
        """Extract and set class attributes from JSON dictionary object."""
        for key, value in json_dict.items():
            # adding underscore as each attribute is stored as protected property
            attribute_name = "_" + key
            setattr(config_object, attribute_name, value)
