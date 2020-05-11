#!/usr/bin/env python

"""Module to parse user-provided, local parameters used for genetic algorithm/neural network.
Overrides default parameters. User-provided configuration file is optional.
"""

# Standard library
import sys
import json

# 3rd party packages

# Local source
from parametrization_clean.infrastructure.config.default import DefaultSettings
from parametrization_clean.infrastructure.exception.exception import ConfigurationError
# Note: the following imports are required so that python can use the relevant algorithm factory in finding the
# user-requested algorithm
from parametrization_clean.domain.selection.factory import SelectionFactory
from parametrization_clean.domain.adaptation.factory import AdaptationFactory
from parametrization_clean.domain.cost.factory import ErrorFactory
from parametrization_clean.domain.crossover.factory import CrossoverFactory
from parametrization_clean.domain.mutation.factory import MutationFactory


# TODO: Need a way to set param bounds, which is required for central uniform mutation
class UserSettings(DefaultSettings):

    def __init__(self, user_config_file_path):
        super().__init__()

        try:
            with open(user_config_file_path, 'r') as in_file:
                all_settings_dict = json.load(in_file)
            self.build_from(all_settings_dict)
        except json.JSONDecodeError:
            raise ConfigurationError("ERROR parsing JSON configuration file at {}."
                                     "Please double check the file and try again.".format(user_config_file_path))
        except FileNotFoundError:
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
                factory_name = "MutationFactory"
            else:
                factory_name = key.capitalize() + "Factory"

            attribute_name = key + "_strategy"
            factory = getattr(sys.modules[__name__], factory_name)
            setattr(self.strategy_settings, attribute_name, factory.create_executor(value))

    def set_ga_settings(self, ga_settings_dict):
        self.set_attributes_from_json(self.ga_settings, ga_settings_dict)

    def set_mutation_settings(self, mutation_settings_dict):
        self.set_attributes_from_json(self.mutation_settings, mutation_settings_dict)

    def set_crossover_settings(self, crossover_settings_dict):
        self.set_attributes_from_json(self.crossover_settings, crossover_settings_dict)

    def set_selection_settings(self, selection_settings_dict):
        self.set_attributes_from_json(self.selection_settings, selection_settings_dict)

    def set_adaptation_settings(self, adaptation_settings_dict):
        self.set_attributes_from_json(self.adaptation_settings, adaptation_settings_dict)

    def set_neural_net_settings(self, neural_net_settings_dict):
        self.set_attributes_from_json(self.neural_net_settings, neural_net_settings_dict)

    @staticmethod
    def set_attributes_from_json(config_object, json_dict):
        """Extract and set class attributes from JSON dictionary object."""
        for key, value in json_dict.items():
            attribute_name = key
            setattr(config_object, attribute_name, value)
