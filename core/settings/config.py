#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os

# 3rd party packages
import yaml

# CONFIG_PATH = "/home/1935/00-Senior-Thesis/Parametrization-GA-ANN/core/settings/config.yaml"
# For normal usage
CONFIG_PATH = os.path.join(os.getcwd(), 'core', 'settings', 'config.yaml')
# For test/experimentation folder
# CONFIG_PATH = 'C:\\Users\\chadd\\Desktop\\repos\\Parametrization-GA-ANN\\core\\settings\\config.yaml'
tail = -1
while tail != "Parametrization-GA-ANN":
    head, tail = os.path.split(CONFIG_PATH)
    CONFIG_PATH = head
CONFIG_PATH = os.path.join(CONFIG_PATH, "Parametrization-GA-ANN", 'core', 'settings', 'config.yaml')


class Messenger:
    """
    A class to convert a nested Dictionary into an object with key-values
    accessibly using attribute notation (AttributeDict.attribute) instead of
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: AttributeDict.attr.attr)
    """
    def __init__(self, **entries):
        self.add_entries(**entries)

    def add_entries(self, **entries):
        for key, value in entries.items():
            if type(value) is dict:
                self.__dict__[key] = Messenger(**value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, key):
        """
        Provides dict-style access to attributes
        """
        return getattr(self, key)


with open(CONFIG_PATH, 'r') as in_file:
    conf = yaml.safe_load(in_file)
settings = Messenger(**conf)
