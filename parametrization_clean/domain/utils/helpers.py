#!/usr/bin/env python

"""Module with helper methods for ffield dict."""

# Standard library
from typing import List, Dict

# 3rd party packages

# Local source


def set_param(key: List[int], ffield: Dict[int, List], param: float):
    """Using a key in the params key list, set the corresponding parameter in the given Child's dictionary.

    :param key: List containing 3 columns: [section # | line/row # | parameter #]
    :param ffield: ffield[section number] = [parameters corresponding to section].
    :param param: Value of the parameter to store in ffield dict at the specified key
    """
    section = key[0]
    row = key[1] - 1  # Indices start at 0 in Python
    ffield[section][row][key[2] - 1] = param


def get_param(key: List[int], ffield: Dict[int, List]) -> float:
    """Using a key in the params key list, find the corresponding value (parameter) in the ffield dictionary.

    :param key: List containing parameter keys that map to the required parameters in the ffield dictionary.
    :param ffield: Dictionary that maps ReaxFF section number to all parameters in that given section.
    :return param: Floating point parameter; value in ffield file at the location of the key.
    """
    section = key[0]
    row = key[1] - 1  # Indices start at 0 in Python
    return ffield[section][row][key[2] - 1]
