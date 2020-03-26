#!/usr/bin/env python

"""Module with class to structure/maintain genetic_algorithm population's "individual"/child."""

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
