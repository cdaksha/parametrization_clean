#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with class to structure/maintain genetic_algorithm population's Child.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List

# 3rd party packages


# Local source
from core.genetic_algorithm.helpers import get_param, set_param, unique_params
from core.utils.reax_io import ReaxIO
from core.settings.config import settings as s


class Individual(object):

    def __init__(self, params=None):
        """Individual/Case in the Genetic Algorithm. Unique based on its own params and corresponding ffield.
        Used as a data storage object.

        Initializes total_error to None.
        Initializes errors to None.
        Initializes reax_predictions to None.
        Initializes energy_curve_data to None.
        Initializes true_vals to None.
        Initializes weights to None.
        Initializes cost to None (cost is used to measure fitness).

        NOTE: Cost must be set in @property `self.cost`; any cost is valid as long as it returns a float.

        Parameters
        ----------
        params: List[float]
            list containing param values mapped from param_keys to ffield.
        """
        self._parent_path = s.path.root
        self._reader = ReaxIO(dir_path=self._parent_path)

        self.ffield, self.atom_types = self._reader.read_ffield()

        # contemplate not reading param_keys, param_bounds, param_increments in __init__ to reduce time cost?
        # Read from params file & remove duplicates
        # '*' unpacks tuple in line
        self._keys, self.param_increments, self.param_bounds = unique_params(*self._reader.read_params())

        if not params:
            self.params = [get_param(key, self.ffield) for key in self._keys]
        else:
            self.params = params
            self.__update()

        self.total_error = None
        self.errors = None
        self.reax_predictions = None
        self.energy_curve_data = None
        self.true_vals = None
        self.weights = None
        self.cost = None
        self.index = None

    def __update(self):
        """Update Child's `ffield` based on `params` (to be used after mutation/mating)."""
        for i, key in enumerate(self._keys):
            set_param(key, self.ffield, self.params[i])
