#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for neural network implementation.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List

# 3rd party packages
import pandas as pd

# Local source
from core.genetic_algorithm import individual as child


# ---------- HELPER FUNCTIONS ----------
# Data Ingestion
def to_df(population: List[child.Individual]) -> pd.DataFrame:
    """Digest population as a pandas DataFrame with columns of ['param1' | ... |'paramN' | 'TotalError']."""
    params_error_list = [case.params + case.reax_predictions for case in population]
    # TESTING usage of total error instead of reax values
    #params_error_list = [case.params + [case.cost] for case in population]
    df = pd.DataFrame(data=params_error_list)
    return df
