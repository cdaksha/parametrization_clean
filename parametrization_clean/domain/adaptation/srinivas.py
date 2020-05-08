#!/usr/bin/env python

# Standard library
from typing import Tuple, List

# 3rd party packages


# Local source
from parametrization_clean.domain.adaptation.strategy import IAdaptationStrategy


class SrinivasAdapt(IAdaptationStrategy):

    @staticmethod
    def adaptation(average_cost: float, minimum_cost: float, parent_costs: Tuple[float, float], **kwargs) \
            -> Tuple[float, List[float]]:
        """Execute Srinivas adaptive algorithm. See Srinivas' 1994 paper on "Adaptive probabilities of crossover
        and mutation in genetic algorithms" for more information.

        Parameters
        ----------
        average_cost: float
            Measure of central tendency of cost of population. Can also be the median.
        minimum_cost: float
            Lowest cost of population of Individuals.
        parent_costs: Tuple[float, float]
            Tuple containing costs associated with two parents.
        kwargs: Dict
            Retrieves parameters associated with executing this algorithm from the passed optional dict; otherwise,
            defaults are used.

        Returns
        -------
        cross_rate: float
            Updated crossover rate.
        mutation_rates: List[float, float]
            Updated mutation rates for each of the two children about to be generated from the two parents.
        """
        k1 = kwargs.get('srinivas_k1', 1.0)
        k2 = kwargs.get('srinivas_k2', 0.5)
        k3 = kwargs.get('srinivas_k3', 1.0)
        k4 = kwargs.get('srinivas_k4', 0.5)
        default_mutation_rate = kwargs.get('srinivas_default_mutation_rate', 0.005)

        spread = average_cost - minimum_cost

        better_parent_cost = min(parent_costs)
        if better_parent_cost <= average_cost:
            cross_rate = k1 * (better_parent_cost - minimum_cost) / spread
        else:
            cross_rate = k3

        mutation_rates = []
        for cost in parent_costs:
            if cost <= average_cost:
                new_mutation_rate = max(default_mutation_rate, k2 * (cost - minimum_cost) / spread)
                mutation_rates.append(new_mutation_rate)
            else:
                mutation_rates.append(k4)

        return cross_rate, mutation_rates
