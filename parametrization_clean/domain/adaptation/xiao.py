#!/usr/bin/env python

# Standard library
from typing import Tuple, List
import math

# 3rd party packages


# Local source
from parametrization_clean.domain.adaptation.strategy import IAdaptationStrategy


class XiaoAdapt(IAdaptationStrategy):

    @staticmethod
    def adaptation(average_cost: float, minimum_cost: float, parent_costs: Tuple[float, float], **kwargs) \
            -> Tuple[float, List[float]]:
        """Execute Xiao adaptive algorithm based on the arctan function.

        :param average_cost: Measure of central tendency of cost of population. Can also be the median.
        :param minimum_cost: Lowest cost of population of Individuals.
        :param parent_costs: Tuple containing costs associated with two parents.
        :return: Tuple with (probability of crossover, [probabilities of mutation]).
        """
        max_crossover_probability = kwargs.get('crossover_rate', None)
        max_mutation_probability = kwargs.get('mutation_rate', None)
        min_crossover_probability = kwargs.get('xiao_min_crossover_rate', max_crossover_probability / 2)
        min_mutation_probability = kwargs.get('xiao_min_mutation_rate', max_mutation_probability / 2)
        scale_factor = kwargs.get('xiao_scale', 0.4)

        spread = minimum_cost - average_cost
        better_parent_cost = min(parent_costs)
        if better_parent_cost < average_cost:
            cross_rate = (min_crossover_probability + max_crossover_probability) / 2 \
                         + (min_crossover_probability - max_crossover_probability) / math.pi \
                         * math.atan(scale_factor * (2 * better_parent_cost - spread) / (2 * spread))
        else:
            cross_rate = max_crossover_probability

        mutation_rates = []
        for cost in parent_costs:
            if cost < average_cost:
                mutation_rate = (min_mutation_probability + max_mutation_probability) / 2 \
                                + (min_mutation_probability - max_mutation_probability) / math.pi \
                                * math.atan(scale_factor * (2 * cost - spread) / (2 * spread))
            else:
                mutation_rate = max_mutation_probability
            mutation_rates.append(mutation_rate)

        return cross_rate, mutation_rates
