#!/usr/bin/env python

"""Uses ANN to propagate population several times, i.e., a nested GA propagator that runs for several iterations
without performing any ReaxFF optimizations. Requires TensorFlow 2.0.
"""

# Standard library
from typing import List

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.neural_network.ann import FeedForwardNet
from parametrization_clean.use_case.port.settings_repository import IAllSettings
from parametrization_clean.use_case.port.population_repository import IPopulationRepository
from parametrization_clean.use_case.population_propagator import PopulationPropagator


class GeneticNeuralNetPropagator:

    def __init__(self, settings_repository: IAllSettings, population_repository: IPopulationRepository):
        self.neural_net_settings = settings_repository.neural_net_settings
        self.population_repository = population_repository

        training_population = population_repository.get_previous_n_populations(
            self.neural_net_settings.num_populations_to_train_on)
        self.neural_net = FeedForwardNet(training_population,
                                         self.neural_net_settings.verbosity,
                                         self.neural_net_settings.train_fraction,
                                         self.neural_net_settings.num_epochs)

        self.population_propagator = PopulationPropagator(settings_repository, population_repository)
        self.root_individual = self.population_repository.get_root_individual()
        self.error_strategy = settings_repository.strategy_settings.error_strategy
        self.population_size = settings_repository.ga_settings.population_size

    def train_neural_net(self):
        model, history = self.neural_net.execute()
        return model, history

    def execute(self, parents: List[Individual]):
        model, history = self.train_neural_net()
        if self.final_ann_accuracy_is_poor(history):
            final_generation = self.run_without_ann(parents, model)
        else:
            final_generation = self.run_with_ann(parents, model)
        return final_generation, model, history

    def run_without_ann(self, parents, model):
        final_generation, _ = self.propagate_first(parents, model)
        return final_generation

    def run_with_ann(self, parents, model):
        next_generation, best_master_parents = self.propagate_first(parents, model)
        final_generation = self.propagate_remaining(next_generation, model)
        return final_generation

    def final_ann_accuracy_is_poor(self, history):
        accuracy_is_poor = False
        final_val_r_squared = history.history['val_r_square'][-1]
        if final_val_r_squared < self.neural_net_settings.minimum_validation_r_squared:
            accuracy_is_poor = True
        return accuracy_is_poor

    def propagate_first(self, parents: List[Individual], model):
        """First propagation/iteration in nested genetic algorithm. Preserve best two parents from master GA."""
        next_generation = self.population_propagator.execute(parents)
        # Update costs for all except top 2 individuals
        y_predicted = self.neural_net.predict_outputs(model, next_generation[2:])
        self.update_costs(next_generation[2:], y_predicted)

        best_master_parents = next_generation[0:2]
        return next_generation, best_master_parents

    def propagate_remaining(self, population: List[Individual], model) -> List[Individual]:
        """Run remaining nested genetic algorithm iterations. Preserve best two parents from master GA."""
        next_generation = population
        for i in range(self.neural_net_settings.num_nested_ga_iterations):
            next_generation = self.population_propagator.execute(next_generation)
            y_predicted = self.neural_net.predict_outputs(model, next_generation[2:])
            self.update_costs(next_generation[2:], y_predicted)
        return next_generation

    def update_costs(self, population: List[Individual], y_predicted):
        costs = self.neural_net.compute_costs(y_predicted, self.root_individual, self.error_strategy)
        for individual, y_pred_row, cost in zip(population, y_predicted, costs):
            individual.reax_energies = y_pred_row
            individual.cost = cost
