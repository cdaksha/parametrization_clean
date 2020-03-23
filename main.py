#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Driver module to run the genetic algorithm.

__author__ = "Chad Daksha"
"""

# Standard library
import sys
import os

# 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local source
import core.genetic_algorithm.helpers as h
import core.genetic_algorithm.population as pop
import core.neural_net.nested_ga as nga
import core.utils.population_io as pop_io
import core.neural_net.ann as ann
from core.utils.report import ReportWriter
from core.utils.reax_io import read_array
from core.settings.config import settings as s
from core.genetic_algorithm import helpers as util
from core.neural_net.ann import normalize


def main(generation_num):
    # GA Population Control & I/O
    master_population_controller = pop.Population()  # For Master GA
    nested_population_controller = pop.Population()  # For Nested GA
    nested_population_controller.mutation_rate = s.NestedGA.mutation.probability
    nested_population_controller.crossover_rate = s.NestedGA.crossover.probability
    pop_io_obj = pop_io.PopulationIO(generation_num)

    # Constants
    population_path = s.path.population
    ga_only = s.GA_ONLY
    epochs = s.ANN.epochs
    val_rsquare_threshold = s.ANN.validationRsquareThreshold
    nested_iters = s.NestedGA.iterations

    print("Generation: {}".format(generation_num))
    if generation_num == 1:
        print("First generation -> Initializing Population...")
        parents = master_population_controller.initialize()
        pop_io_obj.write_population(parents)
    else:
        previous_generation = generation_num - 1  # Data is read from previous_generation

        previous_generation_path = os.path.join(population_path, 'generation-' + str(previous_generation))
        sorted_errors_file = os.path.join(previous_generation_path, '00-all-errors-sorted.txt')
        # Keeping track of generation number & minimum total error
        gen_vs_error_file = os.path.join(population_path, '00-generation-vs-error.txt')
        summary_file = os.path.join(previous_generation_path, '00-gen-summary.txt')

        print("Reading population from generation {}...".format(previous_generation))
        parents = pop_io_obj.read_population()
        case_numbers = [parent.index for parent in parents]

        # Master GA report statistics
        # Report all errors of generation
        print("Generating Master GA report statistics for generation {}...".format(previous_generation))
        sorted_case_numbers_and_costs = h.sort_costs_and_case_numbers(parents, case_numbers)
        ReportWriter.write_rows_to_csv(sorted_errors_file, sorted_case_numbers_and_costs)

        best_child = h.get_best_child(parents)
        ReportWriter.write_summary(parents, best_child, case_numbers, summary_file)
        ReportWriter.append('{}\t{:.5f}\n'.format(previous_generation, best_child.cost),
                            gen_vs_error_file)

        print("Creating generation {}...".format(generation_num))
        if ga_only:  # Run GA only, no Nested GA/ANN
            children = master_population_controller.propagate(parents)
        else:
            # Fitting ANN
            print("Fitting ANN enabled - reading previous generation data...")
            num_generations_to_train_on = 10
            previous_population = pop_io_obj.read_previous_n_populations(num_generations_to_train_on)
            model, history, train_stats = ann.run(previous_population, epochs, generation_num)
            hist = pd.DataFrame(history.history)
            final_results = hist.tail(1)

            # ---------- PLOTTING TRUE VS PREDICTED ERRORS FROM ANN ----------
            # For error calculation
            true_vals = previous_population[0].true_vals
            weights = previous_population[0].weights

            previous_parents_x = pd.DataFrame(data=[parent.params for parent in previous_population])
            cols_to_drop = previous_parents_x.std()[abs(previous_parents_x.std()) <= 1e-4].index
            previous_parents_x = previous_parents_x.drop(cols_to_drop, axis=1)
            parents_predicted_output = model.predict(normalize(previous_parents_x, train_stats))

            reax_total_errors = [parent.cost for parent in previous_population]
            errors = [[util.reax_error(reax_pred=x, true_val=y, weight=z)
                       for x, y, z in zip(vals_row, true_vals, weights)]
                      for vals_row in parents_predicted_output]
            predicted_total_errors = [sum(error_list) for error_list in errors]

            # TESTING predicted objective function vs. true objective function
            generation_path = os.path.join(s.path.population, 'generation-' + str(1))
            counter = 1
            while os.path.isdir(generation_path):
                counter += 1
                generation_path = os.path.join(s.path.population, 'generation-' + str(counter))
            generation_path = os.path.join(s.path.population, 'generation-' + str(counter - 1))
            img_path = os.path.join(generation_path, 'Predicted_vs_True_Total_Error.png')
            plt.scatter(reax_total_errors, predicted_total_errors)
            x = np.linspace(min(reax_total_errors), max(reax_total_errors), 1000)
            plt.plot(x, x)
            plt.xlabel('True Total Error')
            plt.ylabel('ANN Predicted Total Error')
            plt.savefig(img_path)
            # ---------- PLOTTING TRUE VS PREDICTED ERRORS FROM ANN ----------

            # ANN Summary statistics for population
            ann_summary_file = os.path.join(population_path, '00-ann-summary')
            if previous_generation == 1:
                with open(ann_summary_file, 'w') as outf:
                    outf.write('Generation #\t{}\n'.format(list(hist.columns.values)))
            ReportWriter.append('{}\t{}\n'.format(generation_num,
                                                  final_results.to_string(header=False, index=False,
                                                                          index_names=False)), ann_summary_file)

            # ANN model statistics for generation
            # temp fix
            with open(os.path.join(population_path, 'generation-' + str(generation_num - 1), '00-ANN-summary'), 'w') as outf:
                outf.write(hist.to_string())

            if final_results['val_r_square'].values < val_rsquare_threshold:
                # Propagate next generation - GA ONLY
                children = master_population_controller.propagate(parents)
            else:
                # Running Nested GA
                children = nga.nested_ga(num_iters=nested_iters,
                                         master_parents=parents,
                                         model=model,
                                         train_stats=train_stats,
                                         pop_controller=master_population_controller)

                # Appending to report statistics - ANN data
                predicted_best_child = h.get_best_child(children)
                print("Master GA best predicted error: {:.3f}".format(best_child.cost))
                print("Nested GA best predicted error: {:.3f}".format(predicted_best_child.cost))
                ReportWriter.append('Best Predicted Error (from Nested GA): {:.3f}\n'.format(predicted_best_child.cost),
                                    summary_file)
        # finally:
        pop_io_obj.write_population(children)

    print("Adapted crossover and mutation rates: {} and {}".format(master_population_controller.crossover_rate,
                                                                   master_population_controller.mutation_rate))
    adaptive_ga_file = os.path.join(population_path, '00-AGA-cross-mutate-rates')
    ReportWriter.append('generation {}\tcrossover {:.3f}\tmutate {}\n'
                        .format(generation_num, master_population_controller.crossover_rate,
                                master_population_controller.mutation_rate), adaptive_ga_file)

    print("Generation created at {} ... submit ReaxFF optimizations."
          .format(population_path + '/generation-' + str(generation_num) + '/'))
    print("Done!")


if __name__ == "__main__":
    gen_num = int(sys.argv[1])
    main(gen_num)
