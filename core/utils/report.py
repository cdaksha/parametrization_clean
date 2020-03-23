#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temporary module to store functions used in reporting/writing output text files for data insight.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List
import csv

# 3rd party packages

# Local source
from core.settings.config import settings as s


# TODO: Class that has attributes population, case_numbers that can generate several reports?
class ReportWriter(object):

    @staticmethod
    def write_rows_to_csv(file_path, data, delimiter=','):
        with open(file_path, 'w') as outf:
            writer = csv.writer(outf, delimiter=delimiter)
            writer.writerows(data)

    @staticmethod
    def append(formatted_data_string, file_path):
        with open(file_path, 'a+') as outf:
            outf.write(formatted_data_string)

    @staticmethod
    def write_summary(population, best_child, case_numbers, file_path):
        """Generate report for a given generation at location `file_path`."""
        pop_size = s.GA.populationSize

        num_retrieved = len(population)
        num_failed = pop_size - num_retrieved

        best_child_idx = population.index(best_child)
        with open(file_path, 'w') as outf:
            outf.write("---------- GENERATION SUMMARY ----------\n")
            outf.write("Number of successful cases: {:.3f} ({:.2f}%)\n"
                       .format(num_retrieved, 100 * num_retrieved / pop_size))
            outf.write("Number of failed cases: {:.3f} ({:.2f}%)\n"
                       .format(num_failed, 100 * num_failed / pop_size))
            outf.write("Best Cost/Fitness Real ReaxFF Error (from Master GA): {:.3f}\n"
                       .format(best_child.cost))
            outf.write("Corresponding best Master GA case: case-{}\n".
                       format(case_numbers[best_child_idx]))
