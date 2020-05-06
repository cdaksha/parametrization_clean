#!/usr/bin/env python

"""Module to convert data outputs from ReaxFF I/O object to desired data structures for domain objects."""

# Standard library

# 3rd party packages

# Local source


class Fort99Extractor:

    def __init__(self, fort99_data):
        self.data = fort99_data

    def get_reax_energies(self):
        return [row[0] for row in self.data]

    def get_dft_energies(self):
        return [row[1] for row in self.data]

    def get_weights(self):
        return [row[2] for row in self.data]
