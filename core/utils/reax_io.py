#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module to read, structure, and clean data from reference files: ffield, fort.99, params, trainset.in

__author__ = "Chad Daksha"
"""

# Standard library
from collections import OrderedDict
from typing import List, Tuple, Dict
import re
import os
import textwrap  # ffield output formatting purposes

# 3rd party packages
import numpy as np

# Local source


class ReaxIO(object):

    def __init__(self, dir_path):
        """

        Parameters
        ----------
        dir_path: str
            Directory path that would contain ffield, fort.99, params, trainset.in files.
        """
        self.dir_path = dir_path

    def read_ffield(self) -> Tuple[Dict, Dict]:
        """Read ffield file into dictionary, mapping each section to corresponding rows.

        ffield[section number] = [parameters corresponding to that section].
        ASSUMES ffield file name is 'ffield'.

        Returns
        -------
        ffield: Dict[int, List]
            ffield[section number] = [parameters corresponding to section]
        atom_types: Dict[int, List]
            atom_types[section number] = [atom types for each section]
        """
        ffield = {}
        atom_types = {}

        # --- Begin Reading FFIELD File ---
        with open(os.path.join(self.dir_path, 'ffield'), 'r') as f:
            # Reading 1st section - General Info
            f.readline()
            num_general = int(f.readline().split().pop(0))
            ffield[1] = read_array(file=f, max_rows=num_general, columns=(0,))

            # Reading 2nd section - Atom Info
            ffield[2] = []
            atom_types[2] = []
            # Skipping lines
            num_atoms = int(f.readline().split().pop(0))
            for i in range(3):
                f.readline()
            for i in range(num_atoms):
                temp = []
                for j in range(4):
                    temp = temp + f.readline().split()
                atom_types[2].append(temp.pop(0))  # First value is atom type; ex: 'C', 'S', 'N', etc.
                ffield[2].append([float(param) for param in temp])  # Conversion from str to float

            # Reading 3rd section - Bond Info
            ffield[3] = []
            atom_types[3] = []
            num_bonds = int(f.readline().split().pop(0))
            f.readline()
            for i in range(num_bonds):
                temp = []
                for j in range(2):
                    temp = temp + f.readline().split()
                atom_types[3].append([temp.pop(0), temp.pop(0)])  # First two values are atom types
                ffield[3].append([float(param) for param in temp])

            # Reading 4th section - Off Diagonal Terms
            num_offdiag = int(f.readline().split().pop(0))
            ffield[4] = read_array(file=f, max_rows=num_offdiag)
            atom_types[4] = ffield[4][:, [0, 1]]  # First two columns are atom types
            ffield[4] = ffield[4][:, 2:]

            # Reading 5th section - Angular Terms
            num_angles = int(f.readline().split().pop(0))
            ffield[5] = read_array(file=f, max_rows=num_angles)
            atom_types[5] = ffield[5][:, [0, 1, 2]]  # First three columns are atom types
            ffield[5] = ffield[5][:, 3:]

            # Reading 6th section - Torsion Terms
            num_torsions = int(f.readline().split().pop(0))
            ffield[6] = read_array(file=f, max_rows=num_torsions)
            atom_types[6] = ffield[6][:, [0, 1, 2, 3]]  # First four columns are atom types
            ffield[6] = ffield[6][:, 4:]

            # Reading 7th section - Hydrogen Bonds
            num_hbonds = int(f.readline().split().pop(0))
            ffield[7] = read_array(file=f, max_rows=num_hbonds)
            atom_types[7] = ffield[7][:, [0, 1, 2]]  # First three columns are atom types
            ffield[7] = ffield[7][:, 3:]

        return ffield, atom_types

    def read_fort99(self) -> List:
        """Read fort.99 output created from simulation. Store output in list; primarily useful for
        error and total error.

        ASSUMES fort99 file name is 'fort.99'.

        Returns
        -------
        results: List
            List with columns of [ffield value | QM/Lit value | Weight | Error | Total Error].
        """
        results = []

        with open(os.path.join(self.dir_path, 'fort.99'), 'r') as f:
            f.readline()
            for line in f:
                if not line.strip():
                    # Skip empty lines
                    continue
                else:
                    # Extracting all floating points from the line (that are not right next to strings)
                    # All numbers with %.4f formatting are extracted.
                    numbers = re.findall(r"[-+]?\d*\.\d{4}", line)

                    # First three values are reaxFF value, true value, and weight
                    results.append([float(string) for string in numbers[0:3]])

        return results

    def read_fort99_energy_sections(self) -> List:
        """Read only the lines starting with `Energy`."""
        results = []
        with open(os.path.join(self.dir_path, 'fort.99'), 'r') as f:
            f.readline()
            for line in f:
                if not line.startswith('Energy'):
                    # Skip empty lines and only consider 'Energy' lines
                    continue
                # Extracting all floating points from the line (that are not right next to strings)
                # All numbers with %.4f formatting are extracted.
                numbers = re.findall(r"[-+]?\d*\.\d{4}", line)

                # First two values are reaxFF value, true value
                results.append([float(string) for string in numbers[0:3]])
        return results

    def read_params(self) -> Tuple[List, List, List]:
        """Read PARAMS file into lists containing a map for the reference initial parameters and containing the
        parameter min/max bounds, if specified.

        ASSUMES params file name is 'params'.

        Returns
        -------
        param_keys: List[List[int]]
            list containing 3 columns: [section # | line/row # | parameter #].
        param_increments: List[int]
            list containing appropriate increment values for each parameter in the parameter map.
        param_bounds: List[List[float]]
            list containing minimum & maximum values (if specified) for each parameter in the parameter map.
            list format: [minimum value | maximum value].
        """
        param_keys = []
        param_increments = []
        param_bounds = []

        # --- Begin Reading PARAMS File ---
        with open(os.path.join(self.dir_path, 'params'), 'r') as f:
            for line in f:
                temp = line.split()
                cleaned = []
                for item in temp:
                    try:
                        test = float(item)
                        # Only append non-string values to cleaned list
                        cleaned.append(test)
                    except ValueError:
                        # TODO: LOGGING
                        pass
                param_increments.append(cleaned.pop(3))  # 4th column (delta) is the parameter increment
                cleaned = [int(item) if item.is_integer() else item for item in cleaned]
                param_keys.append(cleaned[0:3])  # First three columns are parameter map
                # temporary fix for param bounds
                try:
                    param_bounds.append([item for item in cleaned[3:]])  # 4th and 5th columns contain min and max
                except IndexError:
                    param_bounds.append([])

        return param_keys, param_increments, param_bounds

    def read_trainset(self) -> OrderedDict:
        """Read TRAINSET.IN file into dictionary mapping section titles to the number of lines/constraints in that section.
        ONLY READS THE `ENERGY` SECTIONS.

        ASSUMES section titles always start with one '#'.
        If more than one '#' is used, the line is just a commented line.

        Returns
        -------
        lines_per_section_dict: Dict[str, int]
            Maps quantum/DFT section titles to the number of lines/constraints in that section.
        """
        lines_per_section_dict = OrderedDict()
        # --- Begin Reading TRAINSET.IN File ---
        with open(os.path.join(self.dir_path, 'trainset.in'), 'r') as f:
            for line in f:
                line = line.strip()

                if re.match(r'^#{1} [ a-zA-Z]', line):
                    # Extract section title up till left parenthesis
                    section_title = line[2:].split('(')[0]
                    # print(section_title)
                    lines_per_section_dict[section_title] = 0

                # Do not count constraints that are commented out
                if line.startswith('#') or line.startswith('END'):
                    continue

                try:
                    float(line.split(' ')[0])
                    lines_per_section_dict[section_title] += 1
                except (ValueError, UnboundLocalError):
                    continue

        return lines_per_section_dict


# ---------- HELPER FUNCTIONS ----------
# I/O
def write_ffield(parent_path: str, output_path: str, atom_types: Dict[int, List], ffield: Dict[int, List]):
    """Using the formatted ffield at `parent_path`, write the provided `ffield` to `output_path`.

    Parameters
    ----------
    parent_path: str
        File path with reference ffield file.
    output_path: str
        File path to output new ffield file.
    atom_types: Dict[int, List]
        atom_types[section number] = [atom types for each section]
    ffield: Dict[int, List]
        ffield[section number] = [parameters corresponding to section]
    """
    # TODO: figure out if this is necessary
    import os
    if not os.path.exists(parent_path):
        print("ffield file not found in {}...exiting function".format(parent_path))
        return
    # Commented out to allow overwriting
    #if os.path.exists(output_path):
    #    print("{} already exists...exiting function.".format(output_path))
    #    return

    # --- Begin Reading root/input FFIELD file and writing to output file ---
    with open(parent_path, 'r') as f, open(output_path, 'w') as outf:
        # Writing 1st section - General Info
        # Assumes general parameters aren't changed, so writes based on reference ffield set
        outf.write(f.readline())
        num_general_line = f.readline()
        outf.write(num_general_line)
        num_general = int(num_general_line.split().pop(0))
        for _ in range(num_general):
            outf.write(f.readline())

        # Writing 2nd section - Atom Info
        # Skipping lines
        num_atoms_line = f.readline()
        outf.write(num_atoms_line)
        num_atoms = int(num_atoms_line.split().pop(0))
        for _ in range(3):
            outf.write(f.readline())
        for i in range(num_atoms):
            atom_line = f.readline()  # Line containing atom...ex 'C', 'H', etc.
            atom = [atom_line.split().pop(0)]
            atom_params = ffield[2][i]  # Parameters corresponding to atom in atom section of ffield dict

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            formatted_params = ["{:>{w}.{p}f}".format(param, w=9, p=4) for param in atom_params]
            atom_output = atom + [" "] + formatted_params
            formatted_output = "".join(atom_output)
            formatted_output = textwrap.wrap(formatted_output, width=75)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                if item is formatted_output[0]:  # First item - contains atom identifier...ex 'C', 'H', etc.
                    outf.write(
                        " " + "{:<4}".format(item.split()[0]) + "{:>7.4f}".format(float(item.split()[1])) + "".join(
                            ["{:{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[2:]]) + "\n")
                elif float(item.split().pop(0)) < 0:  # First number is negative
                    outf.write("{:12.4f}".format(float(item.split()[0])) + "".join(
                        ["{:>{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]]) + "\n")
                else:  # First number is positive
                    outf.write("{:12.4f}".format(float(item.split()[0])) + "".join(
                        ["{:>{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]]) + "\n")

            # Skipping lines in input file to reach next atom
            for _ in range(3):
                f.readline()

        # Writing 3rd section - Bond Info
        num_bonds_line = f.readline()
        outf.write(num_bonds_line)
        num_bonds = int(num_bonds_line.split().pop(0))
        outf.write(f.readline())
        for i in range(num_bonds):
            # Combining atom IDs of bonded atoms...ex '1' and '2', etc.
            # with bond parameters corresponding to bonded atoms
            bond_params = [int(atomID) for atomID in atom_types[3][i]] + ffield[3][i]

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            # formatted_params = ["{:>{w}.{p}f}".format(param, w=9, p=4) for param in bond_params]
            formatted_params = []
            for param in bond_params:
                if param is bond_params[0] or param is bond_params[1]:
                    formatted_params.append("{:>{w}d}".format(param, w=3))
                else:
                    formatted_params.append("{:>{w}.{p}f}".format(param, w=9, p=4))
            formatted_output = "".join(formatted_params)
            formatted_output = textwrap.wrap(formatted_output, width=78)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                if item is formatted_output[0]:  # First item - contains bonded atom pair...ex '1' and '2', etc.
                    outf.write(item + "\n")
                else:
                    outf.write(
                        "{:>15.4f}".format(float(item.split()[0]))
                        + "".join(["{:{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]])
                        + "\n")

            # Skipping lines in input file to reach next bond term
            for _ in range(2):
                f.readline()

        # Writing 4th section - Off Diagonal Terms
        num_offdiag_line = f.readline()
        outf.write(num_offdiag_line)
        num_offdiag = int(num_offdiag_line.split().pop(0))
        # Rejoining first two atom type columns
        offdiag_params = np.concatenate((atom_types[4], ffield[4]), axis=1)
        for i in range(num_offdiag):
            temp_params = offdiag_params[i]

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            formatted_params = []
            for index, param in enumerate(temp_params):
                if index in {0, 1}:
                    formatted_params.append("{:>{w}d}".format(int(param), w=3))
                else:
                    formatted_params.append("{:>{w}.{p}f}".format(param, w=9, p=4))
            formatted_output = "".join(formatted_params)
            formatted_output = textwrap.wrap(formatted_output, width=60)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                outf.write(item + "\n")

            # Skipping lines in input file to reach next off diagonal term
            f.readline()

        # Writing 5th section - Angular Terms
        num_angles_line = f.readline()
        outf.write(num_angles_line)
        num_angles = int(num_angles_line.split().pop(0))
        # Rejoining first three atom type columns
        angle_params = np.concatenate((atom_types[5], ffield[5]), axis=1)
        for i in range(num_angles):
            temp_params = angle_params[i]

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            formatted_params = []
            for param in temp_params[0:3]:
                formatted_params.append("{:>{w}d}".format(int(param), w=3))
            for param in temp_params[3:]:
                formatted_params.append("{:>{w}.{p}f}".format(param, w=9, p=4))
            formatted_output = "".join(formatted_params)
            formatted_output = textwrap.wrap(formatted_output, width=72)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                outf.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            f.readline()

        # Writing 6th section - Torsion Terms
        num_torsions_line = f.readline()
        outf.write(num_torsions_line)
        num_torsions = int(num_torsions_line.split().pop(0))
        # Rejoining first four atom type columns
        torsion_params = np.concatenate((atom_types[6], ffield[6]), axis=1)
        for i in range(num_torsions):
            temp_params = torsion_params[i]

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            formatted_params = []
            for param in temp_params[0:4]:
                formatted_params.append("{:>{w}d}".format(int(param), w=3))
            for param in temp_params[4:]:
                formatted_params.append("{:>{w}.{p}f}".format(param, w=9, p=4))
            formatted_output = "".join(formatted_params)
            formatted_output = textwrap.wrap(formatted_output, width=75)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                outf.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            f.readline()

        # Writing 7th section - Hydrogen Bonds
        num_hbonds_line = f.readline()
        outf.write(num_hbonds_line)
        num_hbonds = int(num_hbonds_line.split().pop(0))
        # Rejoining first three atom type columns
        hbonds_params = np.concatenate((atom_types[7], ffield[7]), axis=1)
        for i in range(num_hbonds):
            temp_params = hbonds_params[i]

            # Formatting parameters for output. 'w' = field width, 'p' = decimal precision
            formatted_params = []
            for param in temp_params[0:3]:
                formatted_params.append("{:>{w}d}".format(int(param), w=3))
            for param in temp_params[3:]:
                formatted_params.append("{:>{w}.{p}f}".format(param, w=9, p=4))
            formatted_output = "".join(formatted_params)
            formatted_output = textwrap.wrap(formatted_output, width=45)

            # TODO: Explain complicated formatting procedure!
            for item in formatted_output:
                outf.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            f.readline()


def read_array(file, skip_header: int = 0, max_rows: int = None, columns: Tuple[int] = None, delimiter=None)\
        -> np.ndarray:
    """Read a specified portion of a file into an array using NumPy's genfromtxt function.

    Parameters
    ----------
    file
        string file path OR FileIO object from which the data is extracted
    skip_header : int
        Number of header lines to skip
    max_rows : int
        Maximum number of rows to read into array
    columns : tuple, default=None
        Tuple of columns from data to read; ex (0, 2, 4) reads cols 0, 2, 4

    Returns
    -------
    np.ndarray
        Numpy array with columns of data
    """
    return np.genfromtxt(file, skip_header=skip_header, max_rows=max_rows, usecols=columns, delimiter=delimiter)


# Trainset & fort.99 parsing helpers
def remove_empty_sections(lines_per_section_dict):
    """Remove sections that map to zero lines in the trainset dictionary
    that maps section titles to number of lines.
    """
    return OrderedDict({k: v for k, v in lines_per_section_dict.items() if v != 0})


def get_constraints_per_section(fort99_data, lines_per_section_dict) -> OrderedDict:
    """Given fort.99 data and a dictionary mapping sections to the number of lines in that section, separate the fort99 data accordingly.
    """
    constraints_per_section_dict = OrderedDict()
    current_index = 0
    for k, num_lines in lines_per_section_dict.items():
        final_index = current_index + num_lines
        constraints_per_section_dict[k] = fort99_data[current_index:final_index]
        current_index += num_lines
    return constraints_per_section_dict


def get_energy_curve_sections(constraints_per_section_dict) -> List[List]:
    """Return fort.99 reaxFF & DFT data corresponding to energy curve sections, identified by the
    'volume' or 'restraint' keywords in the `trainset.in` section titles."""
    constraints_per_section_tuples = list(constraints_per_section_dict.items())
    return [fort99_data for title, fort99_data in constraints_per_section_tuples
            if title.split(' ', 1)[0].lower() == 'volume'
            or title.split(' ', 1)[0].lower() == 'restraint'
            ]


def retrieve_energy_curve_data(reax_io_obj, fort99_data) -> List[List]:
    """Return fort.99 data corresponding to energy curve sections, identified by the 'volume' or 'restraint'
    keywords in the `trainset.in` section titles."""
    lines_per_section_dict = reax_io_obj.read_trainset()
    cleaned_lines_per_section_dict = remove_empty_sections(lines_per_section_dict)
    constraints_per_section_dict = get_constraints_per_section(fort99_data, cleaned_lines_per_section_dict)
    return get_energy_curve_sections(constraints_per_section_dict)
