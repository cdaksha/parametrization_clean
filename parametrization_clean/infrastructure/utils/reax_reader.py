#!/usr/bin/env python

"""Module to read, structure, and clean data from reference ReaxFF files:
ffield, fort.99, params.
"""

# Standard library
from typing import List, Tuple, Dict
import os
import re
import textwrap

# 3rd party packages
import numpy as np

# Local source


class ReaxReader(object):

    def __init__(self, dir_path):
        """

        Parameters
        ----------
        dir_path: str
            Directory path containing ffield, fort.99, params files.
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
        with open(os.path.join(self.dir_path, 'ffield'), 'r') as in_file:
            # Reading 1st section - General Info
            in_file.readline()
            num_general = int(in_file.readline().split().pop(0))
            ffield[1] = read_array(file=in_file, max_rows=num_general, columns=(0,))

            # Reading 2nd section - Atom Info
            ffield[2] = []
            atom_types[2] = []
            # Skipping lines
            num_atoms = int(in_file.readline().split().pop(0))
            for i in range(3):
                in_file.readline()
            for i in range(num_atoms):
                temp = []
                for j in range(4):
                    temp = temp + in_file.readline().split()
                atom_types[2].append(temp.pop(0))  # First value is atom type; ex: 'C', 'S', 'N', etc.
                ffield[2].append([float(param) for param in temp])  # Conversion from str to float

            # Reading 3rd section - Bond Info
            ffield[3] = []
            atom_types[3] = []
            num_bonds = int(in_file.readline().split().pop(0))
            in_file.readline()
            for i in range(num_bonds):
                temp = []
                for j in range(2):
                    temp = temp + in_file.readline().split()
                atom_types[3].append([temp.pop(0), temp.pop(0)])  # First two values are atom types
                ffield[3].append([float(param) for param in temp])

            # Reading 4th section - Off Diagonal Terms
            num_offdiag = int(in_file.readline().split().pop(0))
            ffield[4] = read_array(file=in_file, max_rows=num_offdiag)
            atom_types[4] = ffield[4][:, [0, 1]]  # First two columns are atom types
            ffield[4] = ffield[4][:, 2:]

            # Reading 5th section - Angular Terms
            num_angles = int(in_file.readline().split().pop(0))
            ffield[5] = read_array(file=in_file, max_rows=num_angles)
            atom_types[5] = ffield[5][:, [0, 1, 2]]  # First three columns are atom types
            ffield[5] = ffield[5][:, 3:]

            # Reading 6th section - Torsion Terms
            num_torsions = int(in_file.readline().split().pop(0))
            ffield[6] = read_array(file=in_file, max_rows=num_torsions)
            atom_types[6] = ffield[6][:, [0, 1, 2, 3]]  # First four columns are atom types
            ffield[6] = ffield[6][:, 4:]

            # Reading 7th section - Hydrogen Bonds
            num_hbonds = int(in_file.readline().split().pop(0))
            ffield[7] = read_array(file=in_file, max_rows=num_hbonds)
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
            List with columns of [ffield value | QM/Lit value | Weight].
        """
        results = []

        with open(os.path.join(self.dir_path, 'fort.99'), 'r') as in_file:
            in_file.readline()
            for line in in_file:
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
        with open(os.path.join(self.dir_path, 'params'), 'r') as in_file:
            for line in in_file:
                temp = line.split()
                cleaned = []
                for item in temp:
                    try:
                        test = float(item)
                        # Only append non-string values to cleaned list
                        cleaned.append(test)
                    except ValueError:
                        # Skipping string values
                        pass
                param_increments.append(cleaned.pop(3))  # 4th column (delta) is the parameter increment
                cleaned = [int(item) if item.is_integer() else item for item in cleaned]
                param_keys.append(cleaned[0:3])  # First three columns are parameter map
                # temporary fix for param bounds
                try:
                    # 4th and 5th columns contain min and max parameter bounds (if they exist)
                    param_bounds.append(sorted([item for item in cleaned[3:]]))
                except IndexError:
                    param_bounds.append([])

        return param_keys, param_increments, param_bounds


def write_ffield(parent_path: str, output_path: str, ffield: Dict[int, List], atom_types: Dict[int, List]):
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
    if not os.path.exists(parent_path):
        print("ffield file not found in {}...exiting function".format(parent_path))
        return

    # --- Begin Reading root/input FFIELD file and writing to output file ---
    with open(parent_path, 'r') as in_file, open(output_path, 'w') as out_file:
        # Writing 1st section - General Info
        # Assumes general parameters aren't changed, so writes based on reference ffield set
        out_file.write(in_file.readline())
        num_general_line = in_file.readline()
        out_file.write(num_general_line)
        num_general = int(num_general_line.split().pop(0))
        for _ in range(num_general):
            out_file.write(in_file.readline())

        # Writing 2nd section - Atom Info
        # Skipping lines
        num_atoms_line = in_file.readline()
        out_file.write(num_atoms_line)
        num_atoms = int(num_atoms_line.split().pop(0))
        for _ in range(3):
            out_file.write(in_file.readline())
        for i in range(num_atoms):
            atom_line = in_file.readline()  # Line containing atom...ex 'C', 'H', etc.
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
                    out_file.write(
                        " " + "{:<4}".format(item.split()[0]) + "{:>7.4f}".format(float(item.split()[1])) + "".join(
                            ["{:{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[2:]]) + "\n")
                elif float(item.split().pop(0)) < 0:  # First number is negative
                    out_file.write("{:12.4f}".format(float(item.split()[0])) + "".join(
                        ["{:>{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]]) + "\n")
                else:  # First number is positive
                    out_file.write("{:12.4f}".format(float(item.split()[0])) + "".join(
                        ["{:>{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]]) + "\n")

            # Skipping lines in input file to reach next atom
            for _ in range(3):
                in_file.readline()

        # Writing 3rd section - Bond Info
        num_bonds_line = in_file.readline()
        out_file.write(num_bonds_line)
        num_bonds = int(num_bonds_line.split().pop(0))
        out_file.write(in_file.readline())
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
                    out_file.write(item + "\n")
                else:
                    out_file.write(
                        "{:>15.4f}".format(float(item.split()[0]))
                        + "".join(["{:{w}.{p}f}".format(float(param), w=9, p=4) for param in item.split()[1:]])
                        + "\n")

            # Skipping lines in input file to reach next bond term
            for _ in range(2):
                in_file.readline()

        # Writing 4th section - Off Diagonal Terms
        num_offdiag_line = in_file.readline()
        out_file.write(num_offdiag_line)
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
                out_file.write(item + "\n")

            # Skipping lines in input file to reach next off diagonal term
            in_file.readline()

        # Writing 5th section - Angular Terms
        num_angles_line = in_file.readline()
        out_file.write(num_angles_line)
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
                out_file.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            in_file.readline()

        # Writing 6th section - Torsion Terms
        num_torsions_line = in_file.readline()
        out_file.write(num_torsions_line)
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
                out_file.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            in_file.readline()

        # Writing 7th section - Hydrogen Bonds
        num_hbonds_line = in_file.readline()
        out_file.write(num_hbonds_line)
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
                out_file.write(item + "\n")

            # Skipping lines in input file to reach next angle term
            in_file.readline()


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
    delimiter : string, default=None
        Data delimiter; assumes spaces as default.

    Returns
    -------
    np.ndarray
        Numpy array with columns of data
    """
    return np.genfromtxt(file, skip_header=skip_header, max_rows=max_rows, usecols=columns, delimiter=delimiter)
