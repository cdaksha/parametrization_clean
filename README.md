# ReaxFF Parametrization with Clean Architecture
[![Build Status](https://travis-ci.com/cdaksha/parametrization_clean.svg?token=LpB61vRRhRXYf6MrmquF&branch=master)](https://travis-ci.com/cdaksha/parametrization_clean)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Contains Python files and Bash scripts as a basis for automated ReaxFF parametrization. Uses the genetic algorithm (GA)
algorithm as well as an artificial neural network (ANN) to optimize a ReaxFF parameter set. Note that this version
is refactored to attempt to comply with Robert Martin's Clean Architecture guidelines. Using the generational genetic
algorithm and neural net (the latter only if enabled), runs one generation, awaiting submission of standalone
ReaxFF optimizations.

## Getting Started

The project files are available as a GitHub repository [here](https://github.com/cdaksha/parametrization_clean).
The project can also be accessed through PyPi [here](https://pypi.org/project/parametrization-clean-cdaksha/1.0.0/);
the corresponding pip installation command is

```
$pip install parametrization-clean-cdaksha
```

If you don't have pip and/or Python installed, then [this](https://docs.python-guide.org/starting/installation/)
guide may prove helpful in performing a basic setup. If, for whatever reason, a distribution manager such as pip or
conda is not available, then the required packages for running the application are shown in `requirements/prod.txt`.

The current implementation supports a command-line interface with usage

```
$python cli.py --g GENERATION_NUMBER --t TRAINING_PATH --p POPULATION_PATH
```

where *GENERATION_NUMBER* is the current generation number in the generational genetic algorithm, *TRAINING_PATH*
is the file path location of the reference ReaxFF training set files, and *POPULATION_PATH* is the location at which
the user wishes the generational genetic algorithm files to be outputted.

If *GENERATION_NUMBER* = 1, then the first population is initialized, whose ReaxFF optimizations can then be submitted
for evaluation of the parameters. If *GENERATION_NUMBER* > 1, the previous generation's data is read from
*POPULATION_PATH*, and classic genetic operators are applied to generate the next generation and output to the
*POPULATION_PATH* once again, after which the corresponding ReaxFF optimizations may be submitted.

To automate the generational genetic algorithm, an example slurm script is provided in the `scripts` directory. This
allows concurrent submission of ReaxFF optimizations and continuation of the generational genetic algorithm until a
threshold, defined by a maximum generation number, is reached.

Note that several options are provided for potential mutation, crossover, etc., algorithms that the user may use.
Reasonable defaults based on the literature are provided, but they are easy to override by defining a custom
`config.json` file in the project root directory.

### Prerequisites

Project relies on usage of *pip* for installing required dependencies. Additionally, standalone ReaxFF is required to
run the optimizations for the files that are created. Note that reference ReaxFF training files for the system at hand
are required. At the very least, the training set directory must contain

```
training_files
│---ffield
│---geo
│---params
│---control
│---trainset.in
│---fort.99
```

Note that `iopt` files are dynamically created with a single line entry, *0*, to instruct ReaxFF not to use the "manual"
ReaxFF parameter optimization scheme: successful one parameter parabolic extrapolation (SOPPE).

Currently, `fort.99` is required in the training set directory to retrieve DFT energies and weights in the beginning.

## Running the tests

The project can easily be tested by running the test suite through

```
$tox
```

in the project root after installation. Note that TensorFlow is used in building the neural network. If TensorFlow
is unavailable for installation, then the tests corresponding to the neural network will not run. Code coverage can be
checked by running

```
$py.test --cov-report term-missing --cov=parametrization_clean
```

To check for conformation to PEP standards, one can use

```
flake8
```

in the project root directory.

## Authors

* **Chad Daksha** - *Initial work* - [cdaksha](https://github.com/cdaksha)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
