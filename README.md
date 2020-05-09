# ReaxFF Parametrization with Clean Architecture
[![Build Status](https://travis-ci.com/cdaksha/parametrization_clean.svg?token=LpB61vRRhRXYf6MrmquF&branch=master)](https://travis-ci.com/cdaksha/parametrization_clean)
[![codecov](https://codecov.io/gh/cdaksha/parametrization_clean/branch/master/graph/badge.svg?token=AWGP6HY2VD)](https://codecov.io/gh/cdaksha/parametrization_clean)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Contains Python files and Bash scripts as a basis for automated ReaxFF parametrization. Uses the genetic algorithm (GA)
algorithm as well as an artificial neural network (ANN) to optimize a ReaxFF parameter set. Note that this version
is refactored to attempt to comply with Robert Martin's Clean Architecture guidelines. Using the generational genetic
algorithm and neural net (the latter only if enabled), runs one generation, awaiting submission of standalone
ReaxFF optimizations.

## Getting Started

As with all work flows, **it is a good idea
to work within a Python virtual environment** to create an isolated environment for this Python project.
[This](https://docs.python.org/3/tutorial/venv.html) source contains a good tutorial on how to setup a virtual
environment. Another source for help with Python virtual environments is
[here](https://docs.python-guide.org/dev/virtualenvs/).

The project files are available as a GitHub repository [here](https://github.com/cdaksha/parametrization_clean).
The project can also be accessed through PyPi [here](https://pypi.org/project/parametrization-clean-cdaksha/);
the corresponding pip installation command is

```commandline
$pip install parametrization-clean-cdaksha
```

If you don't have pip and/or Python installed, then [this](https://docs.python-guide.org/starting/installation/)
guide may prove helpful in performing a basic setup. If, for whatever reason, a distribution manager such as pip or
conda is not available, then the required packages for running the application are shown in `requirements/prod.txt`.

After installing the package with pip, the current implementation supports a command-line interface with usage

```commandline
$cli --g GENERATION_NUMBER --t TRAINING_PATH --p POPULATION_PATH --c CONFIG_PATH
```

where *GENERATION_NUMBER* is the current generation number in the generational genetic algorithm, *TRAINING_PATH*
is the file path location of the reference ReaxFF training set files, *POPULATION_PATH* is the location at which
the user wishes the generational genetic algorithm files to be outputted, and *CONFIG_PATH* is the location at which
a user-defined JSON configuration file can be entered. The last field is not required, as defaults are provided for
each algorithm and genetic algorithm setting.

All options used in the default configuration are shown in the `example` folder [here](example/config.json). The user
can tune one (or many) of these parameters by defining a config.json file at the *CONFIG_PATH* location, such as the
following:

```json
{
  "strategy_settings": {
    "mutation": "nakata"
  },
  "ga_settings": {
    "population_size": 50,
    "use_neural_network": true
  }
}
```

Note that, **at the very least, the user should define the *population_size* parameter**. This parameter controls the
number of individuals in the genetic algorithm's population.

If *GENERATION_NUMBER* = 1, then the first population is initialized, whose ReaxFF optimizations can then be submitted
for evaluation of the parameters. If *GENERATION_NUMBER* > 1, the previous generation's data is read from
*POPULATION_PATH*, and classic genetic operators are applied to generate the next generation and output to the
*POPULATION_PATH* once again, after which the corresponding ReaxFF optimizations may be submitted.

To automate the generational genetic algorithm, an example bash script is provided in the `example` directory
[here](example/main.sh). This allows concurrent submission of ReaxFF optimizations and continuation of the
generational genetic algorithm until a threshold, defined by a maximum generation number, is reached. In practice,
this example simply uses this Python application to propagate the genetic algorithm from one generation to the next,
then submits ReaxFF optimizations for those created individuals, monitoring their completion. After the generation
is completed (based on job completion status), the cycle repeats.

In practice, this application lends itself to usage with supercomputing. The corresponding supercomputing job for
a SLURM-based environment is also available [here](example/job.qs). This wrapper SLURM script merely calls the bash
script, but makes it so that the user does not need to keep the bash script job running on their own computer; instead,
the bash script will be running on a node in the supercomputer.

Again, note that several options are provided for potential mutation, crossover, etc., algorithms that the user may use.
Reasonable defaults based on the literature are provided, but they are easy to override by defining the custom
`config.json` file and providing the location to the command line interface, as suggested earlier.

Thus, the typical setup for usage of this application is as follows:
  1. Make sure that the application and requirements are properly installed on the machine, as well as
     standalone ReaxFF.
  2. Define the user configuration file with any changes to parameters of interest, **or at least the population size**.
  3. Create a wrapper bash script that automates propagation of generations back-to-back based on the example bash
     script provided, with proper submission details for the standalone ReaxFF jobs.
  4. If using supercomputing resources, create a wrapper job script (SLURM, PBS, etc.) that submits the wrapper bash
     script, configured appropriately based on the environment.

The workflow for running the optimizations is then as follows:
  1. Make sure the user configuration is set as desired and reconfigure beforehand if necessary.
  2. Set the reference/training set directory with the required model files:
     ffield, geo, params, control, trainset.in, and fort.99.
  3. Set the population output directory, where you wish the generational GA files to be outputted.
  4. Set the generation number. If this is the very beginning of the run, then the generation number should equal one.
     If a previous run is being continued, then this generation number should be the last successfully completed
     generation's number incremented by one.
  5. Submit the supercomputer job script (SLURM, PBS, etc.).
  6. Monitor the total error/objective function as the generations go on, as well as any other statistics of interest.

### Dependencies

Through PyPI, the installation should already come with NumPy, Pandas, and Click. **TensorFlow 2.0 is used for building,
training, and using the feed forward neural network, but is NOT automatically installed.** This is because the
application can run without TensorFlow, as long as the option to "use_neural_network" is not *true*, allowing
for compatibility with systems that cannot use TF 2.0. However, those who wish to utilize the neural network can run

```commandline
$pip install -r requirements/prod.txt
```

which will check and install all required modules as listed in the `requirements/prod.txt` file, including TensorFlow.

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
