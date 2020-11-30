=======
History
=======

1.0.0 (2020-05-06)
------------------

* First release on PyPI.

1.0.1 (2020-05-07)
------------------

* Configuration file path now taken as user input through command line application
* Added codecov badge
* Added documentation
* Added example JSON and wrapper bash script
* Refactored factory design patterns used for picking algorithms

1.0.1.4 (2020-05-09)
--------------------

* Added Sphinx-compliant documentation throughout code
* Added example job used for supercomputer
* Added typical workflow for proper setup and for running the application

1.0.1.5 (2020-05-09)
--------------------
* More documentation updates

1.0.1.6 (2020-05-10)
--------------------
* Slight refactoring of nested GA + neural network loop:
  top two parents from master GA will remain now remain untampered throughout nested GA iterations
* Slight refactoring of ANN summary output

1.0.1.7 (2020-05-11)
--------------------
* Slight refactoring of ANN summary output once more
* Added more documentation to README

1.0.1.8 (2020-05-11)
--------------------
* Slight refactoring of error handling for user JSON configuration passed as input
* Addition of configuration specific exception class

1.0.1.9 (2020-05-11)
--------------------
* Fixed ANN summary output
* Added exception handling for when ffield file has merged columns

1.0.2.1 (2020-11-29)
--------------------
* Removed requirement of fort.99 in the training set - now populates fort99 data in the Root Individual
  by finding a valid case from the first generation

1.0.2.2 (2020-11-29)
--------------------
* Changed logic for retrieving fort99 data for root individual - now, only one pass of Generation 1 is required in
  Generation 2 to find the fort99, after which the fort99 file is cached to the generation output directory,
  which is used for subsequent generations.
