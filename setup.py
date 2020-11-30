#!/usr/bin/env python

"""Build script for setuptools. Tells setuptools about your package (such as the name and version)
as well as which code files to include.
"""

from setuptools import setup, find_packages

with open("README.md", 'r') as readme_file:
    readme = readme_file.read()

requirements = [
    # Note that TensorFlow isn't put here - install TensorFlow if you wish to use Neural Network!
    'numpy',
    'pandas',
    'Click'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Chad Daksha",
    author_email='daksha@udel.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="ReaxFF parameter optimization scheme using generational genetic algorithm and neural networks.",
    entry_points={
        'console_scripts': [
            'cli=parametrization_clean.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='parametrization_clean',
    name='parametrization_clean-cdaksha',
    packages=find_packages(include=['parametrization_clean', 'parametrization_clean.*']),
    package_dir={'parametrization_clean-cdaksha': 'parametrization_clean'},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cdaksha/parametrization_clean',
    version='1.0.2.2',
    zip_safe=False,
)
