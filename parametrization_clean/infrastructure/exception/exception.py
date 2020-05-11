#!/usr/bin/env python

"""Module for custom exceptions that can be raised upon errors."""

# Standard library

# 3rd party packages

# Local source


class ApplicationError(Exception):
    """Base class for other exceptions."""
    pass


class ConfigurationError(ApplicationError):
    """Error with user configuration file provided."""
    pass

