#!/usr/bin/env python

"""Response objects for repository output usage. The idea is to return a ResponseSuccess if data writing is
successful, to return a ResponseWarning if there were overcomeable (but potentially problematic) occurrences,
and to return a ResponseFailure if writing individuals was completely unsuccessful.
"""

# Standard library

# 3rd party packages

# Local source


class ResponseSuccess:
    SUCCESS = "SUCCESS"

    def __init__(self, message=None):
        self.type = self.SUCCESS
        self.message = message
        self.status_code = 0

    def __bool__(self):
        return True


class ResponseWarning:
    WARNING = "WARNING"

    def __init__(self, message=None):
        self.type = self.WARNING
        self.message = message
        self.status_code = 0

    def __bool__(self):
        return True


class ResponseFailure:
    RESOURCE_ERROR = "RESOURCE_ERROR"
    PARAMETERS_ERROR = "PARAMETERS_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"

    def __init__(self, type_, message):
        self.type = type_
        self.message = self._format_message(message)
        self.status_code = 1

    @staticmethod
    def _format_message(msg):
        if isinstance(msg, Exception):
            return "{}: {}".format(msg.__class__.__name__, "{}".format(msg))
        return msg

    @property
    def value(self):
        return {'type': self.type, 'message': self.message}

    def __bool__(self):
        return False

    @classmethod
    def build_resource_error(cls, message=None):
        return cls(cls.RESOURCE_ERROR, message)

    @classmethod
    def build_system_error(cls, message=None):
        return cls(cls.SYSTEM_ERROR, message)

    @classmethod
    def build_parameters_error(cls, message=None):
        return cls(cls.PARAMETERS_ERROR, message)
