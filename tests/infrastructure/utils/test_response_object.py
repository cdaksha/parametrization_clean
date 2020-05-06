
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.infrastructure.utils.response_object import (ResponseSuccess,
                                                                        ResponseWarning,
                                                                        ResponseFailure)


def test_response_success():
    success_obj = ResponseSuccess()

    assert success_obj
    assert success_obj.type == ResponseSuccess.SUCCESS
    assert success_obj.status_code == 0
    assert not success_obj.message


def test_response_success_with_message():
    message = "Successful response."
    success_obj = ResponseSuccess(message)

    assert success_obj
    assert success_obj.type == ResponseSuccess.SUCCESS
    assert success_obj.status_code == 0
    assert success_obj.message == message


def test_response_warning():
    warning_obj = ResponseWarning()

    assert warning_obj
    assert warning_obj.type == ResponseWarning.WARNING
    assert warning_obj.status_code == 0
    assert not warning_obj.message


def test_response_warning_with_message():
    message = "WARNING."
    warning_obj = ResponseWarning(message)

    assert warning_obj
    assert warning_obj.type == ResponseWarning.WARNING
    assert warning_obj.status_code == 0
    assert warning_obj.message == message


def test_response_failure_init():
    failure_obj = ResponseFailure(type_=ResponseFailure.RESOURCE_ERROR, message="ERROR.")

    assert not failure_obj
    assert failure_obj.type == ResponseFailure.RESOURCE_ERROR
    assert failure_obj.message == "ERROR."
    assert failure_obj.value == {'type': ResponseFailure.RESOURCE_ERROR,
                                 'message': failure_obj.message}
    assert failure_obj.status_code == 1


def test_response_failure_build_from_resource_error():
    failure_obj = ResponseFailure.build_resource_error(message="ERROR.")

    assert not failure_obj
    assert failure_obj.type == ResponseFailure.RESOURCE_ERROR
    assert failure_obj.message == "ERROR."
    assert failure_obj.value == {'type': ResponseFailure.RESOURCE_ERROR,
                                 'message': failure_obj.message}
    assert failure_obj.status_code == 1


def test_response_failure_build_from_system_error():
    failure_obj = ResponseFailure.build_system_error(message="ERROR.")

    assert not failure_obj
    assert failure_obj.type == ResponseFailure.SYSTEM_ERROR
    assert failure_obj.message == "ERROR."
    assert failure_obj.value == {'type': ResponseFailure.SYSTEM_ERROR,
                                 'message': failure_obj.message}
    assert failure_obj.status_code == 1


def test_response_failure_build_from_parameters_error():
    failure_obj = ResponseFailure.build_parameters_error(message="ERROR.")

    assert not failure_obj
    assert failure_obj.type == ResponseFailure.PARAMETERS_ERROR
    assert failure_obj.message == "ERROR."
    assert failure_obj.value == {'type': ResponseFailure.PARAMETERS_ERROR,
                                 'message': failure_obj.message}
    assert failure_obj.status_code == 1


def test_response_failure_format_message():
    failure_obj = ResponseFailure.build_parameters_error(message=ValueError("Incorrect value entered!"))
    assert failure_obj.message == "ValueError: Incorrect value entered!"
