import logging
from GenomeSigInfer.utils.logging import SingletonLogger


def test_singleton_logger():
    logger1 = SingletonLogger()
    logger2 = SingletonLogger()

    assert logger1 is logger2


def test_log_info(caplog):
    logger = SingletonLogger()
    logger.log_info("This is an informational message.")

    assert "This is an informational message." in caplog.text
    assert caplog.record_tuples[0][1] == logging.INFO
    assert caplog.record_tuples[0][2] == "This is an informational message."


def test_log_warning(caplog):
    logger = SingletonLogger()
    logger.log_warning("This is a warning message.")

    assert "This is a warning message." in caplog.text
    assert caplog.record_tuples[0][1] == logging.WARNING
    assert caplog.record_tuples[0][2] == "This is a warning message."


def test_log_error(caplog):
    logger = SingletonLogger()
    logger.log_error("This is an error message.")

    assert "This is an error message." in caplog.text
    assert caplog.record_tuples[0][1] == logging.ERROR
    assert caplog.record_tuples[0][2] == "This is an error message."


def test_log_exception(caplog):
    logger = SingletonLogger()
    logger.log_exception("This is an exception message.", exc_info=True)

    assert "This is an exception message." in caplog.text
    assert caplog.record_tuples[0][2] == "This is an exception message."
