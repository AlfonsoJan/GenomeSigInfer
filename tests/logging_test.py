#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.utils.logging` module.
"""
import unittest
from GenomeSigInfer.utils.logging import SingletonLogger


class TestSingletonLogger(unittest.TestCase):
	"""
	A test case for the `SingletonLogger` class in the `GenomeSigInfer.utils.logging` module.
	"""

	def test_singleton_logger(self):
		"""
		Test case to verify that the SingletonLogger class returns the same instance
		when called multiple times.
		"""
		logger1 = SingletonLogger()
		logger2 = SingletonLogger()
		self.assertIs(logger1, logger2)

	def test_log_info(self):
		"""
		Test case to verify if an informational message is logged correctly.
		"""
		logger = SingletonLogger()
		with self.assertLogs(logger.logger, level="INFO") as cm:
			logger.log_info("This is an informational message.")
		self.assertIn("This is an informational message.", cm.output[0])

	def test_log_warning(self):
		"""
		Test case to verify if a warning message is logged correctly.

		This test creates a SingletonLogger instance and logs a warning message using the `log_warning` method.
		It then asserts that the warning message is present in the logs captured by the `assertLogs` context manager.
		"""
		logger = SingletonLogger()
		with self.assertLogs(logger.logger, level="WARNING") as cm:
			logger.log_warning("This is a warning message.")
		self.assertIn("This is a warning message.", cm.output[0])

	def test_log_error(self):
		"""
		Test case to verify if the error message is logged correctly.

		It creates a SingletonLogger instance, logs an error message using the `log_error` method,
		and asserts that the error message is present in the logs.
		"""
		logger = SingletonLogger()
		with self.assertLogs(logger.logger, level="ERROR") as cm:
			logger.log_error("This is an error message.")
		self.assertIn("This is an error message.", cm.output[0])

	def test_log_exception(self):
		"""
		Test case to verify if an exception message is logged correctly.

		It asserts that the exception message is present in the logged output.
		"""
		logger = SingletonLogger()
		with self.assertLogs(logger.logger, level="ERROR") as cm:
			logger.log_exception("This is an exception message.", exc_info=True)
		self.assertIn("This is an exception message.", cm.output[0])


if __name__ == "__main__":
	unittest.main()
