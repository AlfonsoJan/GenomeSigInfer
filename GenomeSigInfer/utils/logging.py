#!/usr/bin/env python3
"""
This module provides a singleton logger class for consistent logging configuration.

The SingletonLogger class ensures that only one instance of the logger is created, providing a consistent logging configuration throughout the application.

Usage:
    logger = SingletonLogger()
    logger.log_info("This is an informational message.")
    logger.log_warning("This is a warning message.")
    logger.log_error("This is an error message.")
    logger.log_exception("This is an exception message.", exc_info=True)

Author: J.A. Busker
"""
import logging


class SingletonLogger:
	"""
	A singleton logger class for consistent logging configuration.

	Attributes:
	    _instance (SingletonLogger): The singleton instance of the logger.

	Methods:
	    log_info(message: str)
	    log_warning(message: str)
	    log_error(message: str)
	    log_exception(message: str, exc_info: bool = True)
	"""

	_instance = None

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			cls._instance = super().__new__(cls)
			cls._instance._configure_logger()
		return cls._instance

	def _configure_logger(self):
		formatter = logging.Formatter(
			fmt="%(asctime)s - %(levelname)s - %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.INFO)
		self.logger.addHandler(stream_handler)

	def log_info(self, message):
		"""
		Log an INFO-level message.

		Args:
		    message (str): Log message.
		"""
		self.logger.info(message)

	def log_warning(self, message):
		"""
		Log a WARNING-level message.

		Args:
		    message (str): Log message.
		"""
		self.logger.warning(message)

	def log_error(self, message):
		"""
		Log an ERROR-level message.

		Args:
		    message (str): Log message.
		"""
		self.logger.error(message)

	def log_exception(self, message, exc_info=True):
		"""
		Log an ERROR-level message with exception details.

		Args:
		    message (str): Log message.
		    exc_info (bool): Whether to include exception details (default is True).
		"""
		self.logger.exception(message, exc_info=exc_info)
