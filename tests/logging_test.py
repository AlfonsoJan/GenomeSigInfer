import unittest
import logging
from GenomeSigInfer.utils.logging import SingletonLogger

class TestSingletonLogger(unittest.TestCase):
    def test_singleton_logger(self):
        logger1 = SingletonLogger()
        logger2 = SingletonLogger()
        self.assertIs(logger1, logger2)

    def test_log_info(self):
        logger = SingletonLogger()
        with self.assertLogs(logger.logger, level='INFO') as cm:
            logger.log_info("This is an informational message.")

        self.assertIn("This is an informational message.", cm.output[0])

    def test_log_warning(self):
        logger = SingletonLogger()
        with self.assertLogs(logger.logger, level='WARNING') as cm:
            logger.log_warning("This is a warning message.")

        self.assertIn("This is a warning message.", cm.output[0])

    def test_log_error(self):
        logger = SingletonLogger()
        with self.assertLogs(logger.logger, level='ERROR') as cm:
            logger.log_error("This is an error message.")

        self.assertIn("This is an error message.", cm.output[0])

    def test_log_exception(self):
        logger = SingletonLogger()
        with self.assertLogs(logger.logger, level='ERROR') as cm:
            logger.log_exception("This is an exception message.", exc_info=True)

        self.assertIn("This is an exception message.", cm.output[0])

if __name__ == '__main__':
    unittest.main()
