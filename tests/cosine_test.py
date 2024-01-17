#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.distance.cosine` module.
"""
import unittest
import pandas as pd
from GenomeSigInfer.distance.cosine import cosine_nmf_w


class TestCosineNMFW(unittest.TestCase):
	"""
	A test case for the `cosine_nmf_w` function in the `GenomeSigInfer.distance.cosine` module.
	"""

	def setUp(self):
		"""
		Set up the test case by creating sample dataframes and an optimal_columns dictionary for testing.
		"""
		self.df1 = pd.DataFrame(
			{
				"col1": [1, 2, 3],
				"col2": [4, 5, 6],
			}
		)

		self.df2 = pd.DataFrame(
			{
				"col_a": [1, 2, 3],
				"col_b": [7, 8, 9],
			}
		)

		self.optimal_columns = {
			"col1": "col_a",
			"col2": "col_b",
		}

	def test_cosine_nmf_w(self):
		"""
		Test the `cosine_nmf_w` function by comparing the actual cosine similarity values with the expected values.
		"""
		result = cosine_nmf_w(self.optimal_columns, self.df1, self.df2)
		# Replace the expected values with the actual expected cosine similarity values
		expected_result = {
			"col1": 0.9999999999999998,
			"col2": 0.26261286571944503,
		}
		# Iterate through keys and compare values
		for key, expected_value in expected_result.items():
			self.assertAlmostEqual(result[key], expected_value, places=5)


if __name__ == "__main__":
	unittest.main()
