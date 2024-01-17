#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.matrix.matrix_operations` module.
"""
import unittest
import pandas as pd
from GenomeSigInfer.matrix.matrix_operations import (
	increase_mutations,
	compress_to_96,
)
from GenomeSigInfer.utils import helpers


class TestMatrixOperations(unittest.TestCase):
	"""
	A test case for the `increase_mutations, compress_to_96` fucntions in the `GenomeSigInfer.matrix.matrix_operations` module.
	"""

	def test_increase_mutations_value_error(self):
		"""
		Test if ValueError is raised when the context is less than 3
		"""
		with self.assertRaises(ValueError, msg="Context must be at least 3"):
			increase_mutations(2)

	def test_increase_mutations_context_3(self):
		"""
		Test if the function returns the expected list of increased mutations for context 3.

		This test case checks if the `increase_mutations` function correctly generates the list of increased mutations
		for context 3. It compares the actual result with the expected result and asserts that they are equal.
		"""
		context = 3
		result = increase_mutations(context)

		# Replace the expected values with the actual expected values
		expected_result = [
			"A[C>A]A",
			"A[C>A]C",
			"A[C>A]G",
			"A[C>A]T",
			"C[C>A]A",
			"C[C>A]C",
			"C[C>A]G",
			"C[C>A]T",
			"G[C>A]A",
			"G[C>A]C",
			"G[C>A]G",
			"G[C>A]T",
			"T[C>A]A",
			"T[C>A]C",
			"T[C>A]G",
			"T[C>A]T",
			"A[C>G]A",
			"A[C>G]C",
			"A[C>G]G",
			"A[C>G]T",
			"C[C>G]A",
			"C[C>G]C",
			"C[C>G]G",
			"C[C>G]T",
			"G[C>G]A",
			"G[C>G]C",
			"G[C>G]G",
			"G[C>G]T",
			"T[C>G]A",
			"T[C>G]C",
			"T[C>G]G",
			"T[C>G]T",
			"A[C>T]A",
			"A[C>T]C",
			"A[C>T]G",
			"A[C>T]T",
			"C[C>T]A",
			"C[C>T]C",
			"C[C>T]G",
			"C[C>T]T",
			"G[C>T]A",
			"G[C>T]C",
			"G[C>T]G",
			"G[C>T]T",
			"T[C>T]A",
			"T[C>T]C",
			"T[C>T]G",
			"T[C>T]T",
			"A[T>A]A",
			"A[T>A]C",
			"A[T>A]G",
			"A[T>A]T",
			"C[T>A]A",
			"C[T>A]C",
			"C[T>A]G",
			"C[T>A]T",
			"G[T>A]A",
			"G[T>A]C",
			"G[T>A]G",
			"G[T>A]T",
			"T[T>A]A",
			"T[T>A]C",
			"T[T>A]G",
			"T[T>A]T",
			"A[T>C]A",
			"A[T>C]C",
			"A[T>C]G",
			"A[T>C]T",
			"C[T>C]A",
			"C[T>C]C",
			"C[T>C]G",
			"C[T>C]T",
			"G[T>C]A",
			"G[T>C]C",
			"G[T>C]G",
			"G[T>C]T",
			"T[T>C]A",
			"T[T>C]C",
			"T[T>C]G",
			"T[T>C]T",
			"A[T>G]A",
			"A[T>G]C",
			"A[T>G]G",
			"A[T>G]T",
			"C[T>G]A",
			"C[T>G]C",
			"C[T>G]G",
			"C[T>G]T",
			"G[T>G]A",
			"G[T>G]C",
			"G[T>G]G",
			"G[T>G]T",
			"T[T>G]A",
			"T[T>G]C",
			"T[T>G]G",
			"T[T>G]T",
		]
		self.assertListEqual(result, expected_result)

	def test_compress_to_96_already_compressed(self):
		"""
		Test if the function returns the same DataFrame when it is already compressed.

		This test case checks whether the `compress_to_96` function correctly handles the scenario
		where the input DataFrame is already compressed. It creates a DataFrame with a single column
		named "MutationType" containing a list of mutation types, and a column named "Value" with all
		values set to 1. The `compress_to_96` function is then called on this DataFrame, and the
		resulting compressed DataFrame is compared with the original DataFrame using the
		`pd.testing.assert_frame_equal` method to ensure that they are identical.
		"""
		df = pd.DataFrame(
			{
				"MutationType": helpers.MUTATION_LIST,
				"Value": [1] * len(helpers.MUTATION_LIST),
			}
		)
		compressed_df = compress_to_96(df)
		# Assert that the returned DataFrame is the same as the input DataFrame
		pd.testing.assert_frame_equal(df, compressed_df)

	def test_compress_to_96(self):
		"""
		Test if the function correctly compresses the DataFrame to 96 rows.
		"""
		df = pd.DataFrame(
			{
				"MutationType": helpers.MUTATION_LIST * 2,
				"Value": [1] * len(helpers.MUTATION_LIST) * 2,
			}
		)
		compressed_df = compress_to_96(df)
		# Assert that the compressed DataFrame has 96 rows
		self.assertEqual(compressed_df.shape[0], 96)
		# Assert that the compressed DataFrame has the expected columns
		expected_columns = ["MutationType", "Value"]
		self.assertListEqual(list(compressed_df.columns), expected_columns)
		# Assert that the compressed DataFrame has the correct values in the "Value" column
		expected_values = [2] * 96
		actual_values = compressed_df["Value"].tolist()
		self.assertListEqual(actual_values, expected_values)


if __name__ == "__main__":
	unittest.main()
