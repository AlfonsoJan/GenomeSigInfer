#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.distance.distance` module.
"""
import unittest
import pandas as pd
from scipy.spatial.distance import jensenshannon
from GenomeSigInfer.distance.distance import (
    get_optimal_columns,
    set_optimal_columns,
    get_jensen_shannon_distance,
)


class TestDistance(unittest.TestCase):
    """
    A test case for the `Preprocessing` class in the `GenomeSigInfer.distance.distance` module.
    """
    def test_get_optimal_columns(self):
        """
        Test case for the get_optimal_columns function.

        This test case checks if the get_optimal_columns function correctly assigns optimal columns
        between two DataFrames.
        """
        # Create sample DataFrames
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"X": [7, 8, 9], "Y": [10, 11, 12]})

        # Expected optimal column assignments
        expected_assignments = {"A": "X", "B": "Y"}

        # Call the function
        result = get_optimal_columns(df1, df2)

        # Check if the result matches the expected assignments
        self.assertEqual(result, expected_assignments)

    def test_set_optimal_columns(self):
        """
        Test case for the set_optimal_columns function.

        This test case checks if the set_optimal_columns function correctly sets the optimal columns
        based on the control dataframe and the second dataframe.
        """
        # Create sample dataframes
        control_df = pd.DataFrame(
            {
                "MutationType": ["A", "B", "C"],
                "Column1": [1, 2, 3],
                "Column2": [4, 5, 6],
            }
        )
        df2 = pd.DataFrame(
            {
                "MutationType": ["A", "B", "C"],
                "Column3": [7, 8, 9],
                "Column4": [10, 11, 12],
            }
        )

        # Call the function
        result = set_optimal_columns(control_df, df2)

        # Assert the expected output
        expected_columns = ["MutationType", "Column3", "Column4"]
        self.assertEqual(list(result.columns), expected_columns)

    def test_get_jensen_shannon_distance(self):
        """
        Test case for the get_jensen_shannon_distance function.

        This test case verifies that the calculated Jensen-Shannon distances
        match the expected result when comparing two dataframes.
        """
        # Create sample dataframes
        df1 = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.4, 0.5, 0.6]})
        df2 = pd.DataFrame({"X": [0.7, 0.8, 0.9], "Y": [1.0, 1.1, 1.2]})

        # Define optimal column assignments
        optimal_columns = {"A": "X", "B": "Y"}

        # Calculate Jensen-Shannon distances
        result = get_jensen_shannon_distance(optimal_columns, df1, df2)

        # Define expected result
        expected_result = {
            "A": jensenshannon(df1["A"], df2["X"]),
            "B": jensenshannon(df1["B"], df2["Y"]),
        }

        # Assert the calculated distances match the expected result
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
