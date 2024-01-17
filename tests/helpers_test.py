#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
from GenomeSigInfer.errors import error
from GenomeSigInfer.utils.helpers import calculate_value, custom_sort_column_names, generate_numbers, generate_sequence, alphabet_list, create_signatures_df, must_be_int, check_supported_genome


class TestHelpers(unittest.TestCase):
    def test_calculate_value(self):
        test_cases = [
            (3, 96),
            (5, 96 * 16),
            (7, 96 * 16 * 16),
        ]

        for test_input, expected in test_cases:
            with self.subTest(test_input=test_input, expected=expected):
                self.assertEqual(calculate_value(test_input), expected)

    def test_calculate_value_error_low_value(self):
        with self.assertRaisesRegex(
            ValueError,
            "Input must be an uneven integer greater than or equal to 3.",
        ):
            calculate_value(2)

    def test_calculate_value_error_not_int(self):
        with self.assertRaisesRegex(TypeError, "Input must be an integer."):
            calculate_value("not_an_integer")

    def test_custom_sort_column_names(self):
        self.assertEqual(custom_sort_column_names("A1"), (1, "A", ""))
        self.assertEqual(custom_sort_column_names("B10"), (10, "B", ""))
        self.assertEqual(custom_sort_column_names("C2D"), (2, "C", "D"))

    def test_generate_sequence(self):
        self.assertEqual(generate_sequence(5), [-2, 2])
        self.assertEqual(generate_sequence(7), [-3, -2, 2, 3])

    def test_generate_sequence_error(self):
        with self.assertRaises(ValueError):
            generate_sequence(3)

    def test_generate_numbers(self):
        self.assertEqual(generate_numbers(5), [0, 8])
        self.assertEqual(generate_numbers(7), [0, 1, 9, 10])
        self.assertEqual(generate_numbers(9), [0, 1, 2, 10, 11, 12])
        self.assertEqual(generate_numbers(11), [0, 1, 2, 3, 11, 12, 13, 14])

    def test_generate_numbers_error(self):
        with self.assertRaises(ValueError):
            generate_numbers(4)

    def test_alphabet_list(self):
        self.assertEqual(alphabet_list(3, "SBS"), ["SBSA", "SBSB", "SBSC"])
        self.assertEqual(
            alphabet_list(5, "SBS"), ["SBSA", "SBSB", "SBSC", "SBSD", "SBSE"]
        )

    def test_create_signatures_df(self):
        W = np.array([[1, 2, 3], [4, 5, 6]])
        signatures_df = create_signatures_df(W, 3)
        self.assertIsInstance(signatures_df, pd.DataFrame)
        self.assertEqual(signatures_df.shape, (2, 3))
        self.assertEqual(signatures_df.columns.tolist(), ["SBS2A", "SBS2B", "SBS2C"])

    def test_calculate_value_decorator(self):
        @must_be_int
        def dummy_func(number):
            return number

        self.assertEqual(dummy_func(5), 5)

        with self.assertRaises(TypeError):
            dummy_func("not_an_integer")

    def test_check_supported_genome(self):
        check_supported_genome("GRCh37")
        check_supported_genome("GRCh38")

        with self.assertRaises(error.RefGenomeNotSUpported):
            check_supported_genome("GRCh39")


if __name__ == "__main__":
    unittest.main()
