#!/usr/bin/env python3
"""
Unit tests for the `GenomeSigInfer.data.data_processing` module.
"""
import unittest
import numpy as np
import pandas as pd
from GenomeSigInfer.data.data_processing import Preprocessing


class PreprocessingTest(unittest.TestCase):
    """
    A test case for the `Preprocessing` class in the `GenomeSigInfer.data.data_processing` module.
    """

    def setUp(self):
        """
        Set up the test case by creating a sample genomic data and initializing the Preprocessing object.
        """
        self.genomes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.preprocessing = Preprocessing(self.genomes)

    def test_normalize(self):
        """
        Test the normalize method of the Preprocessing class.
        """
        normalized_genomes = Preprocessing.normalize(
            pd.DataFrame(self.genomes), np.sum(self.genomes), 3
        )
        expected_normalized_genomes = np.array([[0, 2, 3], [0, 5, 6], [0, 8, 9]])
        np.testing.assert_array_almost_equal(
            normalized_genomes, expected_normalized_genomes
        )

    def test_norm_genomes(self):
        """
        Test the total normalized genomic data.
        """
        self.assertAlmostEqual(round(self.preprocessing.norm_genomes.sum()), 45)


if __name__ == "__main__":
    unittest.main()
