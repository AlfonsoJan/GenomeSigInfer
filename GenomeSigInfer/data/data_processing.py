#!/usr/bin/env python3
"""
This module is designed for preprocessing Single Base Substitution (SBS) genomic data, offering a comprehensive set of tools through the Preprocessing class.
The workflow begins with class initialization, where genomic data is provided, and a cutoff value is computed based on a percentile of the data.
In essence, this module encapsulates a streamlined pipeline for preprocessing SBS genomic data, encompassing initialization, data preprocessing, normalization, and denormalization.

Author: J.A. Busker
"""
import pandas as pd
import numpy as np


class Preprocessing:
    """
    A utility class for pre-processing SBS data.

    This class provides methods for normalizing and denormalizing genomic data.

    Attributes:
        genomes (np.ndarray): A DataFrame containing genomic data.
        seed (np.random.Generator): A Numpy random number generator.
        cutoff (int): A cutoff value for data normalization.

    Attributes:
        norm_genomes (np.ndarray): Normalized genomic data.
        total_mutations (np.ndarray): Total mutations per sample.

    Methods:
        normalize(genomes, total_mut, cutoff): Normalize genomic data.
    """

    # Lambda function to calculate the cutoff value
    get_cutoff = lambda data, manual_cutoff: max(np.percentile(data, 95), manual_cutoff)

    def __init__(self, genomes: np.ndarray):
        """
        Initialize the Preprocessing class.

        Args:
            genomes (np.ndarray): A DataFrame containing genomic data.
        """
        # Compute the cutoff value based on the provided genomic data
        self._cutoff = Preprocessing.get_cutoff(
            genomes, manual_cutoff=100 * genomes.shape[0]
        )
        # Generate a random seed for reproducibility
        seed = np.array(np.random.SeedSequence().entropy)
        seed = np.random.SeedSequence(int(seed)).spawn(1)[0]
        self._seed = np.random.Generator(np.random.PCG64DXSM(seed))
        # Convert input data to a DataFrame
        self._genomes = pd.DataFrame(genomes)
        # Perform initial data processing
        self._init()

    def _init(self) -> None:
        """
        Initialize the Preprocessing class.

        This private method performs data preprocessing, including normalization.
        """
        # Calculate the sum of each column in the genomic data
        sum_row_genomes = pd.DataFrame(self._genomes.sum(0)).transpose()
        # Create a matrix to repeat the sum values for each sample
        rep_mat = np.array(sum_row_genomes.values.repeat(self._genomes.shape[0], axis=0))
        # Normalize the genomic data using a random number generator
        n_genomes = self._genomes / rep_mat
        dataframes_list = [
            pd.DataFrame(
                self._seed.multinomial(
                    sum_row_genomes.iloc[:, index], n_genomes.iloc[:, index], 1
                )[0]
            )
            for index in range(n_genomes.shape[1])
        ]
        genomes = pd.concat(dataframes_list, axis=1)
        # Adjust small values in the data for stability
        genomes[genomes < 0.0001] = 0.0001
        genomes = genomes.astype(float)
        # Calculate total mutations per sample and normalize the genomic data
        self._total_mutations = np.sum(genomes, axis=1)
        self._norm_genomes = Preprocessing.normalize(
            genomes, self._total_mutations, self._cutoff
        )

    @property
    def norm_genomes(self) -> np.ndarray:
        """
        np.ndarray: Normalized genomic data.
        """
        return self._norm_genomes

    @property
    def total_mutations(self) -> np.ndarray:
        """
        np.ndarray: Total mutations per sample.
        """
        return self._total_mutations

    @staticmethod
    def normalize(
        genomes: pd.DataFrame, total_mut: np.ndarray, cutoff: int
    ) -> np.ndarray:
        """
        Normalize genomic data.

        Args:
            genomes (pd.DataFrame): Genomic data to normalize.
            total_mut (np.ndarray): Total mutations per sample.
            cutoff (int): A cutoff value for data normalization.

        Returns:
            np.ndarray: Normalized genomic data.
        """
        # Convert DataFrame to NumPy array for efficient computations
        genomes = np.array(genomes)
        # Identify indices where total mutations exceed the cutoff
        indices = np.where(total_mut > cutoff)[0]
        # Normalize genomic data based on total mutations and cutoff
        norm_genome = (
            genomes[:, list(indices)]
            / total_mut.ravel()[list(indices)][:, np.newaxis].T
            * cutoff
        )
        genomes[:, list(indices)] = norm_genome
        return np.array(genomes)

    def __repr__(self) -> str:
        """
        Return a string representation of the Preprocessing class.
        """
        return f"Preprocessing(genomes={self._genomes.shape}, cutoff={self._cutoff})"
