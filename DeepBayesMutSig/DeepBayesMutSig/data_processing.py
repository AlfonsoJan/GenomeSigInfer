#!/usr/bin/env python3
"""
This class provides a preprocessing class for parsing pre-processing the data
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
        denormalize_samples(genomes, original_totals): Denormalize normalized data.

    """

    get_cutoff = lambda data, manual_cutoff: max(np.percentile(data, 95), manual_cutoff)

    def __init__(self, genomes: np.ndarray):
        """
        Initialize the Preprocessing class.

        Args:
            genomes (np.ndarray): A DataFrame containing genomic data.
        """
        self._cutoff = Preprocessing.get_cutoff(
            genomes, manual_cutoff=100 * genomes.shape[0]
        )
        seed = np.array(np.random.SeedSequence().entropy)
        seed = np.random.SeedSequence(int(seed)).spawn(1)[0]
        self._seed = np.random.Generator(np.random.PCG64DXSM(seed))
        self._genomes = pd.DataFrame(genomes)
        self._init()

    def _init(self) -> None:
        """
        Initialize the Preprocessing class.

        This private method performs data preprocessing, including normalization.

        """
        sum_row_genomes = pd.DataFrame(self._genomes.sum(0)).transpose()
        rep_mat = np.array(
            sum_row_genomes.values.repeat(self._genomes.shape[0], axis=0)
        )
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
        genomes[genomes < 0.0001] = 0.0001
        genomes = genomes.astype(float)
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
        genomes = np.array(genomes)
        indices = np.where(total_mut > cutoff)[0]
        norm_genome = (
            genomes[:, list(indices)]
            / total_mut.ravel()[list(indices)][:, np.newaxis].T
            * cutoff
        )
        genomes[:, list(indices)] = norm_genome
        return np.array(genomes)

    @staticmethod
    def denormalize_samples(
        genomes: np.ndarray, original_totals: list[int]
    ) -> np.ndarray:
        """
        Denormalize normalized genomic data.

        Args:
            genomes (np.ndarray): Normalized genomic data to denormalize.
            original_totals (List[int]): Original total mutations per sample.

        Returns:
            np.ndarray: Denormalized genomic data.
        """
        normalized_totals = np.sum(genomes, axis=0)
        original_totals = np.array(original_totals)
        results = genomes / normalized_totals * original_totals
        results = np.round(results, 0)
        results = results.astype(int)
        return results
