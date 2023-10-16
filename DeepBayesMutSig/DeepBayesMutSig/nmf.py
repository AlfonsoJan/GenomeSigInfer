#!/usr/bin/env python3
"""
This module provides classes and functions for performing Non-Negative Matrix Factorization (NMF)

Functions:
- run_multiple(all_genomes: np.ndarray, signatures: int, iters: int) -> np.ndarray: This script runs nmf iters amount of time and get the average result.
- run_nmfs(nmf_combs, all_genomes, sigs, matrix, out) -> pd.DataFrame: Run NMF with different combinations of initialization and beta loss.
"""
import warnings
import shutil
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import NMF
from .data_processing import Preprocessing
from .decompose import Decompose
from .helpers import read_file_decompose


class RunNMF:
    """
    A class for running Non-Negative Matrix Factorization (NMF) on genomic data.

    Attributes:
        genomes (np.ndarray): Genomic data for NMF.
        signatures (int): Number of signatures to extract.
        init (str): Initialization method for NMF.
        beta_loss (str): Beta loss function for NMF.

    Attributes:
        W (np.ndarray): NMF factorization matrix.

    Methods:
        fit(): Fit NMF to the genomic data.
    """

    def __init__(
        self,
        genomes: np.ndarray,
        signatures: int = 1,
        init: str = "nndsvdar",
        beta_loss: str = "frobenius",
    ) -> None:
        """
        Initialize the RunNMF class.

        Args:
            genomes (np.ndarray): Genomic data for NMF.
            signatures (int): Number of signatures to extract.
            init (str): Initialization method for NMF.
            beta_loss (str): Beta loss function for NMF.
        """
        self._solver = "cd" if beta_loss == "frobenius" else "mu"
        self._genomes = genomes
        self._init = None if init == "None" else init
        self._beta_loss = beta_loss
        self._signatures = signatures
        self._W = None

    def fit(self):
        """
        Fit NMF to the genomic data.

        This method performs NMF factorization on the genomic data.
        """
        nmf = NMF(
            n_components=self._signatures,
            init=self._init,
            beta_loss=self._beta_loss,
            solver=self._solver,
            max_iter=1000,
            tol=1e-15,
        )
        self._W = nmf.fit_transform(self._genomes)

    @property
    def W(self) -> np.ndarray:
        """
        self._W (np.ndarray): NMF factorization matrix.
        """
        return self._W

    @property
    def W_norm(self) -> np.ndarray:
        """
        self._W (np.ndarray): NMF factorization matrix normalized between 0 and 1.
        """
        return self._W / self._W.sum(axis=0)[np.newaxis]


def run_multiple(all_genomes: np.ndarray, signatures: int, iters: int) -> np.ndarray:
    """
    This script runs nmf iters amount of time and get the average result.

    Args:
        all_genomes: Genomic mutational data.
        signatures: Numbers of signatures.
        iters: Number of how many times to run NMF.

    Returns:
        np.ndarray: The average result.
    """
    average_w = np.zeros((all_genomes.shape[0], signatures))
    for _ in tqdm(range(iters)):
        preprocessed = Preprocessing(all_genomes)
        nmf_model = RunNMF(preprocessed.norm_genomes, signatures)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            nmf_model.fit()
            average_w += nmf_model.W
    average_w /= iters
    average_w = average_w / average_w.sum(axis=0)[np.newaxis]
    return average_w


def run_nmfs(nmf_combs, all_genomes, sigs, matrix, out) -> pd.DataFrame:
    """
    Run NMF with different combinations of initialization and beta loss.

    Args:
        nmf_combs (list): List of NMF combinations to try.
        all_genomes (numpy.ndarray): Matrix of all genomes.
        sigs (int): Number of signatures.
        matrix (pandas.DataFrame): Data matrix.
        out (str): Output directory.

    Returns:

    """
    df = pd.DataFrame()
    preprocessed = Preprocessing(all_genomes)
    for i in range(len(nmf_combs)):
        combination = nmf_combs[i]
        nmf_model = RunNMF(
            genomes=preprocessed.norm_genomes,
            signatures=sigs,
            init=combination[0],
            beta_loss=combination[1],
        )
        nmf_model.fit()
        folder = f"{combination[0]}_{combination[1]}"
        outpath = Path(out) / folder
        dec = Decompose(
            nmf_model.W_norm,
            matrix[matrix.columns[0]],
            sigs,
            outpath,
            all_genomes,
            matrix.columns,
        )
        dec.decompose()
        decom_file = (
            outpath
            / "result-nmf"
            / "Decompose_Solution"
            / "Solution_Stats"
            / "Cosmic_SBS96_Decomposition_Log.txt"
        )
        read_file_decompose(decom_file, df)
        shutil.rmtree(outpath)
    return df
