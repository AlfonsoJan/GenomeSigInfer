#!/usr/bin/env python3
"""
This module provides classes and functions for performing Non-Negative Matrix Factorization (NMF)
"""
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils.helpers import combinations, get_96_matrix, read_file_decompose
from ..utils.logging import SingletonLogger
from ..data.data_processing import Preprocessing
from ..decompose.decompose import Decompose
from ..distance.cosine import most_similarity_decompose
from ..figures.heatmap import heatmap_best_param
from .sigprofiler import RunSig
from .run_nmf import RunNMF


def run_nmfs(nmf_combs, all_genomes, sigs, matrix, out, result_df) -> pd.DataFrame:
    """
    Run NMF with different combinations of initialization and beta loss.

    Args:
        nmf_combs (list): List of NMF combinations to try.
        all_genomes (numpy.ndarray): Matrix of all genomes.
        sigs (int): Number of signatures.
        matrix (pandas.DataFrame): Data matrix.
        out (str): Output directory.

    Returns:
        Dataframe with the decomposed results
    """
    preprocessed = Preprocessing(all_genomes)
    for index in range(len(nmf_combs)):
        combination = nmf_combs[index]
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
        read_file_decompose(decom_file, result_df)
        shutil.rmtree(outpath)
    return result_df

class NMF_Combinations:
    """
    Run NMF combinations and generate results.
    Of the cosine similarity.
    """

    NMF_COMBS = combinations()
    
    def __init__(self, project: Path, sigs: int) -> None:
        """
        Run NMF combinations and generate results.

        Args:
            project (Path): Path of the project.
            sigs (int): Number of signatures.
        """
        self._logger = SingletonLogger()
        self.project = project
        self.signatures = sigs
        self.results_folder = self.project / "results"
        self._prepare_folder()
        self.matrix = self.load_and_process_matrix()
        self.all_genomes = np.array(self.matrix.iloc[:, 1:])
    
    def run_combinations(self):
        """
        Run NMF on the matrix.
        """
        df_decompose = RunSig.run(
            matrix=self.matrix, signatures=self.signatures, out=self.results_folder
        )
        df_decompose = run_nmfs(
            self.NMF_COMBS,
            self.all_genomes,
            self.signatures,
            self.matrix,
            self.results_folder,
            df_decompose,
        )
        self.df_decompose = df_decompose
    
    def _prepare_folder(self):
        """
        Prepare folders for results.
        """
        self.results_folder.mkdir(parents=True, exist_ok=True)
    
    def load_and_process_matrix(self):
        """
        Load and process the matrix.

        Returns:
            np.ndarray: Processed matrix.
        """
        matrix_file_name = self.project / "SBS" / "sbs.96.txt"
        matrix = get_96_matrix(matrix_file_name)
        return matrix
    
    def cosine_sim(self):
        """
        Calculates and creates a dataframe with the cosine similarity
        """
        cosine_df = most_similarity_decompose(self.df_decompose)
        param_filename = self.results_folder / "param-tuning.txt"
        cosine_df.to_csv(param_filename, index=False, sep="\t")
        self._logger.log_info(f"Written the results to: {param_filename}")
        self.create_heatmap(cosine_df)
    
    def create_heatmap(self, cosine_df: pd.DataFrame):
        """
        Create a heatmap plot of the results.
        """
        figure_folder = self.project / "figures"
        figure_folder.mkdir(parents=True, exist_ok=True)
        image_name = heatmap_best_param(cosine_df, figure_folder=figure_folder)
        self._logger.log_info(f"Created a heatmap plot to: {image_name}")
