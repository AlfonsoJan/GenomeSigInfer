#!/usr/bin/env python3
"""
This module provides classes and functions for performing Non-Negative Matrix Factorization (NMF)

Create NMF files based on the specified number of signatures.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from ..utils.helpers import MutationalSigantures, create_signatures_df, MUTATION_LIST
from ..utils.logging import SingletonLogger
from ..figures.heatmap import heatmap_cosine, heatmap_jens_shan
from ..distance.distance import (
    set_optimal_columns,
    get_optimal_columns,
    get_jensen_shannon_distance,
)
from ..distance.cosine import cosine_nmf_w
from ..data.matrix_operations import compress_to_96
from ..data.data_processing import Preprocessing
from .run_nmf import RunNMF


class NMF_SBS:
    """
    Create NMF files based on the specified number of signatures.

    Args:
        project (Path): project name.
        sigs (int):L signatures.
        cosmic (Path): path of the cosmic file.
        init (str): init for the NMF
        beta_loss (str): beta_loss for the NMF
    """

    def __init__(
        self,
        project: Path,
        sigs: int,
        cosmic: Path,
        init: str = "nndsvda",
        beta_loss: str = "frobenius",
    ) -> None:
        """
        Create NMF files based on the specified number of signatures.

        Args:
            project (Path): project name.
            sigs (int):L signatures.
            cosmic (Path): path of the cosmic file.
            init (str): init for the NMF
            beta_loss (str): beta_loss for the NMF
        """
        self._logger = SingletonLogger()
        self._logger.log_info(f"Creating NMF files with {sigs} signatures")
        self.sigantures = sigs
        self.init = init
        self.beta_loss = beta_loss
        self.control_df = None
        # List to store cosine similarity dat
        self.cosine_similarities = []
        # List to store Jensen Shannon Distance data
        self.jens_shannon_distances = []
        self.project = project
        self.nmf_folder = project / "NMF"
        self.figure_folder = self.project / "figures"
        self._prepare_folder()
        self._cosmic_df = (
            pd.read_csv(cosmic, sep="\t")
            .set_index("Type")
            .reindex(MUTATION_LIST)
            .reset_index()
        )

    def run_nmf(self):
        """
        Create NMF files based on the specified number of signatures.
        """
        self.stats_nmf_list = []
        for index, size in enumerate(MutationalSigantures.SIZES, -1):
            signatures_df = self.perform_nmf(size)
            self._process_figures(size, signatures_df, index)
        self._logger.log_info("Creating cosine and Jensen Shannon Distance plots")
        self._create_distance_figures()
        self.write_stats()

    def write_stats(self):
        """
        Write NMF stats to a file.
        """
        results_df = pd.DataFrame(self.stats_nmf_list)
        filename = self.project / "results" / "nmf.stats.txt"
        self._logger.log_info(f"Written the NMF files to {filename}")
        results_df.to_csv(filename, index=False, sep="\t")

    def _create_distance_figures(self):
        """
        Create cosine and Jensen Shannon Distance figures.
        """
        # Generate heatmap figures
        heatmap_cosine(self.cosine_similarities, self.figure_folder)
        heatmap_jens_shan(self.jens_shannon_distances, self.figure_folder)

        # Log completion
        self._logger.log_info("Created cosine similarity plots from the NMF results")
        self._logger.log_info(
            "Created Jensen Shannon Distance plots from the NMF results"
        )

    def _process_figures(self, size: int, signatures_df: pd.DataFrame, index: int):
        """
        Process figures based on the NMF results.

        Args:
            size (int): Size of the SBS file.
            signatures_df (pd.DataFrame): DataFrame containing NMF results.
            index (int): Index of the current iteration.
        """
        # Process figures based on the size
        decompose_filename = self.nmf_folder / f"decompose.{size}.txt"
        if size == 96:
            signatures_df = (
                signatures_df.set_index("MutationType")
                .reindex(MUTATION_LIST)
                .reset_index()
            )
            control_df = set_optimal_columns(signatures_df, self._cosmic_df)
            control_df.to_csv(decompose_filename, sep="\t", index=False)
            self.control_df = control_df.iloc[:, 1:]
        else:
            # Calculate distances and store in lists
            result_jens_df, result_cosine_df, decomposed_df = self.calculate_distances(
                self.control_df, signatures_df
            )
            decomposed_df.to_csv(decompose_filename, sep="\t", index=False)
            self.cosine_similarities.append(
                {
                    "data": result_cosine_df,
                    "context": MutationalSigantures.CONTEXT_LIST[index],
                }
            )
            self.jens_shannon_distances.append(
                {
                    "data": result_jens_df,
                    "context": MutationalSigantures.CONTEXT_LIST[index],
                }
            )

    def calculate_distances(
        self, control_df: pd.DataFrame, df_not_compressed: pd.DataFrame
    ) -> tuple:
        """
        Calculate Jensen Shannon Distance and Cosine Similarity between two matrices.

        Args:
            control_df (pd.DataFrame): The control matrix.
            df_not_compressed (pd.DataFrame): The matrix to compare.

        Returns:
            Tuple: DataFrames of Jensen Shannon Distance and Cosine Similarity.
        """
        # Create Folder
        result_folder = self.project / "results" / "distance"
        result_folder.mkdir(parents=True, exist_ok=True)

        df_compare = compress_to_96(df_not_compressed)
        df_compare = df_compare.iloc[:, 1:]
        optimal_columns = get_optimal_columns(control_df, df_compare)
        decomposed_df = self.create_decompose_df(optimal_columns, df_not_compressed)
        result_jens_df = self.jens_shannon_dist(optimal_columns, control_df, df_compare)
        result_cosine_df = self.cosine_similarity(
            optimal_columns, control_df, df_compare
        )

        jens_filename = result_folder / f"jensen.{df_not_compressed.shape[0]}.txt"
        cosine_filename = result_folder / f"cosine.{df_not_compressed.shape[0]}.txt"
        result_cosine_df.to_csv(
            cosine_filename,
            sep="\t",
            index=False,
        )
        result_jens_df.to_csv(
            jens_filename,
            sep="\t",
            index=False,
        )
        return result_jens_df, result_cosine_df, decomposed_df

    def cosine_similarity(
        self, optimal_columns: dict, control_df: np.ndarray, df_compare: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate Cosine Similarity between two matrices.

        Args:
            optimal_columns (dict): Optimal columns for comparison.
            control_df (np.ndarray): The control matrix.
            df_compare (np.ndarray): The matrix to compare..

        Returns:
            DataFrame: Cosine Similarity values.
        """
        result_cosine = cosine_nmf_w(
            optimal_columns=optimal_columns, df1=control_df, df2=df_compare
        )
        result_cosine_df = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_cosine])], ignore_index=True
        )
        return result_cosine_df

    def jens_shannon_dist(
        self, optimal_columns: dict, control_df: np.ndarray, df_compare: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate Jensen Shannon Distance between two matrices.

        Args:
            optimal_columns (dict): Optimal columns for comparison.
            control_df (np.ndarray): The control matrix.
            df_compare (np.ndarray): The matrix to compare.

        Returns:
            DataFrame: Jensen Shannon Distance values.
        """
        result_jens = get_jensen_shannon_distance(
            optimal_columns=optimal_columns, df1=control_df, df2=df_compare
        )
        result_jens_df = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_jens])], ignore_index=True
        )
        return result_jens_df

    def create_decompose_df(self, columns: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create decompose dataframe.

        Args:
            columns (dict): A dictionary where keys are column names from df1,
                and values are corresponding column names from df2.
            df (pd.DataFrame): The matrix for the new columns.

        Returns:
            pd.DataFrame: With the updated columns
        """
        df_copy = df.copy()
        df_copy = df_copy.drop("sort_key", axis=1)
        df_copy.rename(
            columns={value: key for key, value in columns.items()}, inplace=True
        )
        return df_copy

    def perform_nmf(self, size):
        """
        Create NMF files based on the specified number of signatures.
        """
        file = self.project / "SBS" / f"sbs.{size}.txt"
        matrix = pd.read_csv(file, sep=",", header=0)
        mutations = matrix[matrix.columns[0]]
        all_genomes = np.array(matrix.iloc[:, 1:])
        self._logger.log_info("Preprocessing the data")
        preprocessed = Preprocessing(all_genomes)
        # Create NMF model and fit
        nmf_model = RunNMF(
            genomes=preprocessed.norm_genomes,
            signatures=self.sigantures,
            init=self.init,
            beta_loss=self.beta_loss,
        )
        self._logger.log_info(f"Fitting the model for size: {size}")
        nmf_model.fit()
        self._logger.log_info("Done fitting")
        # Get NMF results and save to file
        W = nmf_model.W_norm
        # Calculate silhouette scores for each sample
        silhouette_scores, silhouette_avg = self.calculate_sil_score(W=W)
        signatures_df = create_signatures_df(W=W, signatures=self.sigantures)
        signatures_df.insert(0, "MutationType", mutations)

        nmf_filename = self.project / "NMF" / f"nmf.{size}.txt"
        signatures_df.to_csv(nmf_filename, sep="\t", index=False)
        # Save stats of the NMF results
        result = {
            "context": matrix.shape[0],
            "Reconstruction Err": nmf_model.reconstruction_err,
            "Silhouette Scores": silhouette_scores,
            "Silhouette Avg": silhouette_avg,
        }
        self.stats_nmf_list.append(result)
        return signatures_df

    def calculate_sil_score(self, W: np.ndarray) -> None:
        """
        Calculate the silhouette scores for each sample
        """
        kmeans = KMeans(n_clusters=self.sigantures, n_init=10)
        cluster_labels = kmeans.fit_predict(W)
        silhouette_scores = silhouette_samples(W, cluster_labels).tolist()
        silhouette_avg = silhouette_score(W, cluster_labels)
        return silhouette_scores, silhouette_avg

    def _prepare_folder(self):
        """
        Prepare folders for NMF files.
        """
        self.figure_folder.mkdir(parents=True, exist_ok=True)
        self.nmf_folder.mkdir(parents=True, exist_ok=True)
