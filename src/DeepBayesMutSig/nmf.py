#!/usr/bin/env python3
"""
This module provides classes and functions for performing Non-Negative Matrix Factorization (NMF)
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from .matrix_operations import compress_to_96
from .distance import (
    set_optimal_columns,
    get_optimal_columns,
    get_jensen_shannon_distance,
)
from .sbs import SBS
from .heatmap import heatmap_best_param, heatmap_cosine, heatmap_jens_shan
from .logging import SingletonLogger
from .sigprofiler import RunSig
from .data_processing import Preprocessing
from .decompose import Decompose
from .cosine import most_similarity_decompose, cosine_nmf_w
from .helpers import (
    combinations,
    read_file_decompose,
    get_96_matrix,
    create_signatures_df,
    MUTATION_LIST,
)


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
        self._solver: str = "cd" if beta_loss == "frobenius" else "mu"
        self._genomes: np.ndarray = genomes
        self._init: str | None = None if init == "None" else init
        self._beta_loss: str = beta_loss
        self._signatures: int = signatures
        self._W: np.ndarray = None

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
        self._W: np.ndarray = nmf.fit_transform(self._genomes)

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

    """
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
        read_file_decompose(decom_file, result_df)
        shutil.rmtree(outpath)
    return result_df


class NMF_SBS:
    SIZES = SBS(Path(""), Path(""), "GRCh37")._sizes
    MAX_CONTEXT = SBS(Path(""), Path(""), "GRCh37").MAX_CONTEXT

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
        self._context_list: list[int] = list(range(self.MAX_CONTEXT, 2, -2))
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
        for index, size in enumerate(self.SIZES, -1):
            signatures_df = self.perform_nmf(size)
            self._process_figures(size, signatures_df, index)
        self._logger.log_info("Creating cosine and Jensen Shannon Distance plots")
        self._create_distance_figures()

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
                {"data": result_cosine_df, "context": self._context_list[index]}
            )
            self.jens_shannon_distances.append(
                {"data": result_jens_df, "context": self._context_list[index]}
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

        jens_filename = result_folder / f"jensen.{df_compare.shape[0]}.txt"
        cosine_filename = result_folder / f"cosine.{df_compare.shape[0]}.txt"
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
            columns (dict): A dictionary where keys are column names from df1, and values are corresponding column names from df2.
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
        # Create NMF model and fit
        nmf_model = RunNMF(
            genomes=all_genomes,
            signatures=self.sigantures,
            init=self.init,
            beta_loss=self.beta_loss,
        )
        self._logger.log_info(f"Fitting the model for size: {size}")
        nmf_model.fit()
        self._logger.log_info("Done fitting")
        # Get NMF results and save to file
        W = nmf_model.W_norm
        signatures_df = create_signatures_df(W=W, signatures=self.sigantures)
        signatures_df.insert(0, "MutationType", mutations)

        nmf_filename = self.project / "NMF" / f"nmf.{size}.txt"
        signatures_df.to_csv(nmf_filename, sep="\t", index=False)
        return signatures_df

    def _prepare_folder(self):
        """
        Prepare folders for NMF files.
        """
        self.figure_folder.mkdir(parents=True, exist_ok=True)
        self.nmf_folder.mkdir(parents=True, exist_ok=True)


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
        self.matrix = self.load_and_process_matrix()
        self.all_genomes = np.array(self.matrix.iloc[:, 1:])

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
