#!/usr/bin/env python3
"""
NMFMatrixGenerator Module

This module defines the NMFMatrixGenerator class, responsible for generating NMF matrices
from Single Base Substitution (SBS) data. It utilizes the RunNMF class for the NMF process.

Attributes:
    sbs_folder (Path): Path to the folder containing SBS data.
    signatures (int): Number of signatures to extract.
    cosmic (pd.DataFrame): Cosmic mutation data.
    result_folder (Path): Path to the folder where result files will be stored.
    figure_folder (Path): Path to the folder where figure files will be stored.
    nmf_folder (Path): Path to the folder where NMF files will be stored.
    init (str): Initialization method for NMF.
    beta_loss (str): Beta loss function for NMF.
    control_df (pd.DataFrame): Dataframe for the smallest context file.
    cosine_similarities (list): List to store cosine similarity data.
    jens_shannon_distances (list): List to store Jensen Shannon Distance data.

Methods:
    run_nmf_on_sbs_files(): Perform the NMF process on all SBS files.
    sbs_read_and_extract(sbs_path: Path): Read data from an SBS file.
    _prepare_folders(): Create necessary result folders.
    _create_distance_figures(): Create cosine and Jensen Shannon Distance figures.
    _process_results(size: int, signatures_df: pd.DataFrame, index: int): Analyze the NMF results.
    _run_nmf_file(filename: Path) -> pd.DataFrame: Run NMF on a single SBS file.
    write_df_file(df: pd.DataFrame, name: Path, sep: str = "\t") -> None: Write a DataFrame to a file.
    _run_nmf(preprocessed: data_processing.Preprocessing, context: int) -> np.ndarray: Run NMF on preprocessed data.

Author: J.A. Busker
"""
from functools import wraps
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils import helpers, logging
from ..data import data_processing
from ..distance import distance
from ..figures import heatmap
from .run_nmf import RunNMF


def generate_nmf_matrix_arg_checker(func: callable) -> callable:
    """
    Decorator for argument checking in the `generate_nmf_matrix` function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: Decorated function.
    """

    @wraps(func)
    def wrapper(
        sbs_folder: Path,
        signatures: int,
        cosmic: Path,
        nmf_init: str,
        beta_los: str,
        result_folder: Path,
        nmf_folder: Path,
    ):
        """
        Ensure the validity of input arguments for the `generate_nmf_matrix` function.

        Args:
            sbs_folder (Path): Path to the SBS folder.
            signatures (int): Number of signatures to extract.
            cosmic (Path): Path to the cosmic data file.
            nmf_init (str): Initialization method for NMF.
            beta_los (str): Beta loss function for NMF.
            result_folder (Path): Path to the result folder.
            nmf_folder (Path): Path to the NMF folder.

        Raises:
            TypeError: If input types are not valid or within expected values.
        """
        # Ensure Path object are actually Path objects and exist
        cosmic = Path(cosmic)
        sbs_folder = Path(sbs_folder)
        nmf_folder = Path(nmf_folder)
        result_folder = Path(result_folder)
        if not isinstance(signatures, int) or signatures < 1:
            raise TypeError("Input 'signatures' must be a positive integer.")
        # Make sure the init and betaloss are correct
        if nmf_init not in helpers.NMF_INITS:
            raise TypeError(f"NMF init must be one of: {', '.join(helpers.NMF_INITS)}")
        if beta_los not in helpers.BETA_LOSS:
            raise TypeError(
                f"NMF beta loss must be one of: {', '.join(helpers.BETA_LOSS)}"
            )
        return func(
            sbs_folder,
            signatures,
            cosmic,
            nmf_init,
            beta_los,
            result_folder,
            nmf_folder,
        )

    return wrapper


@generate_nmf_matrix_arg_checker
def generate_nmf_matrix(
    sbs_folder: Path,
    signatures: int,
    cosmic: Path,
    nmf_init: str,
    beta_los: str,
    result_folder: Path,
    nmf_folder: Path,
):
    """
    Generate NMF matrix from SBS data and perform analysis.

    Args:
        sbs_folder (Path): Path to the SBS folder.
        signatures (int): Number of signatures to extract.
        cosmic (Path): Path to the cosmic data file.
        nmf_init (str): Initialization method for NMF.
        beta_los (str): Beta loss function for NMF.
        result_folder (Path): Path to the result folder.
        nmf_folder (Path): Path to the NMF folder.

    Raises:
        FileNotFoundError: If any of the required folders or files are not found.
    """
    # Read the COSMIC file
    cosmic = (
        pd.read_csv(cosmic, sep="\t")
        .set_index("Type")
        .reindex(helpers.MUTATION_LIST)
        .reset_index()
    )
    nmfmatrixgen = NMFMatrixGenerator(
        sbs_folder=sbs_folder,
        signatures=signatures,
        cosmic=cosmic,
        result_folder=result_folder,
        nmf_folder=nmf_folder,
        init=nmf_init,
        beta_loss=beta_los,
    )
    nmfmatrixgen.run_nmf_on_sbs_files()


def sbs_read_and_extract(sbs_path: Path):
    """
    Read data from an SBS file.

    Args:
        filename (Path): File name/path.

    Returns:
        tuple: Tuple containing mutations and all_genomes.
    """
    if not sbs_path.exists:
        raise Exception
    # Extract the data from the SBS file
    # And return an numpy array with only the values (all_genomes)
    # And a list of the mutations
    matrix = pd.read_csv(sbs_path, sep=",", header=0)
    mutations = matrix[matrix.columns[0]]
    all_genomes = np.array(matrix.iloc[:, 1:])
    return mutations, all_genomes


class NMFMatrixGenerator:
    """
    NMFMatrixGenerator class is responsible for generating NMF matrices from Single Base Substitution (SBS) data.
    It uses the RunNMF class for the NMF process and performing analysis.
    """

    def __init__(
        self,
        sbs_folder: Path,
        signatures: int,
        cosmic: pd.DataFrame,
        result_folder: Path,
        nmf_folder: Path,
        init: str = "nndsvda",
        beta_loss: str = "frobenius",
    ):
        """
        Initialize the NMFMatrixGenerator.

        Args:
            sbs_folder (Path): Path to the folder containing SBS data.
            signatures (int): Number of signatures to extract.
            cosmic (pd.DataFrame): Cosmic mutation data.
            result_folder (Path): Path to the folder where result files will be stored.
            nmf_folder (Path): Path to the folder where NMF files will be stored.
            init (str): Initialization method for NMF.
            beta_loss (str): Beta loss function for NMF.
        """
        self._logger = logging.SingletonLogger()
        self.signatures = signatures
        self.sbs_folder = sbs_folder
        self.cosmic = cosmic
        self.result_folder = result_folder
        self.figure_folder = result_folder / "figures"
        self.nmf_folder = nmf_folder
        self.beta_loss = beta_loss
        self.init = init
        # The smallest context one
        self.control_df = None
        # List to store cosine similarity dat
        self.cosine_similarities = []
        # List to store Jensen Shannon Distance data
        self.jens_shannon_distances = []
        self._prepare_folders()

    def _prepare_folders(self) -> None:
        """
        Create necessary result folders.
        """
        # Create the folder if it does not exist yet
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.nmf_folder.mkdir(parents=True, exist_ok=True)
        self.figure_folder.mkdir(parents=True, exist_ok=True)

    def run_nmf_on_sbs_files(self) -> None:
        """
        Perform the NMF process on all SBS files.
        """
        self._logger.log_info(f"Creating NMF files with {self.signatures} signatures")
        # For every file run nmf
        for index, context in enumerate(helpers.MutationalSigantures.SIZES, -1):
            filename = self.sbs_folder / f"sbs.{context}.txt"
            signatures_df = self._run_nmf_file(filename)
            # Process the NMF results
            self._process_results(context, signatures_df, index)
        self._logger.log_info("Creating cosine and Jensen Shannon Distance plots")
        # Create figures
        self._create_distance_figures()

    def _create_distance_figures(self) -> None:
        """
        Create cosine and Jensen Shannon Distance figures.
        """
        # Generate heatmap figures
        heatmap.heatmap_cosine(self.cosine_similarities, self.figure_folder)
        heatmap.heatmap_jens_shan(self.jens_shannon_distances, self.figure_folder)
        # Log completion
        self._logger.log_info("Created cosine similarity plots from the NMF results")
        self._logger.log_info(
            "Created Jensen Shannon Distance plots from the NMF results"
        )

    def _process_results(
        self, size: int, signatures_df: pd.DataFrame, index: int
    ) -> None:
        """
        Analyze the NMF results.

        Args:
            size (int): Size of the SBS file.
            signatures_df (pd.DataFrame): DataFrame containing NMF results.
            index (int): Index of the current iteration.
        """
        # Process figures based on the size
        decompose_filename = self.nmf_folder / f"decompose.{size}.txt"
        # If its the smallest context file make it the 'control df'
        # So the larger ones compare against this
        if size == 96:
            signatures_df = (
                signatures_df.set_index("MutationType")
                .reindex(helpers.MUTATION_LIST)
                .reset_index()
            )
            # Set the optimal columns of the cosmic file on this df
            control_df = distance.set_optimal_columns(signatures_df, self.cosmic)
            self.write_df_file(control_df, decompose_filename)
            self.control_df = control_df.iloc[:, 1:]
        else:
            # Calculate distances and store in lists
            (
                result_jens_df,
                result_cosine_df,
                decomposed_df,
            ) = distance.calculate_distances(self.control_df, signatures_df)
            self.cosine_similarities.append(
                {
                    "data": result_cosine_df,
                    "context": helpers.MutationalSigantures.CONTEXT_LIST[index],
                }
            )
            self.jens_shannon_distances.append(
                {
                    "data": result_jens_df,
                    "context": helpers.MutationalSigantures.CONTEXT_LIST[index],
                }
            )
            # Write the calculated similarities to files
            jens_filename = self.result_folder / f"jensen.{size}.txt"
            cosine_filename = self.result_folder / f"cosine.{size}.txt"
            self.write_df_file(result_cosine_df, cosine_filename)
            self.write_df_file(result_jens_df, jens_filename)
            self.write_df_file(decomposed_df, decompose_filename)

    def _run_nmf_file(self, filename):
        """
        Run NMF on a single SBS file.

        Args:
            filename (Path): Path to the SBS file.

        Returns:
            pd.DataFrame: DataFrame containing NMF results.
        """
        self._logger.log_info(f"Performing NMF on '{filename}'")
        mutations, all_genomes = sbs_read_and_extract(filename)
        context = mutations.shape[0]
        self._logger.log_info("Preprocessing the data")
        # Preprocess the data
        preprocessed = data_processing.Preprocessing(all_genomes)
        W = self._run_nmf(preprocessed, context)
        # Create signature dataframe
        signatures_df = helpers.create_signatures_df(W=W, signatures=self.signatures)
        signatures_df.insert(0, "MutationType", mutations)
        nmf_filename = self.nmf_folder / f"nmf.{context}.txt"
        # Write to a file
        self.write_df_file(signatures_df, nmf_filename)
        return signatures_df

    def write_df_file(self, df: pd.DataFrame, name: Path, sep: str = "\t") -> None:
        """
        Write a DataFrame to a file.

        Args:
            df (pd.DataFrame): DataFrame to be written.
            name (Path): File name/path.
            sep (str): Separator for the file (default is "\t").
        """
        self._logger.log_info(f"Written the results to '{name}'")
        df.to_csv(name, sep=sep, index=False)

    def _run_nmf(
        self, preprocessed: data_processing.Preprocessing, context: int
    ) -> np.ndarray:
        """
        Run NMF on preprocessed data.

        Args:
            preprocessed (data_processing.Preprocessing): Preprocessed data.
            context (int): Context size.

        Returns:
            np.ndarray: W matrix from NMF.
        """
        # Create NMF model and fit
        nmf_model = RunNMF(
            genomes=preprocessed.norm_genomes,
            signatures=self.signatures,
            init=self.init,
            beta_loss=self.beta_loss,
        )
        self._logger.log_info(f"Fitting the model for size: {context}")
        nmf_model.fit()
        self._logger.log_info("Done fitting")
        # Get NMF results and save to file
        W = nmf_model.W_norm
        return W
