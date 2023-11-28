#!/usr/bin/env python3
"""
This module provides classes and functions for performing Non-Negative Matrix Factorization (NMF)

Create NMF files based on the specified number of signatures.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils import logging, helpers
from ..data import data_processing
from ..distance import distance
from ..figures import heatmap
from . import run_nmf

class NMF_SBS:
    """
    Create NMF files based on the specified number of signatures.

    Args:
        sbs_folder (Path): sbs_folder folder name.
        nmf_folder (Path): Folder for the computed NMF files.
        result_folder (Path): Folder for the stats resutls.
        sigs (int): signatures in SBS files.
        cosmic (Path): path of the cosmic file.
        init (str): init for the NMF
        beta_loss (str): beta_loss for the NMF
    """

    def __init__(
        self,
        sbs_folder: Path,
        nmf_folder: Path,
        result_folder: Path,
        sigs: int,
        cosmic: Path,
        init: str = "nndsvda",
        beta_loss: str = "frobenius",
    ) -> None:
        """
        Initialize the NMF_SBS instance.

        Args:
            sbs_folder (Path): sbs_folder folder name.
            nmf_folder (Path): Folder for the computed NMF files.
            result_folder (Path): Folder for the stats results.
            sigs (int): signatures in SBS files.
            cosmic (Path): path of the cosmic file.
            init (str): init for the NMF (default is "nndsvda").
            beta_loss (str): beta_loss for the NMF (default is "frobenius").
        """
        self.result_folder = Path(result_folder)
        self.figure_folder = self.result_folder / "figures"
        self.nmf_folder = Path(nmf_folder)
        self.signatures = sigs
        self.init = init
        self.beta_loss = beta_loss
        self._logger = logging.SingletonLogger()
        # THe smallest context one
        self.control_df = None
        # List to store cosine similarity dat
        self.cosine_similarities = []
        # List to store Jensen Shannon Distance data
        self.jens_shannon_distances = []
        # Read cosmic file
        self.cosmic = (
            pd.read_csv(cosmic, sep="\t")
            .set_index("Type")
            .reindex(helpers.MUTATION_LIST)
            .reset_index()
        )
        self._perform_nmf(sbs_folder)
    
    def _perform_nmf(self, sbs_folder: Path) -> None:
        """
        Perform the NMF process.

        Args:
            sbs_folder (Path): sbs_folder folder name.
        """
        # Create Folders
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.nmf_folder.mkdir(parents=True, exist_ok=True)
        self.figure_folder.mkdir(parents=True, exist_ok=True)
        self._logger.log_info(f"Creating NMF files with {self.signatures} signatures")
        sbs_folder = Path(sbs_folder)
        # For every file run nmf
        for index, context in enumerate(helpers.MutationalSigantures.SIZES, -1):
            signatures_df = self._run_nmf_file(sbs_folder, context)
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
    
    def _process_results(self, size: int, signatures_df: pd.DataFrame, index: int) -> None:
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
            result_jens_df, result_cosine_df, decomposed_df = distance.calculate_distances(self.control_df, signatures_df)
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
    
    def write_df_file(self, df: pd.DataFrame, name: Path, sep: str = "\t") -> None:
        """
        Write a DataFrame to a file.

        Args:
            df (pd.DataFrame): DataFrame to be written.
            name (Path): File name/path.
            sep (str): Separator for the file (default is "\t").
        """
        df.to_csv(name, sep=sep, index=False)
    
    def _run_nmf_file(self, sbs_folder: Path, context: int) -> pd.DataFrame:
        """
        Run NMF on a specific file.

        Args:
            sbs_folder (Path): sbs_folder folder name.
            context (int): Context size.

        Returns:
            pd.DataFrame: DataFrame containing NMF results.
        """
        filename = sbs_folder / f"sbs.{context}.txt"
        # Read the SBS file and extract import matrices
        mutations, all_genomes = self._read_sbs_file(filename)
        self._logger.log_info("Preprocessing the data")
        # Preprocess the data
        preprocessed = data_processing.Preprocessing(all_genomes)
        W = self._run_nmf(preprocessed, context)
        # Create signature dataframe
        signatures_df = helpers.create_signatures_df(W=W, signatures=self.signatures)
        signatures_df.insert(0, "MutationType", mutations)
        nmf_filename = self.nmf_folder / f"nmf.{context}.txt"
        self.write_df_file(signatures_df, nmf_filename)
        return signatures_df
    
    def _read_sbs_file(self, filename: Path) -> tuple[pd.Series, np.ndarray]:
        """
        Read data from an SBS file.

        Args:
            filename (Path): File name/path.

        Returns:
            tuple: Tuple containing mutations and all_genomes.
        """
        # Extract the data from the SBS file
        # And return an numpy array with only the values (all_genomes)
        # And a list of the mutations
        matrix = pd.read_csv(filename, sep=",", header=0)
        mutations = matrix[matrix.columns[0]]
        all_genomes = np.array(matrix.iloc[:, 1:])
        return mutations, all_genomes
    
    def _run_nmf(self, preprocessed: data_processing.Preprocessing, context: int) -> np.ndarray:
        """
        Run NMF on preprocessed data.

        Args:
            preprocessed (data_processing.Preprocessing): Preprocessed data.
            context (int): Context size.

        Returns:
            np.ndarray: W matrix from NMF.
        """
        # Create NMF model and fit
        nmf_model = run_nmf.RunNMF(
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
