#!/usr/bin/env python3
"""
SigPlots class is responsible for creating signature plots based on mutation data.
It utilizes the SBS class for accessing genome information and logging using SingletonLogger.
The class provides a method 'create_plots' to generate signature plots for different mutation contexts.

Author: J.A. Busker
"""
from pathlib import Path
import pandas as pd
from ..utils import logging, helpers
from . import barplots


class SigPlots:
    """
    SigPlots class is responsible for creating signature plots based on mutation data.
    It utilizes the SBS class (provide import statement or explanation) for accessing genome information and logging using SingletonLogger.
    The class provides methods to generate signature plots for different mutation contexts.
    """

    EXPECTED_SBS = ["SBS2", "SBS7a", "SBS10a", "SBS13", "SBS17b", "SBS22a"]

    def __init__(self, nmf_folder: Path, figure_folder: Path) -> None:
        """
        Initialize a SigPlots object.

        Args:
            nmf_folder (Path): Path to the folder containing decomposed mutation data.
            figure_folder (Path): Path to the folder where generated plots will be saved.

        This method sets up the SigPlots object with the provided project path.
        It initializes attributes for the project, NMF folder, figures folder, and a SingletonLogger for logging.
        """
        self.nmf_folder = Path(nmf_folder)
        self.figure_folder = Path(figure_folder)
        self.figure_folder.mkdir(parents=True, exist_ok=True)
        self._logger = logging.SingletonLogger()
        self._dfs = self._get_decompose_files()

    def _get_decompose_files(self) -> dict:
        """
        Retrieve decomposed mutation data files.

        Returns:
            dict: A dictionary containing decomposed mutation data DataFrames for different mutation context sizes.

        This method reads the decomposed mutation data for different mutation contexts from the NMF folder and returns a dictionary containing DataFrames for each size.
        """
        decompose_dict = {
            size: pd.read_csv(self.nmf_folder / f"decompose.{size}.txt", sep="\t")
            for size in helpers.MutationalSigantures.SIZES
        }
        return decompose_dict

    def create_plots(self) -> None:
        """
        Create signature plots based on mutation data.

        This method reads the decomposed mutation data for different mutation contexts
        from the NMF folder and generates signature plots. It checks the size of the
        mutation context and chooses the appropriate plotting function accordingly.
        The results are saved in the 'figures' folder within the result directory.
        """
        for size in helpers.MutationalSigantures.SIZES:
            self._logger.log_info(f"Creating siganture plots for context file: '{size}'")
            df = self._dfs[size]
            barplots.signature_pdf_plot(df, self.figure_folder)

    def create_expected_plots(self, sbs: list = None) -> None:
        """
        Create signature plots based on a list that those SBS are having the benefit of a larger context.

        Args:
        sbs (list, optional): List of SBS types to create signature plots for.
            If not provided, defaults to the predefined list in SigPlots.EXPECTED_SBS.
        """
        if sbs is None:
            sbs = SigPlots.EXPECTED_SBS
        barplots.create_expected_larger(self._dfs, sbs, self.figure_folder)

    def __repr__(self) -> str:
        """
        Return a string representation of the SigPlots class.
        """
        return (
            f"SigPlots(nmf_folder={self.nmf_folder}, figure_folder={self.figure_folder})"
        )
