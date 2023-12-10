#!/usr/bin/env python3
"""
SigPlots class is responsible for creating signature plots based on mutation data.
It utilizes the SBS class for accessing genome information and logging using SingletonLogger.
The class provides a method 'create_plots' to generate
signature plots for different mutation contexts.

Author: J.A. Busker
"""
from pathlib import Path
import pandas as pd
from ..utils import logging, helpers
from . import barplots


class SigPlots:
    """
    This method sets up the SigPlots object with the provided project path. It initializes
    attributes for project, NMF folder, figures folder, and a SingletonLogger for logging.
    """

    EXPECTED_SBS = ["SBS2", "SBS7a", "SBS10a", "SBS13", "SBS17a", "SBS17b"]

    def __init__(self, nmf_folder: Path, figure_folder: Path) -> None:
        self.nmf_folder = Path(nmf_folder)
        self.figures = Path(figure_folder)
        self.figures.mkdir(parents=True, exist_ok=True)
        self._logger = logging.SingletonLogger()
        self._dfs = {}

    def create_plots(self):
        """
        Create signature plots based on mutation data.

        This method reads the decomposed mutation data for different mutation contexts
        from the NMF folder and generates signature plots. It checks the size of the
        mutation context and chooses the appropriate plotting function accordingly.
        The results are saved in the 'figures' folder within the result directory.
        """
        for size in helpers.MutationalSigantures.SIZES:
            decompose_file = self.nmf_folder / f"decompose.{size}.txt"
            df = pd.read_csv(decompose_file, sep="\t")
            self._dfs[size] = df
            if df.shape[0] != 96:
                self._logger.log_info(
                    f"Creating siganture plots for context file: '{df.shape[0]}'"
                )
                barplots.larger_context_barplot(df, self.figures)
            else:
                self._logger.log_info("Creating siganture plots for context file: '96'")
                barplots.create_96_barplot(df, self.figures)

    def create_expected_plots(self, sbs: list = None):
        """
        Create signature plots based on a list that
        those SBS are having the benefit of a larger context.

        This method utilizes the pre-processed mutation data stored in the '_dfs'
        attribute and generates signature plots for the SBS types specified in
        'EXPECTED_SBS'. The results are saved in the 'figures' folder within the
        result directory.
        """
        if sbs is None:
            sbs = SigPlots.EXPECTED_SBS
        barplots.create_expected_larger(self._dfs, sbs, self.figures)
