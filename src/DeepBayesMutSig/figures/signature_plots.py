#!/usr/bin/env python3
"""
SigPlots class is responsible for creating signature plots based on mutation data.
It utilizes the SBS class for accessing genome information and logging using SingletonLogger.
The class provides a method 'create_plots' to generate
signature plots for different mutation contexts.
"""
from pathlib import Path
import pandas as pd
from ..utils.logging import SingletonLogger
from ..utils.helpers import MutationalSigantures
from .barplots import larger_context_barplot, create_96_barplot, create_expected_larger


class SigPlots:
    """
    This method sets up the SigPlots object with the provided project path. It initializes
    attributes for project, NMF folder, figures folder, and a SingletonLogger for logging.
    """

    def __init__(self, project: Path) -> None:
        """
        Initialize SigPlots object.

        Args:
            - project (Path): Path to the project folder.
        """
        self.project = project
        self.nmf_folder = self.project / "NMF"
        self.figures = self.project / "figures"
        self._logger = SingletonLogger()
        self._dfs = {}

    def create_plots(self):
        """
        Create signature plots based on mutation data.
        """
        for size in MutationalSigantures.SIZES:
            decompose_file = self.nmf_folder / f"decompose.{size}.txt"
            df = pd.read_csv(decompose_file, sep="\t")
            self._dfs[size] = df
            if df.shape[0] != 96:
                self._logger.log_info(
                    f"Creating siganture plots for context file: '{df.shape[0]}'"
                )
                larger_context_barplot(df, self.figures)
            else:
                self._logger.log_info("Creating siganture plots for context file: '96'")
                create_96_barplot(df, self.figures)

    def create_expected_plots(
        self, sbs=["SBS2", "SBS7a", "SBS10a", "SBS13", "SBS17a", "SBS17b"]
    ):
        """
        Create signature plots based on a list that
        those sbs are having benefit of a larger context.
        """
        create_expected_larger(self._dfs, sbs, self.figures)
