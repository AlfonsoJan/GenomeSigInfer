#!/usr/bin/env python3
"""
This module provides a class and method for running SigProfilerExtractor
to extract genomic signatures.
"""
from pathlib import Path
import shutil
import pandas as pd
from SigProfilerExtractor import sigpro as sig
from .helpers import read_file_decompose


class RunSig:
    """
    A class for running SigProfilerExtractor to extract genomic signatures.
    Attributes:
        None
    Methods:
        run(matrix, signatures, out): Run SigProfilerExtractor with specified arguments.
    """

    @staticmethod
    def run(
        matrix: pd.DataFrame,
        out: str = "output",
        signatures: int = 1,
    ):
        """
        Run SigProfilerExtractor with specified arguments.

        Args:
            matrix (pd.DataFrame): Genomic data matrix.
            signatures (int): Number of extracted signatures
            out (str): Folder where the results are stored
        """
        df = pd.DataFrame()
        output = Path(out).joinpath(f"result-{signatures}-sig")
        sig.sigProfilerExtractor(
            "matrix",
            output.as_posix(),
            matrix,
            maximum_signatures=signatures,
            minimum_signatures=signatures,
            make_decomposition_plots=False,
        )
        decom_file = (
            output
            / "SBS96"
            / "Suggested_Solution"
            / "COSMIC_SBS96_Decomposed_Solution"
            / "Solution_Stats"
            / "Cosmic_SBS96_Decomposition_Log.txt"
        )
        read_file_decompose(decom_file, df, "sigprofiler")
        shutil.rmtree(output)
        return df
