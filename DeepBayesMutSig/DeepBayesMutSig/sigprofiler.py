#!/usr/bin/env python3
"""
This module provides a class and method for running SigProfilerExtractor
to extract genomic signatures.
"""
from pathlib import Path
import pandas as pd
from SigProfilerExtractor import sigpro as sig


class RunSig:
    """
    A class for running SigProfilerExtractor to extract genomic signatures.
    Attributes:
        None
    Methods:
        run(matrix, signatures, out): Run SigProfilerExtractor with specified arguments.
    """

    @staticmethod
    def run(matrix: pd.DataFrame, signatures: int = 1, out: str = "output"):
        """
        Run SigProfilerExtractor with specified arguments.

        Args:
            matrix (pd.DataFrame): Genomic data matrix.
            signatures (int): Number of extracted signatures
            out (str): Folder where the results are stored
        """
        output = (
            Path(out).joinpath(f"result-{signatures}signatures-SigProfiler").as_posix()
        )
        sig.sigProfilerExtractor(
            "matrix",
            output,
            matrix,
            maximum_signatures=signatures,
            minimum_signatures=signatures,
        )
