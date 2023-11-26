#!/usr/bin/env python3
"""
This module defines a class for performing signature decomposition on given genomic data.
It utilizes SigProfilerAssignment.decomposition for the decomposition process and outputs
the results in a formatted text file.
"""
from pathlib import Path
import pandas as pd
from SigProfilerAssignment import decomposition as decomp
from ..utils.helpers import alphabet_list


class Decompose:
    """
    A utility class for decomposing genomic data into mutational signatures.
    """

    def __init__(
        self, W, mutations, signatures, output, all_genomes, col_names
    ) -> None:
        """
        Initialize the Decompose class with necessary parameters.

        Args:
            W: The W matrix from NMF decomposition.
            mutations: List of mutation types.
            signatures: Number of signatures.
            output: Output directory for results.
            all_genomes: Dataframe containing genomic data.
            col_names: Column names for genomic data.
        """
        self._W = W
        self._mutations = mutations
        self._signatures = signatures
        self._output = output
        self._all_genomes = all_genomes
        self._columns = alphabet_list(self._signatures, f"SBS{W.shape[0]}")
        self._col_names = col_names[1:]

    def decompose(self, folder: str = "result-nmf") -> None:
        """
        Perform signature decomposition and save results.

        Args:
            folder (str): Output subfolder name.
        """
        df = pd.DataFrame(self._W)
        df.columns = self._columns
        signatures = df.copy()
        outpath = Path(self._output).joinpath(folder.lower())
        outpath.mkdir(parents=True, exist_ok=True)
        df.insert(0, "MutationType", self._mutations)
        filename = outpath.joinpath("result.txt").as_posix()
        df.to_csv(
            filename,
            encoding="utf-8",
            index=False,
            sep="\t",
        )
        self._decompose(outpath=outpath, signatures=signatures)

    def _decompose(self, outpath: Path, signatures: pd.DataFrame) -> None:
        """
        Perform the actual signature decomposition.

        Args:
            outpath: Output path.
            signatures: Signature data.
        """
        genomes = pd.DataFrame(self._all_genomes)
        genomes.index = self._mutations
        genomes.columns = self._col_names
        decomp.spa_analyze(
            genomes,
            outpath.as_posix(),
            signatures=signatures,
            connected_sigs=True,
            decompose_fit_option=True,
            denovo_refit_option=False,
            cosmic_fit_option=True,
            signature_database=None,
            cosmic_version=3.3,
            exome=False,
            export_probabilities=True,
            export_probabilities_per_mutation=False,
            sample_reconstruction_plots=False,
            make_plots=False,
        )
