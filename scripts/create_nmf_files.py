#!/usr/bin/env python3
"""
This script is a command-line utility that creates a project.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from MutationalSignaturesTools import nmf, install, helpers, arguments


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    args = arguments.arguments_nmf()
    signatures = args.signatures
    project = Path(args.project)
    install.is_valid_project(project)
    file_extension = ["96", "1536", "24576", "393216"]
    nmf_folder = project / "NMF"
    for file in file_extension:
        nmf_filename = nmf_folder / f"nmf.{file}.txt"
        file_str = project / "SBS" / f"sbs.{file}.txt"
        matrix = pd.read_csv(file_str, sep=",", header=0)
        mutations = matrix[matrix.columns[0]]
        all_genomes = np.array(matrix.iloc[:, 1:])
        W = nmf.run_multiple(
            all_genomes=all_genomes, signatures=signatures, iters=1
        )
        signatures_df = helpers.create_signatures_df(W=W, signatures=signatures)
        signatures_df.insert(0, "MutationType", mutations)
        signatures_df.to_csv(nmf_filename, index=False, sep="\t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
