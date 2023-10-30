#!/usr/bin/env python3
"""
This script is a command-line utility that creates a project.
"""
import sys
from pathlib import Path
import numpy as np
from MutationalSignaturesTools import nmf, install, helpers, sigprofiler, arguments, cosine


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
    temp_folder = project / "results"
    result_filename = temp_folder / "param.tuning.decomp.txt"
    matrix = helpers.get_sbs_from_proj(project)
    all_genomes = np.array(matrix.iloc[:, 1:])
    nmf_combs = helpers.combinations()
    df = sigprofiler.RunSig.run(
        matrix=matrix, signatures=signatures, out=temp_folder
    )
    df = nmf.run_nmfs(
        nmf_combs, all_genomes, signatures, matrix, temp_folder, df
    )
    result_df = cosine.most_similarity_decompose(df)
    result_df.to_csv(result_filename, index=False, sep="\t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
