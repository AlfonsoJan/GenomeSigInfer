#!/usr/bin/env python3
"""
Script for running NMF with different combinations of initialization and beta loss,
and performing decomposition on the resulting matrices.
"""
import sys
from pathlib import Path
import numpy as np
from DeepBayesMutSig import arguments, csv_reader, helpers, nmf


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    filename = "param.tuning.decomp.txt"
    args = arguments.arguments_nmf()
    matrix = csv_reader.get_matrix_mut_sig(args.file)
    all_genomes = np.array(matrix.iloc[:, 1:])
    nmf_combs = helpers.combinations()
    result = nmf.run_nmfs(nmf_combs, all_genomes, args.signatures, matrix, args.out)
    result["SigProfiler"] = helpers.SIGPROFILER_DECOMP["SigProfiler"]
    result.to_csv(Path(args.out) / filename, index=False, sep="\t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
