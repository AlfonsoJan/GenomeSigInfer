#!/usr/bin/env python3
"""
This script is a command-line utility that performs specific operations.
For performing Non-Negative Matrix Factorization (NMF) n amount of times
and get the average of it and write it to a file.
"""
import sys
from pathlib import Path
import numpy as np
from DeepBayesMutSig import arguments, csv_reader, nmf, helpers


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    args = arguments.arguments_nmf()
    matrix = csv_reader.get_matrix_mut_sig(file_str=args.file, sep=args.sep)
    mutations = matrix[matrix.columns[0]]
    signatures = args.signatures
    all_genomes = np.array(matrix.iloc[:, 1:])
    W = nmf.run_multiple(
        all_genomes=all_genomes, signatures=signatures, iters=args.iter
    )
    signatures_df = helpers.create_signatures_df(W=W, signatures=signatures)
    signatures_df.insert(0, "MutationType", mutations)
    filename = Path(args.file).name
    cols = ".".join(filename.split(".")[:-1])
    signatures_df.index = mutations
    signatures_df.to_csv(
        Path(args.out) / f"nmf.{cols}.txt",
        index=False,
        sep="\t",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
