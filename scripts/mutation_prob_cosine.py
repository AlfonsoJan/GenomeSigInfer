#!/usr/bin/env python3
"""
This module will get for every sample the normalized (between 0 and 1)
the chance for a mutation type in a SBS
"""
import sys
from pathlib import Path
from DeepBayesMutSig import arguments, cosine, helpers


def main() -> int:
    """
    Main entry point of the script.
    """
    args = arguments.arguments_mutation_prob_cosine()
    folder1, folder2 = args.folders
    df1, df2 = helpers.read_file_mut_prob(folder1), helpers.read_file_mut_prob(folder2)
    result = cosine.cosine_mut_prob(df1, df2)
    result.to_csv(
        Path(args.out).joinpath("cosine.similarity.decompose.txt"),
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    sys.exit(main())
