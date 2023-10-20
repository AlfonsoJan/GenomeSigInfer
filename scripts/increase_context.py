#!/usr/bin/env python3
"""
This module will increase the context size of a file
"""
import sys
from pathlib import Path
from DeepBayesMutSig import arguments, csv_reader, context

def main() -> int:
    """
    Main entry point of the script.
    """
    args = arguments.arguments_incresae_context()
    for c in [5, 7, 9, 11]:
        matrix96 = csv_reader.get_matrix_mut_sig(file_str=args.file)
        new_df = context.increase_context(matrix96, c)
        new_df.to_csv(
            Path(args.out) / f"WGS_PCAWG.{new_df.shape[0]}.csv",
            index=False,
            sep="\t",
        )


if __name__ == "__main__":
    sys.exit(main())
