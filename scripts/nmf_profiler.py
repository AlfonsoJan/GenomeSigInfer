#!/usr/bin/env python3
"""
This script is a command-line utility that performs specific operations
based on the command-line arguments provided.
It can either run signature analysis (--spe) or perform matrix factorization (--nmf) on a CSV file.
"""
import sys
from DeepBayesMutSig import arguments, csv_reader, decompose

def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).

    Raises:
        ValueError: If neither '--spe' nor '--nmf' is selected as a command-line argument.
    """
    args = arguments.arguments_profiler()
    signatures = args.signatures
    if args.spe:
        print("Running analysis with SigProfilerExtractor")
    elif args.nmf:
        print("Running analysis with scikit NMF")
        genomes = csv_reader.get_matrix_mut_sig(args.genomes)
        genomes.index = genomes[genomes.columns[0]]
        genomes.drop(columns=genomes.columns[0], axis=1, inplace=True)
        sigs = csv_reader.get_matrix_nmf_w(args.file)
        decompose.quick_decompose(
            genomes=genomes,
            sigs=sigs,
            output="hello/"
        )
    else:
        raise ValueError("Please select one of '--spe' or '--nmf'")
    return 0

if __name__ == "__main__":
    sys.exit(main())