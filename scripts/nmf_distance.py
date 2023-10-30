#!/usr/bin/env python3
"""
This script is a command-line utility that creates a project.
"""
import sys
from pathlib import Path
import pandas as pd
from MutationalSignaturesTools import install, helpers, arguments, cosine, distance, mutSigMatrix





def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    args = arguments.arguments_only_project()
    project = Path(args.project)
    install.is_valid_project(project)
    file_extension = ["1536", "24576", "393216"]
    file_extension = ["1536", "24576"]
    nmf_folder = project / "NMF"
    cosmic_file = project / "cosmic" / "COSMIC_v3.3.1_SBS_GRCh37.txt"
    cosmic_df = pd.read_csv(cosmic_file, sep="\t").set_index("Type").reindex(helpers.MUTATION_LIST).reset_index()
    control_df = pd.read_csv(nmf_folder / f"nmf.96.txt", sep="\t").set_index("MutationType").reindex(helpers.MUTATION_LIST).reset_index()
    control_df.columns = ["MutationType"] + list(distance.get_optimal_columns(control_df.iloc[:, 1:], cosmic_df.iloc[:, 1:]).values())
    control_df = control_df.iloc[:, 1:]
    for file in file_extension:
        nmf_filename = nmf_folder / f"nmf.{file}.txt"
        matrix = pd.read_csv(nmf_filename, sep="\t", header=0)
        df = mutSigMatrix.compress_to_96(matrix)
        df = df.iloc[:, 1:]
        optimal_columns = distance.get_optimal_columns(control_df, df)
        result_jens = distance.get_jensen_shannon_distance(
            optimal_columns=optimal_columns, df1=control_df, df2=df
        )
        result_jens = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_jens])], ignore_index=True
        )
        result_cosine = cosine.cosine_nmf_w_optimal(
            optimal_columns=optimal_columns, df1=control_df, df2=df
        )
        result_cosine = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_cosine])], ignore_index=True
        )
        jens_filename = project / "results" / f"jensen.{matrix.shape[0]}.txt"
        cosine_filename = project / "results" / f"cosine.{matrix.shape[0]}.txt"
        result_cosine.to_csv(
            cosine_filename,
            sep="\t",
            index=False,
        )

        result_jens.to_csv(
            jens_filename,
            sep="\t",
            index=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
