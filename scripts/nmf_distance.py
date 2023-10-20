#!/usr/bin/env python3
"""
This script is a command-line utility that performs specific operations
based on the command-line arguments provided.
Get the cosine similarity and jensen shannon distance of the most optimal columns.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from DeepBayesMutSig import helpers, csv_reader, data_processing, nmf, distance, cosine


def process(file: str, signatures: int) -> pd.DataFrame:
    """
    Process a file by running NMF and return the resulting DataFrame.

    Args:
        file: Path to the input file.
        signatures: Number of signatures.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    matrix = csv_reader.get_matrix_mut_sig(file)
    all_genomes = np.array(matrix.iloc[:, 1:])
    preprocessed = data_processing.Preprocessing(all_genomes)
    nmf_model = nmf.RunNMF(preprocessed.norm_genomes, signatures)
    nmf_model.fit()
    W = nmf_model.W_norm
    columns = helpers.alphabet_list(signatures, f"SBS{W.shape[0]}")
    df = pd.DataFrame(W)
    df.insert(loc=0, column="MutationType", value=matrix[matrix.columns[0]])
    df.columns = ["MutationType"] + columns
    return df


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    cosmic_df = pd.read_csv("data\COSMIC_v3.3.1_SBS_GRCh37.txt", sep="\t")
    cosmic_df = cosmic_df.set_index("Type").reindex(helpers.MUTATION_LIST).reset_index()
    cosmic_df.drop(columns=cosmic_df.columns[0], axis=1, inplace=True)
    
    nmf_w_file = "data\WGS_PCAWG.96.csv"
    nmw_w = process(nmf_w_file, 48)
    nmw_w.drop(columns=nmw_w.columns[0], axis=1, inplace=True)
    nmw_w.columns = distance.get_optimal_columns(nmw_w, cosmic_df).values()
    files = [
        "output/nmf_files/nmf.WGS_PCAWG.1536.txt",
        "output/nmf_files/nmf.WGS_PCAWG.24576.txt",
        "output/nmf_files/nmf.WGS_PCAWG.393216.txt",
    ]
    
    for file in files:
        jens_filename = Path("results") / f"jensen.{file.split('.')[-2]}.dist.txt"
        cosine_filename = Path("results") / f"cosine.{file.split('.')[-2]}.dist.txt"
        df = pd.read_csv(files[0], sep="\t")
        df = helpers.compress(df)

        df.drop(columns=df.columns[0], axis=1, inplace=True)
        optimal_columns = distance.get_optimal_columns(nmw_w, df)
        result_jens = distance.get_jensen_shannon_distance(
            optimal_columns=optimal_columns, df1=nmw_w, df2=df
        )
        result_jens = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_jens])], ignore_index=True
        )

        result_cosine = cosine.cosine_nmf_w_optimal(
            optimal_columns=optimal_columns, df1=nmw_w, df2=df
        )
        result_cosine = pd.concat(
            [pd.DataFrame(), pd.DataFrame([result_cosine])], ignore_index=True
        )
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
