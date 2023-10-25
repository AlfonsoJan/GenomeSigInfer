#!/usr/bin/env python3
"""
This module contains small, common utility functions used across the project.

Functions:
- alphabet_list(amount: int, genome: str) -> list[str]: Generate a list of column labels.
- create_signatures_df(W: np.ndarray, signatures: int) -> pd.DataFrame: Create a dataframe of the result of the NMF.
- compress(df: pd.DataFrame, mutation: list) -> pd.DataFrame: Compress the DataFrame by summing every 16 rows for each column.
- combinations() -> list[tuple[str, str]]: Generate combinations of initialization and beta loss.
- read_file_decompose(file: str, dataframe: pd.DataFrame) -> None: Read the contents of a file and extract signature data.
"""
import itertools
import string
from pathlib import Path
import pandas as pd
import numpy as np

# Decomposition of 96
SIGPROFILER_DECOMP = {
    "SigProfiler": [
        ["SBS5"],
        ["SBS5", "SBS34"],
        ["SBS5", "SBS17b"],
        ["SBS22"],
        ["SBS28", "SBS58"],
        ["SBS7c"],
        ["SBS1", "SBS5"],
        ["SBS1", "SBS5", "SBS17b"],
        ["SBS1", "SBS54"],
        ["SBS1", "SBS5", "SBS17a"],
        ["SBS5", "SBS86"],
        ["SBS39"],
        ["SBS5", "SBS10d", "SBS91"],
        ["SBS5", "SBS21", "SBS60", "SBS91"],
        ["SBS13", "SBS21"],
        ["SBS5", "SBS43"],
        ["SBS34", "SBS85"],
        ["SBS17a", "SBS17b", "SBS31"],
        ["SBS21"],
        ["SBS5"],
        ["SBS17a", "SBS35"],
        ["SBS87"],
        ["SBS7d", "SBS10a", "SBS13"],
        ["SBS5", "SBS86"],
        ["SBS50", "SBS51"],
        ["SBS1", "SBS5", "SBS50"],
        ["SBS5", "SBS39"],
        ["SBS5", "SBS32"],
        ["SBS1", "SBS5", "SBS19"],
        ["SBS5"],
        ["SBS1", "SBS5"],
        ["SBS5", "SBS27"],
        ["SBS39"],
        ["SBS1", "SBS5", "SBS32"],
        ["SBS2", "SBS9"],
        ["SBS1", "SBS5", "SBS59"],
        ["SBS39"],
        ["SBS5"],
        ["SBS1", "SBS3"],
        ["SBS1", "SBS5"],
        ["SBS1", "SBS5", "SBS24"],
        ["SBS25"],
        ["SBS57"],
        ["SBS5", "SBS48"],
        ["SBS1", "SBS16", "SBS45"],
        ["SBS5", "SBS29"],
        ["SBS5"],
        ["SBS10b", "SBS33"],
    ]
}

# MUTATION TYPES random order
MUTATION_TYPES = ["[C>G]", "[C>A]", "[C>T]", "[T>G]", "[T>C]", "[T>A]"]

# List of the mutation in order
MUTATION_LIST = [
    "A[C>A]A",
    "A[C>A]C",
    "A[C>A]G",
    "A[C>A]T",
    "C[C>A]A",
    "C[C>A]C",
    "C[C>A]G",
    "C[C>A]T",
    "G[C>A]A",
    "G[C>A]C",
    "G[C>A]G",
    "G[C>A]T",
    "T[C>A]A",
    "T[C>A]C",
    "T[C>A]G",
    "T[C>A]T",
    "A[C>G]A",
    "A[C>G]C",
    "A[C>G]G",
    "A[C>G]T",
    "C[C>G]A",
    "C[C>G]C",
    "C[C>G]G",
    "C[C>G]T",
    "G[C>G]A",
    "G[C>G]C",
    "G[C>G]G",
    "G[C>G]T",
    "T[C>G]A",
    "T[C>G]C",
    "T[C>G]G",
    "T[C>G]T",
    "A[C>T]A",
    "A[C>T]C",
    "A[C>T]G",
    "A[C>T]T",
    "C[C>T]A",
    "C[C>T]C",
    "C[C>T]G",
    "C[C>T]T",
    "G[C>T]A",
    "G[C>T]C",
    "G[C>T]G",
    "G[C>T]T",
    "T[C>T]A",
    "T[C>T]C",
    "T[C>T]G",
    "T[C>T]T",
    "A[T>A]A",
    "A[T>A]C",
    "A[T>A]G",
    "A[T>A]T",
    "C[T>A]A",
    "C[T>A]C",
    "C[T>A]G",
    "C[T>A]T",
    "G[T>A]A",
    "G[T>A]C",
    "G[T>A]G",
    "G[T>A]T",
    "T[T>A]A",
    "T[T>A]C",
    "T[T>A]G",
    "T[T>A]T",
    "A[T>C]A",
    "A[T>C]C",
    "A[T>C]G",
    "A[T>C]T",
    "C[T>C]A",
    "C[T>C]C",
    "C[T>C]G",
    "C[T>C]T",
    "G[T>C]A",
    "G[T>C]C",
    "G[T>C]G",
    "G[T>C]T",
    "T[T>C]A",
    "T[T>C]C",
    "T[T>C]G",
    "T[T>C]T",
    "A[T>G]A",
    "A[T>G]C",
    "A[T>G]G",
    "A[T>G]T",
    "C[T>G]A",
    "C[T>G]C",
    "C[T>G]G",
    "C[T>G]T",
    "G[T>G]A",
    "G[T>G]C",
    "G[T>G]G",
    "G[T>G]T",
    "T[T>G]A",
    "T[T>G]C",
    "T[T>G]G",
    "T[T>G]T",
]


def alphabet_list(amount: int, genome: str) -> list[str]:
    """
    Generate a list of column labels in the format [genome + letter(s)].

    Args:
        amount (int): Number of labels to generate.
        genome (str): Prefix for each label.

    Returns:
        list[str]: List of column labels.
    """
    columns = list(
        itertools.chain(
            string.ascii_uppercase,
            (
                "".join(pair)
                for pair in itertools.product(string.ascii_uppercase, repeat=2)
            ),
        )
    )
    return [genome + columns[i] for i in range(amount)]


def create_signatures_df(W: np.ndarray, signatures: int) -> pd.DataFrame:
    """
    Create a dataframe of the result of the NMF.

    Args:
        W (np.ndarray): Np array of the NMF result.
        signatures (int): Number of signatures.

    Returns:
        pd.DataFrame: Dataframe with the result.
    """
    signatures = pd.DataFrame(
        W, columns=alphabet_list(amount=signatures, genome=f"SBS{W.shape[0]}")
    )
    return signatures


def compress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress the DataFrame by summing every 16 rows for each column.
    from {A,C,T,G}{A,C,T,G}[{A,C,T,G}>{A,C,T,G}]{A,C,T,G}{A,C,T,G}
    to sum x{A,C,T,G}[{A,C,T,G}>{A,C,T,G}]{A,C,T,G}y

    Args:
        df: pd.DataFrame to be compressed.

    Returns:
        pd.DataFrame: Compressed DataFrame.
    """
    col = "MutationType"
    df["sort_key"] = df[col].str.extract(r"(\w\[.*\]\w)")
    df = df.sort_values("sort_key")
    df_keys = df["sort_key"].copy()
    df = df.drop(["sort_key", col], axis=1)
    steps = int(df.shape[0] / 96)

    compressed_df = pd.DataFrame()
    compressed_df[col] = df_keys[::steps]
    for col in df.columns:
        chunks = [df[col][i : i + steps] for i in range(0, len(df[col]), steps)]
        chunk_sums = [chunk.sum() for chunk in chunks]
        compressed_df[col] = chunk_sums
    return compressed_df.set_index("MutationType").reindex(MUTATION_LIST).reset_index()


def combinations() -> list[tuple[str, str]]:
    """
    Generate combinations of initialization and beta loss.

    Returns:
        list[tuple[str, str]]: List of tuples representing combinations.
    """
    inits = ["None", "random", "nndsvd", "nndsvda", "nndsvdar"]
    beta_losses = ["frobenius", "kullback-leibler", "itakura-saito"]
    return list(itertools.product(inits, beta_losses))


def read_file_decompose(file: str, dataframe: pd.DataFrame) -> None:
    """
    Read the contents of a file and extract signature data.

    Args:
        file (Path): Path to the file to read.
        df (pd.DataFrame): DataFrame to store the signature data.
    """
    sigs = []
    with open(file, "r", encoding="UTF-8") as open_file:
        final_composition = False
        for line in open_file:
            if line.startswith("#################### Final Composition"):
                final_composition = True
            elif final_composition:
                sigs.append(eval(line.strip()))
                final_composition = False
    dataframe[file.parts[1]] = sigs


def read_file_mut_prob(folder: str) -> pd.DataFrame:
    """
    Will get for every sample the normalized (between 0 and 1)
    the chance for a mutation type in a SBS

    Args:
        folder (str): The path of the folder.

    Returns:
        pd.DataFrame: The df with the chances.
    """
    file_path = (
        Path("result-nmf")
        / "Assignment_Solution"
        / "Activities"
        / "Decomposed_MutationType_Probabilities.txt"
    )
    full_path = str(Path(folder).joinpath(file_path))
    df = pd.read_csv(full_path, sep="\t")
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df = df.groupby("MutationType").sum().reset_index()

    result = pd.DataFrame()
    columns = df.columns[1:]

    for index in range(len(df)):
        row = {"MutationType": df.iloc[index, 0]}
        averages = df.iloc[index, 1:].to_numpy() / df.iloc[index, 1:].to_numpy().sum()
        for sbs, avg in list(zip(columns, averages)):
            row[sbs] = avg
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result
