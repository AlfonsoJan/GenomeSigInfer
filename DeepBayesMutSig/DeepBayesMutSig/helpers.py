#!/usr/bin/env python3
"""
This module contains small, common utility functions used across the project.

Functions:
- alphabet_list(amount: int, genome: str) -> list[str]: Generate a list of column labels.
- create_signatures_df(W: np.ndarray, signatures: int) -> pd.DataFrame: Create a dataframe of the result of the NMF.
"""
import itertools
import string
import pandas as pd
import numpy as np

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


def compress(df: pd.DataFrame, mutation: list) -> pd.DataFrame:
    """
    Compress the DataFrame by summing every 16 rows for each column.
    from {A,C,T,G}{A,C,T,G}[{A,C,T,G}>{A,C,T,G}]{A,C,T,G}{A,C,T,G}
    to sum x{A,C,T,G}[{A,C,T,G}>{A,C,T,G}]{A,C,T,G}y

    Args:
        df: pd.DataFrame to be compressed.
        mutation: List of the mutations.

    Returns:
        pd.DataFrame: Compressed DataFrame.
    """
    col = "MutationType"
    df[col] = mutation
    df["sort_key"] = df[col].str.extract(r"(\w\[.*\]\w)")
    df = df.sort_values("sort_key")
    df_keys = df["sort_key"].copy()
    df = df.drop(["sort_key", "MutationType"], axis=1)
    results_1536 = pd.DataFrame()
    results_1536[col] = df_keys[::16]
    for col in df.columns:
        chunks = [df[col][i : i + 16] for i in range(0, len(df[col]), 16)]
        chunk_sums = [chunk.sum() for chunk in chunks]
        results_1536[col] = chunk_sums
    return results_1536.set_index("MutationType").reindex(MUTATION_LIST).reset_index()
