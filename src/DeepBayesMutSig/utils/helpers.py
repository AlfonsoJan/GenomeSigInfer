#!/usr/bin/env python3
"""
This module contains small, common utility functions and common variables used across the project.

Functions:
    - alphabet_list(amount: int, genome: str) -> list[str]: Generate a list of column labels.
    - read_file_decompose(file: str, dataframe: pd.DataFrame) -> None: Read the contents of a file
        and extract signature data.
    - combinations() -> list[tuple[str, str]]: Generate combinations of
        initialization and beta loss.
    - get_96_matrix(filename: Path) -> pd.DataFrame: Get the 96-context matrix for comparison.
    - prepare_folder(folder: Path) -> None: Prepare folder by cleaning it.
    - calculate_value(number): Calculate the value based on an iterative formula.
        For SBS context/size matrices.
"""
import re
import shutil
import itertools
import string
from pathlib import Path
import pandas as pd
import numpy as np

# Color dict for the mutations
COLOR_DICT_MUTATION = {
    "C>A": "green",
    "C>G": "orange",
    "C>T": "blue",
    "T>A": "red",
    "T>C": "purple",
    "T>G": "brown",
}

# Color for the multiple context bars
COLOR_DICT = {
    "A": "green",
    "C": "blue",
    "T": "orange",
    "G": "red",
}

# DICT for decoding ref genomes files
TSB_REF = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
    4: "A",
    5: "C",
    6: "G",
    7: "T",
    8: "A",
    9: "C",
    10: "G",
    11: "T",
    12: "A",
    13: "C",
    14: "G",
    15: "T",
    16: "N",
    17: "N",
    18: "N",
    19: "N",
}

# MUTATION TYPES random order
MUTATION_TYPES = np.array(["C>G", "C>A", "C>T", "T>G", "T>C", "T>A"])

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

def prepare_folder(folder: Path) -> None:
    """
    Prepare folder by cleaning it.

    Args:
        folder (Path): Folder that needs to be prepared.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

def custom_sort_column_names(column_name: str) -> tuple:
    """
    Custom sorting function for column names.

    Parameters:
    - column_name (str): The column name to be sorted.

    Returns:
    - tuple: A tuple used for sorting, containing (numeric_part, prefix, suffix).
    """
    match = re.match(r"(\D+)(\d+)(\D*)", column_name)
    if match:
        prefix, numeric_part, suffix = match.groups()
        return (int(numeric_part), prefix, suffix)
    return (float("inf"), column_name, "")


def generate_sequence(n: int) -> list[int]:
    """
    Generates a list of the context indices for the barplot.

    Args:
        n (int): The context number

    Returns:
        n (list[int]): The list of the context indices for the barplot.
    """
    if n < 5:
        raise ValueError
    sequence = []
    for i in range(-(n // 2), -1, 1):
        sequence.append(i)
    for i in range(2, n // 2 + 1, 1):
        sequence.append(i)
    return sequence


def generate_numbers(n: int) -> list[int]:
    """
    Generates a list of the context number based of the context number.

    Args:
        n (int): The context number

    Returns:
        n (list[int]): The list for the extra context indices.
    """
    if n == 5:
        return [0, 8]
    if n == 7:
        return [0, 1, 9, 10]
    if n == 9:
        return [0, 1, 2, 10, 11, 12]
    if n == 11:
        return [0, 1, 2, 3, 11, 12, 13, 14]
    raise ValueError


def custom_sort(value: str) -> int | float:
    """
    Custom sorting function for sorting the chromosomes.

    Args:
        value (str): The chromosome value to be sorted.

    Returns:
        int | float: The sorted value. If the value is numeric, it is returned as an integer;
                    otherwise, it is assigned a large float value.
    """
    if value.isdigit():
        return int(value)
    return float("inf")


def calculate_value(number):
    """
    Calculate the value based on an iterative formula.
    For SBS context/size matrices.

    Args:
        number: An uneven integer greater than or equal to 3.

    Returns:
        int: Calculated value.
    """
    if number % 2 == 0 or number < 3:
        raise ValueError("Input must be an uneven integer greater than or equal to 3.")
    if number == 3:
        return 96
    return calculate_value(number - 2) * 16


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
    signatures_df = pd.DataFrame(
        W, columns=alphabet_list(amount=signatures, genome=f"SBS{W.shape[0]}")
    )
    return signatures_df


def combinations() -> list[tuple[str, str]]:
    """
    Generate combinations of initialization and beta loss.

    Returns:
        list[tuple[str, str]]: List of tuples representing combinations.
    """
    inits = ["None", "random", "nndsvd", "nndsvda", "nndsvdar"]
    beta_losses = ["frobenius", "kullback-leibler", "itakura-saito"]
    return list(itertools.product(inits, beta_losses))


def get_96_matrix(filename: Path) -> pd.DataFrame:
    """
    Get the 96-context matrix for comparison.

    Returns:
        pd.DataFrame: 96-context matrix.
    """
    if not filename.is_file():
        raise FileNotFoundError(f"{filename} does not exist")
    df = pd.read_csv(filename, sep=",", header=0)
    return df.set_index("MutationType").reindex(MUTATION_LIST).reset_index()


def read_file_decompose(
    file: Path, dataframe: pd.DataFrame, col: str | None = None
) -> None:
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
    if col is None:
        dataframe[file.parts[-5]] = sigs
    else:
        dataframe[col] = sigs

# Class for default stuff
class MutationalSigantures:
    """
    Class for storing parameters for the analysis, used in multiple scripts
    """
    REF_GENOMES: list = ["GRCh37", "GRCh38"]
    MAX_CONTEXT: int = 7
    SORT_REGEX: dict = {
        13: r"(\w\w\w\w\w\w\[.*\]\w\w\w\w\w\w)",
        11: r"(\w\w\w\w\w\[.*\]\w\w\w\w\w)",
        9: r"(\w\w\w\w\[.*\]\w\w\w\w)",
        7: r"(\w\w\w\[.*\]\w\w\w)",
        5: r"(\w\w\[.*\]\w\w)",
        3: r"(\w\[.*\]\w)",
    }
    CONTEXT_LIST: list[int] = list(range(MAX_CONTEXT, 2, -2))
    SIZES: list[int] = [calculate_value(i) for i in CONTEXT_LIST[::-1]]
