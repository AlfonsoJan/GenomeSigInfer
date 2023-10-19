#!/usr/bin/env python3
""""
CSV Reader

This class provides a CSVReader class for formatting CSV files
based on the presence of specific columns.

Functions:
- get_matrix_mut_sig(file_str: str, columns: int = None) -> pd.DataFrame: Format a CSV file based on the presence of specific columns.
"""
import pandas as pd


class CSVReader:
    """
    CSVReader class for returning mutational signatures
    in the correct matrix with the correct columns
    """

    columns_to_check = ["Trinucleotide", "Pentanucleotide"]

    @staticmethod
    def merge_columns(mutation, nucleotide_row) -> str:
        """
        Merge two columns with square brackets around the middle of col2.

        Args:
            mutation (str): The first column (Mutation type)
            nucleotide_row (str): The second column (One of the columns that we are checking for)

        Returns:
            str: The merged column value.
        """
        index = int(len(nucleotide_row) / 2)
        return f"{nucleotide_row[:index]}[{mutation}]{nucleotide_row[index + 1:]}"


def get_matrix_mut_sig(file_str: str, columns: int = None, sep: str = ",") -> pd.DataFrame:
    """
    Format a CSV file based on the presence of specific columns.

    Args:
        file_str (str): The path to the input CSV file.
        columns (int, optional): Number of columns to limit in the output CSV (default: None).
        sep (str, optional): Seperator of the csv file.

    Returns:
        pd.Dataframe: The matrix with the correct columns
    """
    found_column = None
    dataframe = pd.read_csv(file_str, sep=sep, header=0)
    for column in CSVReader.columns_to_check:
        if column in dataframe.columns:
            found_column = column
            break
    if found_column:
        dataframe["Mutation type"] = dataframe.apply(
            lambda row: CSVReader.merge_columns(
                row["Mutation type"], row[found_column]
            ),
            axis=1,
        )
        dataframe.drop(dataframe.columns[1], axis=1, inplace=True)
    n_columns = dataframe.shape[1] if not columns else columns
    return dataframe.iloc[:, :n_columns]

def get_matrix_nmf_w(file_str: str) -> pd.DataFrame:
    """
    Format a CSV file based on the presence of specific columns.

    Args:
        file_str (str): The path to the input CSV file.

    Returns:
        pd.Dataframe: The matrix with the correct columns
    """
    dataframe = pd.read_csv(file_str, sep="\t", header=0)
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    return dataframe
