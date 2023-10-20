#!/usr/bin/env python3
""""
This module contains functions that calculates the most similar column
based on cosine similarity for a given DataFrame containing signature data.

Functions:
- distribute_value(value: int, num_numbers: int) -> list[int]
- process_column(column: int, shape: int) -> list[int]
- increase_mutations(column: list[str], context: int) -> list[str]
- increase_context(df: pd.DataFrame, context: int = 3) -> pd.DataFrame
"""
import itertools
import pandas as pd


def distribute_value(value: int, num_numbers: int) -> list[int]:
    """
    Distributes a given value into a list of integers based on the number of elements.

    Args:
        value (int): The value to be distributed.
        num_numbers (int): The number of elements in the resulting list.

    Returns:
        list: A list of integers representing the distributed value.
    """
    quotient = value // num_numbers
    remainder = value % num_numbers
    result = [quotient] * num_numbers
    for i in range(remainder):
        result[i] += 1
    return result


def process_column(column: int, shape: int) -> list[int]:
    """
    Processes a column by increasing its values to match a specified shape.

    Args:
        column (int): The column to be processed.
        shape (int): The target shape for the column.

    Returns:
        list: A list of integers representing the processed column.
    """
    increase = int(shape // 96)
    vals = distribute_value(column, increase)
    return vals


def increase_mutations(column: list[str], context: int) -> list[str]:
    """
    Increases mutations in a given column based on a specified context.

    Args:
        column (list): The list of mutations to be increased.
        context (int): The context for increasing mutations.

    Returns:
        list: A list of increased mutations based on the specified context.
    """
    nucleotides = ["A", "C", "G", "T"]
    combinations = list(itertools.product(nucleotides, repeat=context - 3))
    new_mutations = [
        f"{''.join(combo[:len(combo)//2])}{mut}{''.join(combo[len(combo)//2:])}"
        for mut in column
        for combo in combinations
    ]
    return new_mutations


def increase_context(df: pd.DataFrame, context: int = 3) -> pd.DataFrame:
    """
    Increases the context of a DataFrame containing mutation data.

    Args:
        df (DataFrame): The input DataFrame containing mutation data.
        context (int): The desired context for increasing mutations.

    Returns:
        DataFrame: A new DataFrame with increased mutation context.
    """
    if context not in [5, 7, 9, 11]:
        msg = f"Context cannot be: {context}"
        raise AttributeError(msg)
    col = "Mutation type"
    new_df = pd.DataFrame({col: increase_mutations(df[col], context)})
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    for col in df.columns:
        res = df[col].apply(process_column, shape=new_df.shape[0])
        new_df[col] = list(itertools.chain(*res.values.tolist()))
    return new_df
