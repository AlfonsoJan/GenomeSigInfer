#!/usr/bin/env python3
"""
This module provides functions for generating and visualizing mutation data
related to larger context and 96 context mutations. It includes methods to
create bar plots for different mutation contexts and save them in PDF files.
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .helpers import (
    generate_numbers,
    generate_sequence,
    custom_sort_column_names,
    MUTATION_LIST,
    COLOR_DICT,
    COLOR_DICT_MUTATION,
    MUTATION_TYPES
)


def larger_context_barplot(df_MNPRS: pd.DataFrame, folder_path: Path) -> None:
    """
    Generate bar plots for larger context mutations and save them in a PDF file.

    Parameters:
    - df_MNPRS (pd.DataFrame): DataFrame containing mutation data.
    - folder_path (Path): Path to the folder where the PDF file will be saved.
    """
    df_MNPRS["context"] = df_MNPRS["MutationType"].str.extract(r"(\w\[.*\]\w)")
    sorted_columns = sorted(df_MNPRS.columns[1:-1], key=custom_sort_column_names)
    with PdfPages(folder_path / f"signatures.{df_MNPRS.shape[0]}.pdf") as pdf:
        for col in tqdm(sorted_columns):
            result_df = parse_lager_context_df(df_MNPRS, col)
            create_increased_context_barplot(result_df, pdf)


def create_96_barplot(df: pd.DataFrame, figure_folder: Path) -> None:
    """
    Generate bar plots for 96 context mutations and save them in a PDF file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mutation data.
    - figure_folder (Path): Path to the folder where the PDF file will be saved.
    """
    sorted_columns = sorted(df.columns[1:], key=custom_sort_column_names)
    with PdfPages(figure_folder / f"signatures.96.pdf") as pdf:
        for col in tqdm(sorted_columns):
            parse_96_df(df, col, pdf)


def create_increased_context_barplot(df: pd.DataFrame, pdf: PdfPages) -> None:
    """
    Create a bar plot for increased context mutations and save it in a PDF file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mutation data.
    - pdf: PdfPages object for saving the plot.
    """
    title = df["sbs"].unique()[0]
    labels = df["context"].drop_duplicates()
    x0 = np.arange(len(labels))
    data = {
        f"{name}{variable}": group
        for (name, variable), group in df.groupby(["name", "variable"])
    }
    names = df.name.unique()
    stacks = len(names)
    variable = df.variable.unique()
    x1 = []
    w = 0.45
    if stacks == 2:
        x1 = [x0 - w / stacks, x0 + w / stacks]
    elif stacks == 4:
        w = 0.2
        x1 = [
            x0 - w * 4 / stacks - w / 2,
            x0 - w * 2 / stacks,
            x0 + w * 2 / stacks,
            x0 + w * 4 / stacks + w / 2,
        ]
    _, ax = plt.subplots(figsize=(20, 10))
    added_labels = set()
    for x, name in zip(x1, names):
        bottom = 0
        for var in variable:
            height = data[f"{name}{var}"].value.to_numpy()
            color = COLOR_DICT[var]
            if var not in added_labels:
                ax.bar(
                    x=x, height=height, width=w, bottom=bottom, color=color, label=var
                )
                added_labels.add(var)
            else:
                ax.bar(x=x, height=height, width=w, bottom=bottom, color=color)
            bottom += height
    sublist_length = len(x0) // 6
    indices = [x0[i : i + sublist_length] for i in range(0, len(x0), sublist_length)]
    mutation_types = df["context"].str.extract(r"\[(.*?)\]")[0].unique()
    for i, (mutation_type, indices_list) in enumerate(zip(mutation_types, indices)):
        idx = indices_list[-1]
        text_x = indices_list[len(indices_list) // 2]
        plt.text(
            text_x,
            1.05,
            mutation_type,
            rotation=0,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        if i < len(mutation_types) - 1:
            plt.axvline(x=idx + 0.5, color="black", linestyle="-", linewidth=2)
    ax.set_xticks(x0)
    ax.set_xticklabels(labels, rotation=90, ha="right")
    ax.legend()
    ax.set_xlabel("Context")
    ax.set_ylabel("Value")
    plt.xlim(-1, len(labels))
    plt.ylim(0, 1)
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(bbox_inches="tight")
    plt.close()


def parse_lager_context_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse the DataFrame to extract larger context mutation data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mutation data.
    - col (str): Name of the column to parse.

    Returns:
    - pd.DataFrame: Resulting DataFrame with parsed data.
    """
    result_df = pd.DataFrame()
    AMINO = ["A", "C", "T", "G"]
    col_df = pd.DataFrame(
        {
            "MutationType": df["MutationType"],
            col: df[col],
            "context": df["context"],
        }
    )
    for mut in MUTATION_LIST:
        temp_df = col_df[(col_df["context"] == mut)]
        total_sbs_mut = temp_df[col].sum()
        context_index_list = generate_numbers(len(df["MutationType"][0]) - 4)
        seq = generate_sequence(len(df["MutationType"][0]) - 4)
        contex_name_generator = zip(context_index_list, seq)
        for idx, name in contex_name_generator:
            for aa in AMINO:
                row = {"name": name, "context": mut, "variable": aa, "sbs": col}
                if total_sbs_mut == 0:
                    row["value"] = 0
                else:
                    filtered_df = temp_df[temp_df["MutationType"].str[idx] == aa]
                    total_filtered = filtered_df[col].sum()
                    row["value"] = total_filtered
                result_df = pd.concat(
                    [result_df, pd.DataFrame(row, index=[0])], ignore_index=True
                )
    return result_df


def parse_96_df(df: pd.DataFrame, col: str, pdf: PdfPages) -> None:
    """
    Parse the DataFrame to extract 96 context mutation data and create a bar plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mutation data.
    - col (str): Name of the column to parse.
    - pdf: PdfPages object for saving the plot.
    """
    sort_col = "mutation"
    mut_col = "MutationType"
    temp_df = pd.DataFrame({mut_col: df.iloc[:, 0], col: df[col]})
    temp_df[sort_col] = temp_df[mut_col].str.extract(r"\[(.*?)\]")
    muts = temp_df.MutationType.unique()
    x0 = np.arange(len(muts))
    _, ax = plt.subplots(figsize=(20, 10))
    w = 0.8
    added_labels = set()
    for x, mutation in zip(x0, temp_df.mutation):
        bottom = 0
        color = COLOR_DICT_MUTATION[mutation]
        value = temp_df.iloc[x, 1]
        if mutation not in added_labels:
            ax.bar(
                x=x, height=value, width=w, bottom=bottom, label=mutation, color=color
            )
            added_labels.add(mutation)
        else:
            ax.bar(x=x, height=value, width=w, bottom=bottom, color=color)
    groups = temp_df.groupby(sort_col).groups.items()
    for i, (mutation_type, indices) in enumerate(groups):
        idx = indices[-1]
        text_x = indices[len(indices) // 2]
        plt.text(
            text_x,
            1.05,
            mutation_type,
            rotation=0,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
         )
        # Skip the last line
        if i < len(groups) - 1:
            plt.axvline(x=idx + 0.5, color="black", linestyle="-", linewidth=2)
            
    ax.set_xticks(x0)
    ax.set_xticklabels(muts, rotation=90, ha="right")
    ax.legend()
    ax.set_xlabel("Context")
    ax.set_ylabel("Value")
    plt.title(col, fontweight="bold")
    plt.xlim(-1, len(temp_df[mut_col]))
    plt.ylim(0, 1)
    plt.tight_layout()
    pdf.savefig(bbox_inches="tight")
    plt.close()
