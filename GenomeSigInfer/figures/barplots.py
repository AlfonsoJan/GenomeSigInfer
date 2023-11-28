#!/usr/bin/env python3
"""
This module provides functions for generating and visualizing mutation data
related to larger context and 96 context mutations. It includes methods to
create bar plots for different mutation contexts and save them in PDF files.

Functions:
    - larger_context_barplot(df_multi_contexct: pd.DataFrame, folder_path: Path) -> None:
        Generate barplots for larger context mutations and save them in a PDF file.
    - create_96_barplot(df: pd.DataFrame, figure_folder: Path) -> None:
        Generate bar plots for 96 context mutations and save them in a PDF file.
    - create_increased_context_barplot(df: pd.DataFrame, pdf: PdfPages) -> None:
        Create a bar plot for increased context mutations and save it in a PDF file.
    - parse_lager_context_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        Parse the DataFrame to extract larger context mutation data.
    - parse_96_df(df: pd.DataFrame, col: str, pdf: PdfPages) -> None:
        Parse the DataFrame to extract 96 context mutation data and create a bar plot.
"""
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from ..utils.helpers import (
    generate_numbers,
    generate_sequence,
    custom_sort_column_names,
    MUTATION_LIST,
    COLOR_DICT,
    COLOR_DICT_MUTATION,
)
from ..utils.logging import SingletonLogger


def format_xlabels(x_labels: list) -> list:
    """
    Format the x labels for the plots.
    To go from A[C>A]A to ACA

    Args:
        x_labels (list): The mutations to be formatted.

    Returns:
        list: The formatted x labels.
    """
    return [f"{label[0]}{label[2]}{label[-1]}" for label in x_labels]


def percentage_formatter(x: float, pos: int) -> str:
    """
    Format a numeric value as a percentage.

    Args:
        x (float): The numeric value to be formatted.
        pos (int): The tick position (unused in this function).

    Returns:
        str: The formatted percentage string.
    """
    return f"{x:.0%}"


def create_expected_larger(
    df_dict: pd.DataFrame, expected_larger_sbs: list, folder_path: Path
):
    """
    Create a page with all three plots (parse_96_df, create_increased_context_barplot)
    for every SBS in expected_larger_sbs for every DataFrame in df_dict.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing DataFrame for each SBS.
        expected_larger_sbs (List[str]): List of SBS names to create pages for.
        folder_path (Path): Path to the folder where the PDF files will be saved.
    """
    logger = SingletonLogger()
    # Check if all dataframes contains the columns
    # Return a list of the SBS columns that are in all
    expected_larger_sbs = [sbs for sbs in expected_larger_sbs if all([sbs in list(df.columns) for df in df_dict.values()])]
    # Message
    if len(expected_larger_sbs) == 0:
        return
    plot_name = folder_path / "signatures.expected_larger.pdf"
    num_subplots = len(df_dict)
    with PdfPages(plot_name) as pdf:
        for sbs in expected_larger_sbs:
            logger.log_info(f"Creating Signature plot for {sbs}")
            # Create a num_subplotsx1 grid for subplots using GridSpec
            gs = GridSpec(num_subplots, 1, height_ratios=[0.8] * num_subplots)
            _ = plt.figure(figsize=(20, 10 * num_subplots))
            for index, size in enumerate(df_dict.keys()):
                ax = plt.subplot(gs[index])
                data = df_dict[size]
                if size == 96:
                    parse_96_df(data, sbs, pdf, ax)
                else:
                    data["context"] = data["MutationType"].str.extract(r"(\w\[.*\]\w)")
                    result_df = parse_lager_context_df(data, sbs)
                    create_increased_context_barplot(result_df, pdf, ax)
            plt.tight_layout()
            pdf.savefig(bbox_inches="tight")
            plt.close()
            logger.log_info(f"Created Signature plot for {sbs}")


def larger_context_barplot(df_multi_context: pd.DataFrame, folder_path: Path) -> None:
    """
    Generate bar plots for larger context mutations and save them in a PDF file.

    Args:
        df_multi_context (pd.DataFrame): DataFrame containing mutation data.
        folder_path (Path): Path to the folder where the PDF file will be saved.
    """
    df_multi_context["context"] = df_multi_context["MutationType"].str.extract(
        r"(\w\[.*\]\w)"
    )
    sorted_columns = sorted(
        df_multi_context.columns[1:-1], key=custom_sort_column_names
    )
    with PdfPages(folder_path / f"signatures.{df_multi_context.shape[0]}.pdf") as pdf:
        for col in tqdm(sorted_columns):
            result_df = parse_lager_context_df(df_multi_context, col)
            create_increased_context_barplot(result_df, pdf)


def create_96_barplot(df: pd.DataFrame, figure_folder: Path) -> None:
    """
    Generate bar plots for 96 context mutations and save them in a PDF file.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        figure_folder (Path): Path to the folder where the PDF file will be saved.
    """
    sorted_columns = sorted(df.columns[1:], key=custom_sort_column_names)
    with PdfPages(figure_folder / "signatures.96.pdf") as pdf:
        for col in tqdm(sorted_columns):
            parse_96_df(df, col, pdf)


def parse_lager_context_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse the DataFrame to extract larger context mutation data.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to parse.

    Returns:
        pd.DataFrame: Resulting DataFrame with parsed data.
    """
    result_df = pd.DataFrame()
    amino = ["A", "C", "T", "G"]
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
            for aa in amino:
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


def create_increased_context_barplot(
    df: pd.DataFrame, pdf: PdfPages, ax: plt.axes = None
) -> None:
    """
    Create a bar plot for increased context mutations and save it in a PDF file.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        pdf: PdfPages object for saving the plot.
    """
    ax_none = False
    if ax is None:
        ax_none = True
    if ax_none:
        _, ax = plt.subplots(figsize=(20, 10))
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
    # SBS NAME ON THE PLOT
    plt.text(
        -0.05,
        0.99,
        title,
        rotation=0,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="heavy",
    )
    # Formatted x labels
    x_labels_ticks = format_xlabels(labels)
    ax.set_xticks(x0)
    ax.set_xticklabels(x_labels_ticks, rotation=90, ha="center", fontfamily="monospace")
    ax.legend()
    ax.set_xlabel("Context", weight="bold")
    ax.set_ylabel("Percentage OF Single Base Substitution", weight="bold")
    plt.xlim(-1, len(labels))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.ylim(0, 1)
    if ax_none:
        plt.tight_layout()
        pdf.savefig(bbox_inches="tight")
        plt.close()


def parse_96_df(df: pd.DataFrame, col: str, pdf: PdfPages, ax: plt.axes = None) -> None:
    """
    Parse the DataFrame to extract 96 context mutation data and create a bar plot.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to parse.
        pdf: PdfPages object for saving the plot.
    """
    ax_none = False
    if ax is None:
        ax_none = True
    if ax_none:
        _, ax = plt.subplots(figsize=(20, 10))
    sort_col = "mutation"
    mut_col = "MutationType"
    temp_df = pd.DataFrame({mut_col: df.iloc[:, 0], col: df[col]})
    temp_df[sort_col] = temp_df[mut_col].str.extract(r"\[(.*?)\]")
    muts = temp_df.MutationType.unique()
    x0 = np.arange(len(muts))
    # plt.rcParams['font.family'] = 'monospace'
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
    # SBS NAME ON THE PLOT
    plt.text(
        -0.05,
        0.99,
        col,
        rotation=0,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="heavy",
    )
    # Formatted x labels
    x_labels_ticks = format_xlabels(muts)
    ax.set_xticks(x0)
    ax.set_xticklabels(x_labels_ticks, rotation=90, ha="center", fontfamily="monospace")
    ax.legend()
    ax.set_xlabel("Context", weight="bold")
    ax.set_ylabel("Percentage OF Single Base Substitution", weight="bold")
    plt.xlim(-1, len(temp_df[mut_col]))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.ylim(0, 1)
    if ax_none:
        plt.tight_layout()
        pdf.savefig(bbox_inches="tight")
        plt.close()
