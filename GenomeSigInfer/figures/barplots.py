#!/usr/bin/env python3
"""
This module provides functions for generating and visualizing mutation data related to larger context and 96 context mutations.
It includes methods to create bar plots for different mutation contexts and save them in PDF files.

Functions:
* format_xlabels(x_labels: list) -> list: Format x labels for the plots.
* formatted_y_labels(x: float, _: int) -> str: Format numeric values for the y labels.
* signature_pdf_plot(df: pd.DataFrame, figure_folder: Path) -> None: Generate signature plots based on mutation data.
* context_96_barplot(df: pd.DataFrame, figure_folder: Path) -> None: Generate bar plots for 96 context mutations and save them in a PDF file.
* larger_context_barplot(df_multi_context: pd.DataFrame, folder_path: Path) -> None: Generate bar plots for larger context mutations and save them in a PDF file.
* add_title_to_axe(ax: plt.axes, context: int) -> None: Add title to an axes.
* create_expected_larger(df_dict: pd.DataFrame, expected_larger_sbs: list, folder_path: Path): Create a page with all three plots for every SBS and every DataFrame.
* parse_lager_context_df(df: pd.DataFrame, col: str) -> pd.DataFrame: Parse the DataFrame to extract larger context mutation data.
* parse_96_df(df: pd.DataFrame, col: str) -> pd.DataFrame: Parse the DataFrame to extract 96 context.
* create_barplot(df, col: str, pdf: PdfPages, ax: plt.axes = None, write_sbs_title: bool = True): Create a bar plot based on mutation data and save it to a PDF file.
* add_to_plot(info, ax, df, write_sbs_title): Add elements to a plot based on mutation data.
* add_text_lines_to_plot(info, ax, write_sbs_title): Add text lines to a plot based on mutation data.
* add_context_96_elements(info, ax, df): Add elements to the plot for context 96.
* add_larger_context_elements(info, ax, df): Add elements to the plot for larger context.
* plot_context_bar(df, col): Plot a bar for the given mutation data and column.

Author: J.A. Busker
"""
from collections import namedtuple
from pathlib import Path
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
    get_context_given_size,
    COLOR_BG,
    MUTATION_LIST,
    COLOR_DICT,
    COLOR_DICT_MUTATION,
)
from ..utils.logging import SingletonLogger

# A tuple with all the info of the dataframe
# For the barplot
ContextBarInfo = namedtuple(
    "ContextBarInfo",
    [
        "title",
        "labels",
        "x0",
        "x1",
        "names",
        "variable",
        "w",
        "context",
        "groups",
        "mutations_group_length",
    ],
)


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


def formatted_y_labels(x: float, _: int) -> str:
    """
    Format a numeric value for the y labels.

    Args:
        x (float): The numeric value to be formatted.
        _ (int): The tick position (unused in this function).

    Returns:
        str: The formatted string.
    """
    # Uncomment this for percentage
    # return f"{x:.0%}"
    return f"{x:.2f}"


def signature_pdf_plot(df: pd.DataFrame, figure_folder: Path) -> None:
    """
    Generate signature plots based on mutation data.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        figure_folder (Path): Path to the folder where the PDF file will be saved.
    """
    context = df.shape[0]
    if context == 96:
        context_96_barplot(df, figure_folder)
    else:
        larger_context_barplot(df, figure_folder)


def context_96_barplot(df: pd.DataFrame, figure_folder: Path) -> None:
    """
    Generate bar plots for 96 context mutations and save them in a PDF file.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        figure_folder (Path): Path to the folder where the PDF file will be saved.
    """
    # Sort the columns
    sorted_columns = sorted(df.columns[1:], key=custom_sort_column_names)
    # Use PdfPages for creating a multi-page PDF file
    with PdfPages(figure_folder / "signatures.96.pdf") as pdf:
        # Iterate over each column (mutation type) in sorted order
        for col in sorted_columns:
            # Parse the DataFrame to extract 96 context data for the current column
            df_col = parse_96_df(df, col)
            create_barplot(df_col, col, pdf, write_sbs_title=True)


def larger_context_barplot(df_multi_context: pd.DataFrame, folder_path: Path) -> None:
    """
    Generate bar plots for larger context mutations and save them in a PDF file.

    Args:
        df_multi_context (pd.DataFrame): DataFrame containing mutation data.
        folder_path (Path): Path to the folder where the PDF file will be saved.
    """
    # Ectract the smallest context of the mutation
    # eg: smalles context of AAG[C>A]TGA is G[C>A]T
    df_multi_context["context"] = df_multi_context["MutationType"].str.extract(
        r"(\w\[.*\]\w)"
    )
    # Sort the columns
    # The SBS name for consistency
    sorted_columns = sorted(
        df_multi_context.columns[1:-1], key=custom_sort_column_names
    )
    # Use PdfPages for creating a multi-page PDF file
    with PdfPages(folder_path / f"signatures.{df_multi_context.shape[0]}.pdf") as pdf:
        for col in sorted_columns:
            # Parse the DataFrame to extract larger context data for the current column
            df_col = parse_lager_context_df(df_multi_context, col)
            create_barplot(df_col, col, pdf, write_sbs_title=True)


def add_title_to_axe(ax: plt.axes, context: int) -> None:
    """
    Add text with information about the context.

    Args:
        ax (plt.axes): Matplotlib axes.
        context: Context information.
    """
    # Add a text label to the specified axes with information about the context
    ax.text(
        48,
        0.8,
        f"Context of: {get_context_given_size(context)}",
        rotation=0,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )


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
    expected_larger_sbs = [
        sbs
        for sbs in expected_larger_sbs
        if all([sbs in list(df.columns) for df in df_dict.values()])
    ]
    # If none of the SBS that benefit larger context are in the
    # Raise error
    if len(expected_larger_sbs) == 0:
        raise ValueError("No of the SBS are in all the dataframes")
    for sbs in expected_larger_sbs:
        plot_name = folder_path / f"{sbs}.pdf"
        with PdfPages(plot_name) as pdf:
            logger.log_info(f"Creating Signature plot for {sbs}")
            # Create a grid for subplots using GridSpec
            gs = GridSpec(2, 2, height_ratios=[0.8, 0.8])
            _ = plt.figure(figsize=(30, 20))
            # Create for every context a plot and add it to the page
            for index, size in enumerate(df_dict.keys()):
                # Custom Subplot Creation with Matplotlib
                if index == 0:
                    ax = plt.subplot(gs[1, :])
                else:
                    ax = plt.subplot(gs[0, index - 1])
                add_title_to_axe(ax, size)
                data = df_dict[size]
                if size == 96:
                    df = parse_96_df(data, sbs)
                    create_barplot(df, sbs, pdf, ax=ax, write_sbs_title=False)
                else:
                    # Ectract the smallest context of the mutation
                    # eg: smalles context of AAG[C>A]TGA is G[C>A]T
                    data["context"] = data["MutationType"].str.extract(r"(\w\[.*\]\w)")
                    df = parse_lager_context_df(data, sbs)
                    create_barplot(df, sbs, pdf, ax=ax, write_sbs_title=False)
            # Save the plots to the PDF page
            plt.tight_layout()
            pdf.savefig(bbox_inches="tight")
            plt.close()
            logger.log_info(f"Created Signature plot for {sbs}")


def parse_lager_context_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse the DataFrame to extract larger context mutation data.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to parse.

    Returns:
        pd.DataFrame: Resulting DataFrame with parsed data.
    """
    # Result df
    result_df = pd.DataFrame()
    amino = ["A", "C", "T", "G"]
    col_df = pd.DataFrame(
        {
            "MutationType": df["MutationType"],
            col: df[col],
            "context": df["context"],
        }
    )
    # For every uniqe mutation
    for mut in MUTATION_LIST:
        # Create temp df
        temp_df = col_df[(col_df["context"] == mut)]
        total_sbs_mut = temp_df[col].sum()
        # These are the indices of the extra context
        # eg: CA[C>A]AG return [0,8] those are the indices of C and G
        context_index_list = generate_numbers(len(df["MutationType"][0]) - 4)
        # These are the position for the bars
        # eg: CA[C>A]AG return -1, 1
        # So that C is ploitted before G
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
                # Create a df of the values for every extra context
                # eg: CA[C>A]AG
                # eg: C: 0.05
                # eg: G: 0.1
                result_df = pd.concat(
                    [result_df, pd.DataFrame(row, index=[0])], ignore_index=True
                )
    return result_df


def parse_96_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parse the DataFrame to extract 96 context.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to parse.

    Returns:
        pd.DataFrame: Resulting DataFrame with parsed 96 context data.
    """
    mut_col = "MutationType"
    temp_df = pd.DataFrame({mut_col: df.iloc[:, 0], col: df[col]})
    temp_df["mutation"] = temp_df[mut_col].str.extract(r"\[(.*?)\]")
    return temp_df


def create_barplot(
    df, col: str, pdf: PdfPages, ax: plt.axes = None, write_sbs_title: bool = True
) -> None:
    """
    Create a bar plot based on mutation data and save it to a PDF file.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to plot.
        pdf (PdfPages): PDF file to save the plot.
        ax (plt.axes, optional): Matplotlib axes. Defaults to None.
        write_sbs_title (bool, optional): Whether to write the SBS title on the plot. Defaults to True.
    """
    # If ax is not provided, create a new subplot
    ax_none = False
    if ax is None:
        ax_none = True
    if ax_none:
        _, ax = plt.subplots(figsize=(20, 10))
    # Plot the context bar using helper functions
    info = plot_context_bar(df, col)
    add_to_plot(info, ax, df, write_sbs_title)
    # If ax was created inside this function, save the plot to the PDF
    if ax_none:
        plt.tight_layout()
        pdf.savefig(bbox_inches="tight")
        plt.close()


def add_to_plot(
    info: ContextBarInfo, ax: plt.axes, df: pd.DataFrame, write_sbs_title: bool
) -> None:
    """
    Add elements to a plot based on mutation data.

    Args:
        info (ContextBarInfo): Information about the context.
        ax (plt.axes): Matplotlib axes.
        df (pd.DataFrame): DataFrame containing mutation data.
        write_sbs_title (bool): Whether to write the SBS title on the plot.
    """
    # Add vertical lines between each mutation type
    # And add the mutation type at the top
    add_text_lines_to_plot(info, ax, write_sbs_title)
    if info.context == 96:
        add_context_96_elements(info, ax, df)
    else:
        add_larger_context_elements(info, ax, df)
    # Formatted x labels
    x_labels_ticks = format_xlabels(info.labels)
    ax.set_xticks(info.x0)
    ax.set_xticklabels(x_labels_ticks, rotation=90, ha="center", fontfamily="monospace")
    ax.legend(loc="upper right")
    ax.set_xlabel("Context", weight="bold")
    ax.set_ylabel("Percentage OF Single Base Substitution", weight="bold")
    plt.xlim(-1, len(info.labels))
    plt.ylim(0, 1)
    # Format the ylabels to percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(formatted_y_labels))


def add_text_lines_to_plot(
    info: ContextBarInfo, ax: plt.axes, write_sbs_title: bool
) -> None:
    """
    Add text lines to the plot based on mutation data.

    Args:
        info (ContextBarInfo): Information about the context.
        ax (plt.axes): Matplotlib axes.
        write_sbs_title (bool): Whether to write the SBS title on the plot.

    Returns:
        None
    """
    # Iterate over mutation types and add text lines to the plot
    for index, (mutation_type, indices_list) in enumerate(info.groups):
        # Calculate the text position in the middle of the indices list
        text_x = indices_list[len(indices_list) // 2]
        # Add mutation type as text at the top of the plot
        plt.text(
            text_x,
            0.95,
            mutation_type,
            rotation=0,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        # Add a colored background for the mutation type
        bg_color_x_min = indices_list[0] - 1 if index == 0 else indices_list[0] - 0.5
        bg_color_x_max = (
            indices_list[-1] + 1
            if index == (info.mutations_group_length - 1)
            else indices_list[-1] + 0.5
        )
        ax.axvspan(
            bg_color_x_min, bg_color_x_max, facecolor=COLOR_BG[index], alpha=0.25
        )
        # UNCOMMENT THIS
        # Adds a line between each group
        # idx = indices_list[-1]
        # if index < info.mutations_group_length - 1:
        #     plt.axvline(x=idx + 0.5, color="black", linestyle="-", linewidth=1)
    # If specified, add the SBS title to the top-left corner of the plot
    if write_sbs_title:
        # SBS NAME ON THE PLOT
        plt.text(
            0.00,
            0.99,
            info.title,
            rotation=0,
            ha="left",
            va="top",
            fontsize=24,
            fontweight="heavy",
        )


def add_context_96_elements(info: ContextBarInfo, ax: plt.axes, df: pd.DataFrame):
    """
    Add elements to the plot for context 96.

    Args:
        info (ContextBarInfo): Information about the context.
        ax (plt.axes): Matplotlib axes.
        df (pd.DataFrame): DataFrame containing mutation data.
    """
    # Set to track added labels
    added_labels = set()
    for x, mutation in zip(info.x1, info.names):
        bottom = 0
        color = COLOR_DICT_MUTATION[mutation]
        value = df.iloc[x, 1]
        # Otherwise, you get a very large legend with duplicate labels
        if mutation not in added_labels:
            # Add a bar for the mutation type with appropriate color and label
            ax.bar(
                x=x,
                height=value,
                width=info.w,
                bottom=bottom,
                label=mutation,
                color=color,
            )
            # Add the mutation type label to the set of added labels
            added_labels.add(mutation)
        else:
            # If the label is already added, add a bar without a label
            ax.bar(x=x, height=value, width=info.w, bottom=bottom, color=color)


def add_larger_context_elements(
    info: ContextBarInfo, ax: plt.axes, df: pd.DataFrame
) -> None:
    """
    Add elements to the plot for larger context.

    Args:
        info (ContextBarInfo): Information about the context.
        ax (plt.axes): Matplotlib axes.
        df (pd.DataFrame): DataFrame containing mutation data.
    """
    # Set to track added labels
    added_labels = set()
    # Create a dictionary to organize the data by mutation type and variable
    data = {
        f"{name}{variable}": group
        for (name, variable), group in df.groupby(["name", "variable"])
    }
    # Iterate over positions (x) and names
    for x, name in zip(info.x1, info.names):
        bottom = 0
        # Iterate over nucleotides (variables)
        for nucleotide in info.variable:
            height = data[f"{name}{nucleotide}"].value.to_numpy()
            color = COLOR_DICT[nucleotide]
            # Otherwise, you get a very large legend with duplicate labels
            if nucleotide not in added_labels:
                # Add a bar for the nucleotide with appropriate color and label
                ax.bar(
                    x=x,
                    height=height,
                    width=info.w,
                    bottom=bottom,
                    color=color,
                    label=nucleotide,
                )
                # Add the nucleotide label to the set of added labels
                added_labels.add(nucleotide)
            else:
                # If the label is already added, add a bar without a label
                ax.bar(x=x, height=height, width=info.w, bottom=bottom, color=color)
            bottom += height


def plot_context_bar(df: pd.DataFrame, col: str) -> None:
    """
    Plot a bar for the given mutation data and column.

    Args:
        df (pd.DataFrame): DataFrame containing mutation data.
        col (str): Name of the column to plot.

    Returns:
        ContextBarInfo: Information about the context.
    """
    if df.shape[0] == 96:
        # For 96 context, each mutation type has its own bar
        # Get the title and the labels
        title = col
        labels = df.MutationType.unique()
        x0 = np.arange(len(labels))
        names = df.mutation
        w = 0.8
        x1 = x0.copy()
        variable = None
        groups = df.groupby("mutation").groups.items()
        mutations_group_length = len(groups)
    else:
        # For larger context, mutations are grouped and represented in stacked bars
        # Get the title and the labels
        title = df["sbs"].unique()[0]
        labels = df["context"].drop_duplicates()
        x0 = np.arange(len(labels))
        names = df.name.unique()
        stacks = len(names)
        w = 0.45
        # These are for the position for the plots
        # The indices of every small bar for the extra context
        if stacks == 2:
            # For two mutations in a larger context, create two side-by-side bars
            x1 = [x0 - w / stacks, x0 + w / stacks]
        elif stacks == 4:
            # For four mutations, create four bars forming a stacked bar
            w = 0.2
            x1 = [
                x0 - w * 4 / stacks - w / 2,
                x0 - w * 2 / stacks,
                x0 + w * 2 / stacks,
                x0 + w * 4 / stacks + w / 2,
            ]
        variable = df.variable.unique()
        # Split indices into sublists for each mutation type in the larger context
        sublist_length = len(x0) // 6
        indices = [
            x0[i : i + sublist_length] for i in range(0, len(x0), sublist_length)
        ]
        # Extract unique mutation types from the larger context
        mutation_types = df["context"].str.extract(r"\[(.*?)\]")[0].unique()
        groups = zip(mutation_types, indices)
        mutations_group_length = len(mutation_types)
    # Return the information about the context
    return ContextBarInfo(
        title,
        labels,
        x0,
        x1,
        names,
        variable,
        w,
        df.shape[0],
        groups,
        mutations_group_length,
    )
