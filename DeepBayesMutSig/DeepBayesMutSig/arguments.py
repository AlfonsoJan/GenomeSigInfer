#!/usr/bin/env python3
"""
This module provides a utility class for parsing command-line arguments and checking file existence.
And functions for different files.

Functions:
- arguments_nmf() -> argparse.Namespace: Parse command-line arguments for when a file,
    n signatures and iterations is needed.
- arguments_profiler() -> argparse.Namespace: Parse command-line arguments for when you
    want to run SigProfiler or on NMF.
"""
import os.path
import argparse
from pathlib import Path


class Parser:
    """
    A utility class for parsing command-line arguments and checking file existence
    or if a number is between the bounds.
    """

    @staticmethod
    def check_if_folder_exist(folder_path: str) -> str:
        """
        Check if a folder exists.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            str: The input folder path if it exists.

        Raises:
            argparse.ArgumentTypeError: If the folder does not exist.
        """
        if Path(folder_path).is_dir():
            return folder_path
        msg = f"{folder_path} does not exist!"
        raise argparse.ArgumentTypeError(msg)

    @staticmethod
    def check_if_in_border(number: str) -> int:
        """
        Check if the number is 1 >= number >= 100.

        Args:
            number (str): The number

        Returns:
            int: The number.

        Raises:
            argparse.ArgumentTypeError: If the number does not meet the conditions.
        """
        number = abs(int(number))
        max_number = 100
        if not 1 <= number <= max_number:
            msg = f"The number needs to be 1 >= number >= {max_number}"
            raise argparse.ArgumentTypeError(msg)
        return number

    @staticmethod
    def check_if_file_exist(file_str: str) -> str:
        """
        Check if a file exists.

        Args:
            file_str (str): The path to the file to be checked.

        Returns:
            str: The path to the file if it exists.

        Raises:
            argparse.ArgumentTypeError: If the file does not exist.
        """
        if not os.path.isfile(file_str):
            msg = f"{file_str} does not exist!"
            raise argparse.ArgumentTypeError(msg)
        return file_str


def arguments_nmf() -> argparse.Namespace:
    """
    Parse command-line arguments for when a file, n signatures and iterations is needed.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--out", default="results", help="Result folder")
    parser.add_argument("-i", "--iter", default=1, type=Parser.check_if_in_border)
    parser.add_argument("-s", "--signatures", default=1, type=Parser.check_if_in_border)
    parser.add_argument("-f", "--file", required=True, type=Parser.check_if_file_exist)
    return parser.parse_args()

def arguments_profiler() -> argparse.Namespace:
    """
    Parse command-line arguments for when you want to run SigProfiler or on NMF.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--spe", action="store_true", help="Run with SigProfilerExtractor"
    )
    group.add_argument("--nmf", action="store_true", help="Run with scikit NMF")
    parser.add_argument("-o", "--out", default="results", help="Result folder")
    parser.add_argument("-s", "--signatures", default=1, type=Parser.check_if_in_border)
    parser.add_argument("-f", "--file", required=True, type=Parser.check_if_file_exist)
    parser.add_argument("--genomes", required=True, type=Parser.check_if_file_exist)
    return parser.parse_args()
