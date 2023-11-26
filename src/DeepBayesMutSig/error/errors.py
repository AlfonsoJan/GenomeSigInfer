#!/usr/bin/env python3
"""
Custom Exceptions Module

This module defines custom exceptions to handle specific errors in the project.

Classes:
    - RefGenomeNotCorrectError(Exception): Exception raised when ref genome is not supported.
    - NotCorrectProjectError(Exception): Exception raised when the specified project folder
        is not correct.
    - TimeOutError(Exception): Exception raised when the request takes longer
        than a specified duration.
    - SBSNotCorrectError(Exception): Exception raised when the suspected SBS files are not there.

Attributes:
    - RefGenomeNotCorrectError.genome (str): The reference genome.
    - NotCorrectProjectError.path (str): The path of the incorrect project folder.
    - TimeOutError.url (str): The URL path.
    - SBSNotCorrectError.file (str): The file name
"""


class RefGenomeChromosomeNotFound(Exception):
    """
    Exception raised when ref chromosome file is not found.

    Attributes:
        genome (str): The ref genome.
    """

    def __init__(self, chromosome: str) -> None:
        """
        Initialize the exception.

        Args:
            genome (str): The ref genome.
        """
        self.chromosome = chromosome
        super().__init__(f"The chrosome file '{chromosome}' si not found!")


class RefGenomeNotCorrectError(Exception):
    """
    Exception raised when ref genome is not supported.

    Attributes:
        genome (str): The ref genome.
    """

    def __init__(self, genome: str) -> None:
        """
        Initialize the exception.

        Args:
            genome (str): The ref genome.
        """
        self.genome = genome
        super().__init__(f"The ref genome: '{genome}' is not supported.")


class SBSNotCorrectError(Exception):
    """
    Exception raised when the suspected SBS files are not there.

    Attributes:
        file (str): The file path.
    """

    def __init__(self, file: str) -> None:
        """
        Initialize the exception.

        Args:
            file (str): The file path.
        """
        self.file = file
        super().__init__(f"The specified file: '{file}' is not there.")


class NotCorrectProjectError(Exception):
    """
    Exception raised when the specified project folder is not correct.

    Attributes:
        path (str): The path of the incorrect project folder.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the exception.

        Args:
            path (str): The path of the incorrect project folder.
        """
        self.path = path
        super().__init__(f"The specified project folder: '{path}' is not correct.")


class TimeOutError(Exception):
    """
    Exception raised when the request is longer then n seconds.

    Attributes:
        url (str): The url path.
    """

    def __init__(self, url: str) -> None:
        """
        Initialize the exception.

        Args:
            url (str): The url path.
        """
        self.url = url
        super().__init__(f"The specified url: '{url}' is not responsive.")
