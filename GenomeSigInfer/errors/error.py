#!/usr/bin/env python3
"""
Custom Exceptions Module

This module defines custom exceptions to handle specific errors in the project.

Classes:
    - RefGenomeNotCorrectError(Exception): Exception raised when chromosome file is not found.
    - RefGenomeNotSUpported(Exception): Exception raised when ref genome is not supported.

Author: J.A. Busker
"""


class RefGenomeNotSUpported(Exception):
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
        super().__init__(f"The ref genome '{genome}' is not supported!")


class RefGenomeChromosomeNotFound(Exception):
    """
    Exception raised when chromosome file is not found.

    Attributes:
        chromosome (str): The chromosome.
    """

    def __init__(self, chromosome: str) -> None:
        """
        Initialize the exception.

        Args:
            chromosome (str): The chromosome.
        """
        self.chromosome = chromosome
        super().__init__(f"The chrosome file '{chromosome}' si not found!")
