#!/usr/bin/env python3
"""
This script is a command-line utility that creates a project.
"""
import sys
import logging
from MutationalSignaturesTools import arguments, install, mutSigMatrix


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = arguments.arguments_project()
    project = args.project
    genome = args.ref
    install.create_project(
        project, genome=genome, vcf_files=args.files, bash=args.bash
    )
    mutSigMatrix.create_sbs_matrices(project=project, genome=genome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
