#!/usr/bin/env python3
"""
This script is a command-line utility that creates a project.
"""
import sys
from MutationalSignaturesTools import arguments, install


def main() -> int:
    """
    Main entry point of the script.

    Returns:
        int: Exit status (0 for success).
    """
    args = arguments.arguments_project()
    install.create_project(
        args.project, genome=args.ref, vcf_files=args.files, bash=args.bash
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
