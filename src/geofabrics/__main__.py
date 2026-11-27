# -*- coding: utf-8 -*-
"""
Command-line entry point for running GeoFabrics from a JSON instruction file.
"""

from geofabrics import runner
import argparse

import warnings

# Turn off future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    """Parse command-line arguments; ``--instructions`` must name a JSON file."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instructions",
        metavar="path",
        required=True,
        action="store",
        help="path to a JSON instruction file",
    )

    return parser.parse_args()


def cli_run_from_dict():
    """Pass the parsed ``--instructions`` value to the dictionary runner."""

    # Load the instructions
    args = parse_args()
    # Run the pipeline
    runner.from_instructions_dict(instructions=args.instructions)


def cli_run_from_file():
    """Run the standard workflow using the JSON instruction file from the CLI."""
    args = parse_args()
    runner.from_instructions_file(instructions_path=args.instructions)


if __name__ == "__main__":
    """If called as a script."""

    cli_run_from_file()
