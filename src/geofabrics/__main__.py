# -*- coding: utf-8 -*-
"""
A convenience script for running the DEM generation pipelines contained in the processor
 module of geofabrics.
"""
from geofabrics import runner
import argparse

import warnings

# Turn off future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    """Expect a command line argument of the form:
    '--instructions path/to/json/instruction/file'"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instructions",
        metavar="path",
        required=True,
        action="store",
        help="the instructions - either a path or dict - depending if running from file"
        " or from dictionary. If running as a script must be a path",
    )

    return parser.parse_args()


def cli_run_from_dict():
    """Run the DEM generation pipeline(s) given the specified instructions as a
    dictionary. If a benchmark is specified compare the result to the benchmark"""

    # Load the instructions
    args = parse_args()
    # Run the pipeline
    runner.from_instructions_dict(instructions=args.instructions)


def cli_run_from_file():
    """The standard script entry point to geofabrics & a CLI entry point to geofabrics.
    Run standard workflow from an instruction file intput."""
    args = parse_args()
    runner.from_instructions_file(instructions_path=args.instructions)


if __name__ == "__main__":
    """If called as a script."""

    cli_run_from_file()
