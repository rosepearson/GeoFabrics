# -*- coding: utf-8 -*-
"""
A convenience script for running the DEM generation pipelines contained in the processor
 module of geofabrics.
"""
from geofabrics import runner
import json
import argparse
import pathlib
import typing

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
        help="the path to instruction file",
    )

    return parser.parse_args()


def run_from_file(
    instructions_path: typing.Union[str, pathlib.Path],
):
    """Run the DEM generation pipeline(s) given the specified instructions.
    If a benchmark is specified compare the result to the benchmark"""

    # Load the instructions
    with open(instructions_path, "r") as file_pointer:
        instructions = json.load(file_pointer)
    # Run the pipeline
    runner.run_from_dict(instructions=instructions)


def cli_run_from_file():
    """The script entry point to geofabrics & a CLI entry point to geofabrics.
    Run standard workflow from an instruction file intput."""
    args = parse_args()
    instructions_path = args.instructions
    runner.run_from_file(instructions_path=instructions_path)


if __name__ == "__main__":
    """If called as a script."""

    cli_run_from_file()
