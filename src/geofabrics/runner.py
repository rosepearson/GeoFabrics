# -*- coding: utf-8 -*-
"""
A convenience script for running the DEM generation pipelines contained in the processor
 module of geofabrics.
"""
from . import processor
import json
import datetime
import logging
import pathlib
import typing


def setup_logging_for_run(instructions: dict, label: str):
    """Setup logging for the current processor run"""

    assert "local_cache" in instructions["data_paths"], (
        "A local_cache must be spcified in the instruction file"
        "this is where the log file will be written."
    )

    log_path = pathlib.Path(pathlib.Path(instructions["data_paths"]["local_cache"]))
    if "subfolder" in instructions["data_paths"].keys():
        log_path = log_path / instructions["data_paths"]["subfolder"]
    else:
        log_path = log_path / "results"
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / f"geofabrics_{label}.log",
        encoding="utf-8",
        level=logging.INFO,
        force=True,
    )
    print(f"Log file is located at: geofabrics_{label}.log")
    logging.info(instructions)


def run_processor_class(processor_class, processor_label: str, instructions: dict):
    """Run a processor class recording outputs in a unique log file and timing the
    execution."""

    start_time = datetime.datetime.now()
    print(f"Run {processor_class.__name__} at {start_time}")
    run_instructions = instructions[processor_label]
    setup_logging_for_run(instructions=run_instructions, label=processor_label)
    runner = processor_class(run_instructions)
    runner.run()
    message = (
        f"Execution time is {datetime.datetime.now() - start_time} for the "
        f"{processor_class.__name__}"
    )
    print(message)
    logging.info(message)
    return runner


def from_instructions_dict(instructions: dict):
    """Run the DEM generation pipeline(s) given the specified instructions.
    If a benchmark is specified compare the result to the benchmark"""

    # Run the pipeline
    initial_start_time = datetime.datetime.now()
    if "rivers" in instructions:
        # Estimate river channel bathymetry
        run_processor_class(
            processor_class=processor.RiverBathymetryGenerator,
            processor_label="rivers",
            instructions=instructions,
        )
    if "waterways" in instructions:
        # Estimate waterway elevations
        run_processor_class(
            processor_class=processor.WaterwayBedElevationEstimator,
            processor_label="waterways",
            instructions=instructions,
        )
    if "dem" in instructions:
        run_instructions = instructions["dem"]
        dem_paths = run_instructions["data_paths"]
        if "raw_dem" not in dem_paths or not (
            pathlib.Path(dem_paths["raw_dem"]).is_file()
            or (
                pathlib.Path(dem_paths["local_cache"])
                / dem_paths["subfolder"]
                / dem_paths["raw_dem"]
            ).is_file()
        ):
            # Create a raw DEM from LiDAR / reference DEM
            run_processor_class(
                processor_class=processor.RawLidarDemGenerator,
                processor_label="dem",
                instructions=instructions,
            )
        # Add bathymetry information to a raw DEM
        run_processor_class(
            processor_class=processor.HydrologicDemGenerator,
            processor_label="dem",
            instructions=instructions,
        )
    if "roughness" in instructions:
        # Create a roughness map and add to the hydrological DEM
        run_processor_class(
            processor_class=processor.RoughnessLengthGenerator,
            processor_label="roughness",
            instructions=instructions,
        )
    print(f"Total execution time is {datetime.datetime.now() - initial_start_time}")


def from_instructions_file(
    instructions_path: typing.Union[str, pathlib.Path],
):
    """Run the DEM generation pipeline(s) given the specified instructions.
    If a benchmark is specified compare the result to the benchmark"""

    # Load the instructions
    with open(instructions_path, "r") as file_pointer:
        instructions = json.load(file_pointer)
    # Run the pipeline
    from_instructions_dict(instructions=instructions)