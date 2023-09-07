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
import copy


def setup_logging_for_run(instructions: dict, label: str):
    """Setup logging for the current processor run"""

    assert "local_cache" in instructions["data_paths"], (
        "A local_cache must be spcified in the instruction file"
        "this is where the log file will be written."
    )

    log_path = pathlib.Path(
        pathlib.Path(instructions["data_paths"]["local_cache"])
    )
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


def run_processor_class(
    processor_class, processor_label: str, instructions: dict
):
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


def merge_dicts(dict_a: dict, dict_b: dict, replace_a: bool):
    """Merge the contents of the dict_a and dict_b. Use recursion to merge
    any nested dictionaries. replace_a determines if the dict_a values are
    replaced or not if different values are in the dict_b.

    Adapted from https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries

    Parameters:
            dict_a  The dict to
            dict_b  The location of the centre of the river mouth
            replace_a If True any dict_a values are replaced if different values are in dict_b
    """

    def recursive_merge_dicts(
        base_dict: dict, new_dict: dict, replace_base: bool, path: list = []
    ):
        """Recurively add the new_dict into the base_dict. dict_a is mutable."""
        for key in new_dict:
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(
                    new_dict[key], dict
                ):
                    recursive_merge_dicts(
                        base_dict=base_dict[key],
                        new_dict=new_dict[key],
                        replace_base=replace_base,
                        path=path + [str(key)],
                    )
                elif base_dict[key] == new_dict[key]:
                    pass  # same leaf value
                else:
                    if replace_base:
                        print(
                            f"Conflict with both dictionaries containing different values at {path + [str(key)]}."
                            " Value replaced."
                        )
                        base_dict[key] = new_dict[key]
                    else:
                        print(
                            f"Conflict with both dictionaries containing different values at {path + [str(key)]}"
                            ". Value ignored."
                        )
            else:
                base_dict[key] = new_dict[key]
        return base_dict

    return recursive_merge_dicts(
        copy.deepcopy(dict_a), dict_b, replace_base=replace_a
    )


def from_instructions_dict(instructions: dict):
    """Run the DEM generation pipeline(s) given the specified instructions.
    If a benchmark is specified compare the result to the benchmark"""

    # Construct the full instructions by adding the shared entries to each stage
    instructions = copy.deepcopy(instructions)
    if "shared" in instructions:
        shared = instructions.pop("shared")
        # Auto-add dem and roughness keys if not included and outputs specified
        if (
            "dem" not in instructions
            and "data_paths" in shared
            and (
                "raw_dem" in shared["data_paths"]
                or "result_dem" in shared["data_paths"]
            )
        ):
            instructions["dem"] = {}
        if (
            "roughness" not in instructions
            and "data_paths" in shared
            and "result_geofabric" in shared["data_paths"]
        ):
            instructions["roughness"] = {}
        # Construct the full instructions
        for key in instructions:
            instructions[key] = merge_dicts(
                dict_a=instructions[key], dict_b=shared, replace_a=False
            )

    # Run the pipeline
    initial_start_time = datetime.datetime.now()
    if "measured" in instructions:
        # Estimate river channel bathymetry
        run_processor_class(
            processor_class=processor.MeasuredRiverGenerator,
            processor_label="measured",
            instructions=instructions,
        )
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
        # Only run if raw doesn't exist
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
        # Only run if the dem doesn't already exist
        if "result_dem" not in dem_paths or not (
            pathlib.Path(dem_paths["result_dem"]).is_file()
            or (
                pathlib.Path(dem_paths["local_cache"])
                / dem_paths["subfolder"]
                / dem_paths["result_dem"]
            ).is_file()
        ):
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
    print(
        f"Total execution time is {datetime.datetime.now() - initial_start_time}"
    )


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
