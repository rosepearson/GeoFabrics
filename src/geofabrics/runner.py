# -*- coding: utf-8 -*-
"""
Orchestrate the GeoFabrics processing pipelines from an instruction dictionary.
"""

from . import processor
import json
import datetime
import logging
import pathlib
import typing
import copy


def config_logging(logging_filepath: pathlib):
    """Configure the root logger inherited by other loggers."""
    log_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "stream_handler": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
            "file_handler": {
                "level": "DEBUG",
                "filename": logging_filepath,
                "class": "logging.FileHandler",
                "formatter": "standard",
                "encoding": "utf-8",
                "mode": "a",
            },
        },
        "root": {
            "handlers": ["file_handler", "stream_handler"],
            "level": "DEBUG",
            "propagate": False,
        },
    }
    logging.config.dictConfig(log_dict)


def setup_logging_for_run(instructions: dict, label: str):
    """Setup logging for the current processor run"""

    if label == "runner":
        # In this case expecting the top level instruction dictionary instead of a subsection
        log_path = pathlib.Path(
            pathlib.Path(
                instructions[next(iter(instructions))]["data_paths"]["local_cache"]
            )
        )
        if "subfolder" in instructions[next(iter(instructions))]["data_paths"].keys():
            log_path = (
                log_path
                / instructions[next(iter(instructions))]["data_paths"]["subfolder"]
            )
        else:
            log_path = log_path / "results"
    else:
        log_path = pathlib.Path(pathlib.Path(instructions["data_paths"]["local_cache"]))
        if "subfolder" in instructions["data_paths"].keys():
            log_path = log_path / instructions["data_paths"]["subfolder"]
        else:
            log_path = log_path / "results"
    log_path.mkdir(parents=True, exist_ok=True)

    config_logging(log_path / f"geofabrics_{label}.log")
    logger = logging.getLogger(f"{__name__}.{label}")

    logger.info(f"Log file is located at: geofabrics_{label}.log")
    logger.debug(instructions)
    return logger


def run_processor_class(processor_class, processor_label: str, instructions: dict):
    """Run a processor class recording outputs in a unique log file and timing the
    execution."""

    start_time = datetime.datetime.now()
    run_instructions = instructions[processor_label]
    logger = setup_logging_for_run(instructions=run_instructions, label=processor_label)
    logger.info(f"Run {processor_class.__name__} at {start_time}")
    runner = processor_class(run_instructions)
    runner.run()
    message = (
        f"Execution time is {datetime.datetime.now() - start_time} for the "
        f"{processor_class.__name__}"
    )
    logger.info(message)
    del runner
    del logger
    return


def merge_dicts(dict_a: dict, dict_b: dict, logger: logging.Logger, replace_a: bool):
    """Recursively merge ``dict_b`` into a deep copy of ``dict_a``.

    Nested dictionaries are merged recursively, and keys present only in
    ``dict_b`` are added. For conflicting leaf values, ``replace_a`` controls
    whether the value from ``dict_b`` replaces the value from ``dict_a``.
    Conflicts are reported through ``logger``. Neither input dictionary is
    modified.

    Parameters
    ----------
    dict_a
        Base dictionary to copy and update.
    dict_b
        Dictionary whose values are merged into the copy.
    logger
        Logger used to report conflicting values.
    replace_a
        If true, replace conflicting values from ``dict_a`` with values from
        ``dict_b``; otherwise keep the values from ``dict_a``.

    Returns
    -------
    dict
        The merged dictionary.
    """

    def recursive_merge_dicts(
        base_dict: dict,
        new_dict: dict,
        replace_base: bool,
        logger: logging.Logger,
        path: list = [],
    ):
        """Merge ``new_dict`` into ``base_dict`` in place."""
        for key in new_dict:
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(new_dict[key], dict):
                    recursive_merge_dicts(
                        base_dict=base_dict[key],
                        new_dict=new_dict[key],
                        logger=logger,
                        replace_base=replace_base,
                        path=path + [str(key)],
                    )
                elif base_dict[key] == new_dict[key]:
                    pass  # same leaf value
                else:
                    if replace_base:
                        logger.warning(
                            f"Conflict with both dictionaries containing different values at {path + [str(key)]}."
                            " Value replaced."
                        )
                        base_dict[key] = new_dict[key]
                    else:
                        logger.warning(
                            f"Conflict with both dictionaries containing different values at {path + [str(key)]}"
                            ". Value ignored."
                        )
            else:
                base_dict[key] = new_dict[key]
        return base_dict

    return recursive_merge_dicts(
        copy.deepcopy(dict_a), dict_b, replace_base=replace_a, logger=logger
    )


def from_instructions_dict(instructions: dict):
    """Run the processing pipelines specified by an instruction dictionary.

    If a ``default`` section is present, merge its values into each pipeline's
    instructions. Requested processing stages run in pipeline order. DEM
    generation steps are skipped when their configured output files already
    exist.
    """

    # Construct the full instructions by adding the default entries to each stage
    logger = setup_logging_for_run(instructions=instructions, label="runner")
    instructions = copy.deepcopy(instructions)
    if "default" in instructions:
        default = instructions.pop("default")
        # Auto-add dem and roughness keys if not included and outputs specified
        if (
            "dem" not in instructions
            and "data_paths" in default
            and (
                "raw_dem" in default["data_paths"]
                or "result_dem" in default["data_paths"]
            )
        ):
            instructions["dem"] = {}
        if (
            "roughness" not in instructions
            and "data_paths" in default
            and "result_geofabric" in default["data_paths"]
        ):
            instructions["roughness"] = {}
        # Construct the full instructions
        for key in instructions:
            instructions[key] = merge_dicts(
                dict_a=instructions[key], dict_b=default, logger=logger, replace_a=False
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
    if "stopbanks" in instructions:
        # Estimate stopbank crest elevations
        run_processor_class(
            processor_class=processor.StopbankCrestElevationEstimator,
            processor_label="stopbanks",
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
    if "patch" in instructions:
        # Add patch to the hydrological dem or geofabric
        run_processor_class(
            processor_class=processor.PatchDemGenerator,
            processor_label="patch",
            instructions=instructions,
        )
    logger = setup_logging_for_run(instructions=instructions, label="runner")
    logger.info(
        f"Total execution time is {datetime.datetime.now() - initial_start_time}"
    )
    del logger


def from_instructions_file(
    instructions_path: typing.Union[str, pathlib.Path],
):
    """Load a JSON instruction file and run its requested processing pipelines.

    This function delegates pipeline execution to :func:`from_instructions_dict`.
    """

    # Load the instructions
    with open(instructions_path, "r") as file_pointer:
        instructions = json.load(file_pointer)
    # Run the pipeline
    from_instructions_dict(instructions=instructions)
