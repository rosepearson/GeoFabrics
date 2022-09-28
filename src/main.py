# -*- coding: utf-8 -*-
"""
A convenience script for running the DEM generation pipelines contained in the processor
 module of geofabrics.
"""
from geofabrics import processor
import json
import argparse
import rioxarray
import numpy
import matplotlib
import time
import logging
import pathlib

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


def setup_logging_for_run(instructions: dict):
    """Setup logging for the current processor run"""

    assert "local_cache" in instructions["data_paths"], (
        "A local_cache must be spcified in the instruction file"
        "this is where the log file will be written."
    )

    log_path = pathlib.Path(instructions["data_paths"]["local_cache"])
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "geofabrics.log",
        encoding="utf-8",
        level=logging.INFO,
        force=True,
    )
    print(f"Log file is located at: {log_path / 'geofabrics.log'}")
    logging.info(instructions)


def check_for_benchmarks(instructions: dict, runner: processor.BaseProcessor):
    """Compare against a benchmark DEM if one is specified"""

    # Load in benchmark DEM and compare - if specified in the instructions
    if "benchmark_dem" in instructions["data_paths"]:
        with rioxarray.rioxarray.open_rasterio(
            instructions["data_paths"]["benchmark_dem"], masked=True
        ) as benchmark_dem:
            benchmark_dem.load()
        result_dem = runner.dense_dem.dem
        logging.info(
            "Comparing the generated DEM saved at "
            f"{instructions['data_paths']['result_dem']} against the "
            f"benchmark DEM stored "
            "{instructions['data_paths']['benchmark_dem']}. \nAny "
            "difference will be reported."
        )
        # Compare the generated and benchmark DEMs - plot
        diff = benchmark_dem.copy()
        diff.data = result_dem.data - benchmark_dem.data

        f = matplotlib.pyplot.figure(figsize=(15, 5))
        gs = f.add_gridspec(1, 3)

        ax1 = f.add_subplot(gs[0, 0])
        benchmark_dem.plot(cmap="viridis", ax=ax1)

        ax2 = f.add_subplot(gs[0, 1])
        result_dem.plot(cmap="viridis", ax=ax2)

        ax3 = f.add_subplot(gs[0, 2])
        diff.plot(cmap="viridis", ax=ax3)

        ax1.set_title("Benchmark")
        ax2.set_title("Generated")
        ax3.set_title("Difference")

        # assert different
        numpy.testing.assert_array_equal(
            result_dem.data,
            benchmark_dem.data,
            "The generated result_dem has different data from the benchmark_dem",
        )


def launch_processor(args):
    """Run the DEM generation pipeline(s) given the specified instructions.
    If a benchmark is specified compare the result to the benchmark"""

    # Load the instructions
    with open(args.instructions, "r") as file_pointer:
        instructions = json.load(file_pointer)
    # Run the pipeline
    initial_start_time = time.time()
    if "rivers" in instructions:
        # Estimate river channel bathymetry
        start_time = time.time()
        print(f"Run processor.RiverBathymetryGenerator at {start_time}")
        run_instructions = instructions["rivers"]
        setup_logging_for_run(run_instructions)
        runner = processor.RiverBathymetryGenerator(run_instructions)
        runner.run()
        message = (
            f"Execution time is {time.time() - start_time} for "
            "processor.RiverBathymetryGenerator"
        )
        print(message)
        logging.info(message)
    if "waterways" in instructions:
        # Estimate waterway elevations
        start_time = time.time()
        print("Run processor.WaterwayBedElevationEstimator")
        run_instructions = instructions["waterways"]
        setup_logging_for_run(run_instructions)
        runner = processor.WaterwayBedElevationEstimator(run_instructions)
        runner.run()
        message = (
            f"Execution time is {time.time() - start_time} for "
            "processor.WaterwayBedElevationEstimator"
        )
        print(message)
        logging.info(message)
    if "dem" in instructions:
        start_time = time.time()
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
            start_time = time.time()
            print("Run processor.RawLidarDemGenerator")
            setup_logging_for_run(run_instructions)
            runner = processor.RawLidarDemGenerator(run_instructions)
            runner.run()
            message = (
                f"Execution time is {time.time() - start_time} for "
                "processor.RawLidarDemGenerator"
            )
            print(message)
            logging.info(message)
        # Add bathymetry information to a raw DEM
        start_time = time.time()
        print("Run processor.HydrologicDemGenerator")
        setup_logging_for_run(run_instructions)
        runner = processor.HydrologicDemGenerator(run_instructions)
        runner.run()
        message = (
            f"Execution time is {time.time() - start_time} for "
            "processor.HydrologicDemGenerator"
        )
        print(message)
        logging.info(message)
        check_for_benchmarks(run_instructions, runner)
    if "roughness" in instructions:
        # Create a roughness map and add to the hydrological DEM
        start_time = time.time()
        print("Run processor.RoughnessLengthGenerator")
        run_instructions = instructions["roughness"]
        setup_logging_for_run(run_instructions)
        runner = processor.RoughnessLengthGenerator(run_instructions)
        runner.run()
        message = (
            f"Execution time is {time.time() - start_time} for "
            "processor.RoughnessLengthGenerator"
        )
        print(message)
        logging.info(message)
    print(f"Total execution time is {time.time() - initial_start_time}")


def main():
    """The entry point to geofabrics."""
    args = parse_args()
    launch_processor(args)


if __name__ == "__main__":
    main()
