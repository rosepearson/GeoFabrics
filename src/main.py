# -*- coding: utf-8 -*-
"""
A convenience script for running many combinations of chunk_sizes and numbers_of_cores for a small sub-set of a
catchment to help with the selection of an appropiate chink_size and number_of_cores before processing an entire
catchment.
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


def parse_args():
    """ Expect a command line argument of the form '--instructions path/to/json/instruction/file' """

    parser = argparse.ArgumentParser()

    parser.add_argument('--instructions', metavar='path', required=True, action='store',
                        help='the path to instruction file')

    return parser.parse_args()


def launch_processor(args):
    """ Run the DEM generation pipeline given the specified instructions.
    If a benchmark is specified compare the result to the benchmark """

    # Load the instructions
    with open(args.instructions, 'r') as file_pointer:
        instructions = json.load(file_pointer)

    assert 'local_cache' in instructions['instructions']['data_paths'], "A local_cache must be spcified in the instruction file" \
        "this is where the log file will be written."

    # Setup logging
    log_path = pathlib.Path(instructions['instructions']['data_paths']['local_cache'])
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path / 'geofabrics.log', encoding='utf-8', level=logging.INFO, force=True)
    print(f"Log file is located at: {log_path / 'geofabrics.log'}")

    # Run the pipeline
    start_time = time.time()
    if 'dense_dem' in instructions['instructions']['data_paths']:
        # Update a dense DEM with offshore values
        print("Run processor.OffshoreDemGenerator")
        runner = processor.OffshoreDemGenerator(instructions)
        runner.run()
    else:
        # Create a DEM from dense data (LiDAR, reference DEM) and bathymetry if specified
        print("Run processor.DemGenerator")
        runner = processor.DemGenerator(instructions)
        runner.run()
    end_time = time.time()

    print(f"Execution time is {end_time - start_time}")

    # Load in benchmark DEM and compare - if specified in the instructions
    if 'benchmark_dem' in instructions['instructions']['data_paths']:
        with rioxarray.rioxarray.open_rasterio(instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()
        logging.info(f"Comparing the generated DEM saved at {instructions['instructions']['data_paths']['result_dem']} "
                     f"against the benchmark DEM stored {instructions['instructions']['data_paths']['benchmark_dem']} "
                     "\nAny difference will be reported.")
        # Compare the generated and benchmark DEMs - plot
        diff = benchmark_dem.copy()
        diff.data = runner.result_dem.data - benchmark_dem.data

        f = matplotlib.pyplot.figure(figsize=(15, 5))
        gs = f.add_gridspec(1, 3)

        ax1 = f.add_subplot(gs[0, 0])
        benchmark_dem.plot(cmap="viridis", ax=ax1)

        ax2 = f.add_subplot(gs[0, 1])
        runner.result_dem.plot(cmap="viridis", ax=ax2)

        ax3 = f.add_subplot(gs[0, 2])
        diff.plot(cmap="viridis", ax=ax3)

        ax1.set_title("Benchmark")
        ax2.set_title("Generated")
        ax3.set_title("Difference")

        # assert different
        numpy.testing.assert_array_equal(runner.result_dem.data, benchmark_dem.data,
                                         "The generated result_dem has different data from the benchmark_dem")


def main():
    """ The entry point to geofabrics. """
    args = parse_args()
    launch_processor(args)


if __name__ == "__main__":
    main()
