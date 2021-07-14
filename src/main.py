# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
from geofabrics import processor
import json
import argparse
import rioxarray
import numpy
import matplotlib


def parse_args():
    """ Expect a command line argument of the form '--instructions path/to/json/instruction/file' """

    parser = argparse.ArgumentParser()

    parser.add_argument('--instructions', metavar='path', required=True, action='store',
                        help='the path to instruction file')

    return parser.parse_args()


def launch_processor(args):
    """ Run the pipeline over the specified instructions and compare the result to the benchmark """

    # load the instructions
    with open(args.instructions, 'r') as file_pointer:
        instructions = json.load(file_pointer)

    # run the pipeline
    runner = processor.GeoFabricsGenerator(instructions)
    runner.run()

    # load in benchmark DEM and compare - if specified in the instructions
    if 'benchmark_dem' in instructions['instructions']['data_paths']:
        with rioxarray.rioxarray.open_rasterio(instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()
        print(f"Comparing the generated DEM saved at {instructions['instructions']['data_paths']['result_dem']} " +
              f"against the benchmark DEM stored {instructions['instructions']['data_paths']['benchmark_dem']}\n Any " +
              "difference will be reported.")
        # compare the generated and benchmark DEMs - plot
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
    args = parse_args()
    launch_processor(args)


if __name__ == "__main__":
    main()
