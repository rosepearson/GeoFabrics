# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
from geofabrics import processor
import json
import argparse
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


def benchmark_processing(args):
    """ Run the DEM generation pipeline given the specified instructions for a range of different 'number_of_cores'
    and 'chunk_sizes' specified in the instructions. Optionally save each DEM exparately. Plot the execution times of
    all differnet processing configurations at the end."""

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
    print("Benchmarking processor.DemGenerator for chunk_sizes: "
          f"{instructions['instructions']['benchmarking']['chunk_sizes']} and numbers_of_cores: "
          f"{instructions['instructions']['benchmarking']['numbers_of_cores']}")

    resolution = instructions['instructions']['output']['grid_params']['resolution']
    cache_path = pathlib.Path(instructions['instructions']['data_paths']['local_cache'])

    # Cycle through different chunk sizes and number for cores
    results = {'execution_time': [], 'number_of_cores': [], 'chunk_sizes': []}
    for chunk_size in instructions['instructions']['benchmarking']['chunk_sizes']:
        for number_of_cores in instructions['instructions']['benchmarking']['numbers_of_cores']:
            instructions['instructions']['data_paths']['result_dem'] = cache_path / \
                f"benchmarking_{resolution}res_{number_of_cores}cores_{chunk_size}chunk"
            instructions['instructions']['processing']['chunk_size'] = chunk_size
            instructions['instructions']['processing']['number_of_cores'] = number_of_cores
            # Run the pipeline
            start_time = time.time()
            # Create a DEM from dense data (LiDAR, reference DEM) and bathymetry if specified
            runner = processor.DemGenerator(instructions)
            runner.run()
            end_time = time.time()
            if instructions['instructions']['benchmarking']['delete_dems']:
                pathlib.unlink(pathlib.Path(instructions['instructions']['data_paths']['result_dem']))

            # record results
            results['execution_time'].append(end_time - start_time)
            results['number_of_cores'].append(number_of_cores)
            results['chunk_sizes'].append(chunk_size)
            print(f"Time: {end_time - start_time}, Cores: {number_of_cores}, Chunk size: {chunk_size}")

    print(results)
    logging.info(results)

    times = numpy.asarray(results['execution_time'])
    cores = numpy.asarray(results['number_of_cores'])
    chunks = numpy.asarray(results['chunk_sizes'])
    for i in instructions['instructions']['benchmarking']['numbers_of_cores']:
        matplotlib.pyplot.plot(chunks[cores == i], times[cores == i], label=f'{i} cores')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Chunk size in pixels')
    matplotlib.pyplot.ylabel('Execution time')
    matplotlib.pyplot.title(instructions['instructions']['benchmarking']['title'] + f"\nResolution = {resolution}")
    matplotlib.pyplot.savefig(cache_path / f"benchmarking_plot_{resolution}res.png")


def main():
    """ The entry point to geofabrics. """
    args = parse_args()
    benchmark_processing(args)


if __name__ == "__main__":
    main()
