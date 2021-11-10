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
    print("Run processor.DemGenerator")

    # Cycle through different chunk sizes and number for cores
    results = {'execution_time': [], 'number_of_cores': [], 'chunk_sizes': []}
    result_name_stub = instructions['instructions']['general']['name_stub']
    core_range = list(range(1, 5, 1))
    for chunk_size in range(75, 226, 25):
        for number_of_cores in core_range:
            instructions['instructions']['data_paths']['result_dem'] = instructions['instructions']['data_paths']['local_cache'] \
                 + result_name_stub + f"_{number_of_cores}cores_{chunk_size}chunk"
            instructions['instructions']['processing']['chunk_size'] = chunk_size
            instructions['instructions']['processing']['number_of_cores'] = number_of_cores
            # Run the pipeline
            start_time = time.time()
            # Create a DEM from dense data (LiDAR, reference DEM) and bathymetry if specified
            runner = processor.DemGenerator(instructions)
            runner.run()
            end_time = time.time()

            # record results
            results['execution_time'].append(end_time - start_time)
            results['number_of_cores'].append(number_of_cores)
            results['chunk_sizes'].append(chunk_size)
            print(f"Time: {end_time - start_time}, Cores: {number_of_cores}, Chunk size: {chunk_size}")

    print(results)

    times = numpy.asarray(results['execution_time'])
    cores = numpy.asarray(results['number_of_cores'])
    chunks = numpy.asarray(results['chunk_sizes'])
    for i in range(1, len(core_range) + 1, 1):
        matplotlib.pyplot.plot(chunks[cores == i], times[cores == i], label=f'{i} cores')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Chunk size in pixels')
    matplotlib.pyplot.ylabel('Execution time')
    matplotlib.pyplot.title('10m resolution test2 catchment Wellington_2013 LiDAR')


def main():
    """ The entry point to geofabrics. """
    args = parse_args()
    launch_processor(args)


if __name__ == "__main__":
    main()
