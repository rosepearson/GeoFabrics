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

def parse_args():
    """ Expect a command line argument of the form '--instructions path/to/json/instruction/file' """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--instructions', metavar='path', required=True, action='store',
                        help='the path to instruction file')
    
    return parser.parse_args()

def launch_processor(args):
    """ Run the pipeline over ht especified instructions and compare the result 
    to the benchmark """
    
    # load the instructions
    with open(args.instructions, 'r') as file_pointer:
        instructions = json.load(file_pointer)

    # ru the pipeline
    runner = processor.GeoFabricsGenerator(instructions)
    runner.run()
    
    # load in bencmark dem and compare - if specified in the insturctions
    if 'benchmark_dem' in instructions['instructions']['data_paths']:
        benchmark_dem = rioxarray.rioxarray.open_rasterio(instructions['instructions']['data_paths']['benchmark_dem'], masked=True)
        print('Comparing the generated DEM saved at ' + instructions['instructions']['data_paths']['result_dem'] + 
              ' against the benchmark DEM stored ' + instructions['instructions']['data_paths']['benchmark_dem'] + '\n Any difference will be reported.')
        # compare the generated and benchmark dems
        numpy.testing.assert_array_equal(runner.result_dem.data, benchmark_dem.data, "The generated result_dem has different data from the benchmark_dem")
        
def main():
    args = parse_args()
    launch_processor(args)

if __name__ == "__main__":
    main()
        
        
