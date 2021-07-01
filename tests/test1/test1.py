# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import rioxarray
import numpy

from src.geofabrics import processor

class Test1(unittest.TestCase):
    """ A class to test the basic dem generation pipeline for a simple example
    with land, offshore, a reference DEM and lidar using the data specified in 
    the test1/instruction.json """

    def test_result_dem(self):
        """ A basic comparison between the generated and benchmark dem """
        
        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test1/instruction.json")
        with open(file_path, 'r') as file_pointer:
            instructions = json.load(file_pointer)
        
        # Run pipeline
        runner = processor.GeoFabricsGenerator(instructions)
        runner.run()
        
        # load in bencmark dem
        benchmark_dem = rioxarray.rioxarray.open_rasterio(instructions['instructions']['data_paths']['benchmark_dem'], masked=True)
        
        # compare the generated and benchmark dems
        numpy.testing.assert_array_equal(runner.result_dem.data, benchmark_dem.data, "The generated result_dem has different data from the benchmark_dem")


if __name__ == '__main__':
    unittest.main()