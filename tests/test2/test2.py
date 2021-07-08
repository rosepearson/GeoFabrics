# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import shapely
import geopandas
import shutil
import numpy

from src.geofabrics import lidar_fetch
from src.geofabrics import geometry

class Test2(unittest.TestCase):
    """ A class to test the basic lidar_fetch functionality by downloading files from OpenTopography 
    within a small region. All files are deleted after checking their names and size."""
    
    DATASET = "Wellington_2013"
    FILES = ["ot_CL1_WLG_2013_1km_085033.laz", "ot_CL1_WLG_2013_1km_086033.laz", 
             "ot_CL1_WLG_2013_1km_085032.laz", "ot_CL1_WLG_2013_1km_086032.laz", 
             DATASET + "_TileIndex.zip"]
    SIZES = [ 6795072, 5712485, 1670549, 72787, 598532]

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the test. """
        
        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test2/instruction.json")
        with open(file_path, 'r') as file_pointer:
            instructions = json.load(file_pointer)
            
        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(instructions['instructions']['data_paths']['local_cache'])
        
        # ensure the cache directory doesn't exist - i.e. clean up from last test occured correctly
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
        cls.cache_dir.mkdir()
        # create fake catchment boundary
        x0 = 1764410; y0 = 5470382; x1 = 1765656; y1 = 5471702;
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(instructions['instructions']['projection'])
        
        # save faked catchment boundary
        catchment_dir = cls.cache_dir / "catchment"
        catchment.to_file(catchment_dir)
        shutil.make_archive(base_name=catchment_dir, format='zip', root_dir=catchment_dir)
        shutil.rmtree(catchment_dir)
        
        # create a catchment_geometry
        catchment_dir = pathlib.Path(str(catchment_dir) + ".zip")
        cls.catchment_geometry = geometry.CatchmentGeometry(catchment_dir, catchment_dir,  # all land
                                                            instructions['instructions']['projection'],
                                                            instructions['instructions']['grid_params']['resolution'])
        
    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory and included created and downloaded files at the end of the test. """
        
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)

    def test_download_all(self):
        """ A test to see if all expected files are downloaded - check files and size """
        
        # Run pipeline
        runner = lidar_fetch.OpenTopography(self.catchment_geometry, self.cache_dir, verbose = True)
        runner.run()
        
        # check the right dataset is downloaded - self.DATASET
        self.assertEqual(len(list(self.cache_dir.glob('*/**'))), 1, "There should only be one dataset named " + self.DATASET + " instead there are " + \
            str(len(list(self.cache_dir.glob('*/**')))) + " list " + str(list(self.cache_dir.glob('*/**'))))
        
        dataset_dir = self.cache_dir / self.DATASET
        self.assertEqual(len([file for file in self.cache_dir.iterdir() if file.is_dir() and file==dataset_dir]), 1, "Only the " + self.DATASET + \
            " directory should have been downloaded. Insead we have: " + str([file for file in self.cache_dir.iterdir() if file.is_dir()]))
        
        # check files are correct
        downloaded_files = [dataset_dir / file for file in self.FILES]
        self.assertEqual(len(list(dataset_dir.glob('*'))), len(downloaded_files), "There should have been " + str(len(downloaded_files)) + " files downloaded into the " \
            + self.DATASET + " directory, instead there are " + str(len(list(dataset_dir.glob('*')))) + " files/dirs in the directory")
        self.assertTrue(numpy.all([file in downloaded_files for file in dataset_dir.glob('*')]), "The downloaded files " + str(list(dataset_dir.glob('*'))) + \
            " do not match the expected files " + str(downloaded_files))
        
        # check sizes are correct
        self.assertTrue(numpy.all([downloaded_file.stat().st_size == self.SIZES[i] for i, downloaded_file in enumerate(downloaded_files)]), "There is a missmatch between the size of the downloaded files " \
            + str([downloaded_file.stat().st_size for downloaded_file in downloaded_files]) + " and the expected sizes of " + str(self.SIZES))

if __name__ == '__main__':
    unittest.main()
