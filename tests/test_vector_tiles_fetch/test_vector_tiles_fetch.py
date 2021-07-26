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

from src.geofabrics import vector_fetch
from src.geofabrics import geometry


class LinzTilesTest(unittest.TestCase):
    """ A class to test the basic lidar_fetch class OpenTopography functionality by downloading files from
    OpenTopography within a small region. All files are deleted after checking their names and size."""

    TILE_NAMES = ['BR20_1000_4014', 'BR20_1000_4015', 'BR20_1000_4114', 'BR20_1000_4115', 'BR20_1000_4013',
                  'BR20_1000_4113', 'BR20_1000_4016', 'BR20_1000_4116']

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the tests and also download the files used
        in the tests. """

        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test_vector_tiles_fetch/instruction.json")
        with open(file_path, 'r') as file_pointer:
            instructions = json.load(file_pointer)

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(instructions['instructions']['data_paths']['local_cache'])

        # ensure the cache directory doesn't exist - i.e. clean up from last test occurred correctly
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
        cls.cache_dir.mkdir()
        # create fake catchment boundary
        x0 = 1473821
        y0 = 5376526
        x1 = 1475457
        y1 = 5378203
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(instructions['instructions']['projection'])

        # save faked catchment file
        catchment_dir = cls.cache_dir / "catchment"
        catchment.to_file(catchment_dir)
        shutil.make_archive(base_name=catchment_dir, format='zip', root_dir=catchment_dir)
        shutil.rmtree(catchment_dir)

        # cconvert catchment file to zipfile
        catchment_dir = pathlib.Path(str(catchment_dir) + ".zip")
        catchment_geometry = geometry.CatchmentGeometry(catchment_dir, catchment_dir,  # all land
                                                        instructions['instructions']['projection'],
                                                        instructions['instructions']['grid_params']['resolution'])

        # Run pipeline - download files
        cls.runner = vector_fetch.LinzTiles(instructions['instructions']['linz_api']['key'],
                                            instructions['instructions']['linz_api']['layers']['lidar_tile'],
                                            catchment_geometry, cls.cache_dir, verbose=True)

    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory and included created and downloaded files at the end of the test. """

        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)

    def test_correct_file_list(self):
        """ A test to see if all expected tiles name are located """

        self.runner.run()
        print(f"The returned tile names are: {self.runner.tile_names}")

        # check files are correct
        self.assertEqual(self.runner.tile_names, self.TILE_NAMES, f"The returned tile names `{self.runner.tile_names}` "
                         + f"differ from those expected `{self.TILE_NAMES}`")


if __name__ == '__main__':
    unittest.main()
