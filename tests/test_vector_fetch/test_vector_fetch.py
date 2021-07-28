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

    TILE_NAMES = ['BR20_1000_4012', 'BR20_1000_4013', 'BR20_1000_4014', 'BR20_1000_4112',
                  'BR20_1000_4212', 'BR20_1000_4213', 'BR20_1000_4214']

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the tests and also download the files used
        in the tests. """

        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test_vector_fetch/instruction.json")
        with open(file_path, 'r') as file_pointer:
            cls.instructions = json.load(file_pointer)

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(cls.instructions['instructions']['data_paths']['local_cache'])

        # makes sure the data directory exists and only contains benchmark data
        cls.clean_data_folder()

        # create fake catchment boundary
        x0 = 1477354
        x1 = 1484656
        y0 = 5374408
        y1 = 5383411
        catchment = shapely.geometry.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions['instructions']['projection'])

        # save faked catchment file
        catchment_dir = cls.cache_dir / "catchment"
        catchment.to_file(catchment_dir)
        shutil.make_archive(base_name=catchment_dir, format='zip', root_dir=catchment_dir)
        shutil.rmtree(catchment_dir)

        # cconvert catchment file to zipfile
        catchment_dir = pathlib.Path(str(catchment_dir) + ".zip")
        catchment_geometry = geometry.CatchmentGeometry(catchment_dir, catchment_dir,  # all land
                                                        cls.instructions['instructions']['projection'],
                                                        cls.instructions['instructions']['grid_params']['resolution'])

        # Run pipeline - download files
        cls.runner = vector_fetch.LinzVectors(cls.instructions['instructions']['linz_api']['key'],
                                              catchment_geometry, verbose=True)
        cls.catchment_geometry = catchment_geometry

    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory and included created and downloaded files at the end of the test. """

        assert cls.cache_dir.exists(), "The data directory that should include the comparison benchmark files " + \
            "doesn't exist"
        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """ Remove all generated or downloaded files from the data directory """

        assert cls.cache_dir.exists(), "The data directory that should include the comparison benchmark file " + \
            "doesn't exist"

        benchmark_files = [cls.cache_dir / "land.zip", cls.cache_dir / "bathymetry_contours.zip"]
        for file in cls.cache_dir.glob('*'):
            if file not in benchmark_files:
                file.unlink()

    def test_land(self):
        """ A test to check expected island is loaded """

        land = self.runner.run(self.instructions['instructions']['linz_api']['land']['layers'][0],
                               self.instructions['instructions']['linz_api']['land']['type'])

        # Load in benchmark
        land_dir = self.cache_dir / "land.zip"
        benchmark = geopandas.read_file(land_dir)

        # check files are correct
        self.assertEqual(land.difference(benchmark).area.sum(), 0, f"The returned land polygon ``f{land}` differs by " +
                         f"`{land.difference(benchmark).area.sum()}` in area from the benchmark of f{benchmark}")

    def test_bathymetry(self):
        """ A test to check expected bathyemtry contours are loaded """

        bathymetry_contours = self.runner.run(
            self.instructions['instructions']['linz_api']['bathymetry_contours']['layers'][0],
            self.instructions['instructions']['linz_api']['bathymetry_contours']['type'])

        # Load in benchmark
        bathymetry_dir = self.cache_dir / "bathymetry_contours.zip"
        benchmark = geopandas.read_file(bathymetry_dir)

        # check files are correct
        self.assertEqual(bathymetry_contours.difference(benchmark).area.sum(), 0, "The returned land polygon " +
                         f"`{bathymetry_contours}` differs by `{bathymetry_contours.difference(benchmark).area.sum()}" +
                         f"` in area from the benchmark of f{benchmark}")


if __name__ == '__main__':
    unittest.main()
