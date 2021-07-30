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
import dotenv
import os

from src.geofabrics import vector_fetch
from src.geofabrics import geometry


class LinzVectorsTest(unittest.TestCase):
    """ A class to test the basic vector_fetch class Linz functionality by downloading files from
    OpenTopography within a small region. All files are deleted after checking their names and size.

    Tests run include:
        1. test_land - Test that the expected land dataset is downloaded from LINZ
        2. test_bathymetry - Test that the expected land dataset is downloaded from LINZ
    """

    # The expected datasets and files to be downloaded - used for comparison in the later tests
    LAND = {"area": 150539169542.39142, "geometryType": 'Polygon', 'length': 6006036.039821969}
    BATHYMETRY_CONTOURS = {"area": 0.0, "geometryType": 'LineString', 'length': 144353.73387463146}

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the tests and also download the files used
        in the tests. """

        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test_vector_fetch/instruction.json")
        with open(file_path, 'r') as file_pointer:
            cls.instructions = json.load(file_pointer)

        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get('LINZ_API', None)
        cls.instructions['instructions']['apis']['linz']['key'] = linz_key

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(cls.instructions['instructions']['data_paths']['local_cache'])

        # makes sure the data directory exists but is empty
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
        cls.cache_dir.mkdir()

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
        catchment_geometry = geometry.CatchmentGeometry(catchment_dir,
                                                        cls.instructions['instructions']['projection'],
                                                        cls.instructions['instructions']['grid_params']['resolution'])
        catchment_geometry.land = catchment_dir  # all land

        # Run pipeline - download files
        cls.runner = vector_fetch.Linz(cls.instructions['instructions']['apis']['linz']['key'],
                                       catchment_geometry, verbose=True)
        cls.catchment_geometry = catchment_geometry

    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory. """

        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)

    def test_land(self):
        """ A test to check expected island is loaded """

        land = self.runner.run(self.instructions['instructions']['apis']['linz']['land']['layers'][0],
                               self.instructions['instructions']['apis']['linz']['land']['type'])

        # check various shape attributes match those expected
        self.assertEqual(land.geometry.area.sum(), self.LAND['area'], "The area of the returned land polygon " +
                         f"`{land.geometry.area.sum()}` differs from the expected {self.LAND['area']}")
        self.assertEqual(land.geometry.length.sum(), self.LAND['length'], "The length of the returned land polygon " +
                         f"`{land.geometry.length.sum()}` differs from the expected {self.LAND['length']}")
        self.assertEqual(land.loc[0].geometry.geometryType(), self.LAND['geometryType'], "The geometryType of the " +
                         f"returned land polygon `{land.loc[0].geometry.geometryType()}` differs from the expected " +
                         f"{self.LAND['length']}")

    def test_bathymetry(self):
        """ A test to check expected bathyemtry contours are loaded """

        bathymetry_contours = self.runner.run(
            self.instructions['instructions']['apis']['linz']['bathymetry_contours']['layers'][0],
            self.instructions['instructions']['apis']['linz']['bathymetry_contours']['type'])

        # check various shape attributes match those expected
        self.assertEqual(bathymetry_contours.geometry.area.sum(), self.BATHYMETRY_CONTOURS['area'], "The area of the " +
                         f"returned bathymetry_contours polygon `{bathymetry_contours.geometry.area.sum()}` differs " +
                         "from the expected {self.BATHYMETRY_CONTOURS['area']}")
        self.assertEqual(bathymetry_contours.geometry.length.sum(), self.BATHYMETRY_CONTOURS['length'], "The area of " +
                         f"the returned bathymetry_contours polygon `{bathymetry_contours.geometry.length.sum()}` " +
                         "differs from the expected {self.BATHYMETRY_CONTOURS['length']}")
        self.assertEqual(bathymetry_contours.loc[0].geometry.geometryType(), self.BATHYMETRY_CONTOURS['geometryType'],
                         "The geometryType of the returned land polygon " +
                         f"`{bathymetry_contours.loc[0].geometry.geometryType()}` differs from the expected " +
                         f"{self.BATHYMETRY_CONTOURS['length']}")


if __name__ == '__main__':
    unittest.main()
