# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import shutil
import rioxarray
import numpy
import shapely
import geopandas
import pdal
import pytest
import sys

from src.geofabrics import processor


class ProcessorLocalFilesTest(unittest.TestCase):
    """ A class to test the basic DEM generation pipeline for a simple example with land, offshore bathymetry, and
    LiDAR using the data  generated in the set-up routine and referenced in instruction.json

    The dem.DenseDem.CACHE_SIZE is exceded and the offshore sampled points are re-sampled at a lower resolution.

    Tests run include:
        1. test_result_dem  Check the generated DEM matches the benchmark DEM exactly in Windows, or approximately in
        Linux. It aims to ensure the offshore resolution is reduced.
    """

    LAS_GROUND = 2

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the tests and also download the files used
        in the tests. """

        test_path = pathlib.Path().cwd() / pathlib.Path("tests/test_processor_local_files_limited_offshore_res")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, 'r') as file_pointer:
            cls.instructions = json.load(file_pointer)

        # remove any files from last test in the cache directory
        cls.cache_dir = test_path / "data"
        assert cls.cache_dir.exists(), "The data directory that should include the comparison benchmark dem file " + \
            "doesn't exist"
        cls.tearDownClass()

        # generate catchment data
        catchment_dir = cls.cache_dir / "catchment_boundary"
        x0 = 0
        y0 = -5
        x1 = 3000
        y1 = 1
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions['instructions']['output']['crs']['horizontal'])
        catchment.to_file(catchment_dir)
        shutil.make_archive(base_name=catchment_dir, format='zip', root_dir=catchment_dir)
        shutil.rmtree(catchment_dir)

        # generate land data - wider than catchment
        land_dir = cls.cache_dir / "land"
        x0 = -50
        y0 = 0
        x1 = 3050
        y1 = 5
        land = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        land = geopandas.GeoSeries([land])
        land = land.set_crs(cls.instructions['instructions']['output']['crs']['horizontal'])
        land.to_file(land_dir)
        shutil.make_archive(base_name=land_dir, format='zip', root_dir=land_dir)
        shutil.rmtree(land_dir)

        # generate bathymetry data - wider than catchment
        bathymetry_dir = cls.cache_dir / "bathymetry"
        x0 = -50
        x1 = 3050
        y0 = -1
        y1 = -3
        y2 = -5
        contour_0 = shapely.geometry.LineString([(x0, y0, -y0 * 0.1), ((x0 + x1) / 2, y0, 0), (x1, y0, -y0 * 0.1)])
        contour_1 = shapely.geometry.LineString([(x0, y1, -y1 * 0.1), ((x0 + x1) / 2, y1, 0), (x1, y1, -y1 * 0.1)])
        contour_2 = shapely.geometry.LineString([(x0, y2, -y2 * 0.1), ((x0 + x1) / 2, y2, 0), (x1, y2, -y2 * 0.1)])
        contours = geopandas.GeoSeries([contour_0, contour_1, contour_2])
        contours = contours.set_crs(cls.instructions['instructions']['output']['crs']['horizontal'])
        contours.to_file(bathymetry_dir)
        shutil.make_archive(base_name=bathymetry_dir, format='zip', root_dir=bathymetry_dir)
        shutil.rmtree(bathymetry_dir)

        # create LiDAR - wider than catchment
        lidar_file = cls.cache_dir / "lidar.laz"
        x0 = -50
        y0 = -5  # drop offshore LiDAR is True in the instruction file so only LiDAR in the foreshore will be kept
        x1 = 3050
        y1 = 5
        dxy = 0.1
        grid_lidar_x, grid_lidar_y = numpy.meshgrid(numpy.arange(x0, x1, dxy), numpy.arange(y0, y1, dxy))
        grid_lidar_z = grid_lidar_y / 10
        lidar_array = numpy.empty([len(grid_lidar_x.flatten())], dtype=[('X', '<f8'), ('Y', '<f8'), ('Z', '<f8'),
                                                                        ('Classification', 'u1')])
        lidar_array['X'] = grid_lidar_x.flatten()
        lidar_array['Y'] = grid_lidar_y.flatten()
        lidar_array['Z'] = grid_lidar_z.flatten()
        lidar_array['Classification'] = cls.LAS_GROUND

        pdal_pipeline_instructions = [
            {"type":  "writers.las",
             "a_srs": f"EPSG:{cls.instructions['instructions']['output']['crs']['horizontal']}+" +
             f"{cls.instructions['instructions']['output']['crs']['vertical']}",
             "filename": str(lidar_file),
             "compression": "laszip"}
        ]

        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [lidar_array])
        pdal_pipeline.execute()

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows test - this is strict")
    def test_result_dem_windows(self):
        """ A basic comparison between the generated and benchmark DEM """

        # Run pipeline
        runner = processor.DemGenerator(self.instructions)
        runner.run()

        # load in benchmark DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # load in result DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['result_dem'],
                                               masked=True) as saved_dem:
            saved_dem.load()

        # compare DEMs - load from file as rioxarray.rioxarray.open_rasterio ignores index order
        diff_array = saved_dem.data-benchmark_dem.data
        print(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_equal(saved_dem.data, benchmark_dem.data,
                                         err_msg="The generated result_dem has different data from the benchmark_dem")

    @pytest.mark.skipif(sys.platform != 'linux', reason="Linux test - this is less strict")
    def test_result_dem_test(self):
        """ A basic comparison between the generated and benchmark DEM """

        # Run pipeline
        runner = processor.DemGenerator(self.instructions)
        runner.run()

        # load in benchmark DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # load in result DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['result_dem'],
                                               masked=True) as saved_dem:
            saved_dem.load()

        # compare DEMs - load from file as rioxarray.rioxarray.open_rasterio ignores index order
        diff_array = saved_dem.data-benchmark_dem.data
        print(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(saved_dem.data, benchmark_dem.data,
                                                err_msg="The generated result_dem has different data from the " +
                                                "benchmark_dem")

    @classmethod
    def tearDownClass(cls):
        """ Remove created files in the cache directory as part of the testing process at the end of the test. """

        benchmark_file = cls.cache_dir / "benchmark_dem.nc"
        for file in cls.cache_dir.glob('*'):
            if file != benchmark_file:
                file.unlink()


if __name__ == '__main__':
    unittest.main()
