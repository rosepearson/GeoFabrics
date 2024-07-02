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
import logging
import gc

from src.geofabrics import processor


class Test(unittest.TestCase):
    """A class to test the basic DemGenerator processor class for a simple example with
    land, offshore bathymetry, and LiDAR using the data generated in the set-up routine
    and referenced in the instruction.json.

    The dem.DenseDem.CACHE_SIZE is exceded and the offshore sampled points are
    re-sampled at a lower resolution.

    Tests run include:
        1. test_result_dem  Check the generated DEM matches the benchmark DEM exactly in
           Windows, or approximately in Linux. It aims to ensure the offshore resolution
           is reduced.
    """

    LAS_GROUND = 2

    @classmethod
    def setUpClass(cls):
        """Create a cache directory and CatchmentGeometry object for use in the tests
        and also download the files used in the tests."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_dem_generation_local_2"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_processor_local_files_limited_offshore.py")

        # Load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # Generate catchment data
        catchment_file = cls.results_dir / "catchment_boundary.geojson"
        x0 = 0
        x1 = 3000
        y0 = -5
        y1 = 1
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        catchment.to_file(catchment_file)

        # Generate land data - wider than catchment
        land_file = cls.results_dir / "land.geojson"
        x0 = -50
        x1 = 3050
        y0 = 0
        y1 = 5
        land = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        land = geopandas.GeoSeries([land])
        land = land.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        land.to_file(land_file)

        # Generate bathymetry data - wider than catchment
        bathymetry_file = cls.results_dir / "bathymetry.geojson"
        x0 = -50
        x1 = 3050
        y0 = -1
        y1 = -3
        y2 = -5
        contour_0 = shapely.geometry.LineString(
            [(x0, y0, -y0 * 0.1), ((x0 + x1) / 2, y0, 0), (x1, y0, -y0 * 0.1)]
        )
        contour_1 = shapely.geometry.LineString(
            [(x0, y1, -y1 * 0.1), ((x0 + x1) / 2, y1, 0), (x1, y1, -y1 * 0.1)]
        )
        contour_2 = shapely.geometry.LineString(
            [(x0, y2, -y2 * 0.1), ((x0 + x1) / 2, y2, 0), (x1, y2, -y2 * 0.1)]
        )
        contours = geopandas.GeoSeries([contour_0, contour_1, contour_2])
        contours = contours.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        contours.to_file(bathymetry_file)

        # Create LiDAR - wider than catchment
        lidar_file = cls.results_dir / "lidar.laz"
        x0 = -50
        x1 = 3050
        y0 = (
            -5
        )  # drop offshore LiDAR is True in the instruction file so only LiDAR in the
        # foreshore will be kept
        y1 = 5
        dxy = 0.1
        grid_lidar_x, grid_lidar_y = numpy.meshgrid(
            numpy.arange(x0, x1, dxy), numpy.arange(y0, y1, dxy)
        )
        grid_lidar_z = grid_lidar_y / 10
        lidar_array = numpy.empty(
            [len(grid_lidar_x.flatten())],
            dtype=[("X", "<f8"), ("Y", "<f8"), ("Z", "<f8"), ("Classification", "u1")],
        )
        lidar_array["X"] = grid_lidar_x.flatten()
        lidar_array["Y"] = grid_lidar_y.flatten()
        lidar_array["Z"] = grid_lidar_z.flatten()
        lidar_array["Classification"] = cls.LAS_GROUND

        pdal_pipeline_instructions = [
            {
                "type": "writers.las",
                "a_srs": f"EPSG:{cls.instructions['output']['crs']['horizontal']}+"
                + f"{cls.instructions['output']['crs']['vertical']}",
                "filename": str(lidar_file),
                "compression": "laszip",
            }
        ]

        pdal_pipeline = pdal.Pipeline(
            json.dumps(pdal_pipeline_instructions), [lidar_array]
        )
        pdal_pipeline.execute()

        # Run geofabrics processing pipeline
        runner = processor.RawLidarDemGenerator(cls.instructions)
        runner.run()
        runner = processor.HydrologicDemGenerator(cls.instructions)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """Remove created files in the cache directory as part of the testing process at
        the end of the test."""
        gc.collect()
        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """Remove all generated or downloaded files from the data directory"""

        assert cls.cache_dir.exists(), (
            "The data directory that should include the comparison benchmark dem file "
            "doesn't exist"
        )

        # Cycle through all folders within the cache dir deleting their contents
        for path in cls.cache_dir.iterdir():
            if path.is_dir():
                for file in path.glob("*"):  # only files
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                shutil.rmtree(path)

    def test_result_dem(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in result DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare DEMs - load from file as rioxarray.rioxarray.open_rasterio ignores
        # index order
        diff_array = test_dem.z.data - benchmark_dem.z.data
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_equal(
            test_dem.z.data,
            benchmark_dem.z.data,
            err_msg="The generated result_dem differs from the benchmark_dem",
        )

        # Compare DEMs data source classification
        diff_array = test_dem.data_source.data - benchmark_dem.data_source.data
        logging.info(f"DEM z array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.data_source.data,
            benchmark_dem.data_source.data,
            err_msg="The generated test data_source layer has different data "
            "from the benchmark",
        )

        # Compare DEMs lidar source classification
        diff_array = test_dem.lidar_source.data - benchmark_dem.lidar_source.data
        logging.info(f"DEM z array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.lidar_source.data,
            benchmark_dem.lidar_source.data,
            err_msg="The generated test lidar_source layer has different data "
            "from the benchmark",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem


if __name__ == "__main__":
    unittest.main()
