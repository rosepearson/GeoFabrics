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
import xarray
import numpy
import shapely
import geopandas
import pdal
import logging
import gc

from src.geofabrics import processor


class ProcessorLocalFilesTest(unittest.TestCase):
    """Tests the basic DemGenerator processor class for a simple example with land,
    offshore, a reference DEM and
    LiDAR using the data specified in the instruction.json

    Tests run include:
        1. test_result_dem  Check the generated DEM matches the benchmark DEM within a
        tolerance
    """

    LAS_GROUND = 2

    @classmethod
    def setUpClass(cls):
        """Create a cache directory and CatchmentGeometry object for use in the tests
        and also download the files used
        in the tests."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_processor_local_files_with_roughness"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_processor_local_files_with_roughness.py")

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
        x0 = 250
        y0 = -250
        x1 = 1250
        y1 = 750
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        catchment.to_file(catchment_file)

        # Generate land data
        land_file = cls.results_dir / "land.geojson"
        x0 = 0
        y0 = 0
        x1 = 1500
        y1 = 1000
        land = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        land = geopandas.GeoSeries([land])
        land = land.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        land.to_file(land_file)

        # Generate bathymetry data
        bathymetry_file = cls.results_dir / "bathymetry.geojson"
        x0 = 0
        x1 = 1500
        y0 = -50
        y1 = -100
        y2 = -200
        contour_0 = shapely.geometry.LineString(
            [(x0, y0, -y0 / 10), (x1, y0, -y0 / 10)]
        )
        contour_1 = shapely.geometry.LineString(
            [(x0, y1, -y1 / 10), (x1, y1, -y1 / 10)]
        )
        contour_2 = shapely.geometry.LineString(
            [(x0, y2, -y2 / 10), (x1, y2, -y2 / 10)]
        )
        contours = geopandas.GeoSeries([contour_0, contour_1, contour_2])
        contours = contours.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        contours.to_file(bathymetry_file)

        # Create a reference DEM
        dem_file = cls.results_dir / "reference_dem.nc"
        dxy = 15
        grid_dem_x, grid_dem_y = numpy.meshgrid(
            numpy.arange(200, 1300, dxy), numpy.arange(-25, 800, dxy)
        )
        grid_dem_z = numpy.zeros_like(grid_dem_x, dtype=numpy.float64)
        grid_dem_z[grid_dem_y < 0] = grid_dem_y[grid_dem_y < 0] / 10
        grid_dem_z[grid_dem_y > 0] = (
            grid_dem_y[grid_dem_y > 0]
            / 10
            * (numpy.abs(grid_dem_x[grid_dem_y > 0] - 750) / 500 + 0.1)
            / 1.1
        )
        dem = xarray.DataArray(
            grid_dem_z,
            coords={"x": grid_dem_x[0], "y": grid_dem_y[:, 0]},
            dims=["y", "x"],
            attrs={"scale_factor": 1.0, "add_offset": 0.0},
        )
        dem.rio.write_crs(
            cls.instructions["output"]["crs"]["horizontal"],
            inplace=True,
        )
        dem.rio.write_transform(inplace=True)
        dem.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
        dem.name = "z"
        dem.to_netcdf(dem_file, format="NETCDF4", engine="netcdf4")

        # Create LiDAR
        lidar_file = cls.results_dir / "lidar.laz"
        dxy = 1
        grid_lidar_x, grid_lidar_y = numpy.meshgrid(
            numpy.arange(500, 1000, dxy), numpy.arange(-25, 475, dxy)
        )
        # ground points
        grid_lidar_z = numpy.zeros_like(grid_lidar_x, dtype=numpy.float64)
        grid_lidar_z[grid_lidar_y < 0] = grid_lidar_y[grid_lidar_y < 0] / 10
        grid_lidar_z[grid_lidar_y > 0] = (
            grid_lidar_y[grid_lidar_y > 0]
            / 10
            * (numpy.abs(grid_lidar_x[grid_lidar_y > 0] - 750) / 500 + 0.1)
            / 1.1
        )
        # trees
        # scrub
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
                "a_srs": f"EPSG:"
                f"{cls.instructions['output']['crs']['horizontal']}+"
                f"{cls.instructions['output']['crs']['vertical']}",
                "filename": str(lidar_file),
                "compression": "laszip",
            }
        ]

        pdal_pipeline = pdal.Pipeline(
            json.dumps(pdal_pipeline_instructions), [lidar_array]
        )
        pdal_pipeline.execute()

        # Run pipeline
        runner = processor.RawLidarDemGenerator(cls.instructions)
        runner.run()
        runner = processor.HydrologicDemGenerator(cls.instructions)
        runner.run()
        runner = processor.RoughnessLengthGenerator(cls.instructions)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """Remove created files in the cache directory as part of the testing process at
        the end of the test."""

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
            benchmark_dem = benchmark_dem.squeeze("band", drop=True)
        # Load in result DEM
        file_path = (
            self.results_dir / self.instructions["data_paths"]["result_geofabric"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem = test_dem.squeeze("band", drop=True)
        # Compare DEMs z - load both from file as rioxarray.rioxarray.open_rasterio
        # ignores index order
        diff_array = test_dem.z.data - benchmark_dem.z.data
        logging.info(f"DEM z array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data,
            benchmark_dem.z.data,
            err_msg="The generated result_dem z has different data from the "
            "benchmark_dem",
        )

        # Compare DEMs source classification
        diff_array = test_dem.source_class.data - benchmark_dem.source_class.data
        logging.info(f"DEM source class array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.source_class.data,
            benchmark_dem.source_class.data,
            err_msg="The generated result_dem source_class has different data "
            "from the benchmark_dem",
        )

        # Compare DEMs roughness
        diff_array = test_dem.zo.data - benchmark_dem.zo.data
        logging.info(f"DEM zo array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.source_class.data,
            benchmark_dem.source_class.data,
            err_msg="The generated result_dem zo has different data "
            "from the benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem
        gc.collect()


if __name__ == "__main__":
    unittest.main()
