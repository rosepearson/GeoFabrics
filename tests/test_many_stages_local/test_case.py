# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import rioxarray
import xarray
import numpy
import shapely
import geopandas
import pdal
import logging

from geofabrics import runner
from tests import base_test


class Test(base_test.Test):
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
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # Generate catchment data
        catchment_file = cls.results_dir / "catchment_boundary.geojson"
        x0 = 250
        y0 = -250
        x1 = 1250
        y1 = 750
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(
            cls.instructions["default"]["output"]["crs"]["horizontal"]
        )
        catchment.to_file(catchment_file)

        # Generate land data
        land_file = cls.results_dir / "land.geojson"
        x0 = 0
        y0 = 0
        x1 = 1500
        y1 = 1000
        land = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        land = geopandas.GeoSeries([land])
        land = land.set_crs(cls.instructions["default"]["output"]["crs"]["horizontal"])
        land.to_file(land_file)

        # Generate bathymetry data
        bathymetry_file = cls.results_dir / "bathymetry.geojson"
        x0 = 0
        x1 = 1500
        y0 = -50
        y1 = -100
        y2 = -200
        y3 = -300
        contour_0 = shapely.geometry.LineString(
            [(x0, y0, -y0 / 10), (x1, y0, -y0 / 10)]
        )
        contour_1 = shapely.geometry.LineString(
            [(x0, y1, -y1 / 10), (x1, y1, -y1 / 10)]
        )
        contour_2 = shapely.geometry.LineString(
            [(x0, y2, -y2 / 10), (x1, y2, -y2 / 10)]
        )
        contour_3 = shapely.geometry.LineString(
            [(x0, y3, -y3 / 10), (x1, y3, -y3 / 10)]
        )
        contours = geopandas.GeoSeries([contour_0, contour_1, contour_2, contour_3])
        contours = contours.set_crs(
            cls.instructions["default"]["output"]["crs"]["horizontal"]
        )
        contours.to_file(bathymetry_file)

        # Create a reference DEM
        dem_file = cls.results_dir / "coarse_dem.nc"
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
            cls.instructions["default"]["output"]["crs"]["horizontal"],
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
            dtype=[
                ("X", "<f8"),
                ("Y", "<f8"),
                ("Z", "<f8"),
                ("Classification", "u1"),
            ],
        )
        lidar_array["X"] = grid_lidar_x.flatten()
        lidar_array["Y"] = grid_lidar_y.flatten()
        lidar_array["Z"] = grid_lidar_z.flatten()
        lidar_array["Classification"] = cls.LAS_GROUND

        pdal_pipeline_instructions = [
            {
                "type": "writers.las",
                "a_srs": f"EPSG:"
                f"{cls.instructions['default']['output']['crs']['horizontal']}+"
                f"{cls.instructions['default']['output']['crs']['vertical']}",
                "filename": str(lidar_file),
                "compression": "laszip",
            }
        ]

        pdal_pipeline = pdal.Pipeline(
            json.dumps(pdal_pipeline_instructions), [lidar_array]
        )
        pdal_pipeline.execute()

        # Run pipeline
        runner.from_instructions_dict(cls.instructions)

    def test_result_dem(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark DEM
        print(self.instructions)
        file_path = (
            self.cache_dir / self.instructions["default"]["data_paths"]["benchmark_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem = benchmark_dem.squeeze("band", drop=True)
        # Load in result DEM
        file_path = (
            self.results_dir
            / self.instructions["default"]["data_paths"]["result_geofabric"]
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

        # Compare DEMs roughness
        diff_array = test_dem.zo.data - benchmark_dem.zo.data
        logging.info(f"DEM zo array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.zo.data,
            benchmark_dem.zo.data,
            err_msg="The generated result_dem zo has different data "
            "from the benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem


if __name__ == "__main__":
    unittest.main()
