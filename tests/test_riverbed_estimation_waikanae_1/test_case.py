# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import pathlib
import geopandas
import dotenv
import os
import sys
import pytest
import numpy

from geofabrics import processor
from tests import base_test


class Test(base_test.Test):
    """A class to test the basic river bathymetry estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_river_polygon_(linux/windows) - Test that the expected river polygon is
        created
        2. test_river_bathymetry_(linux/windows) - Test that the expected river
        bathymetry is created
        3. test_fan_(linux/windows) - Test that the expected fan polygon and geometry
        are created
    """

    @classmethod
    def setUpClass(cls):
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["datasets"]["vector"]["linz"]["key"] = linz_key

        # Run pipeline - download files and generated DEM
        runner = processor.RiverBathymetryGenerator(cls.instructions, debug=False)
        runner.run()

    @pytest.mark.skipif(sys.platform != "win32", reason="Test windows - less strict")
    def test_river_polygon_to_tolerance(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon")
        decimal_places = 2
        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["extents"]
        )

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

    @pytest.mark.skipif(sys.platform != "linux", reason="Test linux - strict")
    def test_river_polygon_strict(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon")
        decimal_places = 3
        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["extents"]
        )

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

    @pytest.mark.skipif(
        sys.platform != "win32" and sys.platform != "linux",
        reason="Test windows - this is less",
    )
    def test_river_bathymetry_to_tolerance(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - with tolerance")
        decimal_places = 6
        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["elevations"]
        )

        # check some of the bathymetry columns match
        column_name = "bed_elevation_Neal_et_al"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. {test_comparison} vs "
            f"{benchmark_comparison}",
        )
        column_name = "width"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. {test_comparison} vs "
            f"{benchmark_comparison}",
        )

        column_name = "geometry"
        comparison = test[column_name].distance(benchmark[column_name]).array
        print(
            f"Distances between the test and benchmark points {numpy.array(comparison)}"
        )
        self.assertAlmostEqual(
            comparison,
            numpy.zeros(len(test[column_name])),
            places=decimal_places,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. They are separated by "
            f"distances of {comparison}",
        )

    @pytest.mark.skipif(True, reason="Skip")
    def test_river_bathymetry_strict(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - Strict")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["elevations"]
        )

        # check the bathymetries match - exclude other columns where there may be NaN
        comparison = (
            (test == benchmark)[
                ["bed_elevation_Neal_et_al", "bed_elevation_Rupp_and_Smart"]
            ]
            .all()
            .all()
        )
        self.assertTrue(
            comparison,
            "The geneated river"
            f"bathymetry {test} doesn't equal the river benchmark "
            f"river bathymetry {benchmark}: {comparison}",
        )


if __name__ == "__main__":
    unittest.main()
