# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import pathlib
import geopandas
import sys
import pytest
import numpy

from src.geofabrics import processor
from tests import base_test


class Test(base_test.Test):
    """A class to test the basic measured river interpolation functionality
    contained in processor.MeasuredRiverGenerator.

    Tests run include:
        1. test_river_polygon(linux/windows) - Test that the expected river polygon is
        created
        2. test_river_elevations(linux/windows) - Test that the expected river
        bathymetry is created
    """

    @classmethod
    def setUpClass(cls):
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # Run pipeline - download files and generated DEM
        runner = processor.MeasuredRiverGenerator(
            cls.instructions["measured"], debug=False
        )
        runner.run()

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is less strict"
    )
    def test_river_polygon_windows(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon  - All OS")

        data_path_instructions = self.instructions["measured"]["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["extents"]
        )
        decimal_threshold = 6

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.sum()
        benchmark_comparison = benchmark[column_name].area.sum()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux test - this is strict")
    def test_river_polygon_linux(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon  - All OS")

        data_path_instructions = self.instructions["measured"]["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["extents"]
        )
        decimal_threshold = 6

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.sum()
        benchmark_comparison = benchmark[column_name].area.sum()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is less strict"
    )
    def test_river_elevations_windows(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - Windows")

        data_path_instructions = self.instructions["measured"]["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_elevations.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["elevations"]
        )
        decimal_threshold = 6

        # check some of the bathymetry columns match
        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
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
            places=decimal_threshold,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. They are separated by "
            f"distances of {comparison}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is more strict"
    )
    def test_river_elevations_linux(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - Linux")

        data_path_instructions = self.instructions["measured"]["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_elevations.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["benchmark"]["elevations"]
        )
        decimal_threshold = 6

        # check some of the bathymetry columns match
        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
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
            places=decimal_threshold,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. They are separated by "
            f"distances of {comparison}",
        )


if __name__ == "__main__":
    unittest.main()
