# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import geopandas
import shutil
import dotenv
import os
import sys
import pytest
import logging
import numpy

from src.geofabrics import processor


class ProcessorRiverBathymetryOsmTest(unittest.TestCase):
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
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_river_bathymetry_osm_wellington"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_river_bathymetry_osm_wellington.py")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["apis"]["vector"]["linz"]["key"] = linz_key

        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # Run pipeline - download files and generated DEM
        runner = processor.RiverBathymetryGenerator(cls.instructions, debug=False)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """Remove created cache directory and included created and downloaded files at
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

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_river_polygon_windows(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon  - All OS")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["river_polygon_benchmark"]
        )

        # check the polygons match
        self.assertTrue(
            (test == benchmark).all().all(),
            "The geneated river"
            f"polygon {test} doesn't equal the river benchmark "
            f"river polygon {benchmark}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_river_polygon_linux(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated first."""

        print("Compare river polygon  - All OS")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["river_polygon_benchmark"]
        )

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=6,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_river_bathymetry_windows(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - Windows")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["river_bathymetry_benchmark"]
        )

        # check the bathymetries match
        self.assertTrue(
            (test == benchmark).all().all(),
            "The geneated river"
            f"bathymetry {test} doesn't equal the river benchmark "
            f"river bathymetry {benchmark}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_river_bathymetry_linux(self):
        """A test to see if the correct river polygon is generated. This is
        tested individually as it is generated on its own."""

        print("Compare river bathymetry - Linux")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "river_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["river_bathymetry_benchmark"]
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
            places=7,
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
            places=7,
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
            places=7,
            msg=f"The geneated river {column_name} does not"
            f" match the benchmark. They are separated by "
            f"distances of {comparison}",
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_fan_windows(self):
        """A test to see if the correct fan polygon and bathymetry are
        generated. These are generated and tested together."""

        print("Compare fan bathymetry and polygon - Windows")

        data_path_instructions = self.instructions["data_paths"]

        # Compare the polygons
        test = geopandas.read_file(self.results_dir / "fan_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["fan_polygon_benchmark"]
        )

        self.assertTrue(
            (test == benchmark).all().all(),
            "The geneated river"
            f"fan {test} doesn't equal the river benchmark "
            f"fan polygon {benchmark}",
        )

        # Compare the bathymetries
        test = geopandas.read_file(self.results_dir / "fan_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["fan_bathymetry_benchmark"]
        )

        self.assertTrue(
            (test == benchmark).all().all(),
            "The geneated fan"
            f"bathymetry {test} doesn't equal the fan benchmark "
            f"fan bathymetry {benchmark}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_fan_linux(self):
        """A test to see if the correct fan polygon and bathymetry are
        generated. These are generated and tested together."""

        print("Compare fan bathymetry and polygon - Linux")

        data_path_instructions = self.instructions["data_paths"]

        # Compare the polygons
        test = geopandas.read_file(self.results_dir / "fan_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["fan_polygon_benchmark"]
        )

        # check some of the bathymetry columns match
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=5,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

        # Compare the bathymetries
        test = geopandas.read_file(self.results_dir / "fan_bathymetry.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["fan_bathymetry_benchmark"]
        )

        # check some of the bathymetry columns match
        column_name = "bed_elevation_Neal_et_al"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            "{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=7,
            msg="The geneated  river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

        column_name = "bed_elevation_Rupp_and_Smart"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"{column_name} difference "
            "{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=7,
            msg="The geneated  river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )

        column_name = "geometry"
        comparison = test[column_name].distance(benchmark[column_name]).array
        print(
            f"Distances between the test and benchmark points {numpy.array(comparison)}"
        )
        self.assertAlmostEqual(
            comparison,
            numpy.zeros(len(test[column_name])),
            places=7,
            msg=f"The geneate driver {column_name}"
            f" does not match the benchmark. They are"
            f" separated by distances of {comparison}",
        )


if __name__ == "__main__":
    unittest.main()
