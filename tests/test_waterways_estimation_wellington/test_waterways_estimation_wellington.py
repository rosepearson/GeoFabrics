# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import geopandas
import shapely
import shutil
import dotenv
import os
import sys
import pytest
import logging
import numpy
import gc

from src.geofabrics import processor


class Test(unittest.TestCase):
    """A class to test the basic waterway elevation estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_open_waterways_windows - Test the expected waterways are created in
        windows
        2. test_open_waterways_linux - Test the expected waterways are created in linux
    """

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_waterways_estimation_wellington"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_waterways_estimation_wellington")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["datasets"]["vector"]["linz"]["key"] = linz_key

        # define the cache directory location - from the instruction file
        path_instructions = cls.instructions["data_paths"]
        cls.cache_dir = pathlib.Path(path_instructions["local_cache"])

        # Remove any files from last test, then create a results directory
        cls.results_dir = cls.cache_dir / path_instructions["subfolder"]
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # create fake catchment boundary
        x0 = 1770797
        y0 = 5472879
        x1 = 1770969
        y1 = 5472707
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner = processor.WaterwayBedElevationEstimator(cls.instructions, debug=False)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """Remove created cache directory and included created and downloaded files at
        the end of the test."""

        gc.collect()
        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """Remove all generated or downloaded files from the data directory,
        but with only warnings if files can't be removed."""

        assert cls.cache_dir.exists(), (
            "The data directory that should include the comparison benchmark dem file "
            "doesn't exist"
        )

        # Cycle through all folders within the cache dir deleting their contents
        for path in cls.cache_dir.iterdir():
            if path.is_dir():
                for file in path.glob("*"):  # only files
                    if file.is_file():
                        try:
                            file.unlink()
                        except (Exception, PermissionError) as caught_exception:
                            logging.warning(
                                f"Caught error {caught_exception} during "
                                f"rmtree of {file}. Supressing error. You "
                                "will have to manually delete."
                            )
                    elif file.is_dir():
                        try:
                            shutil.rmtree(file)
                        except (Exception, PermissionError) as caught_exception:
                            logging.warning(
                                f"Caught error {caught_exception} during "
                                f"rmtree of {file}. Supressing error. You "
                                "will have to manually delete."
                            )
                try:
                    shutil.rmtree(path)
                except (Exception, PermissionError) as caught_exception:
                    logging.warning(
                        f"Caught error {caught_exception} during rmtree of "
                        f"{path}. Supressing error. You will have to manually "
                        "delete."
                    )

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is less strict"
    )
    def test_open_waterways_windows(self):
        """A test to see if the correct open waterways polygon and bathymetry are
        generated."""

        decimal_places = 5
        delta = 40
        print(f"Compare river polygon - with tolerance {delta}")

        data_path_instructions = self.instructions["data_paths"]

        test = geopandas.read_file(self.results_dir / "open_waterways_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["open_benchmark"]["extents"]
        )

        # check the polygons match
        column_name = "geometry"
        test_comparison = test[column_name].area.sum()
        benchmark_comparison = benchmark[column_name].area.sum()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            delta=delta,
            msg=f"The geneated open waterways polygon {column_name} does not match the "
            f"benchmark. {test_comparison} vs {benchmark_comparison}",
        )

        print(f"Compare open waterways bathymetry - with tolerance {decimal_places}")

        test = geopandas.read_file(
            self.results_dir / "open_waterways_elevation.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["open_benchmark"]["elevations"]
        )

        # check the polygons match closely
        column_name = "geometry"
        test_comparison = test[column_name].area.sum()
        benchmark_comparison = benchmark[column_name].area.sum()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The geneated river {column_name} does"
            f" not match the benchmark. {test_comparison} "
            f"vs {benchmark_comparison}",
        )
        # check some of the bathymetry columns match
        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        # Temporary simple max test
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_places,
            msg=f"The maximum open waterways bathymetry {column_name} does not"
            f" match the benchmark. {test_comparison.max()} vs "
            f"{benchmark_comparison.max()}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_open_waterways_linux(self):
        """A test to see if the correct open waterways polygon and bathymetry are
        generated."""

        data_path_instructions = self.instructions["data_paths"]

        print("Compare river polygon  - Linux")

        # Compare the polygons
        test = geopandas.read_file(self.results_dir / "open_waterways_polygon.geojson")
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["open_benchmark"]["extents"]
        )

        # check the polygons match
        column_name = "geometry"
        test_comparison = test[column_name].area.sum()
        benchmark_comparison = benchmark[column_name].area.sum()
        print(f"test area {test_comparison}, and benchmark area {benchmark_comparison}")
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            delta=50,
            msg=f"The geneated open waterways polygon {column_name} does not match the "
            f"benchmark. {test_comparison} vs {benchmark_comparison}",
        )

        print("Compare open waterways bathymetry - Linux")

        test = geopandas.read_file(
            self.results_dir / "open_waterways_elevation.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["open_benchmark"]["elevations"]
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
        # check some of the bathymetry columns match
        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        # Temporary simple max test
        self.assertAlmostEqual(
            test_comparison.max(),
            benchmark_comparison.max(),
            places=1,
            msg=f"The maximum open waterways bathymetry {column_name} does not"
            f" match the benchmark. {test_comparison.max()} vs "
            f"{benchmark_comparison.max()}",
        )

        # TODO - trun back on more robust comparison
        """print(
            f"Open waterways elevation {column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=7,
            msg=f"The geneated open waterways bathymetry {column_name} does not"
            f" match the benchmark. {test_comparison} vs {benchmark_comparison}",
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
        )"""

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is less strict"
    )
    def test_closed_waterways_windows(self):
        """A test to see if the correct close waterways polygon and bathymetry are
        generated."""

        print("Compare closed waterways - Windows")

        data_path_instructions = self.instructions["data_paths"]

        # Compare the polygons
        test = geopandas.read_file(
            self.results_dir / "closed_waterways_polygon.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["closed_benchmark"]["extents"]
        )
        decimal_threshold = 6
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(
            f"Closed waterways polygon test area {test_comparison}, and benchmark area "
            f"{benchmark_comparison}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
            msg=f"The geneated closed waterways polygon {column_name} does not match the"
            f" benchmark. {test_comparison} vs {benchmark_comparison}",
        )

        # Compare the bathymetries
        test = geopandas.read_file(
            self.results_dir / "closed_waterways_elevation.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["closed_benchmark"]["elevations"]
        )

        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"Close waterways bathymetry {column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=decimal_threshold,
            msg="The geneated closed waterways bathymetry {column_name} does not match "
            f"the benchmark. {test_comparison} vs {benchmark_comparison}",
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
            msg=f"The geneated closed waterways bathymetry {column_name} does not match the"
            f"benchmark. They are separated by distances of {comparison}",
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_closed_waterways_linux(self):
        """A test to see if the correct close waterways polygon and bathymetry are
        generated."""

        print("Compare close waterways - Linux")

        data_path_instructions = self.instructions["data_paths"]

        # Compare the polygons
        test = geopandas.read_file(
            self.results_dir / "closed_waterways_polygon.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["closed_benchmark"]["extents"]
        )

        # check some of the bathymetry columns match
        column_name = "geometry"
        test_comparison = test[column_name].area.item()
        benchmark_comparison = benchmark[column_name].area.item()
        print(
            f"Closed waterways polygon test area {test_comparison}, and benchmark area "
            f"{benchmark_comparison}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=6,
            msg=f"The geneated closed waterways polygon {column_name} does not match the"
            f" benchmark. {test_comparison} vs {benchmark_comparison}",
        )

        # Compare the bathymetries
        test = geopandas.read_file(
            self.results_dir / "closed_waterways_elevation.geojson"
        )
        benchmark = geopandas.read_file(
            self.cache_dir / data_path_instructions["closed_benchmark"]["elevations"]
        )

        # check some of the bathymetrt columns match
        column_name = "z"
        test_comparison = test[column_name].array
        benchmark_comparison = benchmark[column_name].array
        print(
            f"Close waterways bathymetry {column_name} difference "
            f"{numpy.array(test_comparison) - numpy.array(benchmark_comparison)}"
        )
        self.assertAlmostEqual(
            test_comparison,
            benchmark_comparison,
            places=7,
            msg="The geneated closed waterways bathymetry {column_name} does not match "
            f"the benchmark. {test_comparison} vs {benchmark_comparison}",
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
            msg=f"The geneated closed waterways bathymetry {column_name} does not match the"
            f"benchmark. They are separated by distances of {comparison}",
        )


if __name__ == "__main__":
    unittest.main()
