# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import numpy
import json
import pathlib
import shutil
from . import geometry
from . import lidar
import geoapis.lidar
import geoapis.vector
from . import dem


class GeoFabricsGenerator:
    """ A class executing a pipeline for creating geofabric derivatives.

    The pipeline is controlled by the contents of the json_instructions file.
    See the README.md for usage examples or GeoFabrics/tests/test1 for an example of usage and an instruction file.
    """

    def __init__(self, json_instructions: json):
        self.instructions = json_instructions

        self.catchment_geometry = None
        self.dense_dem = None
        self.reference_dem = None
        self.bathy_contours = None
        self.result_dem = None

    def get_instruction_path(self, key: str) -> str:
        """ Return the file path from the instruction file, or default if there is a default value and the local cache
        is specified. Raise an error if the key is not in the instructions. """

        defaults = {'temp_raster': "temp_dense_dem.tif", 'result_dem': "generated_dem.nc"}

        if key in self.instructions['instructions']['data_paths']:
            return self.instructions['instructions']['data_paths'][key]
        elif "local_cache" in self.instructions['instructions']['data_paths'] and key in defaults.keys():
            return pathlib.Path(self.instructions['instructions']['data_paths']['local_cache']) / defaults[key]
        else:
            assert False, f"Either the key `{key}` missing from data paths, or either the `local_cache` is not " + \
                f"specified in the data paths or the key is not specified in the defaults: {defaults}"

    def check_instruction_path(self, key: str) -> bool:
        """ Return True if the file path exists in the instruction file, or True if there is a default value and the
        local cache is specified. """

        defaults = ['temp_raster', 'result_dem']

        if key in self.instructions['instructions']['data_paths']:
            return True
        else:
            return 'local_cache' in self.instructions['instructions']['data_paths'] and key in defaults

    def get_resolution(self) -> float:
        """ Return the resolution from the instruction file. Raise an error if not in the instructions. """

        assert 'output' in self.instructions['instructions'], "'output' is not a key-word in the instructions. It " + \
            "should exist and is where the resolution and optionally the CRS of the output DEM is defined."
        assert 'resolution' in self.instructions['instructions']['output']['grid_params'], \
            "'resolution' is not a key-word in the instructions"
        return self.instructions['instructions']['output']['grid_params']['resolution']

    def get_crs(self, key: str) -> float:
        """ Return the CRS projection from the instruction file. Raise an error if 'output' is not in the instructions.
        If no 'crs' or 'horizontal' or 'vertical' values are specified then use the defaults."""

        defaults = {'horizontal': 2193, 'vertical': 7839}

        assert 'output' in self.instructions['instructions'], "'output' is not a key-word in the instructions. It " + \
            "should exist and is where the resolution and optionally the CRS of the output DEM is defined."

        if 'crs' not in self.instructions['instructions']['output']:
            assert key in defaults.keys(), "The CRS is not specified in the 'output' with the 'crs' key-word, and " + \
                f"the key: {key} is not included in the defaults {defaults}"
            return defaults[key]
        elif key in self.instructions['instructions']['output']['crs']:
            return self.instructions['instructions']['output']['crs'][key]
        elif key in defaults.keys():
            return defaults[key]
        else:
            assert False, f"The key `{key}` is both missing from instructions 'crs', and is not defined in the " + \
                f"{defaults}"

    def get_instruction_general(self, key: str):
        """ Return the general instruction from the instruction file or return the default value if not specified in
        the instruction file. Raise an error if the key is not in the instructions and there is no default value. """

        defaults = {'filter_lidar_holes_area': None, 'verbose': True, 'set_dem_shoreline': True,
                    'bathymetry_contours_z_label': None}

        assert key in defaults or key in self.instructions['instructions']['general'], f"The key: {key} is missing " \
            + "from the general instructions, and does not have a default value"
        if 'general' in self.instructions['instructions'] and key in self.instructions['instructions']['general']:
            return self.instructions['instructions']['general'][key]
        else:
            return defaults[key]

    def check_apis(self, key) -> bool:
        """ Check to see if APIs are included in the instructions and if the key is included in specified apis """

        if "apis" in self.instructions['instructions']:
            # 'apis' included instructions and Key included in the APIs
            return key in self.instructions['instructions']['apis']
        else:
            return False

    def check_vector(self, key) -> bool:
        """ Check to see if vector key (i.e. land, bathymetry_contours, etc) is included either as a file path, or
        within any of the vector API's (i.e. LINZ or LRIS). """

        data_services = ["linz", "lris"]  # This list will increase as geopais is extended to support more vector APIs

        if "data_paths" in self.instructions['instructions'] and key in self.instructions['instructions']['data_paths']:
            # Key included in the data paths
            return True
        elif "apis" in self.instructions['instructions']:
            for data_service in data_services:
                if data_service in self.instructions['instructions']['apis'] \
                        and key in self.instructions['instructions']['apis'][data_service]:
                    # Key is included in one or more of the data_service's APIs
                    return True
        else:
            return False

    def get_vector_paths(self, key, verbose: bool = False) -> list:
        """ Get the path to the vector key data included either as a file path or as a LINZ API. Return all paths
        where the vector key is specified. In the case that an API is specified ensure the data is fetched as well. """

        paths = []

        # Check the instructions for vector data specified as a data_paths
        if "data_paths" in self.instructions['instructions'] and key in self.instructions['instructions']['data_paths']:
            # Key included in the data paths - add - either list or individual path
            if type(self.instructions['instructions']['data_paths'][key]) == list:
                paths.extend(self.instructions['instructions']['data_paths'][key])
            else:
                paths.append(self.instructions['instructions']['data_paths'][key])

        # Check the instructions for LINZ hoster vector data
        data_services = {"linz": geoapis.vector.Linz, "lris": geoapis.vector.Lris}  # API name and geoapis class pairs
        for data_service in data_services.keys():
            if self.check_apis(data_service) and key in self.instructions['instructions']['apis'][data_service]:
                cache_dir = pathlib.Path(self.get_instruction_path('local_cache'))
                api_key = self.instructions['instructions']['apis'][data_service]['key']

                assert self.check_instruction_path('local_cache'), "Local cache file path must exist to specify the" + \
                    f" location to download vector data from the vector APIs: {data_services}"
                assert self.catchment_geometry is not None, "The `self.catchment_directory` object must exist " + \
                    "before avector is downloaded using `vector.Linz`"

                # Instantiate object for downloading vectors from the data service. Download layers with the run method
                vector_fetcher = data_services[data_service](api_key,
                                                             bounding_polygon=self.catchment_geometry.catchment,
                                                             verbose=True)

                vector_instruction = self.instructions['instructions']['apis'][data_service][key]
                geometry_type = vector_instruction['type'] if 'type' in vector_instruction else None

                if verbose:
                    print(f"Downloading vector layers {vector_instruction['layers']} from the {data_service} data" +
                          "service")

                # Cycle through all layers specified - save each and add to the path list
                for layer in vector_instruction['layers']:
                    vector = vector_fetcher.run(layer, geometry_type)

                    # Ensure directory for layer and save vector file
                    layer_dir = cache_dir / str(layer)
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    vector_dir = layer_dir / key
                    vector.to_file(vector_dir)
                    shutil.make_archive(base_name=vector_dir, format='zip', root_dir=vector_dir)
                    shutil.rmtree(vector_dir)
                    paths.append(layer_dir / f"{key}.zip")
        return paths

    def get_lidar_file_list(self, verbose: bool = False) -> list:
        """ Load or construct a list of lidar tiles to construct a DEM from. """

        lidar_dataset_index = 0  # currently only support one LiDAR dataset

        if self.check_apis('open_topography'):

            assert self.check_instruction_path('local_cache'), "A 'local_cache' must be specified under the " + \
                "'file_paths' in the instruction file if you are going to use an API - like 'open_topography'"

            # download from OpenTopography - then get the local file path
            self.lidar_fetcher = geoapis.lidar.OpenTopography(self.catchment_geometry.catchment,
                                                              self.get_instruction_path('local_cache'),
                                                              verbose=verbose)
            self.lidar_fetcher.run()
            lidar_file_paths = sorted(pathlib.Path(self.lidar_fetcher.cache_path /
                                      self.lidar_fetcher.dataset_prefixes[lidar_dataset_index]).glob('*.laz'))
        else:
            # get the specified file paths from the instructions
            lidar_file_paths = self.get_instruction_path('lidars')

        return lidar_file_paths

    def run(self):
        """ This method executes the geofabrics generation pipeline to produce geofabric derivatives. """

        # Note correctly only consider one LiDAR dataset.
        area_threshold = 10.0/100  # 10%

        # load in instruction values or set to defaults
        verbose = self.get_instruction_general('verbose')

        # create the catchment geometry object
        catchment_dirs = self.get_instruction_path('catchment_boundary')
        assert type(catchment_dirs) is not list, f"A list of catchment_boundary's is provided: {catchment_dirs}, " + \
            "where only one is supported."
        self.catchment_geometry = geometry.CatchmentGeometry(catchment_dirs,
                                                             self.get_crs('horizontal'), self.get_crs('vertical'),
                                                             self.get_resolution(), foreshore_buffer=2)
        land_dirs = self.get_vector_paths('land', verbose)
        assert len(land_dirs) == 1, f"{len(land_dirs)} catchment_boundary's provided, where only one is supported." + \
            f" Specficially land_dirs = {land_dirs}."
        self.catchment_geometry.land = land_dirs[0]

        # Define PDAL/GDAL griding parameter values
        radius = self.catchment_geometry.resolution * numpy.sqrt(2)
        window_size = 0
        idw_power = 2

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_file_paths = self.get_lidar_file_list(verbose)

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDem(self.catchment_geometry, self.get_instruction_path('temp_raster'),
                                      verbose=verbose)
        self.catchment_lidar = lidar.CatchmentLidar(
            self.catchment_geometry, area_to_drop=self.get_instruction_general('filter_lidar_holes_area'),
            verbose=verbose)

        # Load in LiDAR tiles
        for index, lidar_file_path in enumerate(lidar_file_paths):
            if verbose:
                print(f"Looking at LiDAR tile {index + 1} of {len(lidar_file_paths)}: {lidar_file_path}")

            # load in LiDAR tile
            self.catchment_lidar.load_tile(lidar_file_path)

            # update the dense DEM with a patch created from the LiDAR tile
            self.dense_dem.add_tile(self.catchment_lidar.tile_array, window_size, idw_power, radius)
            del self.catchment_lidar.tile_array

        # Filter the LiDAR extents based on the area_to_drop
        self.catchment_lidar.filter_lidar_extents_for_holes()

        # Load in reference DEM if any significant land/foreshore not covered by LiDAR
        area_without_lidar = \
            self.catchment_geometry.land_and_foreshore_without_lidar(self.catchment_lidar.extents).geometry.area.sum()
        if (self.check_instruction_path('reference_dems') and
                area_without_lidar > self.catchment_geometry.land_and_foreshore.area.sum() * area_threshold):

            assert len(self.get_instruction_path('reference_dems')) == 1, \
                f"{len(self.get_instruction_path('reference_dems'))} reference_dems specified, but only one supported" \
                + f" currently. reference_dems: {self.get_instruction_path('reference_dems')}"

            if verbose:
                print(f"Incorporting background DEM: {self.get_instruction_path('reference_dems')}")

            # Load in background DEM - cut away within the LiDAR extents
            self.reference_dem = dem.ReferenceDem(self.get_instruction_path('reference_dems')[0],
                                                  self.catchment_geometry,
                                                  self.get_instruction_general('set_dem_shoreline'),
                                                  exclusion_extent=self.catchment_lidar.extents)

            # update the dense DEM with a patch created from the reference DEM where there isn't LiDAR
            self.dense_dem.add_tile(self.reference_dem.points, window_size, idw_power, radius)

        # Load in bathymetry and interpolate offshore if significant offshore is not covered by LiDAR
        area_without_lidar = \
            self.catchment_geometry.offshore_without_lidar(self.catchment_lidar.extents).geometry.area.sum()
        if (self.check_vector('bathymetry_contours') and
                area_without_lidar > self.catchment_geometry.offshore.area.sum() * area_threshold):

            # Get the bathymetry data directory
            bathy_contour_dirs = self.get_vector_paths('bathymetry_contours')
            assert len(bathy_contour_dirs) == 1, f"{len(bathy_contour_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {catchment_dirs}. Support has not yet been added for multiple datasets."

            if verbose:
                print(f"Incorporting Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.bathy_contours = geometry.BathymetryContours(
                bathy_contour_dirs[0], self.catchment_geometry,
                z_label=self.get_instruction_general('bathymetry_contours_z_label'),
                exclusion_extent=self.catchment_lidar.extents)

            # interpolate
            self.dense_dem.interpolate_offshore(self.bathy_contours, self.catchment_lidar.extents)

        # fill combined dem
        self.result_dem = self.dense_dem.dem

        # save results
        self.result_dem.to_netcdf(self.get_instruction_path('result_dem'))
