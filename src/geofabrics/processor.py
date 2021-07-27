# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import numpy
import json
import pathlib
from . import geometry
from . import lidar
from . import lidar_fetch
from . import vector_fetch
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
        """ Return the file path from the instruction file. Raise an error if the key is not in the instructions. """

        assert key in self.instructions['instructions']['data_paths'], "Key missing from data paths"
        return self.instructions['instructions']['data_paths'][key]

    def check_instruction_path(self, key: str) -> bool:
        """ Return True if the file path exists in the instruction file. """

        return key in self.instructions['instructions']['data_paths']

    def get_resolution(self) -> float:
        """ Return the resolution from the instruction file. Raise an error if not in the instructions. """

        assert 'resolution' in self.instructions['instructions']['grid_params'], \
            "'resolution' is not a key-word in the instructions"
        return self.instructions['instructions']['grid_params']['resolution']

    def get_projection(self) -> float:
        """ Return the CRS projection from the instruction file. Raise an error if not in the instructions. """

        assert 'projection' in self.instructions['instructions'], \
            "'projection' is not a key-word in the instructions"
        return self.instructions['instructions']['projection']

    def get_instruction_general(self, key: str):
        """ Return the general instruction from the instruction file or return the default value if not specified in
        the instruction file. Raise an error if the key is not in the instructions and there is no default value. """

        defaults = {'filter_lidar_holes_area': None, 'verbose': True, 'set_dem_shoreline': True,
                    'bathymetry_contours_z_label': None}

        assert key in defaults or key in self.instructions['instructions']['general'], f"The key: {key} is missing " \
            + "from the general instructions, and does not have a default value"
        if key in self.instructions['instructions']['general']:
            return self.instructions['instructions']['general'][key]
        else:
            return defaults[key]

    def check_instruction_linz(self, key: str) -> bool:
        """ Check if the specified LINZ layer is specified """

        if "linz_api" in self.instructions['instructions']:
            return key in self.instructions['instructions']['linz_api']
        else:
            return False

    def get_instruction_linz(self, key: str) -> str:
        """ Return the linz API info from the instruction file.
        Raise an error if the key is not in the instructions. """

        assert key in self.instructions['instructions']['linz_api'], "Key missing from data paths"
        return self.instructions['instructions']['linz_api'][key]

    def get_lidar_file_list(self, verbose) -> list:
        """ Load or construct a list of lidar tiles to construct a DEM from. """

        lidar_dataset_index = 0  # currently only support one LiDAR dataset

        if self.check_instruction_path('local_cache'):

            # download from OpenTopography - then get the local file path
            self.lidar_fetcher = lidar_fetch.OpenTopography(self.catchment_geometry,
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
        self.catchment_geometry = geometry.CatchmentGeometry(self.get_instruction_path('catchment_boundary'),
                                                             self.get_instruction_path('land'),
                                                             self.get_projection(),
                                                             self.get_resolution(),
                                                             foreshore_buffer=2)

        # Define PDAL/GDAL griding parameter values
        radius = self.catchment_geometry.resolution * numpy.sqrt(2)
        window_size = 0
        idw_power = 2

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_file_paths = self.get_lidar_file_list(verbose)

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDem(self.catchment_geometry, self.get_instruction_path('tmp_raster'), verbose=verbose)
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
        if (self.check_instruction_path('bathymetry_contours') and
                area_without_lidar > self.catchment_geometry.offshore.area.sum() * area_threshold):

            # Load in bathymetry
            self.bathy_contours = geometry.BathymetryContours(
                self.get_instruction_path('bathymetry_contours')[0], self.catchment_geometry,
                z_label=self.get_instruction_general('bathymetry_contours_z_label'),
                exclusion_extent=self.catchment_lidar.extents)

            # interpolate
            self.dense_dem.interpolate_offshore(self.bathy_contours, self.catchment_lidar.extents)

        # fill combined dem
        self.result_dem = self.dense_dem.dem

        # save results
        self.result_dem.to_netcdf(self.get_instruction_path('result_dem'))
