# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:36:41 2021

@author: pearsonra
"""
import pdal
import json
import typing
import pathlib
import shapely
import geopandas
import numpy
from . import geometry


class CatchmentLidar:
    """ A class to manage LiDAR data in a catchment context

    Specifically, this supports the addition of LiDAR data tile by tile.
    """
    LAS_GROUND = 2

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, source_crs: dict = None,
                 drop_offshore_lidar: bool = True, keep_only_ground_lidar: bool = True,
                 verbose: bool = True):
        """ Load in LiDAR with relevant processing chain """

        self.catchment_geometry = catchment_geometry
        self.source_crs = source_crs
        self.drop_offshore_lidar = drop_offshore_lidar
        self.keep_only_ground_lidar = keep_only_ground_lidar
        self.verbose = verbose

        self._pdal_pipeline = None
        self._tile_array = None
        self._tile_extent = None

    def _set_up(self):
        """ Ensure the source_crs is either None or fully and correctly populated """

        if self.source_crs is not None:
            assert 'horizontal' in self.source_crs, "The horizontal component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            assert 'vertical' in self.source_crs, "The vertical component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"

    def load_tile(self, lidar_file: typing.Union[str, pathlib.Path]):
        """ Function loading in a LiDAR tile and its extent """

        # Define instructions for loading in LiDAR
        pdal_pipeline_instructions = [{"type":  "readers.las", "filename": str(lidar_file)}]

        # Specify reprojection - if a source_crs is specified use this to define the 'in_srs'
        if self.source_crs is None:
            pdal_pipeline_instructions.append(
                {"type": "filters.reprojection",
                 "out_srs": f"EPSG:{self.catchment_geometry.crs['horizontal']}+" +
                 f"{self.catchment_geometry.crs['vertical']}"})
        else:
            pdal_pipeline_instructions.append(
                {"type": "filters.reprojection",
                 "in_srs": f"EPSG:{self.source_crs['horizontal']}+{self.source_crs['vertical']}",
                 "out_srs": f"EPSG:{self.catchment_geometry.crs['horizontal']}+" +
                 f"{self.catchment_geometry.crs['vertical']}"})

        # Add instructions for clip within either the catchment, or the land and foreshore
        if self.drop_offshore_lidar:
            pdal_pipeline_instructions.append(
                {"type": "filters.crop", "polygon": str(self.catchment_geometry.land_and_foreshore.loc[0].geometry)})
        else:
            pdal_pipeline_instructions.append(
                {"type": "filters.crop", "polygon": str(self.catchment_geometry.catchment.loc[0].geometry)})

        # Add instructions for creating a polygon extents of the remaining point cloud
        pdal_pipeline_instructions.append({"type": "filters.hexbin"})

        # Load in LiDAR and perform operations
        self._pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
        self._pdal_pipeline.execute()

        self._tile_array = self._pdal_pipeline.arrays[0]

        # update the catchment geometry with the LiDAR extents - note has to run on imported LAS file not point data
        metadata = json.loads(self._pdal_pipeline.get_metadata())
        tile_extents_string = metadata['metadata']['filters.hexbin']['boundary']

        # Optionally filter the points by classification code - to keep only ground coded points
        if self.keep_only_ground_lidar:
            self._tile_array = self._tile_array[self._tile_array['Classification'] == self.LAS_GROUND]

        # Only care about horizontal extents
        self._tile_extent = geopandas.GeoDataFrame({'geometry': [shapely.wkt.loads(tile_extents_string)]},
                                                   crs=self.catchment_geometry.crs['horizontal'])

    @property
    def tile_array(self) -> numpy.ndarray:
        """ Function returning the LiDAR point values. """

        return self._tile_array

    @tile_array.deleter
    def tile_array(self):
        """ Delete the LiDAR array and pdal_pipeline """

        # Set to None and let automatic garbage collection free memory
        self._tile_array = None
        self._pdal_pipeline = None

    @property
    def tile_extent(self) -> geopandas.GeoDataFrame:
        """ Function returning the extent for the last LiDAR tile. """

        return self._tile_extent
