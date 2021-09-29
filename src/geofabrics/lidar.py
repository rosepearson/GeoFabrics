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
    """ A class to manage LiDAR data in a catchment context by supporting the addition of LiDAR data tile by tile. LiDAR
    files are loaded using PDAL and the point cloud, and it's extents are returned.
    """
    LAS_GROUND = 2  # As specified in the LAS/LAZ format

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, source_crs: dict = None,
                 drop_offshore_lidar: bool = True, keep_only_ground_lidar: bool = True,
                 verbose: bool = True, tile_index_file: typing.Union[str, pathlib.Path]= None):
        """ Specify the catchment_geometry, which LiDAR files are loaded inside. All other inputs are optional.
        source_crs - specify if the CRS encoded in the LiDAR files are incorrect/only partially defined (i.e. missing
                     vertical CRS) and need to be overwritten.
        drop_offshore_lidar - if True, trim any LiDAR values that are offshore as specified by the catchment_geometry
        keep_only_ground_lidar - if True, only keep LiDAR values that are coded '2' of ground
        verbose - if True, print out status updates
        tile_index_file - if not None load in the specified tile_index file and use to clip all LiDAR tile data within
                          the relevant LiDAR tiles extents as defined in the tile_index file."""

        self.catchment_geometry = catchment_geometry
        self.source_crs = source_crs
        self.drop_offshore_lidar = drop_offshore_lidar
        self.keep_only_ground_lidar = keep_only_ground_lidar
        self.verbose = verbose

        self._tile_index_extents = geopandas.read_file(tile_index_file) if tile_index_file is not None else None
        self._tile_index_name_column = None

        self._set_up()

    def _set_up(self):
        """ Ensure the source_crs is either None or fully and correctly populated """

        if self.source_crs is not None:
            assert 'horizontal' in self.source_crs, "The horizontal component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            assert 'vertical' in self.source_crs, "The vertical component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"

        # If there is a tile_index_file - remove tiles outside the catchment & get the 'file name' column
        if self._tile_index_extents is not None:
            self._tile_index_extents = self._tile_index_extents.to_crs(self.catchment_geometry.crs['horizontal'])
            self._tile_index_extents = geopandas.sjoin(self._tile_index_extents, self.catchment_geometry.catchment)
            self._tile_index_extents = self._tile_index_extents.reset_index(drop=True)

            column_names = self._tile_index_extents.columns
            self._tile_index_name_column = column_names[["filename" == name.lower() or "file_name" == name.lower()
                                                         for name in column_names]][0]

    def load_tile(self, lidar_file: typing.Union[str, pathlib.Path]):
        """ Function loading in a LiDAR tile and its extent. The array and extent is returned. """

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
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
        pdal_pipeline.execute()

        # Load LiDAR points from pipeline
        tile_array = pdal_pipeline.arrays[0]

        # Optionally filter the points by classification code - to keep only ground coded points
        if self.keep_only_ground_lidar:
            tile_array = tile_array[tile_array['Classification'] == self.LAS_GROUND]

        # update the catchment geometry with the LiDAR extents - note has to run on imported LAS file not point data
        metadata = json.loads(pdal_pipeline.get_metadata())
        tile_extents_string = metadata['metadata']['filters.hexbin']['boundary']

        # Only care about horizontal extents
        tile_extent = geopandas.GeoDataFrame({'geometry': [shapely.wkt.loads(tile_extents_string)]},
                                                   crs=self.catchment_geometry.crs['horizontal'])

        if self._tile_index_extents is not None and tile_extent.geometry.area.sum() > 0:
            tile_index_extent = self._tile_index_extents[lidar_file.name==self._tile_index_extents[self._tile_index_name_column]]
            tile_extent = geopandas.clip(tile_extent, tile_index_extent)

        return tile_array, tile_extent
