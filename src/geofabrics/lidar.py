# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:36:41 2021

@author: pearsonra
"""
import pdal
import json
import typing
import pathlib
import geopandas
import shapely
from . import geometry


class CatchmentLidar:
    """ A class to manage LiDAR data in a catchment context

    Specifically, this supports the addition of LiDAR data tile by tile.
    """

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, source_crs: dict = None,
                 drop_offshore_lidar: bool = True, area_to_drop: float = None, verbose: bool = True):
        """ Load in LiDAR with relevant processing chain """

        self.catchment_geometry = catchment_geometry
        self.source_crs = source_crs
        self.drop_offshore_lidar = drop_offshore_lidar
        self.area_to_drop = area_to_drop
        self.verbose = verbose

        self._pdal_pipeline = None
        self._tile_array = None
        self._extents = None

    def _set_up(self):
        """ Ensure the source_crs is either None or fully and correctly populated """

        if self.source_crs is not None:
            assert 'horizontal' in self.source_crs, "The horizontal component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            assert 'vertical' in self.source_crs, "The vertical component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"

    def load_tile(self, lidar_file: typing.Union[str, pathlib.Path]):
        """ Function loading in the LiDAR

        This updates the LiDAR extents in the catchment_geometry

        In future we may want to have the option of filtering by foreshore / land """

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

        # update the catchment geometry with the LiDAR extents
        metadata = json.loads(self._pdal_pipeline.get_metadata())
        tile_extents_string = metadata['metadata']['filters.hexbin']['boundary']

        self._update_extents(tile_extents_string)
        self._tile_array = self._pdal_pipeline.arrays[0]

    def _update_extents(self, tile_extents_string: str):
        """ Update the extents of all LiDAR tiles updated """

        tile_extents = shapely.wkt.loads(tile_extents_string)

        if tile_extents.area > 0:  # check polygon isn't empty

            if self._extents is None:
                self._extents = geopandas.GeoSeries([tile_extents], crs=self.catchment_geometry.crs['horizontal'])
                self._extents = geopandas.GeoDataFrame(
                    index=[0], geometry=self._extents, crs=self.catchment_geometry.crs['horizontal'])
            else:
                self._extents = geopandas.GeoSeries(
                    shapely.ops.cascaded_union([self._extents.loc[0].geometry, tile_extents]),
                    crs=self.catchment_geometry.crs['horizontal'])
                self._extents = geopandas.GeoDataFrame(
                    index=[0], geometry=self._extents, crs=self.catchment_geometry.crs['horizontal'])
            self._extents = geopandas.clip(self.catchment_geometry.catchment, self._extents)

    @property
    def tile_array(self):
        """ Function returning the LiDAR point values. """

        return self._tile_array

    @tile_array.deleter
    def tile_array(self):
        """ Delete the LiDAR array and pdal_pipeline """

        # Set to None and let automatic garbage collection free memory
        self._tile_array = None
        self._pdal_pipeline = None

    @property
    def extents(self):
        """ The combined extents for all added LiDAR tiles """

        assert self._extents is not None, "No tiles have been added yet"
        return self._extents

    def filter_lidar_extents_for_holes(self):
        """ Remove holes below a filter size within the extents """

        if self.area_to_drop is None:
            return  # do nothing

        polygon = self._extents.loc[0].geometry

        if polygon.geometryType() == "Polygon":
            polygon = shapely.geometry.Polygon(
                polygon.exterior.coords, [interior for interior in polygon.interiors if
                                          shapely.geometry.Polygon(interior).area > self.area_to_drop])
            self._extents = geopandas.GeoSeries([polygon], crs=self.catchment_geometry.crs['horizontal'])
            self._extents = geopandas.GeoDataFrame(index=[0], geometry=self._extents,
                                                   crs=self.catchment_geometry.crs['horizontal'])
            self._extents = geopandas.clip(self.catchment_geometry.catchment, self._extents)
        else:
            if self.verbose:
                print("Warning filtering holes in CatchmentLidar using filter_lidar_extents_for_holes is not yet "
                      + f"supported for {polygon.geometryType()}")
