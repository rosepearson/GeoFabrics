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
    """ A class to manage lidar data in a catchment context
    
    Specifically, this supports the addition of LiDAR data tile by tile.
    """
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, area_to_drop: float = None, verbose: bool = True):
        """ Load in lidar with relevant processing chain """
        
        self.catchment_geometry = catchment_geometry
        self.area_to_drop = area_to_drop
        self.verbose = verbose
        
        self._pdal_pipeline = None
        self._tile_array = None
        self._extents = None
        
    def load_tile(self, lidar_file: typing.Union[str, pathlib.Path]):
        """ Function loading in the lidar
        
        This updates the lidar extents in the catchment_geometry
        
        In future we may want to have the option of filtering by foreshore / 
        land """
        
        pdal_pipeline_instructions = [
            {"type":  "readers.las", "filename": str(lidar_file)},
            {"type": "filters.reprojection","out_srs":"EPSG:" + str(self.catchment_geometry.crs)}, # reproject to NZTM
            {"type": "filters.crop", "polygon":str(self.catchment_geometry.catchment.loc[0].geometry)}, # filter within boundary
            {"type": "filters.hexbin"} # create a polygon boundary of the LiDAR
        ]
        
        self._pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
        self._pdal_pipeline.execute()
        
        # update the catchment geometry with the LiDAR extents
        metadata = json.loads(self._pdal_pipeline.get_metadata())
        tile_extents_string = metadata['metadata']['filters.hexbin']['boundary']

        self._update_extents(tile_extents_string)
        self._tile_array = self._pdal_pipeline.arrays[0]

    def _update_extents(self, tile_extents_string: str):
        """ Update the extents of all lidar tiles updated """

        tile_extents = shapely.wkt.loads(tile_extents_string)

        if tile_extents.area > 0: # check polygon isn't empty
        
            if self._extents is None:
                self._extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries([tile_extents], crs=self.catchment_geometry.crs),
                                                       crs=self.catchment_geometry.crs)
            else:
                self._extents = geopandas.GeoDataFrame(index=[0],
                                                       geometry=geopandas.GeoSeries(shapely.ops.cascaded_union([self._extents.loc[0].geometry, tile_extents]),
                                                                                    crs=self.catchment_geometry.crs), crs=self.catchment_geometry.crs)
            self._extents = geopandas.clip(self.catchment_geometry.catchment, self._extents)

    @property
    def tile_array(self):
        """ function returing the lidar point values. """
        
        return self._tile_array
    
    @tile_array.deleter
    def tile_array(self):
        """ Delete the lidar array and pdal_pipeline """
        
        # Set to None and let automatic garbage collection free memory
        self._tile_array = None
        self._pdal_pipeline = None

    @property
    def extents(self):
        """ The combined extents for all added lidar tiles """

        assert self._extents is not None, "No tiles have been added yet"
        return self._extents

    def filter_lidar_extents_for_holes(self):
        """ Remove holes below a filter size within the extents """

        if self.area_to_drop is None:
            return # do nothing

        polygon = self._extents.loc[0].geometry

        if polygon.geometryType() == "Polygon":
            polygon = shapely.geometry.Polygon(polygon.exterior.coords, [interior for interior in polygon.interiors
                                                                         if shapely.geometry.Polygon(interior).area > self.area_to_drop])
            self._extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries([polygon], crs=self.catchment_geometry.crs),
                                                   crs=self.catchment_geometry.crs)
            self._extents = geopandas.clip(self.catchment_geometry.catchment, self._extents)
        else:
            if self.verbose:
                "Warning filtering holes in CatchmentLidar using filter_lidar_extents_for_holes is not yet supported for {polygon.geometryType()}"
