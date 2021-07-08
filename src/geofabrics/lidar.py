# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:36:41 2021

@author: pearsonra
"""
import pdal
import json
import typing
import pathlib
from . import geometry


class CatchmentLidar:
    """ A class to manage lidar data in a catchment context
    
    Specifically, this supports the import, and manipulation if LiDAR data.
    """
    
    def __init__(self, lidar_file: typing.Union[str, pathlib.Path], catchment_geometry: geometry.CatchmentGeometry):
        """ Load in lidar with relevant processing chain """
        
        self.catchment_geometry = catchment_geometry
        self._pdal_pipeline = None
        self._lidar_array = None
        
        self._load_lidar(lidar_file)
        
    def _load_lidar(self, lidar_file):
        """ Function loading in the lidar
        
        In future we may want to have the option of filtering by foreshore / 
        land """
        
        pdal_pipeline_instructions = [
            {"type":  "readers.las", "filename": str(lidar_file)},
            {"type":"filters.reprojection","out_srs":"EPSG:" + str(self.catchment_geometry.crs)}, # reproject to NZTM
            {"type":"filters.crop", "polygon":str(self.catchment_geometry.catchment.loc[0].geometry)}, # filter within boundary
            {"type" : "filters.hexbin"} # create a polygon boundary of the LiDAR
        ]
        
        self._pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
        self._pdal_pipeline.execute()
        
        # update the catchment geometry with the LiDAR extents
        metadata=json.loads(self._pdal_pipeline.get_metadata())
        self.catchment_geometry.load_lidar_extents(metadata['metadata']['filters.hexbin']['boundary'])
        
    @property
    def lidar_array(self):
        """ function returing the lidar point values - 
        
        The array is loaded from the PDAL pipeline the first time it is 
        called. """
        
        if self._lidar_array is None:
            self._lidar_array = self._pdal_pipeline.arrays[0]
        return self._lidar_array
    
    @lidar_array.deleter
    def lidar_array(self):
        """ Delete the lidar array
        
        should check how it is stored in the pdal pieline"""
        
        del self._lidar_array