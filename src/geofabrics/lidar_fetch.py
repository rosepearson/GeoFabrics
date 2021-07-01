# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:10:55 2021

@author: pearsonra
"""

import urllib
import pathlib
import requests
import json
from . import geometry

class OpenTopography:
    """ A class to manage fetching LiDAR data from Open Topography
    """
    
    SCHEME = "https"
    NETPATH = "portal.opentopography.org"
    APIPATH = "/API/otCatalog"
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry):
        """ Load in lidar with relevant processing chain """
        
        self.catchment_geometry = catchment_geometry
        
        self.api_query = None
        
        self._set_up()
        
        self._lidar_array = None
        
    
    def _set_up(self):
        """ create the API query and url """
        
        self.api_queary = {
            "productFormat": "PointCloud",
            "minx": self.catchment_geometry.catchment.geometry.bounds['minx'].min(),
            "miny": self.catchment_geometry.catchment.geometry.bounds['miny'].min(),
            "maxx": self.catchment_geometry.catchment.geometry.bounds['maxx'].max(),
            "maxy": self.catchment_geometry.catchment.geometry.bounds['maxy'].max(),
            "detail": False,
            "outputFormat": "json",
            "inlcude_federated": True
            }
        
    def lookup(self):
        """ Function to check for data in search region """
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETPATH, self.APIPATH, "", "", ""))
        
        response = requests.get(data_url, params=self.api_queary, stream=True)
        response.raise_for_status()
        
        print(response.json())
        
    @property
    def lidar_array(self):
        """ function returing the lidar point values - 
        
        The array is loaded from the PDAL pipeline the first time it is 
        called. """
        
        if self._lidar_array is None:
            self._lidar_array = None
        return self._lidar_array