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
    NETLOC_API = "portal.opentopography.org"
    PATH_API = "/API/otCatalog"
    CRS = "EPSG:4326"
    NETLOC_DATA = "opentopography.s3.sdsc.edu"
    PATH_DATA = "/minio/pc-bulk/"
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, cache_path: str):
        """ Load in lidar with relevant processing chain.
        
        Note in case of multiple datasets could select by name, 
        spatial extent, or most recent"""
        
        self.catchment_geometry = catchment_geometry
        self.cache_path = pathlib.Path(cache_path)
        
        self.api_query = None
        
        self._set_up()
        
        self._json_response = None
        self._lidar_array = None
        
    
    def _set_up(self):
        """ create the API query and url """
        
        self.api_queary = {
            "productFormat": "PointCloud",
            "minx": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['minx'].min(),
            "miny": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['miny'].min(),
            "maxx": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['maxx'].max(),
            "maxy": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['maxy'].max(),
            "detail": False,
            "outputFormat": "json",
            "inlcude_federated": True
            }
        
    def send_query(self):
        """ Function to check for data in search region """
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))
        
        response = requests.get(data_url, params=self.api_queary, stream=True)
        response.raise_for_status()
        self.response = response
        self._json_response = response.json()
        
    def get_tile_info(self):
        """ Download the tile shapefile to determine which data tiles to download """ 
        
        short_name = self._json_response['Datasets'][0]['Dataset']['alternateName']
        
        bulk_download_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_DATA, 
                                            self.PATH_DATA + '/' + short_name, 0, 0, 0))
        
        
        tile_name = short_name + "_TileIndex.zip"
        tile_url = bulk_download_url + tile_name
        
        # download tile data
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', self.USER_AGENT)
        filename, headers = opener.retrieve(tile_url, self.cache_path / pathlib.Path(tile_name))
        
        # load in tile information - next part
        
        
    @property
    def json_response(self):
        """ JSON api query response """
        
        assert self._json_response is not None, "json_return have not been set, and need to be set explicitly"
        
        return self._json_response()
