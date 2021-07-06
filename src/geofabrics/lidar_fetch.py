# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:10:55 2021

@author: pearsonra
"""

import urllib
import pathlib
import requests
import boto3
import botocore
import geopandas
import typing
from . import geometry

class OpenTopography:
    """ A class to manage fetching LiDAR data from Open Topography
    """
    
    SCHEME = "https"
    NETLOC_API = "portal.opentopography.org"
    PATH_API = "/API/otCatalog"
    CRS = "EPSG:4326"
    NETLOC_DATA = "opentopography.s3.sdsc.edu"
    OT_BUCKET = 'pc-bulk'
    
    
    PATH_DATA = "/minio/pc-bulk/"
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, cache_path: typing.Union[str, pathlib.Path]):
        """ Load in lidar with relevant processing chain.
        
        Note in case of multiple datasets could select by name, 
        spatial extent, or most recent"""
        
        self.catchment_geometry = catchment_geometry
        self.cache_path = pathlib.Path(cache_path)
        
        self.api_query = None
        self.tile_info = None
        
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
        
    def _ensure_dir(self, directory: pathlib.Path):
        """ Checks if a repository exists and creates it if it doesn't. Note 
        could use exist_ok to move this to a oneline operation. """
        if not directory.exists():
           directory.mkdir(parents=True, exist_ok=True) 
        
    def query_inside_catchment(self):
        """ Function to check for data in search region """
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))
        
        response = requests.get(data_url, params=self.api_queary, stream=True)
        response.raise_for_status()
        self._json_response = response.json()
        
    def download_tile_info(self, client, response, short_name):
        """ Download the tile shapefile to determine which data tiles to download """ 
        TileIndexExists = False
        
        for obj in response['Contents']:
            if 'TileIndex' in obj['Key']:
                
                assert TileIndexExists == False, "Support for multiple tile index files not yet added. Multiple tile index files in the OpenTopography Bucket: " + short_name
                    
                # download tile information
                local_path = self.cache_path / obj['Key']
                local_path.parent.mkdir(parents=True, exist_ok=True) 
                client.download_file(self.OT_BUCKET, obj['Key'], str(local_path))
                TileIndexExists = True
                    
        assert TileIndexExists, "No tile index file exists in the OpenTopography Bucket: " + short_name
        
        # load in tile information
        tile_info = geometry.TileInfo(local_path, self.catchment_geometry)
        
        return tile_info
        
    def download_lidar_in_catchment(self):
        """ Download the lidar data within the catchment """ 
        
        
        short_name = self._json_response['Datasets'][0]['Dataset']['alternateName']
        
        ot_endpoint_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_DATA, "", "", "", ""))
        client = boto3.client('s3', endpoint_url=ot_endpoint_url, 
                              config=botocore.config.Config(signature_version=botocore.UNSIGNED))
        
        
        response = client.list_objects_v2(Bucket=self.OT_BUCKET, Prefix=short_name)
        
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "List objects for " + short_name + " wasn't successful"
        
        self.tile_info = self.download_tile_info(client, response, short_name)
        
        
        for tile_name in self.tile_info.tile_names: #self.tile_info['Filename']:
            
            # download tile information
            local_path = self.cache_path / short_name / tile_name
            print(short_name + "/" + tile_name)
            #client.download_file(self.OT_BUCKET, short_name / tile_name, str(local_path))
        
