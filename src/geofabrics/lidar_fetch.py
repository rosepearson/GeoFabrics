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
    
    
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, cache_path: typing.Union[str, pathlib.Path], 
                 redownload_files: bool = False, download_limit: typing.Union[int, float] = 100, verbose: bool = False):
        """ Load in lidar with relevant processing chain.
        
        Note in case of multiple datasets could select by name, 
        spatial extent, or most recent. download_size is in GB. """
        
        self.catchment_geometry = catchment_geometry
        self.cache_path = pathlib.Path(cache_path)
        self.redownload_files = redownload_files
        self.download_limit = download_limit
        self.verbose = verbose
        
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
        
    def query_inside_catchment(self):
        """ Function to check for data in search region """
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))
        
        response = requests.get(data_url, params=self.api_queary, stream=True)
        response.raise_for_status()
        self._json_response = response.json()
        
    def _get_tile_info(self, client, short_name):
        """ Download the tile shapefile to determine which data tiles to download """ 
        
        # first try the expect path/name fo the tile
        expected_key = short_name + "/" + short_name + "_TileIndex.zip"
        local_path = self.cache_path / expected_key
        if self.redownload_files or not local_path.exists():
            response = client.head_object(Bucket=self.OT_BUCKET, Key=expected_key)
            assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "No tile index file exists with key: " + expected_key
            self.response = response
            
            # ensure folder exists before download
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            client.download_file(self.OT_BUCKET, expected_key, str(local_path))
            
        # load in tile information
        tile_info = geometry.TileInfo(local_path, self.catchment_geometry)
        
        return tile_info
    
    def _calculate_lidar_download_size(self, client, short_name):
        """ Sum up the size of the LiDAR data in catchment """
        
        lidar_size = 0
        tile_names = self.tile_info.tile_names
        
        for tile_name in tile_names:
            expected_key = short_name + "/" + tile_name
            local_path = self.cache_path / expected_key
            if self.redownload_files or not local_path.exists():
                response = client.head_object(Bucket=self.OT_BUCKET, Key=expected_key)
                assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "No tile file exists with key: " + expected_key
                lidar_size += response['ContentLength']
                if(self.verbose):
                    print("checking size: " + expected_key + ": " + str(response['ContentLength']) + ", total: " + str(lidar_size))
                
        return lidar_size
        
        
    def download_lidar_in_catchment(self):
        """ Download the lidar data within the catchment """ 
        
        short_name = self._json_response['Datasets'][0]['Dataset']['alternateName']
        
        ot_endpoint_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_DATA, "", "", "", ""))
        client = boto3.client('s3', endpoint_url=ot_endpoint_url, 
                              config=botocore.config.Config(signature_version=botocore.UNSIGNED))
        self.client = client
        
        self.tile_info = self._get_tile_info(client, short_name)
        
        lidar_size = self._calculate_lidar_download_size(client, short_name)
        
        assert lidar_size/1000/1000/1000 < self.download_limit, "The size of the LiDAR to be " \
            + "downloaded is greater than the specified download limit of " + str(self.download_limit)
    
        tile_names = self.tile_info.tile_names
    
        for tile_name in tile_names:
            expected_key = short_name + "/" + tile_name
            local_path = self.cache_path / expected_key
            if self.redownload_files or not local_path.exists():
                if(self.verbose):
                    print('Downloading file: ' + expected_key)
                client.download_file(self.OT_BUCKET, expected_key, str(local_path))
                    
        
