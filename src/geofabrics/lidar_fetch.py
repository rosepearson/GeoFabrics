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
        
    def run(self):
        """ Querey for  LiDAR data within a catchment and download any that hasn't already been downloaded """
        
        json_response = self.query_for_datasets_inside_catchment()
        
        ot_endpoint_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_DATA, "", "", "", ""))
        client = boto3.client('s3', endpoint_url=ot_endpoint_url, 
                              config=botocore.config.Config(signature_version=botocore.UNSIGNED))
        
        # cycle through each dataset within a region
        for i in range(len(json_response['Datasets'])):
            dataset_prefix = json_response['Datasets'][i]['Dataset']['alternateName']
            
            tile_info = self._get_tile_info(client, dataset_prefix)
            
            lidar_size = self._calculate_lidar_download_size(client, dataset_prefix, tile_info)
            
            assert lidar_size/1000/1000/1000 < self.download_limit, "The size of the LiDAR to be " \
                + "downloaded is greater than the specified download limit of " + str(self.download_limit)
            
            # check for tiles and download as needed
            self.download_lidar_in_catchment(client, dataset_prefix, tile_info)
            
        
    def query_for_datasets_inside_catchment(self):
        """ Function to check for data in search region using hte otCatalogue API
        https://portal.opentopography.org/apidocs/#/Public/getOtCatalog """
        
        api_queary = {
            "productFormat": "PointCloud",
            "minx": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['minx'].min(),
            "miny": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['miny'].min(),
            "maxx": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['maxx'].max(),
            "maxy": self.catchment_geometry.catchment.geometry.to_crs(self.CRS).bounds['maxy'].max(),
            "detail": False,
            "outputFormat": "json",
            "inlcude_federated": True
        }
        
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))
        
        response = requests.get(data_url, params=api_queary, stream=True)
        response.raise_for_status()
        return response.json()
        
    def _get_tile_info(self, client, dataset_prefix):
        """ Check for the tile index shapefile and download as needed, then load in 
        and trim to the catchment to determine which data tiles to download. """ 
        
        
        file_prefex = dataset_prefix + "/" + dataset_prefix + "_TileIndex.zip"
        local_file_path = self.cache_path / file_prefex
        
        # Download the file if needed
        if self.redownload_files or not local_file_path.exists():
            response = client.head_object(Bucket=self.OT_BUCKET, Key=file_prefex)
            assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "No tile index file exists with key: " + file_prefex
            self.response = response
            
            # ensure folder exists before download
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            client.download_file(self.OT_BUCKET, file_prefex, str(local_file_path))
            
        # load in tile information
        tile_info = geometry.TileInfo(local_file_path, self.catchment_geometry)
        
        return tile_info
    
    def _calculate_lidar_download_size(self, client, short_name, tile_info):
        """ Sum up the size of the LiDAR data in catchment """
        
        lidar_size = 0
        
        for tile_name in tile_info.tile_names:
            file_prefex = short_name + "/" + tile_name
            local_path = self.cache_path / file_prefex
            if self.redownload_files or not local_path.exists():
                response = client.head_object(Bucket=self.OT_BUCKET, Key=file_prefex)
                assert response['ResponseMetadata']['HTTPStatusCode'] == 200, "No tile file exists with key: " + file_prefex
                lidar_size += response['ContentLength']
                if(self.verbose):
                    print("checking size: " + file_prefex + ": " + str(response['ContentLength']) + ", total: " + str(lidar_size))
                
        return lidar_size
        
        
    def download_lidar_in_catchment(self, client, dataset_prefix, tile_info):
        """ Download the lidar data within the catchment """
    
        for tile_name in tile_info.tile_names:
            file_prefex = dataset_prefix + "/" + tile_name
            local_path = self.cache_path / file_prefex
            print('Check file: ' + file_prefex)
            if self.redownload_files or not local_path.exists():
                if(self.verbose):
                    print('Downloading file: ' + file_prefex)
                client.download_file(self.OT_BUCKET, file_prefex, str(local_path))
                    
        
