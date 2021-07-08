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
import typing
from . import geometry

class OpenTopography:
    """ A class to manage fetching LiDAR data from Open Topography
    """
    
    SCHEME = "https"
    NETLOC_API = "portal.opentopography.org"
    PATH_API = "/API/otCatalog"
    OT_CRS = "EPSG:4326"
    NETLOC_DATA = "opentopography.s3.sdsc.edu"
    OT_BUCKET = 'pc-bulk'
    
    PATH_DATA = "/minio/pc-bulk/"
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, cache_path: typing.Union[str, pathlib.Path], 
                 redownload_files: bool = False, download_limit_gbytes: typing.Union[int, float] = 100, verbose: bool = False):
        """ Load in lidar with relevant processing chain.
        
        Note in case of multiple datasets could select by name, 
        spatial extent, or most recent. download_size is in GB. """
        
        self.catchment_geometry = catchment_geometry
        self.cache_path = pathlib.Path(cache_path)
        self.redownload_files_bool = redownload_files
        self.download_limit_gbytes = download_limit_gbytes
        self.verbose = verbose
        
        self._dataset_prefixes = None
        
    def _to_gbytes(self, bytes_number):
        """ convert bytes into gigabytes"""
        
        return bytes_number/1000/1000/1000
        
    def run(self):
        """ Query for  LiDAR data within a catchment and download any that hasn't already been downloaded """
        self._dataset_prefixes = []
        
        json_response = self.query_for_datasets_inside_catchment()
        
        ot_endpoint_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_DATA, "", "", "", ""))
        client = boto3.client('s3', endpoint_url=ot_endpoint_url, 
                              config=botocore.config.Config(signature_version=botocore.UNSIGNED))
        
        # cycle through each dataset within a region
        for json_dataset in json_response['Datasets']:
            dataset_prefix = json_dataset['Dataset']['alternateName']
            self._dataset_prefixes.append(dataset_prefix)
            
            tile_info = self._get_dataset_tile_info(client, dataset_prefix)
            
            # check download size limit is not exceeded
            lidar_size_bytes = self._calculate_dataset_download_size(client, dataset_prefix, tile_info)
            
            assert self._to_gbytes(lidar_size_bytes) < self.download_limit_gbytes, "The size of the LiDAR to be " \
                + f"downloaded is greater than the specified download limit of {self.download_limit_gbytes}"
            
            # check for tiles and download as needed
            self._download_tiles_in_catchment(client, dataset_prefix, tile_info)
                   
    def query_for_datasets_inside_catchment(self):
        """ Function to check for data in search region using the otCatalogue API
        https://portal.opentopography.org/apidocs/#/Public/getOtCatalog """
        
        catchment_bounds = self.catchment_geometry.catchment.geometry.to_crs(self.OT_CRS).bounds
        api_queary = {
            "productFormat": "PointCloud",
            "minx": catchment_bounds['minx'].min(),
            "miny": catchment_bounds['miny'].min(),
            "maxx": catchment_bounds['maxx'].max(),
            "maxy": catchment_bounds['maxy'].max(),
            "detail": False,
            "outputFormat": "json",
            "inlcude_federated": True
        }
        
        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))
        
        response = requests.get(data_url, params=api_queary, stream=True)
        response.raise_for_status()
        return response.json()
        
    def _get_dataset_tile_info(self, client, dataset_prefix):
        """ Check for the tile index shapefile and download as needed, then load in 
        and trim to the catchment to determine which data tiles to download. """ 
        
        
        file_prefix = f"{dataset_prefix}/{dataset_prefix}_TileIndex.zip"
        local_file_path = self.cache_path / file_prefix
        
        # Download the file if needed
        if self.redownload_files_bool or not local_file_path.exists():
            response = client.head_object(Bucket=self.OT_BUCKET, Key=file_prefix)
            assert response['ResponseMetadata']['HTTPStatusCode'] == 200, f"No tile index file exists with key: {file_prefix}"
            self.response = response
            
            # ensure folder exists before download
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            client.download_file(self.OT_BUCKET, file_prefix, str(local_file_path))
            
        # load in tile information
        tile_info = geometry.TileInfo(local_file_path, self.catchment_geometry)
        
        return tile_info
    
    def _calculate_dataset_download_size(self, client, dataset_prefix, tile_info):
        """ Sum up the size of the LiDAR data in catchment """
        
        lidar_size_bytes = 0
        
        for tile_name in tile_info.tile_names:
            file_prefix = dataset_prefix + "/" + tile_name
            local_path = self.cache_path / file_prefix
            if self.redownload_files_bool or not local_path.exists():
                response = client.head_object(Bucket=self.OT_BUCKET, Key=file_prefix)
                assert response['ResponseMetadata']['HTTPStatusCode'] == 200, f"No tile file exists with key: {file_prefix}"
                lidar_size_bytes += response['ContentLength']
                if(self.verbose):
                    print(f"checking size: {file_prefix}: {response['ContentLength']}, total (GB): {self._to_gbytes(lidar_size_bytes)}")
                
        return lidar_size_bytes
        
    def _download_tiles_in_catchment(self, client, dataset_prefix, tile_info):
        """ Download the lidar data within the catchment """
    
        for tile_name in tile_info.tile_names:
            file_prefix = f"{dataset_prefix}/{tile_name}"
            local_path = self.cache_path / file_prefix
            
            if self.redownload_files_bool or not local_path.exists():
                if(self.verbose):
                    print(f"Downloading file: {file_prefix}")
                client.download_file(self.OT_BUCKET, file_prefix, str(local_path))
                    
    @property
    def dataset_prefixes(self):
        
        assert self._dataset_prefixes is not None, "The run command needs to be called before 'dataset_prefixes' can be called."
        
        return self._dataset_prefixes
        
