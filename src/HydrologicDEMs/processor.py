# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import geopandas
import rioxarray

class HydrologicalDemGenerator:
    def __init__(self, json_instructions):
        self.instructions = json_instructions
        
    def run(self):
        print(self.instructions)
        
        # load in data
        catchment_boundary = geopandas.read_file(self.instructions['instructions']['data_paths']['catchment_boundary'])
        islands = geopandas.read_file(self.instructions['instructions']['data_paths']['shoreline'])
        bathymetry = geopandas.read_file(self.instructions['instructions']['data_paths']['bathymetries'][0])
        background_dem = rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['background_dems'][0], masked=True)
        
        
