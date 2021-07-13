# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import rioxarray
import rioxarray.merge
import numpy
import json
import pathlib
from . import geometry
from . import lidar
from . import lidar_fetch
from . import dem

class GeoFabricsGenerator:
    """ A class execuiting a pipeline for creating geo-fabric derivatives. 
    
    The pipeline is controlled by the contents of the json_instructions file. 
    See the README.md for usage examples or GeoFabrics/tests/test1 for an 
    example of usage and an instruction file. 
    """
    
    def __init__(self, json_instructions: json):
        self.instructions = json_instructions
        
        self.catchment_geometry = None
        self.dense_dem = None
        self.reference_dem = None
        self.bathy_contours = None
        self.result_dem = None
        
    def run(self):
        """ This method executes the geofabrics generation pipeline to produce
        geofabric derivatives. """
        
        # Note corrently only consider one LiDAR dataset.
        lidar_dataset_index = 0
        area_threshold = 10.0/100 # 10%
        
        # load in instruction values or set to defaults
        verbose = self.instructions['instructions']['instructions']['verbose'] if 'verbose' in \
            self.instructions['instructions']['instructions'] else True
        area_to_drop = self.instructions['instructions']['instructions']['filter_lidar_holes_area'] if  \
            'filter_lidar_holes_area' in self.instructions['instructions']['instructions'] else None
        
        # create the catchment geometry object
        self.catchment_geometry = geometry.CatchmentGeometry(self.instructions['instructions']['data_paths']['catchment_boundary'],
                                                        self.instructions['instructions']['data_paths']['land'], 
                                                        self.instructions['instructions']['projection'],
                                                        self.instructions['instructions']['grid_params']['resolution'], 
                                                        foreshore_buffer = 2)
        
        
        # Define PDAL/GDAL dridding parameter values
        radius =  self.catchment_geometry.resolution * numpy.sqrt(2)
        window_size = 0
        idw_power = 2
        
        # Get LiDAR data filelist
        if 'local_cache' in self.instructions['instructions']['data_paths']:  # download from OpenTopography - then get the local file path
            self.lidar_fetcher = lidar_fetch.OpenTopography(self.catchment_geometry, self.instructions['instructions']['data_paths']['local_cache'], verbose = verbose)
            self.lidar_fetcher.run()
            lidar_file_paths = list(pathlib.Path(self.lidar_fetcher.cache_path / self.lidar_fetcher.dataset_prefixes[lidar_dataset_index]).glob('*.laz'))
        else:  # get the specified file paths from the instructions
            lidar_file_paths = self.instructions['instructions']['data_paths']['lidars']
            
        # setup dense dem and catchment lidar objects
        self.dense_dem = dem.DenseDem(self.catchment_geometry, self.instructions['instructions']['data_paths']['tmp_raster'], verbose=verbose)
        self.catchment_lidar = lidar.CatchmentLidar(self.catchment_geometry, area_to_drop=area_to_drop, verbose=verbose)

        # Load in LiDAR tiles
        for index, lidar_file_path in enumerate(lidar_file_paths):
            if verbose:
                print(f"Looking at LiDAR tile {index + 1} of {len(lidar_file_paths)}: {lidar_file_path}")
            
            # load in lidar tile
            self.catchment_lidar.load_tile(lidar_file_path)
            
            # update the dense dem with a patch created from the lidar tile 
            self.dense_dem.add_tile(self.catchment_lidar.tile_array, window_size, idw_power, radius)
            del self.catchment_lidar.tile_array
        
        # Filter the lidar extents based on the area_to_drop
        self.catchment_lidar.filter_lidar_extents_for_holes()

        # Load in reference DEM if any significant land/foreshore not covered by lidar
        if (self.catchment_geometry.land_and_foreshore_without_lidar(self.catchment_lidar.extents).geometry.area.sum()
            > self.catchment_geometry.land_and_foreshore.area.sum() * area_threshold):

            # if True set any dem values used along the foreshore to zero
            set_dem_foreshore = self.instructions['instructions']['instructions']['set_dem_shoreline'] if  \
                'set_dem_shoreline' in self.instructions['instructions']['instructions'] else True

            # Load in background DEM - cut away within the lidar extents
            self.reference_dem = dem.ReferenceDem(self.instructions['instructions']['data_paths']['reference_dems'][0],
                                                  self.catchment_geometry, set_dem_foreshore, exclusion_extent=self.catchment_lidar.extents)

            # update the dense dem with a patch created from the reference dem where there isn't LiDAR
            self.dense_dem.add_tile(self.reference_dem.points, window_size, idw_power, radius)

        # Load in bathy and interpolate offshore if significant offshore is not covered by lidar
        if (self.catchment_geometry.offshore_without_lidar(self.catchment_lidar.extents).geometry.area.max() >
            self.catchment_geometry.offshore.area.sum() * area_threshold):

            # Load in bathy
            z_label = self.instructions['instructions']['instructions']['bathymetry_contours_z_label'] if  \
                'bathymetry_contours_z_label' in self.instructions['instructions']['instructions'] else None
            self.bathy_contours = geometry.BathymetryContours(self.instructions['instructions']['data_paths']['bathymetry_contours'][0],
                                                              self.catchment_geometry, z_label = z_label, exclusion_extent = self.catchment_lidar.extents)

            # interpolate
            self.dense_dem.interpolate_offshore(self.bathy_contours, self.catchment_lidar.extents)

            # combine rasters
            self.result_dem = rioxarray.merge.merge_arrays([self.dense_dem.dem, self.dense_dem.offshore]) # should be the same for either (method='first' or 'last')
        else:
            self.result_dem = self.dense_dem.dem

        self.result_dem = self.result_dem.rio.interpolate_na()  
        self.result_dem = self.result_dem.rio.clip(self.catchment_geometry.catchment.geometry)

        ### save results
        self.result_dem.to_netcdf(self.instructions['instructions']['data_paths']['result_dem'])
