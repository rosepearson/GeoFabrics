# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import rioxarray
import rioxarray.merge
import pdal
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
    
    def __init__(self, json_instructions):
        self.instructions = json_instructions
        
        self.catchment_geometry = None
        self.dense_dem = None
        self.reference_dem = None
        self.bathy_contours = None
        self.result_dem = None
        
    def run(self):
        """ This method executes the geofabrics generation pipeline to produce
        geofabric derivatives. """
        
        ### instruction values and other set values
        area_to_drop = self.instructions['instructions']['instructions']['filter_lidar_holes_area'] if  \
            'filter_lidar_holes_area' in self.instructions['instructions']['instructions'] else None
        
        ### load in boundary data
        self.catchment_geometry = geometry.CatchmentGeometry(self.instructions['instructions']['data_paths']['catchment_boundary'],
                                                        self.instructions['instructions']['data_paths']['land'], 
                                                        self.instructions['instructions']['projection'],
                                                        self.instructions['instructions']['grid_params']['resolution'], 
                                                        foreshore_buffer = 2, area_to_drop = area_to_drop)
        
        
        ### Other set values
        radius =  self.catchment_geometry.resolution * numpy.sqrt(2)
        window_size = 0
        idw_power = 2
        
        ### Ensure LiDAR data is downloaded within the catchment region from OpenTopography
        if 'local_cache' in self.instructions['instructions']['data_paths']:  # download from OpenTopography - then get the local file path
            self.lidar_fetcher = lidar_fetch.OpenTopography(self.catchment_geometry, self.instructions['instructions']['data_paths']['local_cache'], verbose = True)
            self.lidar_fetcher.run()
            
            lidar_file_paths = list(pathlib.Path(self.lidar_fetcher.cache_path / self.lidar_fetcher.dataset_prefixes[0]).glob('*.laz'))
        else:  # already downloaded - get the specified file path
            lidar_file_paths = self.instructions['instructions']['data_paths']['lidars']
            
        ### Load in LiDAR files using PDAL - for now just take one to test basic pipeline
        catchment_lidar = lidar.CatchmentLidar(lidar_file_paths[0], self.catchment_geometry)
        
        ### Load in reference DEM if any land/foreshore not covered by lidar
        if (self.catchment_geometry.foreshore_without_lidar.geometry.area.max() > 0) or (self.catchment_geometry.land_without_lidar.geometry.area.max() > 0):
            
            # if True set any dem values used along the foreshore to zero
            set_dem_foreshore = self.instructions['instructions']['instructions']['set_dem_shoreline'] if  \
                'set_dem_shoreline' in self.instructions['instructions']['instructions'] else True
        
            # Load in background DEM
            self.reference_dem = dem.ReferenceDem(self.instructions['instructions']['data_paths']['reference_dems'][0], self.catchment_geometry, set_dem_foreshore)
            
            # Get all DEM values
            dem_points = numpy.zeros_like(catchment_lidar.lidar_array, shape=[len(self.reference_dem.land['x']) + len(self.reference_dem.foreshore['x'])])
            dem_points['X'] = numpy.concatenate([self.reference_dem.land['x'], self.reference_dem.foreshore['x']])
            dem_points['Y'] = numpy.concatenate([self.reference_dem.land['y'], self.reference_dem.foreshore['y']])
            dem_points['Z'] = numpy.concatenate([self.reference_dem.land['z'], self.reference_dem.foreshore['z']])
            
            combined_dense_points_array = numpy.concatenate([catchment_lidar.lidar_array, dem_points])  
        else:
           combined_dense_points_array = catchment_lidar.lidar_array     
        del catchment_lidar.lidar_array
        
        ### Create dense raster - note currently involves writing out a temp file
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": self.catchment_geometry.resolution, "gdalopts": "a_srs=EPSG:" + str(self.catchment_geometry.crs), "output_type":["idw"], 
             "filename": self.instructions['instructions']['data_paths']['tmp_raster'], 
             "window_size": window_size, "power": idw_power, "radius": radius, 
             "origin_x": self.catchment_geometry.raster_origin[0], "origin_y": self.catchment_geometry.raster_origin[1], 
             "width": self.catchment_geometry.raster_size[0], "height": self.catchment_geometry.raster_size[1]}
        ]
        
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [combined_dense_points_array])
        pdal_pipeline.execute();
        
        ### load in dense DEM 
        metadata=json.loads(pdal_pipeline.get_metadata())
        self.dense_dem = dem.DenseDem(metadata['metadata']['writers.gdal']['filename'][0], self.catchment_geometry)
        
        ### Load in bathy
        z_label = self.instructions['instructions']['instructions']['bathymetry_contours_z_label'] if  \
            'bathymetry_contours_z_label' in self.instructions['instructions']['instructions'] else None
        self.bathy_contours = geometry.BathymetryContours(self.instructions['instructions']['data_paths']['bathymetry_contours'][0], self.catchment_geometry, z_label = z_label)
        
        ### sparse/offshore interpolant
        self.dense_dem.interpolate_offshore(self.bathy_contours)
        
        ### combine rasters
        self.result_dem = rioxarray.merge.merge_arrays([self.dense_dem.dem, self.dense_dem.offshore], method= "last") # important for this to be last as otherwise values that
        self.result_dem = self.result_dem.rio.interpolate_na()  
        self.result_dem = self.result_dem.rio.clip(self.catchment_geometry.catchment.geometry)
        
        ### save results
        self.result_dem.to_netcdf(self.instructions['instructions']['data_paths']['result_dem'])
