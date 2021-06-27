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
from . import geometry
from . import lidar
from . import dem

class GeoFabricsGenerator:
    def __init__(self, json_instructions):
        self.instructions = json_instructions
        
    def run(self):
        
        ### instruction values and other set values
        area_to_drop = self.instructions['instructions']['instructions']['filter_lidar_holes_area'] if  \
            'filter_lidar_holes_area' in self.instructions['instructions']['instructions'] else None
        
        ### load in boundary data
        catchment_geometry = geometry.CatchmentGeometry(self.instructions['instructions']['data_paths']['catchment_boundary'],
                                                        self.instructions['instructions']['data_paths']['shoreline'], 
                                                        self.instructions['instructions']['projection'],
                                                        self.instructions['instructions']['grid_params']['resolution'], 
                                                        foreshore_buffer = 2, area_to_drop = area_to_drop)
        
        ### Other set values
        radius =  catchment_geometry.resolution * numpy.sqrt(2)
        window_size = 0
        idw_power = 2
        
        ### Load in LiDAR using PDAL
        catchment_lidar = lidar.CatchmentLidar(self.instructions['instructions']['data_paths']['lidars'][0], catchment_geometry)
        
        ### Load in reference DEM if any land/foreshore not covered by lidar
        if (catchment_geometry.foreshore_without_lidar.geometry.area.max() > 0) or (catchment_geometry.land_without_lidar.geometry.area.max() > 0):
            
            # if True set any dem values used along the foreshore to zero
            set_dem_foreshore = self.instructions['instructions']['instructions']['set_dem_shoreline'] if  \
                'set_dem_shoreline' in self.instructions['instructions']['instructions'] else True
        
            # Load in background DEM
            reference_dem = dem.ReferenceDem(self.instructions['instructions']['data_paths']['reference_dems'][0], catchment_geometry, set_dem_foreshore)
            
            # Get all DEM values
            dem_points = numpy.zeros_like(catchment_lidar.lidar_array, shape=[len(reference_dem.land['x']) + len(reference_dem.foreshore['x'])])
            dem_points['X'] = numpy.concatenate([reference_dem.land['x'], reference_dem.foreshore['x']])
            dem_points['Y'] = numpy.concatenate([reference_dem.land['y'], reference_dem.foreshore['y']])
            dem_points['Z'] = numpy.concatenate([reference_dem.land['z'], reference_dem.foreshore['z']])
            
            combined_dense_points_array = numpy.concatenate([catchment_lidar.lidar_array, dem_points])
            
        else:
           combined_dense_points_array = catchment_lidar.lidar_array     
        del catchment_lidar.lidar_array
        
        ### Create dense raster - note currently involves writing out a temp file
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": catchment_geometry.resolution, "gdalopts": "a_srs=EPSG:" + str(catchment_geometry.crs), "output_type":["idw"], 
             "filename": self.instructions['instructions']['data_paths']['tmp_raster_path'], 
             "window_size": window_size, "power": idw_power, "radius": radius, 
             "origin_x": catchment_geometry.raster_origin[0], "origin_y": catchment_geometry.raster_origin[1], 
             "width": catchment_geometry.raster_size[0], "height": catchment_geometry.raster_size[1]}
        ]
        
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [combined_dense_points_array])
        pdal_pipeline.execute();
        
        ### load in dense DEM 
        metadata=json.loads(pdal_pipeline.get_metadata())
        dense_dem = dem.DenseDem(metadata['metadata']['writers.gdal']['filename'][0], catchment_geometry)
        
        ### Load in bathy
        bathy_contours = geometry.BathymetryContours(self.instructions['instructions']['data_paths']['bathymetry_contours'][0], catchment_geometry, z_label = 'valdco')
        #bathy_points = geometry.BathymetryPoints(self.instructions['instructions']['data_paths']['bathymetry_points'][0], catchment_geometry)
        
        ### sparse/offshore interpolant
        dense_dem.interpolate_offshore(bathy_contours)
        
        ### combine rasters
        combined_dem = rioxarray.merge.merge_arrays([dense_dem.dem, dense_dem.offshore], method= "last") # important for this to be last as otherwise values that
        combined_dem_filled = combined_dem.rio.interpolate_na()
        
        ### save results
        combined_dem_filled.to_netcdf(self.instructions['instructions']['data_paths']['final_raster_path'])
        comp_dem = rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['final_raster_path_comp'], masked=True)
        
        print(numpy.max(numpy.abs(combined_dem_filled.data[0] 
                                  - comp_dem.data[0])))