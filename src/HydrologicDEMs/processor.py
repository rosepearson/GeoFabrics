# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import geopandas
import rioxarray
import rioxarray.merge
import pdal
import shapely
import scipy.interpolate
import numpy
import json
from . import geometry
from . import lidar
from . import dem

class GeoFabricsGenerator:
    def __init__(self, json_instructions):
        self.instructions = json_instructions
        
    def run(self):
        #print(self.instructions)
        
        ## load in boundary data
        catchment_geometry = geometry.CatchmentGeometry(self.instructions['instructions']['data_paths']['catchment_boundary'],
                                                        self.instructions['instructions']['data_paths']['shoreline'], 
                                                        self.instructions['instructions']['projection'],
                                                        self.instructions['instructions']['grid_params']['resolution'], foreshore_buffer = 2)
        
        ## key values in instructions
        window_size = 0
        idw_power = 2
        radius =  catchment_geometry.resolution * numpy.sqrt(2)
        set_dem_foreshore = True
        
        
        
        ### Load in LiDAR using PDAL
        catchment_lidar = lidar.CatchmentLidar(self.instructions['instructions']['data_paths']['lidars'][0], catchment_geometry)
        
        
        ### Load in background DEM
        reference_dem = dem.ReferenceDem(self.instructions['instructions']['data_paths']['reference_dems'][0], catchment_geometry, set_dem_foreshore)
        
        
        ### Dense raster generation
        dem_points = numpy.zeros_like(catchment_lidar.lidar_array, shape=[len(reference_dem.land['x']) + len(reference_dem.foreshore['x'])])
        dem_points['X'] = numpy.concatenate([reference_dem.land['x'], reference_dem.foreshore['x']])
        dem_points['Y'] = numpy.concatenate([reference_dem.land['y'], reference_dem.foreshore['y']])
        dem_points['Z'] = numpy.concatenate([reference_dem.land['z'], reference_dem.foreshore['z']])
        
        combined_dense_points_array = numpy.concatenate([catchment_lidar.lidar_array, dem_points])
        
        del catchment_lidar.lidar_array
       
        ### Create raster
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
        
        
        ### Load in and cut bathy
        bathy_contours = geometry.BathymetryContours(self.instructions['instructions']['data_paths']['bathymetry_contours'][0], catchment_geometry, z_label = 'valdco')
        #bathy_points = geometry.BathymetryPoints(self.instructions['instructions']['data_paths']['bathymetry_points'][0], catchment_geometry)S
        
        
        offshore_x = numpy.concatenate([dense_dem.offshore_edge['x'], bathy_contours.x])
        offshore_y = numpy.concatenate([dense_dem.offshore_edge['y'], bathy_contours.y])
        offshore_z = numpy.concatenate([dense_dem.offshore_edge['z'], bathy_contours.z])


        ### interpolate offshore
        offshore_dem=dense_dem.dem.copy()
        offshore_dem.rio.set_crs(catchment_geometry.crs)
        offshore_dem.data[0]=0
        offshore_dem = offshore_dem.rio.clip(catchment_geometry.offshore_dense_data.geometry);
        # interpolate
        offshore_rbf = scipy.interpolate.Rbf(offshore_x, offshore_y, offshore_z, function='linear')
        
        # evaluate rbf function
        dem_x, dem_y = numpy.meshgrid(offshore_dem.x, offshore_dem.y)
        dem_z = offshore_dem.data[0].flatten()
        dem_offshore_x = dem_x.flatten()[~numpy.isnan(dem_z)]
        dem_offshore_y = dem_y.flatten()[~numpy.isnan(dem_z)]
        dem_z[~numpy.isnan(dem_z)] = offshore_rbf(dem_offshore_x, dem_offshore_y)
        offshore_dem.data[0] = dem_z.reshape(dem_x.shape)
        
        ### combine rasters
        combined_dem = rioxarray.merge.merge_arrays([dense_dem.dem, offshore_dem], method= "last") # important for this to be last as otherwise values that
        combined_dem_filled = combined_dem.rio.interpolate_na()
        
        ### save results
        combined_dem_filled.to_netcdf(self.instructions['instructions']['data_paths']['final_raster_path'])
        comp_dem = rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['final_raster_path_comp'], masked=True)
        
        print(
            numpy.max(numpy.abs(combined_dem_filled.data[0] - comp_dem.data[0])))