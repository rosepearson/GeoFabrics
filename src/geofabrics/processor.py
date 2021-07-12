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
        
        ### Note corrently only consider one LiDAR dataset and only the first tile from that dataset. Later use all tiles and eventually combine datasets based on preference by data or extent.
        lidar_dataset_index = 0
        
        ### instruction values and other set values
        verbose = self.instructions['instructions']['instructions']['verbose'] if 'verbose' in \
            self.instructions['instructions']['instructions'] else True
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
            self.lidar_fetcher = lidar_fetch.OpenTopography(self.catchment_geometry, self.instructions['instructions']['data_paths']['local_cache'], verbose = verbose)
            self.lidar_fetcher.run()
            
            # take the first 
            lidar_file_paths = list(pathlib.Path(self.lidar_fetcher.cache_path / self.lidar_fetcher.dataset_prefixes[lidar_dataset_index]).glob('*.laz'))
        else:  # already downloaded - get the specified file path
            lidar_file_paths = self.instructions['instructions']['data_paths']['lidars']
            
            # setup dense dem object
        self.dense_dem = dem.DenseDem(self.catchment_geometry, self.instructions['instructions']['data_paths']['tmp_raster'])
            
        ### Load in LiDAR files using PDAL - in turn - and add to the dense_dem
        for index, lidar_file_path in enumerate(lidar_file_paths):
            if verbose:
                print(f"Looking at LiDAR tile {index} of {len(lidar_file_paths)}: {lidar_file_path}")
            
            catchment_lidar = lidar.CatchmentLidar(lidar_file_path, self.catchment_geometry)
            
            ### Create dense raster - note currently involves writing out a temp file
            temp_dem_file = pathlib.Path(self.instructions['instructions']['data_paths']['tmp_raster'])
            if temp_dem_file.exists():
                temp_dem_file.unlink()
            pdal_pipeline_instructions = [
                {"type":  "writers.gdal", "resolution": self.catchment_geometry.resolution, "gdalopts": "a_srs=EPSG:" + str(self.catchment_geometry.crs), "output_type":["idw"], 
                 "filename": str(temp_dem_file), 
                 "window_size": window_size, "power": idw_power, "radius": radius, 
                 "origin_x": self.dense_dem.raster_origin[0], "origin_y": self.dense_dem.raster_origin[1], 
                 "width": self.dense_dem.raster_size[0], "height": self.dense_dem.raster_size[1]}
            ]
            
            pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [catchment_lidar.lidar_array])
            pdal_pipeline.execute();
            del catchment_lidar.lidar_array
            
            ### load in dense DEM 
            metadata=json.loads(pdal_pipeline.get_metadata())
            self.dense_dem.add_tile(metadata['metadata']['writers.gdal']['filename'][0])
            
        
        ### Load in reference DEM if any land/foreshore not covered by lidar
        self.catchment_geometry.filter_lidar_extents_for_holes() 
        if (self.catchment_geometry.foreshore_without_lidar.geometry.area.max() > 0) or (self.catchment_geometry.land_without_lidar.geometry.area.max() > 0):
            
            # if True set any dem values used along the foreshore to zero
            set_dem_foreshore = self.instructions['instructions']['instructions']['set_dem_shoreline'] if  \
                'set_dem_shoreline' in self.instructions['instructions']['instructions'] else True
        
            # Load in background DEM
            self.reference_dem = dem.ReferenceDem(self.instructions['instructions']['data_paths']['reference_dems'][0], self.catchment_geometry, set_dem_foreshore)
            '''
            # Get all DEM values
            combined_dense_points_array = numpy.empty([len(self.dense_dem.points['x']) + len(self.reference_dem.land['x']) 
                                                       + len(self.reference_dem.foreshore['x'])],
                                                       dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])
            combined_dense_points_array['X'] = numpy.concatenate([self.dense_dem.points['x'], self.reference_dem.land['x'], self.reference_dem.foreshore['x']])
            combined_dense_points_array['Y'] = numpy.concatenate([self.dense_dem.points['y'], self.reference_dem.land['y'], self.reference_dem.foreshore['y']])
            combined_dense_points_array['Z'] = numpy.concatenate([self.dense_dem.points['z'], self.reference_dem.land['z'], self.reference_dem.foreshore['z']])
            '''
            # Get all DEM values
            combined_dense_points_array = numpy.empty([len(self.reference_dem.land['x']) + len(self.reference_dem.foreshore['x'])],
                                                       dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])
            combined_dense_points_array['X'] = numpy.concatenate([self.reference_dem.land['x'], self.reference_dem.foreshore['x']])
            combined_dense_points_array['Y'] = numpy.concatenate([self.reference_dem.land['y'], self.reference_dem.foreshore['y']])
            combined_dense_points_array['Z'] = numpy.concatenate([self.reference_dem.land['z'], self.reference_dem.foreshore['z']])
            
            ### Create dense raster - note currently involves writing out a temp file
            temp_dem_file = pathlib.Path(self.instructions['instructions']['data_paths']['tmp_raster'])
            if temp_dem_file.exists():
                temp_dem_file.unlink()
            pdal_pipeline_instructions = [
                {"type":  "writers.gdal", "resolution": self.catchment_geometry.resolution, "gdalopts": "a_srs=EPSG:" + str(self.catchment_geometry.crs), "output_type":["idw"], 
                 "filename": str(temp_dem_file), 
                 "window_size": window_size, "power": idw_power, "radius": radius, 
                 "origin_x": self.dense_dem.raster_origin[0], "origin_y": self.dense_dem.raster_origin[1], 
                 "width": self.dense_dem.raster_size[0], "height": self.dense_dem.raster_size[1]}
            ]
            
            pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [combined_dense_points_array])
            pdal_pipeline.execute();
            
            ### load in dense DEM - currently add the same as lidar. In future may want to treat differently - add with a separate routetine to add tile
            metadata=json.loads(pdal_pipeline.get_metadata())
            self.dense_dem.add_tile(metadata['metadata']['writers.gdal']['filename'][0])
        
        ### Load in bathy
        z_label = self.instructions['instructions']['instructions']['bathymetry_contours_z_label'] if  \
            'bathymetry_contours_z_label' in self.instructions['instructions']['instructions'] else None
        self.bathy_contours = geometry.BathymetryContours(self.instructions['instructions']['data_paths']['bathymetry_contours'][0], self.catchment_geometry, z_label = z_label)
        
        ### sparse/offshore interpolant
        self.dense_dem.interpolate_offshore(self.bathy_contours)
        
        ### combine rasters
        self.result_dem = rioxarray.merge.merge_arrays([self.dense_dem.dem, self.dense_dem.offshore]) # can be overlayed in either order (method='first' or 'last') as both clipped that the foreshore is the only overlap and that should be guarenteed to be the same. 
        self.result_dem = self.result_dem.rio.interpolate_na()  
        self.result_dem = self.result_dem.rio.clip(self.catchment_geometry.catchment.geometry)
        
        ### save results
        self.result_dem.to_netcdf(self.instructions['instructions']['data_paths']['result_dem'])
