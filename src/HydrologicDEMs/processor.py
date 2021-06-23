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

class GeoFabricsGenerator:
    def __init__(self, json_instructions):
        self.instructions = json_instructions
        
    def run(self):
        print(self.instructions)
        
        ## key values in instructions
        crs = self.instructions['instructions']['projection']
        resolution = self.instructions['instructions']['grid_params']['resolution']
        
        window_size = 0
        idw_power = 2
        radius =  resolution * numpy.sqrt(2)
        foreshore_buffer = 2
        
        ## load in boundary data
        catchment = geopandas.read_file(self.instructions['instructions']['data_paths']['catchment_boundary']) # should check thir is only one boundary polygon
        land = geopandas.read_file(self.instructions['instructions']['data_paths']['shoreline'])
        land = land.to_crs(crs)
        
        # define raster dimensions - if not specified in the instruction file  
        raster_origin = [catchment.loc[0].geometry.bounds[0], catchment.loc[0].geometry.bounds[1]]
        raster_size = [int((catchment.loc[0].geometry.bounds[2] - catchment.loc[0].geometry.bounds[0]) / resolution), int((catchment.loc[0].geometry.bounds[3] - catchment.loc[0].geometry.bounds[1]) / resolution)]
        
        # define land, foreshore and offshore
        land = geopandas.clip(catchment, land)
        land_and_foreshore = geopandas.GeoDataFrame(index=[0], geometry=land.buffer(resolution * foreshore_buffer), crs=crs)
        land_and_foreshore = geopandas.clip(catchment, land_and_foreshore)
        foreshore = geopandas.overlay(land_and_foreshore, land, how='difference')
        foreshore_and_offshore = geopandas.overlay(catchment, land, how='difference')
        offshore = geopandas.overlay(catchment, land_and_foreshore, how='difference')
        
        ### Load in LiDAR using PDAL
        pdal_pipeline_instructions = [
            {"type":  "readers.las", "filename": self.instructions['instructions']['data_paths']['lidars'][0]},
            {"type":"filters.reprojection","out_srs":"EPSG:" + str(crs)}, # reproject to NZTM
            {"type":"filters.crop", "polygon":str(catchment.loc[0].geometry)}, # filter within boundary
            {"type" : "filters.hexbin"} # create a polygon boundary of the LiDAR
        ]
        
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
        pdal_pipeline.execute();
        
        # pull out key LiDAR results
        metadata=json.loads(pdal_pipeline.get_metadata())
        lidar_array = pdal_pipeline.arrays[0]
        
        # define LiDAR extents
        lidar_extents=shapely.wkt.loads(metadata['metadata']['filters.hexbin']['boundary'])
        
        # filter out holes in the middle of the LiDAR
        if self.instructions['instructions']['instructions'].get('filter_holes_in_lidar') != None:
            # drop interio holes under a certain area
            area_to_drop = self.instructions['instructions']['instructions']['filter_holes_in_lidar']['area_to_drop']
        else:
            # or, drop all interior holes
            area_to_drop = shapely.geometry.Polygon(lidar_extents.exterior).area 
        lidar_extents = shapely.geometry.Polygon(lidar_extents.exterior.coords,
            [interior for interior in lidar_extents.interiors if shapely.geometry.Polygon(interior).area > area_to_drop])
        lidar_extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries([lidar_extents], crs=crs), crs=crs)
        
        ### Update geometries to include filtering by lidar
        foreshore_with_lidar = geopandas.clip(lidar_extents, foreshore)
        foreshore_without_lidar = geopandas.overlay(foreshore, foreshore_with_lidar, how="difference")
        
        
        ### Load in background DEM
        reference_dem = rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['reference_dems'][0], masked=True)
        reference_dem.rio.set_crs(crs);
        
        # filter spatially
        reference_dem = reference_dem.rio.clip(catchment.geometry)
        reference_dem = reference_dem.rio.clip([lidar_extents.loc[0].geometry], invert=True) # should clip with the biggest or maybe each in turn?
        reference_dem_land = reference_dem.rio.clip(catchment.geometry)
        reference_dem_foreshore = reference_dem.rio.clip(foreshore_without_lidar.geometry)
        
        # set values to zero on foreshore
        reference_dem_foreshore.data[0][reference_dem_foreshore.data[0]>0] = 0
        
        
        
        ### Dense raster generation
        dem_x, dem_y = numpy.meshgrid(reference_dem_land.x, reference_dem_land.y)
        dem_z = reference_dem_land.data[0].flatten()
        dem_land_x = dem_x.flatten()[~numpy.isnan(dem_z)]
        dem_land_y = dem_y.flatten()[~numpy.isnan(dem_z)]
        dem_land_z = dem_z[~numpy.isnan(dem_z)]
        
        
        dem_x, dem_y = numpy.meshgrid(reference_dem_foreshore.x, reference_dem_foreshore.y)
        dem_z = reference_dem_foreshore.data[0].flatten()
        dem_foreshore_x = dem_x.flatten()[~numpy.isnan(dem_z)]
        dem_foreshore_y = dem_y.flatten()[~numpy.isnan(dem_z)]
        dem_foreshore_z = dem_z[~numpy.isnan(dem_z)]
        
        dem_points = numpy.zeros_like(lidar_array, shape=[len(dem_land_x) + len(dem_foreshore_x)])
        dem_points['X'] = numpy.concatenate([dem_land_x, dem_foreshore_x])
        dem_points['Y'] = numpy.concatenate([dem_land_y, dem_foreshore_y])
        dem_points['Z'] = numpy.concatenate([dem_land_z, dem_foreshore_z])
        
        combined_dense_points_array = numpy.concatenate([lidar_array, dem_points])
        
       
        ### Create raster
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": resolution, "gdalopts": "a_srs=EPSG:" + str(crs), "output_type":["idw"], 
             "filename": self.instructions['instructions']['data_paths']['tmp_raster_path'], 
             "window_size": window_size, "power": idw_power, "radius": radius, 
             "origin_x": raster_origin[0], "origin_y": raster_origin[1], "width": raster_size[0], "height": raster_size[1]}
        ]
        
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [combined_dense_points_array])
        pdal_pipeline.execute();
        
        
        ### Define geometries where the dense dem
        dense_dem_extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries(shapely.ops.cascaded_union([land_and_foreshore.loc[0].geometry, lidar_extents.loc[0].geometry])), crs=crs)
        offshore_dense_dem = geopandas.overlay(catchment, dense_dem_extents, how='difference')
        deflated_dense_dem = geopandas.GeoDataFrame(index=[0], geometry=dense_dem_extents.buffer(resolution * -1 * foreshore_buffer), crs=crs)
        offshore_edge_dense_dem = geopandas.overlay(dense_dem_extents, deflated_dense_dem, how='difference')
        offshore_edge_dense_dem = geopandas.clip(offshore_edge_dense_dem, foreshore_and_offshore)
        
        
        ### load in dense DEM 
        metadata=json.loads(pdal_pipeline.get_metadata())
        dense_dem = rioxarray.rioxarray.open_rasterio(metadata['metadata']['writers.gdal']['filename'][0], masked=True)
        dense_dem.rio.set_crs(crs);
        # trim
        dense_dem_offshore_edge = dense_dem.rio.clip(offshore_edge_dense_dem.geometry) # the bit to use for interpolation
        
        ### Load in and cut bathy
        bathy_countours = geopandas.read_file(self.instructions['instructions']['data_paths']['bathymetry_contours'][0])
        bathy_points = geopandas.read_file(self.instructions['instructions']['data_paths']['bathymetry_points'][0])
        bathy_countours = bathy_countours.to_crs(crs)
        bathy_points = bathy_points.to_crs(crs)
        
        # trim
        bathy_points = geopandas.clip(bathy_points, offshore_dense_dem)
        bathy_points = bathy_points.reset_index(drop=True)
        
        bathy_countours = geopandas.clip(bathy_countours, offshore_dense_dem)
        bathy_countours = bathy_countours.reset_index(drop=True)
        
        # sub sample the contours 
        bathy_countours['points']=bathy_countours.geometry.apply(lambda row : shapely.geometry.MultiPoint([ row.interpolate(i * resolution) for i in range(int(numpy.ceil(row.length/resolution)))]))
        
        
        ### combine bathy and offshore edge of the dense raster
        lidar_array = pdal_pipeline.arrays[0]

        dem_x, dem_y = numpy.meshgrid(dense_dem_offshore_edge.x, dense_dem_offshore_edge.y)
        dem_z = dense_dem_offshore_edge.data[0].flatten()
        dense_dem_foreshore_x = dem_x.flatten()[~numpy.isnan(dem_z)]
        dense_dem_foreshore_y = dem_y.flatten()[~numpy.isnan(dem_z)]
        dense_dem_foreshore_z = dem_z[~numpy.isnan(dem_z)]
        
        '''bathymetry_x = bathy_points.apply(lambda x : x['geometry'][0].x,axis=1).to_numpy()
        bathymetry_y = bathy_points.apply(lambda x : x['geometry'][0].y,axis=1).to_numpy()
        bathymetry_z = bathy_points.apply(lambda x : x['geometry'][0].z,axis=1).to_numpy() * -1 # map depth to elevatation'''
        
        bathy_x = numpy.concatenate(bathy_countours['points'].apply(lambda row : [row[i].x for i in range(len(row))]).to_list())
        bathy_y = numpy.concatenate(bathy_countours['points'].apply(lambda row : [row[i].y for i in range(len(row))]).to_list())
        bathy_z = numpy.concatenate(bathy_countours.apply(lambda row : (row['valdco'] * numpy.ones(len(row['points']))), axis=1).to_list()) * -1 # map depth to elevatation
        
        offshore_x = numpy.concatenate([dense_dem_foreshore_x, bathy_x])
        offshore_y = numpy.concatenate([dense_dem_foreshore_y, bathy_y])
        offshore_z = numpy.concatenate([dense_dem_foreshore_z, bathy_z])


        ### interpolate offshore
        offshore_dem=dense_dem.copy()
        offshore_dem.rio.set_crs(dense_dem.rio.crs)
        offshore_dem.data[0]=0
        offshore_dem = offshore_dem.rio.clip(offshore_dense_dem.geometry);
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
        combined_dem = rioxarray.merge.merge_arrays([dense_dem, offshore_dem], method= "last") # important for this to be last as otherwise values that
        combined_dem_filled = combined_dem.rio.interpolate_na()
        
        ### save results
        combined_dem_filled.to_netcdf(self.instructions['instructions']['data_paths']['final_raster_path'])