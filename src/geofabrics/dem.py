# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import rioxarray
import numpy
import pdal
import json
import typing
import pathlib
import scipy.interpolate 
from . import geometry

class ReferenceDem:
    """ A class to manage the reference DEM in a catchment context
    
    Specifically, clip within the catchment and outside any LiDAR. If
    set_foreshore is True set all positive DEM values in the foreshore to zero
    """
    
    def __init__(self, dem_file, catchment_geometry: geometry.CatchmentGeometry, set_foreshore: bool = True):
        """ Load in dem """
        
        self.catchment_geometry = catchment_geometry
        self.set_foreshore = set_foreshore
        with rioxarray.rioxarray.open_rasterio(dem_file, masked=True) as self._dem:
            self._dem.load()
        
        self._set_up()
        
        self._land = None
        self._foreshore = None
        
        
    def _set_up(self):
        """ Set dem crs and trim the dem to size """
        self._dem.rio.set_crs(self.catchment_geometry.crs);
        self._dem = self._dem.rio.clip(self.catchment_geometry.catchment.geometry)
        self._dem = self._dem.rio.clip([self.catchment_geometry.lidar_extents.loc[0].geometry], invert=True) 
        
    @property
    def land(self):
        """ Return the dem cells outside any LiDAR on land """
        
        if self._land is None:
            land_dem = self._dem.rio.clip(self.catchment_geometry.land.geometry)
            land_grid_x, land_grid_y = numpy.meshgrid(land_dem.x, land_dem.y)
            land_flat_z = land_dem.data[0].flatten()
            land_mask_z = ~numpy.isnan(land_flat_z)
            
            self._land = {'x': land_grid_x.flatten()[land_mask_z],
                          'y': land_grid_y.flatten()[land_mask_z], 
                          'z': land_flat_z[land_mask_z]}
        
        return self._land
    
    @property
    def foreshore(self):
        """ Return the dem cells outside any LiDAR on the foreshore """
        
        if self._foreshore is None:
            
            foreshore_dem = self._dem.rio.clip(self.catchment_geometry.foreshore_without_lidar.geometry)
            
            if self.set_foreshore:
                foreshore_dem.data[0][foreshore_dem.data[0]>0] = 0
            
            foreshore_grid_x, foreshore_grid_y = numpy.meshgrid(foreshore_dem.x, foreshore_dem.y)
            foreshore_flat_z = foreshore_dem.data[0].flatten()
            foreshore_mask_z = ~numpy.isnan(foreshore_flat_z)
            
            self._foreshore = {'x': foreshore_grid_x.flatten()[foreshore_mask_z],
                          'y': foreshore_grid_y.flatten()[foreshore_mask_z], 
                          'z': foreshore_flat_z[foreshore_mask_z]}
        
        return self._foreshore
    
    
class DenseDem:
    """ A class to manage the dense DEM in a catchment context
    
    Specifically, clip within the region there is dense data and provide the
    offshore edge of this dense region """
    
    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, temp_raster_path: typing.Union[str, pathlib.Path]):
        """ Load in dem """
        
        self.catchment_geometry = catchment_geometry
        self._dem = None
        
        self.raster_origin = None
        self.raster_size = None
        
        self._offshore_edge = None
        self._offshore = None
        self._offshore_interpolated = False
        
        self._set_up(temp_raster_path)
        
    def _set_up(self, temp_raster_path: typing.Union[str, pathlib.Path]):
        """ Create the dense DEM to file and define the raster size and origin """
        
        catchment_bounds = self.catchment_geometry.catchment.loc[0].geometry.bounds
        self.raster_origin = [catchment_bounds[0], 
                              catchment_bounds[1]]
        
        self.raster_size = [int((catchment_bounds[2] - 
                                 catchment_bounds[0]) / self.catchment_geometry.resolution), 
                            int((catchment_bounds[3] - 
                                 catchment_bounds[1]) / self.catchment_geometry.resolution)]
        
        ### create a dummy DEM for updated origin and size
        empty_points = numpy.zeros([1], dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": self.catchment_geometry.resolution, "gdalopts": "a_srs=EPSG:" + str(self.catchment_geometry.crs), 
             "output_type":["idw"], "filename": str(temp_raster_path), 
             "origin_x": self.raster_origin[0], "origin_y": self.raster_origin[1], 
             "width": self.raster_size[0], "height": self.raster_size[1]}
                ]
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [empty_points])
        pdal_pipeline.execute();
        metadata=json.loads(pdal_pipeline.get_metadata())
        with rioxarray.rioxarray.open_rasterio(metadata['metadata']['writers.gdal']['filename'][0], masked=True) as dem_temp:
            dem_temp.load()
            dem_temp.rio.set_crs(self.catchment_geometry.crs)
        if self.raster_origin[0] != dem_temp.x.data.min() or self.raster_origin[1] != dem_temp.y.data.min():
            raster_origin = [dem_temp.x.data.min() - self.catchment_geometry.resolution/2, 
                             dem_temp.y.data.min() - self.catchment_geometry.resolution/2]
            print('In process: The generated dense DEM has an origin differing from ' + 
                  'the one specified. Updating the catchment geometry raster origin from ' 
                  + str(self.raster_origin) + ' to ' + str(raster_origin))
            self.raster_origin = raster_origin
            
        # set empty dem - all nan - to add tiles too
        dem_temp.data[0] = numpy.nan 
        self._dem = dem_temp.rio.clip(self.catchment_geometry.catchment.geometry)
        
        # setup the empty offshore area ready for interpolation later
        self._offshore = self.dem.rio.clip(self.catchment_geometry.offshore.geometry);
        self._offshore.data[0] = 0 # set all to zero then clip out dense region where we don't need to interpolate
        self._offshore = self._offshore.rio.clip(self.catchment_geometry.offshore_dense_data.geometry);
        
    def add_tile(self, dem_file: str):
        """ Set dem crs and trim the dem to size """
        with rioxarray.rioxarray.open_rasterio(dem_file, masked=True) as tile:
            tile.load()
        tile.rio.set_crs(self.catchment_geometry.crs)
        
        # ensure the tile is lined up with the whole dense dem - i.e. that that raster orgin values match 
        raster_origin = [tile.x.data.min() - self.catchment_geometry.resolution/2, 
                         tile.y.data.min() - self.catchment_geometry.resolution/2]
        assert self.raster_origin[0] == raster_origin[0] and self.raster_origin[1] == raster_origin[1], "The generated tile is not " \
            + f"aligned with the overall dense dem. The DEM raster origin is {raster_origin} instead of {self.raster_origin}"
        
        # trim to only include cells where there is dense data then merge onto base dem
        tile = tile.rio.clip(self.catchment_geometry.dense_data_extents.geometry)
        self._dem = rioxarray.merge.merge_arrays([self._dem, tile], method='last')
        
        
    @property
    def dem(self):
        """ Return the dem """
        
        return self._dem
        
    @property
    def offshore_edge(self):
        """ Return the offshore edge cells to be used for offshore 
        interpolation """
        
        if self._offshore_edge is None:
            offshore_edge_dem = self._dem.rio.clip(self.catchment_geometry.offshore_edge_dense_data.geometry) 
            offshore_grid_x, offshore_grid_y = numpy.meshgrid(offshore_edge_dem.x, offshore_edge_dem.y)
            offshore_flat_z = offshore_edge_dem.data[0].flatten()
            offshore_mask_z = ~numpy.isnan(offshore_flat_z)
            
            self._offshore_edge = {'x': offshore_grid_x.flatten()[offshore_mask_z],
                                   'y': offshore_grid_y.flatten()[offshore_mask_z], 
                                   'z': offshore_flat_z[offshore_mask_z]}
            
        return self._offshore_edge
    
    def interpolate_offshore(self, bathy_contours):
        """ Performs interpolation offshore using the scipy Rbf function. """
        
        x = numpy.concatenate([self.offshore_edge['x'], bathy_contours.x])
        y = numpy.concatenate([self.offshore_edge['y'], bathy_contours.y])
        z = numpy.concatenate([self.offshore_edge['z'], bathy_contours.z])
        
        ### interpolate offshore
        rbf_function = scipy.interpolate.Rbf(x, y, z, function='linear')
        
        # Interpolate over offshore region
        grid_x, grid_y = numpy.meshgrid(self._offshore.x, self._offshore.y)
        flat_z = self._offshore.data[0].flatten()
        mask_z = ~numpy.isnan(flat_z)
        flat_z[mask_z] = rbf_function(grid_x.flatten()[mask_z], grid_y.flatten()[mask_z])
        self._offshore.data[0] = flat_z.reshape(self._offshore.data[0].shape)
        
        self._offshore_interpolated = True
    
    
    @property
    def offshore(self):
        """ Return the offshore dem - must be called after 
        'intepolate_offshore() """
        
        assert self._offshore_interpolated is True, "The offshore has to be interpolated explicitly"
    
        return self._offshore