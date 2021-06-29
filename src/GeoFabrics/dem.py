# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import rioxarray
import numpy
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
        self._dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True)
        
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
    
    def __init__(self, dem_file: str, catchment_geometry: geometry.CatchmentGeometry):
        """ Load in dem """
        
        self.catchment_geometry = catchment_geometry
        self._dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True)
        
        self._set_up()
        
        self._offshore_edge = None
        self._offshore = None
        
    def _set_up(self):
        """ Set dem crs and trim the dem to size """
        self._dem.rio.set_crs(self.catchment_geometry.crs);
        
        # trim to only include cells where there is dense data
        self._dem = self._dem.rio.clip(self.catchment_geometry.dense_data_extents.geometry)
        
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
        
        # initalise offshore dem
        self._offshore = self.dem.rio.clip(self.catchment_geometry.offshore.geometry);
        self._offshore.data[0] = 0 # set all to zero then clip out dense region where we don't need to interpolate
        self._offshore = self._offshore.rio.clip(self.catchment_geometry.offshore_dense_data.geometry);
        
        grid_x, grid_y = numpy.meshgrid(self._offshore.x, self._offshore.y)
        flat_z = self._offshore.data[0].flatten()
        mask_z = ~numpy.isnan(flat_z)
        flat_z[mask_z] = rbf_function(grid_x.flatten()[mask_z], grid_y.flatten()[mask_z])
        self._offshore.data[0] = flat_z.reshape(self._offshore.data[0].shape)
    
    @property
    def offshore(self):
        """ Return the offshore dem - must be called after 
        'intepolate_offshore() """
        
        assert self._offshore is not None, "The offshore has to be interpolated explicitly"
    
        return self._offshore