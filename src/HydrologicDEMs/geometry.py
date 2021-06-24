# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import geopandas
import shapely

class CatchmentGeometry:
    """ A class defining all relevant regions in a catchment
    
    Specifically, this defines regions like 'land', 'foreshore', 'offshore'. 
    It also defines the extents of different data like lidar and any generated
    dense raster. Finally, it ensures all regions are defined in the same crs, 
    and defines the origin and extents of any generated raster.
    """
    
    def __init__(self, catchment_file, land_file, crs, resolution, foreshore_buffer = 2, area_to_drop = None):
        self._catchment = geopandas.read_file(catchment_file)
        self._land = geopandas.read_file(land_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer
        self.area_to_drop = area_to_drop
        
        
        self.__set_up()
        
        # values set only when called for the first time
        self._raster_origin = None 
        self._raster_size = None
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None
        
        # values that require load_lidar_extents to be called first
        self._lidar_extents = None
        self._foreshore_with_lidar = None
        self._foreshore_without_lidar = None
        self._dense_data_extents = None
        self._offshore_dense_data = None
        self._offshore_edge_dense_data = None
        
    def __set_up(self):
        """ Called in init to ensure correct setup of the input values """
        
        self._catchment = self._catchment.to_crs(self.crs)
        self._land = self._land.to_crs(self.crs)
        
        self._land = geopandas.clip(self._catchment, self._land)
        
    @property
    def raster_origin(self):
        """ Return the origin (LLHS) of the catchment bbox """
        
        if self._raster_origin is None:
            self._raster_origin = [self.catchment.loc[0].geometry.bounds[0], 
                                   self.catchment.loc[0].geometry.bounds[1]]
        return self._raster_origin

    @property
    def raster_size(self):
        """ Return the size of the catchment bbox in raster cells given the 
        resolution """
        
        if self._raster_size is None:
            self._raster_size = [int((self.catchment.loc[0].geometry.bounds[2] - 
                                     self.catchment.loc[0].geometry.bounds[0]) / self.resolution), 
                                int((self.catchment.loc[0].geometry.bounds[3] - 
                                     self.catchment.loc[0].geometry.bounds[1]) / self.resolution)]
        return self._raster_size
    
    @property
    def catchment(self):
        """ Return the catchment region """
        
        return self._catchment
    
    @property
    def land(self):
        """ Return the catchment land region """
        
        return self._land
    
    @property
    def foreshore(self):
        """ Return the catchment foreshore region """
        
        if self._foreshore is None:
            self._foreshore = geopandas.overlay(self.land_and_foreshore, 
                                                self.land, how='difference')
        return self._foreshore
    
    @property    
    def land_and_foreshore(self):
        """ Return the catchment and and foreshore region """
        
        if self._land_and_foreshore is None:
            self._land_and_foreshore = geopandas.GeoDataFrame(index=[0], geometry=self.land.buffer(self.resolution * self.foreshore_buffer), crs=self.crs)
            self._land_and_foreshore = geopandas.clip(self.catchment, self._land_and_foreshore)
        return self._land_and_foreshore
    
    @property
    def foreshore_and_offshore(self):
        """ Return the catchment land and offshore region """
        
        if self._foreshore_and_offshore is None:
            self._foreshore_and_offshore = geopandas.overlay(self.catchment, 
                                                self.land, how='difference')
        return self._foreshore_and_offshore
    
    @property
    def offshore(self):
        """ Return the catchment offshore region """
        
        if self._offshore is None:
            self._offshore = geopandas.overlay(self.catchment, 
                                                self.land_and_foreshore, how='difference')
        return self._offshore
    
    @property
    def lidar_extents(self):
        """ Return the catchment lidar extents """
        
        assert self._lidar_extents is not None, "lidar_extents have not been set, and need to be set explicitly"
            
        return self._lidar_extents
    
    def load_lidar_extents(self, lidar_extents_string): # expect to be filepath
        """ Load the lidar extents and clip within the catchment region """
        
        self._lidar_extents=shapely.wkt.loads(lidar_extents_string)
        
        if self.area_to_drop is None:
            area_to_drop = shapely.geometry.Polygon(self._lidar_extents.exterior).area
        
        self._lidar_extents = shapely.geometry.Polygon(self._lidar_extents.exterior.coords,
            [interior for interior in self._lidar_extents.interiors if shapely.geometry.Polygon(interior).area > area_to_drop])
        self._lidar_extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries([self._lidar_extents], crs=self.crs), crs=self.crs)
        self._lidar_extents = geopandas.clip(self._catchment, self._lidar_extents)
        
        
    @property
    def foreshore_with_lidar(self):
        """ Return the catchment foreshore within lidar region """
        
        if self._foreshore_with_lidar is None:
            self._foreshore_with_lidar = geopandas.clip(self.lidar_extents, self.foreshore)
        return self._foreshore_with_lidar    
    
    @property
    def foreshore_without_lidar(self):
        """ Return the catchment foreshore without lidar region """
        
        if self._foreshore_without_lidar is None:
            self._foreshore_without_lidar = geopandas.overlay(self.foreshore, self.foreshore_with_lidar, how="difference")
        return self._foreshore_without_lidar 
    
    @property
    def dense_data_extents(self):
        """ Return the extents of where 'dense data' exists """
        
        if self._dense_data_extents is None:
            self._dense_data_extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries(shapely.ops.cascaded_union([self.land_and_foreshore.loc[0].geometry, self.lidar_extents.loc[0].geometry])), crs=self.crs)
        return self._dense_data_extents 
    
    @property
    def offshore_dense_data(self):
        """ Return the region offshore of where 'dense data' exists """
        
        if self._offshore_dense_data is None:
            self._offshore_dense_data = geopandas.overlay(self.catchment, self.dense_data_extents, how='difference')
        return self._offshore_dense_data
    
    @property
    def offshore_edge_dense_data(self):
        """ Return the offshore edge of where there is 'dense data' """
        
        if self._offshore_edge_dense_data is None:
            deflated_dense_data = geopandas.GeoDataFrame(index=[0], geometry=self.dense_data_extents.buffer(self.resolution * -1 * self.foreshore_buffer), crs=self.crs)
            self._offshore_edge_dense_data = geopandas.overlay(self.dense_data_extents, deflated_dense_data, how='difference')
            self._offshore_edge_dense_data = geopandas.clip(self._offshore_edge_dense_data, self.foreshore_and_offshore)
        return self._offshore_edge_dense_data 


     
    
