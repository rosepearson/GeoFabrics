# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import geopandas
import shapely
import numpy

class CatchmentGeometry:
    """ A class defining all relevant regions in a catchment
    
    Specifically, this defines regions like 'land', 'foreshore', 'offshore'. 
    It also defines the extents of different data like lidar and any generated
    dense raster. Finally, it ensures all regions are defined in the same crs, 
    and defines the origin and extents of any generated raster.
    """
    
    def __init__(self, catchment_file: str, land_file: str, crs, resolution, foreshore_buffer = 2, area_to_drop = None):
        self._catchment = geopandas.read_file(catchment_file)
        self._land = geopandas.read_file(land_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer
        self.area_to_drop = area_to_drop
        
        
        self._set_up()
        
        # values set only when called for the first time
        self._raster_origin = None 
        self._raster_size = None
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None
        
        # values that require load_lidar_extents to be called first
        self._lidar_extents = None
        self._land_without_lidar = None
        self._foreshore_with_lidar = None
        self._foreshore_without_lidar = None
        self._dense_data_extents = None
        self._offshore_dense_data = None
        self._offshore_edge_dense_data = None
        
    def _set_up(self):
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
    
    @raster_origin.setter
    def raster_origin(self, raster_origin):
        """ Overwrite the raster origin - this is supported to ensure the 
        origin matches the generated dense_dem - as this seems to round in the 
        pdal writers.gdal case. """
        
        self._raster_origin = raster_origin

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
        """ Return the catchment land and foreshore region """
        
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
    
    def load_lidar_extents(self, lidar_extents_string: str): 
        """ Load the lidar extents and clip within the catchment region """
        
        self._lidar_extents=shapely.wkt.loads(lidar_extents_string)
        
        if self.area_to_drop is None:
            area_to_drop = shapely.geometry.Polygon(self._lidar_extents.exterior).area
        else:
            area_to_drop = self.area_to_drop
        
        self._lidar_extents = shapely.geometry.Polygon(self._lidar_extents.exterior.coords,
            [interior for interior in self._lidar_extents.interiors if shapely.geometry.Polygon(interior).area > area_to_drop])
        self._lidar_extents = geopandas.GeoDataFrame(index=[0], geometry=geopandas.GeoSeries([self._lidar_extents], crs=self.crs), crs=self.crs)
        self._lidar_extents = geopandas.clip(self._catchment, self._lidar_extents)
        
    @property
    def land_without_lidar(self):
        """ Return the catchment land without lidar """
        
        if self._land_without_lidar is None:
            land_with_lidar = geopandas.clip(self.lidar_extents, self.land)
            self._land_without_lidar = geopandas.overlay(self.land, land_with_lidar, how="difference")
        return self._land_without_lidar 
        
        
    @property
    def foreshore_with_lidar(self):
        """ Return the catchment foreshore within lidar region """
        
        if self._foreshore_with_lidar is None:
            self._foreshore_with_lidar = geopandas.clip(self.lidar_extents, self.foreshore)
        return self._foreshore_with_lidar    
    
    @property
    def foreshore_without_lidar(self):
        """ Return the catchment foreshore without lidar """
        
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


class BathymetryContours:
    """ A class working with bathymetry contours.
    
    Assumes contours to be sampled to the catchment_geometry resolution
    """
    
    def __init__(self, contour_file: str, catchment_geometry: CatchmentGeometry, z_label = None):
        self._contour = geopandas.read_file(contour_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label
        self._points_label = 'points'
        
        
        self._set_up()
        
        self._x = None
        self._y = None
        self._z = None
        
        
    def _set_up(self):
        """ Set crs and clip to catchment """
        
        self._contour = self._contour.to_crs(self.catchment_geometry.crs) 
        
        self._contour = geopandas.clip(self._contour, self.catchment_geometry.offshore_dense_data)
        self._contour = self._contour.reset_index(drop=True)
    
    @property
    def points(self):
        """ Return the offshore edge of where there is 'dense data' """
        
        resolution = self.catchment_geometry.resolution
        
        if 'points' not in self._contour.columns:
            self._contour[self._points_label]=self._contour.geometry.apply(lambda row : shapely.geometry.MultiPoint([ row.interpolate(i * resolution) for i in range(int(numpy.ceil(row.length/resolution)))]))
        return self._contour[self._points_label] 
    
    @property
    def x(self):
    
        if self._x is None:
            self._x = numpy.concatenate(self.points.apply(lambda row : [row[i].x for i in range(len(row))]).to_list())
            
        return self._x
            
    @property
    def y(self):
    
        if self._y is None:
            self._y = numpy.concatenate(self.points.apply(lambda row : [row[i].y for i in range(len(row))]).to_list())
            
        return self._y
    
    @property
    def z(self):
    
        if self._z is None:
            # map depth to elevatation
            if self.z_label is None:
                self._z = numpy.concatenate(self.points.apply(lambda row : [row[i].z for i in range(len(row))]).to_list()) * -1
            else:
                self._z = numpy.concatenate(self._contour.apply(lambda row : (row[self.z_label] * numpy.ones(len(row[self._points_label]))), axis=1).to_list()) * -1
        
        return self._z
    
    
class BathymetryPoints:
    """ A class working with bathymetry points """
    
    def __init__(self, points_file: str, catchment_geometry: CatchmentGeometry):
        self._points = geopandas.read_file(points_file)
        self.catchment_geometry = catchment_geometry
        
        
        self._set_up()
        
        
    def _set_up(self):
        """ Set crs and clip to catchment """
        
        self._points = self._points.to_crs(self.catchment_geometry.crs) 
        
        self._points = geopandas.clip(self._points, self.catchment_geometry.offshore_dense_data)
        self._points = self._points.reset_index(drop=True)
    
    @property
    def points(self):
        """ Return the points """
        
        return self._points
    
    @property
    def x(self):
    
        if self._x is None:
            self._x = self._points.points.apply(lambda row : row['geometry'][0].x,axis=1).to_numpy()
            
        return self._x
            
    @property
    def y(self):
    
        if self._y is None:
            self._y = self._points.points.apply(lambda row : row['geometry'][0].y,axis=1).to_numpy()
            
        return self._y
    
    @property
    def z(self):
    
        if self._z is None:
            # map depth to elevatation
            self._z = self._points.points.apply(lambda row : row['geometry'][0].z,axis=1).to_numpy() * -1 
        
        return self._z


class TileInfo:
    """ A class for working with tiling information
    """
    
    def __init__(self, tile_file: str, catchment_geometry: CatchmentGeometry):
        self._tile_info = geopandas.read_file(tile_file)
        self.catchment_geometry = catchment_geometry       
        
        self._set_up()
        
        
    def _set_up(self):
        """ Set crs and select all tiles partially within the catchment """
        
        self._tile_info = self._tile_info.to_crs(self.catchment_geometry.crs)
        self._tile_info = geopandas.sjoin(self._tile_info, self.catchment_geometry.catchment)
        self._tile_info = self._tile_info.reset_index(drop=True)
        
    @property
    def tile_names(self):
        """ Return the names of all tiles within the catchment"""
        
        return self._tile_info['Filename']