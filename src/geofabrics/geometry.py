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

    Specifically, this defines regions like 'land', 'foreshore', 'offshore', and ensures all regions are defined in the
    same CRS.

    It also supports functions for determining how much of a a region is outside an exclusion zone. """

    def __init__(self, catchment_file: str, land_file: str, crs, resolution, foreshore_buffer=2):
        self._catchment = geopandas.read_file(catchment_file)
        self._land = geopandas.read_file(land_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer

        # values set in setup
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None

        self._set_up()

    def _set_up(self):
        """ Define the main catchment regions and set CRS """

        self._catchment = self._catchment.to_crs(self.crs)
        self._land = self._land.to_crs(self.crs)

        self._land = geopandas.clip(self._catchment, self._land)

        self._foreshore_and_offshore = geopandas.overlay(self.catchment, self.land, how='difference')

        self._land_and_foreshore = geopandas.GeoDataFrame(
            index=[0], geometry=self.land.buffer(self.resolution * self.foreshore_buffer), crs=self.crs)
        self._land_and_foreshore = geopandas.clip(self.catchment, self._land_and_foreshore)

        self._foreshore = geopandas.overlay(self.land_and_foreshore, self.land, how='difference')

        self._offshore = geopandas.overlay(self.catchment, self.land_and_foreshore, how='difference')

        assert len(self._catchment) == 1, "The catchment is made of multiple separate polygons it must be a single " + \
            "polygon object - a MultiPolygon is fine"

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

        return self._foreshore

    @property
    def land_and_foreshore(self):
        """ Return the catchment land and foreshore region """

        return self._land_and_foreshore

    @property
    def foreshore_and_offshore(self):
        """ Return the catchment foreshore and offshore region """

        return self._foreshore_and_offshore

    @property
    def offshore(self):
        """ Return the catchment offshore region """

        return self._offshore

    def land_and_foreshore_without_lidar(self, lidar_extents):
        """ Return the land and foreshore region without LiDAR """

        land_and_foreshore_with_lidar = geopandas.clip(lidar_extents, self.land_and_foreshore)
        land_and_foreshore_without_lidar = geopandas.overlay(
            self.land_and_foreshore, land_and_foreshore_with_lidar, how="difference")

        return land_and_foreshore_without_lidar

    def offshore_without_lidar(self, lidar_extents):
        """ Return the offshore region without LiDAR """

        offshore_with_lidar = geopandas.clip(lidar_extents, self.offshore)
        offshore_without_lidar = geopandas.overlay(self.offshore, offshore_with_lidar, how="difference")

        return offshore_without_lidar

    def offshore_dense_data_edge(self, lidar_extents):
        """ Return the offshore edge of where there is 'dense data' i.e. LiDAR or reference DEM """

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        # the foreshore and whatever lidar extents are offshore
        dense_data_extents = geopandas.GeoSeries(shapely.ops.cascaded_union([self.foreshore.loc[0].geometry,
                                                                             lidar_extents.loc[0].geometry]))
        dense_data_extents = geopandas.GeoDataFrame(index=[0], geometry=dense_data_extents, crs=self.crs)
        dense_data_extents = geopandas.clip(dense_data_extents, self.foreshore_and_offshore)

        # deflate this
        deflated_dense_data_extents = geopandas.GeoDataFrame(index=[0],
                                                             geometry=dense_data_extents.buffer(
                                                                 self.resolution * -1 * self.foreshore_buffer),
                                                             crs=self.crs)

        # get the difference between them
        offshore_dense_data_edge = geopandas.overlay(dense_data_extents, deflated_dense_data_extents, how='difference')
        offshore_dense_data_edge = geopandas.clip(offshore_dense_data_edge, self.foreshore_and_offshore)
        return offshore_dense_data_edge

    def offshore_no_dense_data(self, lidar_extents):
        """ Return the offshore area where there is no 'dense data' i.e. LiDAR """

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        # lidar extents are offshore
        offshore_dense_data = geopandas.clip(lidar_extents, self.offshore)

        # get the difference between them
        offshore_no_dense_data = geopandas.overlay(self.offshore, offshore_dense_data, how='difference')

        return offshore_no_dense_data


class BathymetryContours:
    """ A class working with bathymetry contours.

    Assumes contours to be sampled to the catchment_geometry resolution """

    def __init__(self, contour_file: str, catchment_geometry: CatchmentGeometry, z_label=None, exclusion_extent=None):
        self._contour = geopandas.read_file(contour_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label
        self._points_label = 'points'

        self._extents = None

        self._x = None
        self._y = None
        self._z = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """ Set CRS and clip to catchment """

        self._contour = self._contour.to_crs(self.catchment_geometry.crs)

        if exclusion_extent is not None:
            exclusion_extent = geopandas.clip(exclusion_extent, self.catchment_geometry.offshore)
            self._extent = geopandas.overlay(self.catchment_geometry.offshore, exclusion_extent, how="difference")
        else:
            self._extent = self.catchment_geometry.offshore

        self._contour = geopandas.clip(self._contour, self._extent)
        self._contour = self._contour.reset_index(drop=True)

        resolution = self.catchment_geometry.resolution
        assert self._points_label not in self._contour.columns, "The bathymetry data already has a points column that" \
            + " will be overridden"
        self._contour[self._points_label] = self._contour.geometry.apply(lambda row: shapely.geometry.MultiPoint(
            [row.interpolate(i * resolution) for i in range(int(numpy.ceil(row.length/resolution)))]))

    @property
    def points(self):
        """ Return the sampled points column with points along each contour """

        return self._contour[self._points_label]

    @property
    def x(self):
        """ The sampled contour x values """
        if self._x is None:
            self._x = numpy.concatenate(self.points.apply(lambda row: [row[i].x for i in range(len(row))]).to_list())

        return self._x

    @property
    def y(self):
        """ The sampled contour y values """
        if self._y is None:
            self._y = numpy.concatenate(self.points.apply(lambda row: [row[i].y for i in range(len(row))]).to_list())

        return self._y

    @property
    def z(self):
        """ The sampled contour z values """
        if self._z is None:
            # map depth to elevation
            if self.z_label is None:
                self._z = numpy.concatenate(self.points.apply(
                    lambda row: [row[i].z for i in range(len(row))]).to_list()) * -1
            else:
                self._z = numpy.concatenate(self._contour.apply(lambda row: (
                    row[self.z_label] * numpy.ones(len(row[self._points_label]))), axis=1).to_list()) * -1

        return self._z


class BathymetryPoints:
    """ A class working with bathymetry points """

    def __init__(self, points_file: str, catchment_geometry: CatchmentGeometry, exclusion_extent=None):
        self._points = geopandas.read_file(points_file)
        self.catchment_geometry = catchment_geometry

        self._extents = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """ Set CRS and clip to catchment """

        self._points = self._points.to_crs(self.catchment_geometry.crs)

        if exclusion_extent is not None:
            exclusion_extent = geopandas.clip(exclusion_extent, self.catchment_geometry.offshore)
            self._extent = geopandas.overlay(self.catchment_geometry.offshore, exclusion_extent, how="difference")
        else:
            self._extent = self.catchment_geometry.offshore

        self._points = geopandas.clip(self._points, self._extent)
        self._points = self._points.reset_index(drop=True)

    @property
    def points(self):
        """ Return the points """

        return self._points

    @property
    def x(self):
        """ The x values """

        if self._x is None:
            self._x = self._points.points.apply(lambda row: row['geometry'][0].x, axis=1).to_numpy()

        return self._x

    @property
    def y(self):
        """ The y values """

        if self._y is None:
            self._y = self._points.points.apply(lambda row: row['geometry'][0].y, axis=1).to_numpy()

        return self._y

    @property
    def z(self):
        """ The z values """

        if self._z is None:
            # map depth to elevation
            self._z = self._points.points.apply(lambda row: row['geometry'][0].z, axis=1).to_numpy() * -1

        return self._z


class TileInfo:
    """ A class for working with tiling information """

    def __init__(self, tile_file: str, catchment_geometry: CatchmentGeometry):
        self._tile_info = geopandas.read_file(tile_file)
        self.catchment_geometry = catchment_geometry

        self._set_up()

    def _set_up(self):
        """ Set CRS and select all tiles partially within the catchment """

        self._tile_info = self._tile_info.to_crs(self.catchment_geometry.crs)
        self._tile_info = geopandas.sjoin(self._tile_info, self.catchment_geometry.catchment)
        self._tile_info = self._tile_info.reset_index(drop=True)

    @property
    def tile_names(self):
        """ Return the names of all tiles within the catchment """

        return self._tile_info['Filename']
