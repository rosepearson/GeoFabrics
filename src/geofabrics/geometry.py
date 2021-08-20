# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import geopandas
import shapely
import numpy
import pathlib
import typing


class CatchmentGeometry:
    """ A class defining all relevant regions as defined by polygons in a catchment.

    The CRS is a dictionary containing the EPSG code for a 'horizontal' and 'vertical' datum.

    Specifically, this defines polygon regions like 'land', 'foreshore', 'offshore', and ensures all regions are
    defined in the same CRS.

    The land, foreshore and offshore regions are clipped within the catchment, but do not overlap. The land is
    polygon is defined by a specified polygon which is clipped within the catchment. The foreshore is defined
    as a 'foreshore_buffer' x 'resolution' outward buffer from the land and within the catchment extents. The offshore
    region is any remaining region within the catchment region and outside the land and foreshore polygons.

    It also supports functions for determining how much of a region is outside an exclusion zone. I.E. is outside
    `lidar_extents` see class method land_and_foreshore_without_lidar for an example.

    It is initalised with the catchment. The land must be added before the other regions (i.e. land, offshore,
    foreshore, etc) can be accessed. """

    def __init__(self, catchment_file: typing.Union[str, pathlib.Path], crs: dict, resolution: float,
                 foreshore_buffer: int = 2):
        self._catchment = geopandas.read_file(catchment_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer

        # set catchment CRS
        self._catchment = self._catchment.to_crs(self.crs['horizontal'])

        # values set in setup once land has been specified
        self._land = None
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None

    def _set_up(self):
        """ Define the main catchment regions and ensure the CRS is set for each region """

        self._land = self._land.to_crs(self.crs['horizontal'])

        self._land = geopandas.clip(self._catchment, self._land)

        self._foreshore_and_offshore = geopandas.overlay(self.catchment, self.land, how='difference')

        self._land_and_foreshore = geopandas.GeoDataFrame(
            index=[0], geometry=self.land.buffer(self.resolution * self.foreshore_buffer), crs=self.crs['horizontal'])
        self._land_and_foreshore = geopandas.clip(self.catchment, self._land_and_foreshore)

        self._foreshore = geopandas.overlay(self.land_and_foreshore, self.land, how='difference')

        self._offshore = geopandas.overlay(self.catchment, self.land_and_foreshore, how='difference')

        assert len(self._catchment) == 1, "The catchment is made of multiple separate polygons it must be a single " + \
            "polygon object - a MultiPolygon is fine"

    def _assert_land_set(self):
        assert self._land is not None, "Land has not been set yet. This must be set before anything other than the " + \
            "`catchment` can be returned from a `CatchmentGeometry` object"

    @property
    def catchment(self):
        """ Return the catchment region """

        return self._catchment

    @property
    def land(self):
        """ Return the catchment land region """

        self._assert_land_set()
        return self._land

    @land.setter
    def land(self, land_file: typing.Union[str, pathlib.Path]):
        """ Set the land region and finish setup. """

        self._land = geopandas.read_file(land_file)

        self._set_up()

    @property
    def foreshore(self):
        """ Return the catchment foreshore region """

        self._assert_land_set()
        return self._foreshore

    @property
    def land_and_foreshore(self):
        """ Return the catchment land and foreshore region """

        self._assert_land_set()
        return self._land_and_foreshore

    @property
    def foreshore_and_offshore(self):
        """ Return the catchment foreshore and offshore region """

        self._assert_land_set()
        return self._foreshore_and_offshore

    @property
    def offshore(self):
        """ Return the catchment offshore region """

        self._assert_land_set()
        return self._offshore

    def land_and_foreshore_without_lidar(self, lidar_extents):
        """ Return the land and foreshore region without LiDAR """

        self._assert_land_set()

        land_and_foreshore_with_lidar = geopandas.clip(lidar_extents, self.land_and_foreshore)
        land_and_foreshore_without_lidar = geopandas.overlay(
            self.land_and_foreshore, land_and_foreshore_with_lidar, how="difference")

        return land_and_foreshore_without_lidar

    def offshore_without_lidar(self, lidar_extents):
        """ Return the offshore region without LiDAR """

        self._assert_land_set()

        offshore_with_lidar = geopandas.clip(lidar_extents, self.offshore)
        offshore_without_lidar = geopandas.overlay(self.offshore, offshore_with_lidar, how="difference")

        return offshore_without_lidar

    def offshore_dense_data_edge(self, lidar_extents):
        """ Return the offshore edge of where there is 'dense data' i.e. LiDAR or reference DEM """

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        assert self._land is not None, "Land has not been set yet. This must be set before anything other than the " + \
            "`catchment` can be returned from a `CatchmentGeometry` object"

        # the foreshore and whatever lidar extents are offshore
        dense_data_extents = geopandas.GeoSeries(shapely.ops.cascaded_union([self.foreshore.loc[0].geometry,
                                                                             lidar_extents.loc[0].geometry]))
        dense_data_extents = geopandas.GeoDataFrame(index=[0], geometry=dense_data_extents, crs=self.crs['horizontal'])
        dense_data_extents = geopandas.clip(dense_data_extents, self.foreshore_and_offshore)

        # deflate this
        deflated_dense_data_extents = geopandas.GeoDataFrame(index=[0],
                                                             geometry=dense_data_extents.buffer(
                                                                 self.resolution * -1 * self.foreshore_buffer),
                                                             crs=self.crs['horizontal'])

        # get the difference between them
        offshore_dense_data_edge = geopandas.overlay(dense_data_extents, deflated_dense_data_extents, how='difference')
        offshore_dense_data_edge = geopandas.clip(offshore_dense_data_edge, self.foreshore_and_offshore)
        return offshore_dense_data_edge

    def offshore_no_dense_data(self, lidar_extents):
        """ Return the offshore area where there is no 'dense data' i.e. LiDAR """

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        assert self._land is not None, "Land has not been set yet. This must be set before anything other than the " + \
            "`catchment` can be returned from a `CatchmentGeometry` object"

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

        self._contour = self._contour.to_crs(self.catchment_geometry.crs['horizontal'])

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

        self._points = self._points.to_crs(self.catchment_geometry.crs['horizontal'])

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

        self.file_name = None

        self._set_up()

    def _set_up(self):
        """ Set CRS and select all tiles partially within the catchment, and look up the file column name """

        self._tile_info = self._tile_info.to_crs(self.catchment_geometry.crs['horizontal'])
        self._tile_info = geopandas.sjoin(self._tile_info, self.catchment_geometry.catchment)
        self._tile_info = self._tile_info.reset_index(drop=True)

        column_names = self._tile_info.columns

        column_name_matches = [name for name in column_names if "filename" == name.lower()]
        column_name_matches.extend([name for name in column_names if "file_name" == name.lower()])

        assert len(column_name_matches) == 1, "No single `file name` column detected in the tile file with" + \
            f" columns: {column_names}"
        self.file_name = column_name_matches[0]

    @property
    def tile_names(self):
        """ Return the names of all tiles within the catchment """

        return self._tile_info[self.file_name]
