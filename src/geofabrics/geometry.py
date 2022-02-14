# -*- coding: utf-8 -*-
"""
This module contains classes associated with manipulating vector data.
"""
import geopandas
import shapely
import numpy
import pathlib
import typing
import logging


class CatchmentGeometry:
    """ A class defining all relevant regions as defined by polygons in a catchment.

    The CRS is a dictionary containing the EPSG code for a 'horizontal' and 'vertical' datum.

    Specifically, this defines polygon regions like 'land', 'foreshore', 'offshore', and ensures all regions are
    defined in the same CRS.

    The 'land', 'foreshore' and 'offshore' regions are clipped within the 'catchment', but do not overlap. The 'land' is
    polygon is defined by a specified polygon which is clipped within the 'catchment'. The 'foreshore' is defined
    as a 'foreshore_buffer' x 'resolution' outward buffer from the 'land' and within the 'catchment' extents. The
    'offshore' region is any remaining region within the 'catchment' and outside the 'land' and 'foreshore' polygons.

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

        # Clip and remove any sub pixel regions
        self._land = self._catchment.clip(self._land, keep_geom_type=True)
        self._land = self._land[self._land.area > self.resolution * self.resolution]

        self._foreshore_and_offshore = self.catchment.overlay(self.land, how='difference')

        # Buffer and clip and remove any sub pixel regions
        self._land_and_foreshore = geopandas.GeoDataFrame(
            {'geometry': self.land.buffer(self.resolution * self.foreshore_buffer)}, crs=self.crs['horizontal'])
        self._land_and_foreshore = self._catchment.clip(self._land_and_foreshore, keep_geom_type=True)
        self._land_and_foreshore = self._land_and_foreshore[self._land_and_foreshore.area
                                                            > self.resolution * self.resolution]

        self._foreshore = self.land_and_foreshore.overlay(self.land, how='difference')

        self._offshore = self.catchment.overlay(self.land_and_foreshore, how='difference')

        assert len(self._catchment) == 1, "The catchment is made of multiple separate polygons it must be a single " + \
            "polygon object - a MultiPolygon is fine"

    def _assert_land_set(self):
        """ Check to make sure the 'land' has been set. """
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

    def land_and_foreshore_without_lidar(self, dense_extents: geopandas.GeoDataFrame):
        """ Return the land and foreshore region without LiDAR """

        self._assert_land_set()

        if dense_extents is None:
            logging.warning("In CatchmentGeometry dense extents are `None` so `land_and_foreshore_without_lidar`" +
                            "is returning `land_and_foreshore`")
            return self.land_and_foreshore

        # Clip to remove any offshore regions before doing a difference overlay. Drop any sub-pixel polygons.
        land_and_foreshore_with_lidar = dense_extents.clip(self.land_and_foreshore, keep_geom_type=True)
        land_and_foreshore_with_lidar = land_and_foreshore_with_lidar[land_and_foreshore_with_lidar.area
                                                                      > self.resolution * self.resolution]
        land_and_foreshore_without_lidar = self.land_and_foreshore.overlay(
            land_and_foreshore_with_lidar, how="difference")

        return land_and_foreshore_without_lidar

    def offshore_without_lidar(self, dense_extents: geopandas.GeoDataFrame):
        """ Return the offshore region without LiDAR """

        self._assert_land_set()

        if dense_extents is None:
            logging.warning("In CatchmentGeometry dense extents are `None` so `offshore_without_lidar`" +
                            "is returning `offshore`")
            return self.offshore
        elif self.offshore.area.sum() == 0:  # There is no offshore region - return an empty dataframe
            return self.offshore

        # Clip to remove any offshore regions before doing a difference overlay. Drop any sub-pixel polygons.
        offshore_with_lidar = dense_extents.clip(self.offshore, keep_geom_type=True)
        offshore_with_lidar = offshore_with_lidar[offshore_with_lidar.area > self.resolution ** 2]
        offshore_without_lidar = geopandas.overlay(self.offshore, offshore_with_lidar, how="difference")

        return offshore_without_lidar

    def offshore_dense_data_edge(self, dense_extents: geopandas.GeoDataFrame):
        """ Return the offshore edge of where there is 'dense data' i.e. LiDAR or reference DEM """

        self._assert_land_set()

        if dense_extents is None:
            logging.warning("In CatchmentGeometry dense extents are `None` so `offshore_dense_data_edge`" +
                            "is returning `None`")
            return None

        # the foreshore and whatever lidar extents are offshore
        offshore_foreshore_dense_data_extents = geopandas.GeoDataFrame(
            {'geometry': [shapely.ops.cascaded_union([self.foreshore.loc[0].geometry, dense_extents.loc[0].geometry])]},
            crs=self.crs['horizontal'])
        offshore_foreshore_dense_data_extents = offshore_foreshore_dense_data_extents.clip(self.foreshore_and_offshore,
                                                                                           keep_geom_type=True)

        # deflate this - this will be taken away from the offshore_foreshore_dense_data_extents to give the edge
        deflated_dense_data_extents = geopandas.GeoDataFrame(
                {'geometry': offshore_foreshore_dense_data_extents.buffer(self.resolution * - self.foreshore_buffer)},
                crs=self.crs['horizontal'])

        # get the difference between them
        if deflated_dense_data_extents.area.sum() > 0:
            offshore_dense_data_edge = offshore_foreshore_dense_data_extents.overlay(
                deflated_dense_data_extents, how='difference')
        else:
            offshore_dense_data_edge = offshore_foreshore_dense_data_extents
        offshore_dense_data_edge = offshore_dense_data_edge.clip(self.foreshore_and_offshore, keep_geom_type=True)
        return offshore_dense_data_edge

    def offshore_no_dense_data(self, lidar_extents):
        """ Return the offshore area where there is no 'dense data' i.e. LiDAR """

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        assert self._land is not None, "Land has not been set yet. This must be set before anything other than the " + \
            "`catchment` can be returned from a `CatchmentGeometry` object"

        # lidar extents are offshore - drop any sub pixel areas
        offshore_dense_data = lidar_extents.clip(self.offshore, keep_geom_type=True)
        offshore_dense_data = offshore_dense_data[offshore_dense_data.area > self.resolution * self.resolution]

        # get the difference between them
        offshore_no_dense_data = self.offshore.overlay(offshore_dense_data, how='difference')

        return offshore_no_dense_data


class BathymetryContours:
    """ A class working with bathymetry contours.

    Assumes contours to be sampled to the catchment_geometry resolution """

    def __init__(self, contour_file: str, catchment_geometry: CatchmentGeometry, z_label=None, exclusion_extent=None):
        self._contour = geopandas.read_file(contour_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label

        self._extent = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """ Set CRS and clip to catchment """

        self._contour = self._contour.to_crs(self.catchment_geometry.crs['horizontal'])

        if exclusion_extent is not None:
            # Remove areas already covered by LiDAR - drop any polygons less than a pixel in area
            exclusion_extent = exclusion_extent.clip(self.catchment_geometry.offshore, keep_geom_type=True)
            exclusion_extent = exclusion_extent[exclusion_extent.area > self.catchment_geometry.resolution ** 2]
            self._extent = self.catchment_geometry.offshore.overlay(exclusion_extent, how="difference")
        else:
            self._extent = self.catchment_geometry.offshore

        # Keep only contours in the 'extents' i.e. inside the catchment and outside any exclusion_extent
        self._contour = self._contour.clip(self._extent, keep_geom_type=True)
        self._contour = self._contour.reset_index(drop=True)

        # Convert any 'GeometryCollection' objects to 'MultiLineString' objects - dropping any points
        if (self._contour.geometry.type == 'GeometryCollection').any():
            geometry_list = []
            for geometry_row in self._contour.geometry:
                if geometry_row.geometryType() in {'LineString', 'MultiLineString'}:
                    geometry_list.append(geometry_row)
                elif geometry_row.geometryType() == 'GeometryCollection':
                    geometry_list.append(
                        shapely.geometry.MultiLineString([geometry_element for geometry_element in geometry_row if
                                                          geometry_element.geometryType() == 'LineString']))
            self._contour.set_geometry(geometry_list, inplace=True)

    def sample_contours(self, resolution: float) -> numpy.ndarray:
        """ Sample the contours at the specified resolution. """

        assert resolution > 0, f"The sampling resolution must be greater than 0. Instead, it is {resolution}."

        points_df = self._contour.geometry.apply(lambda row: shapely.geometry.MultiPoint(
            [row.interpolate(i * resolution) for i in range(int(numpy.ceil(row.length/resolution)))]))

        points = numpy.empty([points_df.apply(lambda row: len(row)).sum()],
                             dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a depth column
        points['X'] = numpy.concatenate(points_df.apply(lambda row: [row[i].x for i in range(len(row))]).to_list())
        points['Y'] = numpy.concatenate(points_df.apply(lambda row: [row[i].y for i in range(len(row))]).to_list())
        if self.z_label is None:
            points['Z'] = numpy.concatenate(points_df.apply(lambda row:
                                                            [row[i].z for i in range(len(row))]).to_list()) * -1
        else:
            points['Z'] = numpy.concatenate([numpy.ones(len(points_df.loc[i])) * self._contour[self.z_label].loc[i]
                                             for i in range(len(points_df))]) * -1

        return points


class BathymetryPoints:
    """ A class working with bathymetry points """

    def __init__(self, points_file: str, catchment_geometry: CatchmentGeometry, exclusion_extent=None):
        self._points = geopandas.read_file(points_file)
        self.catchment_geometry = catchment_geometry

        self._extent = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """ Set CRS and clip to catchment """

        self._points = self._points.to_crs(self.catchment_geometry.crs['horizontal'])

        if exclusion_extent is not None:
            exclusion_extent = exclusion_extent.clip(self.catchment_geometry.offshore, keep_geom_type=True)
            self._extent = self.catchment_geometry.offshore.overlay(exclusion_extent, how="difference")
        else:
            self._extent = self.catchment_geometry.offshore

        self._points = self._points.clip(self._extent, keep_geom_type=True)
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


class RiverBathymetryPoints:
    """ A class working with bathymetry points """

    def __init__(self,
                 points_file: str,
                 polygon_file: str,
                 catchment_geometry: CatchmentGeometry,
                 z_label: str = None):
        self._points = geopandas.read_file(points_file)
        self.polygon = geopandas.read_file(polygon_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label

        self._set_up()

    def _set_up(self):
        """ Set CRS and clip to catchment and within the flat water polygon """

        self._points = self._points.to_crs(self.catchment_geometry.crs['horizontal'])
        self._polygon = self._points.to_crs(self.catchment_geometry.crs['horizontal'])

        self._points = self._points.clip(self.polygon, keep_geom_type=True)
        self._points = self._points.clip(self.catchment_geometry.catchment, keep_geom_type=True)
        self._points = self._points.reset_index(drop=True)

    def points_array(self) -> numpy.ndarray:
        """ Sample the contours at the specified resolution. """

        points = numpy.empty([len(self._points)],
                             dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a depth column
        points['X'] = self._points.apply(lambda row: row.geometry.x, axis=1).to_list()
        points['Y'] = self._points.apply(lambda row: row.geometry.y, axis=1).to_list()
        if self.z_label is None:
            points['Z'] = self._points.apply(lambda row: row.geometry.z, axis=1).to_list()
        else:
            points['Z'] = self._points.apply(lambda row: row[self.z_label], axis=1).to_list()

        return points

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
