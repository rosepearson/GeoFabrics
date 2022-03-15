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

    DEPTH_LABEL = 'depths'

    def __init__(self,
                 points_files: str,
                 polygon_files: str,
                 catchment_geometry: CatchmentGeometry,
                 z_labels: str = None):

        self.catchment_geometry = catchment_geometry

        self.z_label = z_labels is not None
        self._points = None
        self.polygon = None

        self._set_up(points_files, polygon_files, z_labels)

    def _set_up(self, points_files, polygon_files, z_labels):
        """ Set CRS and clip to catchment and within the flat water polygon """

        assert len(points_files) == len(polygon_files), "The polygon and point lists should all be the " \
            f"same length. Instead there are {len(points_files)} points files and {len(polygon_files)} polygon files"
        assert z_labels is None or len(points_files) == len(z_labels), \
            "Either all points should include z-values, or all have a label."

        points = geopandas.read_file(points_files[0])
        if z_labels is not None:
            points[self.DEPTH_LABEL] = points[z_labels[0]]
        polygon = geopandas.read_file(polygon_files[0])
        for i in range(1, len(points_files)):
            points_i = geopandas.read_file(points_files[i])
            if z_labels is not None:
                points[self.DEPTH_LABEL] = points[z_labels[i]]
            points = points.append(points_i)
            polygon_i = geopandas.read_file(polygon_files[i])
            polygon = polygon.append(polygon_i)

        self._points = points
        self.polygon = polygon

        self._points = self._points.to_crs(self.catchment_geometry.crs['horizontal'])
        self.polygon = self.polygon.to_crs(self.catchment_geometry.crs['horizontal'])

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
        if self.z_label:
            points['Z'] = self._points.apply(lambda row: row[self.DEPTH_LABEL], axis=1).to_list()
        else:
            points['Z'] = self._points.apply(lambda row: row.geometry.z, axis=1).to_list()

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

        if self.z_label:
            self._z = self._points.apply(lambda row: row[self.DEPTH_LABEL], axis=1).to_list()
        else:
            self._z = self._points.apply(lambda row: row.geometry.z, axis=1).to_list()

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


class RiverMouthFan:
    """ A class for creating an appropiate river mouth fan to transition from
    river depth estimates to the ocean bathymetry values. This fan region defines a
    transition from river to coast within a fan shaped polygon (15 degrees on each side). The fan
    begins with the most downstream river width estimate, and ends with the
    first contour of either more than 2x the depth of the mouth. This aims to ensure
    their is a defined fan that is slightly deeper than the surrounding during
    the transition.
    TODO In future,it may move to defining the width as 10x the mouth width.
    TODO deal with no width at the mouth and work upstream.

    Attributes:
        crs  The horizontal CRS to be used. i.e. EPSG:2193
        transect_spacing  The spacing in (m) of the sampled cross sections.
        aligned_channel_file  Thefile name for the aligned river channel file.
        river_bathymetry_file  The file name for the river bathymetry values.
        ocean_contour_file  The file name for the ocean contours.
        ocean_contour_depth_label  The column label for the depth values.
    """

    FAN_ANGLE = 30
    FAN_MAX_LENGTH = 10_000

    def __init__(self,
                 aligned_channel_file: str,
                 river_bathymetry_file: str,
                 ocean_contour_file: str,
                 crs: int,
                 transect_spacing: float,
                 ocean_contour_depth_label: str = None):

        self.crs = crs
        self.transect_spacing = transect_spacing
        self.aligned_channel_file = aligned_channel_file
        self.river_bathymetry_file = river_bathymetry_file
        self.ocean_contour_file = ocean_contour_file
        self.ocean_contour_depth_label = ocean_contour_depth_label

    def _get_mouth_alignment(self):
        """ Get the location and alignment of the river mouth. """

        aligned_channel = geopandas.read_file(self.aligned_channel_file)
        (x, y) = aligned_channel.loc[0].geometry.xy

        # Get the midpoint of the river mouth
        mouth_point = shapely.geometry.Point([x[0], y[0]])

        # Calculate the normal and tangent to the channel segment at the mouth
        segment_dx = x[0] - x[1]
        segment_dy = y[0] - y[1]
        segment_length = numpy.sqrt(segment_dx**2 + segment_dy**2)
        mouth_tangent = shapely.geometry.Point([segment_dx / segment_length,
                                                segment_dy / segment_length])
        mouth_normal = shapely.geometry.Point([-mouth_tangent.y,
                                               mouth_tangent.x])

        return mouth_point, mouth_tangent, mouth_normal

    def _get_mouth_bathymetry(self):
        """ Get the width and depth at the river mouth. """

        river_bathymetry = geopandas.read_file(self.river_bathymetry_file)
        river_mouth_depth = river_bathymetry['bed_elevation_Rupp_and_Smart'].iloc[0]
        river_mouth_width = river_bathymetry['widths'].iloc[0]
        return river_mouth_depth, river_mouth_width

    def _get_ocean_contours(self, river_mouth_depth,
                            depth_sign: int = -1,
                            depth_multiplier: int = 2):
        """ Load in the ocean contours.

        Parameters:
            river_mouth_depth  The depth in m of the river mouth
            depth_sign  The sign of the depths (-1 means depths are positive)
            depth_multiplier  Number of times deeped the end contour should be
                than the river mouth
        """

        assert self.ocean_contour_depth_label is not None, "Support not yet " \
            "added for z values within the geometry values"

        # Load in the ocean contours and find the contours to terminate against
        ocean_contours = geopandas.read_file(self.ocean_contour_file).to_crs(self.crs)
        depth_label = self.ocean_contour_depth_label

        # Determine the end depth and filter the contours to include only these contours
        end_depth = ocean_contours[depth_label][ocean_contours[depth_label] >
                                                depth_multiplier * river_mouth_depth * depth_sign].min()
        ocean_contours = ocean_contours[ocean_contours[depth_label] == end_depth].reset_index(drop=True)

        assert len(ocean_contours) > 0, "No contours exist with a depth 2x the river mouth depth. "

        return ocean_contours, end_depth

    def _bathymetry(self, intersection_line: shapely.geometry.LineString,
                    river_mouth_depth: float,
                    end_depth: float,
                    mouth_point: shapely.geometry.Point,
                    mouth_tangent: shapely.geometry.Point):
        """ Calculate and return the fan bathymetry values.
        
        Parameters:
            intersection_line  The contour line defining the end of the fan
            river_mouth_depth  The depth in m of the river mouth
            end_depth  The depth of the end contour of the fan in m
            mouth_point  The location of the centre of the river mouth
            mouth_tangent The tangent to the river mouth (along channel axis)
        """

        # Get the length of the fan centreline
        fan_centre = shapely.geometry.LineString([mouth_point,
                                                  [mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                                                   mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y]])
        distance = fan_centre.intersection(intersection_line).distance(mouth_point)

        # Setup the fan data values
        fan_depths = {'geometry': [], 'depths': []}
        number_of_samples = int(distance / self.transect_spacing)
        depth_increment = (-1 * end_depth - river_mouth_depth) / number_of_samples

        # Iterate through creating fan bathymetry
        for i in range(1, number_of_samples):
            fan_depths['geometry'].append(shapely.geometry.Point([mouth_point.x + mouth_tangent.x
                                                                  * i * self.transect_spacing,
                                                                  mouth_point.y + mouth_tangent.y
                                                                  * i * self.transect_spacing]))
            fan_depths['depths'].append(river_mouth_depth + i * depth_increment)
        fan_depths = geopandas.GeoDataFrame(fan_depths, crs=self.crs)
        return fan_depths

    def _max_length_polygon(self, river_mouth_width: float,
                            mouth_point: float,
                            mouth_tangent: float,
                            mouth_normal: float):
        """ Return the fan polygon of maximum length. This will be used to
        produce data to the first contour at least 2x the depth of the river
        mouth.

        Parameters:
            river_mouth_depth  The depth in m of the river mouth
            mouth_point  The location of the centre of the river mouth
            mouth_tangent The tangent to the river mouth (along channel axis)
            mouth_normal  The normal to the river mouth (cross channel axis)
        """

        end_width = river_mouth_width + 2 * self.FAN_MAX_LENGTH * numpy.tan(numpy.pi/180 * self.FAN_ANGLE)
        fan_end_point = shapely.geometry.Point([mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                                                mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y])
        fan_polygon = shapely.geometry.Polygon([[mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                                                 mouth_point.y - mouth_normal.y * river_mouth_width / 2],
                                                [mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                                                 mouth_point.y + mouth_normal.y * river_mouth_width / 2],
                                                [fan_end_point.x + mouth_normal.x * end_width / 2,
                                                 fan_end_point.y + mouth_normal.y * end_width / 2],
                                                [fan_end_point.x - mouth_normal.x * end_width / 2,
                                                 fan_end_point.y - mouth_normal.y * end_width / 2]])
        return fan_polygon

    def polygon_and_bathymetry(self):
        """ Calculate and return the fan polygon values. """

        # Load in river mouth alignment and bathymetry
        mouth_point, mouth_tangent, mouth_normal = self._get_mouth_alignment()
        river_mouth_depth, river_mouth_width = self._get_mouth_bathymetry()

        # Create maximum fan polygon
        fan_polygon = self._max_length_polygon(river_mouth_width=river_mouth_width,
                                               mouth_point=mouth_point,
                                               mouth_tangent=mouth_tangent,
                                               mouth_normal=mouth_normal)

        # Load in ocean depth contours
        ocean_contours, end_depth = self._get_ocean_contours(river_mouth_depth)

        # Cycle through contours finding the nearest contour to intersect the fan
        distance = numpy.inf
        intersection_line = shapely.geometry.Point()

        for i, row in ocean_contours.iterrows():
            if row.geometry.intersects(fan_polygon):
                intersection_line_i = row.geometry.intersection(fan_polygon)
                if intersection_line_i.distance(mouth_point) < distance:
                    distance = intersection_line_i.distance(mouth_point)
                    intersection_line = intersection_line_i

        assert distance < numpy.inf, "There must be at least one ocean " \
            "contour within the max length fan polygon."

        # Construct a fan ending at the contour
        (x, y) = intersection_line.xy
        polygon_points = [[xi, yi] for (xi, yi) in zip(x, y)]

        # Construct the bottom edge of the fan to work out the direction of the intersection line
        bottom_mouth_point = shapely.geometry.Point([mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                                                     mouth_point.y + mouth_normal.y * river_mouth_width / 2])
        bottom_fan_edge = shapely.geometry.LineString([bottom_mouth_point,
                                                       [bottom_mouth_point.x + distance * mouth_tangent.x,
                                                        bottom_mouth_point.y + distance * mouth_tangent.y]])

        # Determine line direction before adding the mouth points in the correct direction
        first_point = shapely.geometry.Point([x[0], y[0]])
        last_point = shapely.geometry.Point([x[0], y[0]])
        if first_point.distance(bottom_fan_edge) < last_point.distance(bottom_fan_edge):
            # keep line order
            polygon_points.extend([[mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                                    mouth_point.y - mouth_normal.y * river_mouth_width / 2],
                                   [mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                                    mouth_point.y + mouth_normal.y * river_mouth_width / 2]])
        else:
            # reverse fan order
            polygon_points.extend([[mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                                    mouth_point.y + mouth_normal.y * river_mouth_width / 2],
                                   [mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                                    mouth_point.y - mouth_normal.y * river_mouth_width / 2]])
        fan_polygon = shapely.geometry.Polygon(polygon_points)
        fan_polygon = geopandas.GeoDataFrame(geometry=[fan_polygon], crs=self.crs)

        # Get bathymetry values
        
        bathymetry = self._bathymetry(intersection_line=intersection_line,
                                      end_depth=end_depth,
                                      river_mouth_depth=river_mouth_depth,
                                      mouth_point=mouth_point,
                                      mouth_tangent=mouth_tangent)
        return fan_polygon, bathymetry
