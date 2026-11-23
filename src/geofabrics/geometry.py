# -*- coding: utf-8 -*-
"""
This module contains classes associated with manipulating vector data.
"""

import geopandas
import pandas
import shapely
import numpy
import pathlib
import typing
import logging

RASTER_TYPE = numpy.float32


class CatchmentGeometry:
    """Define catchment regions as polygons.

    The CRS is a dictionary containing the EPSG code for a 'horizontal' and
    'vertical' datum.

    This class defines the ``land``, ``foreshore``, and ``offshore`` regions
    and ensures that they use the same CRS.

    The regions are clipped to the catchment and do not overlap. The land
    region is defined by a supplied polygon. The foreshore is an outward buffer
    from the land region, with width ``foreshore_buffer * resolution``. The
    offshore region is the remaining area outside the land and foreshore.

    The class also provides methods to exclude areas, such as LiDAR extents,
    from a region; see ``land_and_foreshore_without_lidar`` for an example.

    Initialize the class with the catchment, then set the land region before
    accessing the other regions.
    """

    def __init__(
        self,
        catchment_file: typing.Union[str, pathlib.Path],
        crs: dict,
        resolution: float,
        foreshore_buffer: int = 2,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._catchment = geopandas.read_file(catchment_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer

        # set catchment CRS
        self._catchment = self._catchment.to_crs(self.crs["horizontal"])

        # values set in setup once land has been specified
        self._land = None
        self._full_land = None
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None

    def _set_up(self):
        """Define the main catchment regions and ensure the CRS is set for
        each region"""

        self._full_land = self._full_land.to_crs(self.crs["horizontal"])

        # Clip land to the catchment and remove sub-pixel regions.
        self._land = self._catchment.clip(self._full_land, keep_geom_type=True)
        self._land = self._land[self._land.area > self.resolution * self.resolution]
        if self._land.area.sum() > 0:
            # Define the area outside land, then buffer land to create the shore.
            self._foreshore_and_offshore = self.catchment.overlay(
                self.land, how="difference"
            )
            self._land_and_foreshore = geopandas.GeoDataFrame(
                {"geometry": self.land.buffer(self.resolution * self.foreshore_buffer)},
                crs=self.crs["horizontal"],
            )
            self._land_and_foreshore = self._catchment.clip(
                self._land_and_foreshore, keep_geom_type=True
            )
            self._land_and_foreshore = self._land_and_foreshore[
                self._land_and_foreshore.area > self.resolution * self.resolution
            ]

            # Subtract land to isolate the foreshore, then subtract both from the
            # catchment to define offshore.
            self._foreshore = self._land_and_foreshore.overlay(
                self.land, how="difference"
            )

            self._offshore = self.catchment.overlay(
                self.land_and_foreshore, how="difference"
            )
        else:
            # Assume no foreshore
            self._foreshore_and_offshore = self.catchment
            self._foreshore = self._land
            self._land_and_foreshore = self._land
            self._offshore = self.catchment

        assert len(self._catchment) == 1, (
            "The catchment is made of multiple separate polygons it must be a single "
            + "polygon object - a MultiPolygon is fine"
        )

    def _assert_land_set(self):
        """Ensure that the land region has been set."""
        assert self._land is not None, (
            "The land region must be set before accessing any region other than "
            "the catchment."
        )

    @property
    def catchment(self):
        """Return the catchment region"""

        return self._catchment

    @property
    def land(self):
        """Return the catchment land region"""

        self._assert_land_set()
        return self._land

    @property
    def full_land(self):
        """Return the full land region"""

        self._assert_land_set()
        return self._full_land

    @land.setter
    def land(self, land_file: typing.Union[str, pathlib.Path]):
        """Set the land region and finish setup."""

        self._full_land = geopandas.read_file(land_file)

        self._set_up()

    @property
    def foreshore(self):
        """Return the catchment foreshore region"""

        self._assert_land_set()
        return self._foreshore

    @property
    def land_and_foreshore(self):
        """Return the catchment land and foreshore region"""

        self._assert_land_set()
        return self._land_and_foreshore

    @property
    def foreshore_and_offshore(self):
        """Return the catchment foreshore and offshore region"""

        self._assert_land_set()
        return self._foreshore_and_offshore

    @property
    def offshore(self):
        """Return the catchment offshore region"""

        self._assert_land_set()
        return self._offshore


class BathymetryContours:
    """A class for sampling from bathymetry contours.

    Assumes contours to be sampled to the catchment_geometry resolution"""

    def __init__(
        self,
        contour_file: str,
        catchment_geometry: CatchmentGeometry,
        z_label=None,
        region_of_interest=None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._contour = geopandas.read_file(contour_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label

        self._extent = None

        self._set_up(region_of_interest)

    def _set_up(self, region_of_interest):
        """Set CRS and clip to catchment"""

        self._contour = self._contour.to_crs(self.catchment_geometry.crs["horizontal"])

        # Define a large offshore area to keep ocean contours in - to ensure correct
        # interpolation behaviour near the edges
        if region_of_interest is None:
            region_of_interest = self.catchment_geometry.offshore
        region_of_interest = geopandas.GeoDataFrame(
            geometry=region_of_interest.buffer(
                numpy.sqrt(self.catchment_geometry.catchment.area.sum())
            )
        )
        region_of_interest = region_of_interest.overlay(
            self.catchment_geometry.land, how="difference"
        )
        self._extent = region_of_interest[
            region_of_interest.area > self.catchment_geometry.resolution**2
        ]
        # Keep only contours in the 'extents' i.e. inside the catchment region of interest
        self._contour = self._contour.clip(self._extent, keep_geom_type=True)
        self._contour = self._contour.reset_index(drop=True)

        # Convert any 'GeometryCollection' objects to 'MultiLineString' objects
        # - dropping any points
        if (self._contour.geometry.type == "GeometryCollection").any():
            geometry_list = []
            for geometry_row in self._contour.geometry:
                if geometry_row.geometryType() in {
                    "LineString",
                    "MultiLineString",
                }:
                    geometry_list.append(geometry_row)
                elif geometry_row.geometryType() == "GeometryCollection":
                    geometry_list.append(
                        shapely.geometry.MultiLineString(
                            [
                                geometry_element
                                for geometry_element in geometry_row
                                if geometry_element.geometryType() == "LineString"
                            ]
                        )
                    )
            self._contour.set_geometry(geometry_list, inplace=True)

    def sample_contours(self, resolution: float) -> numpy.ndarray | list:
        """Sample contour lines at regular intervals.

        Parameters
        ----------
        resolution
            Positive spacing between samples, in the horizontal coordinate
            units of the catchment CRS.

        Returns
        -------
        numpy.ndarray or list
            Structured array with ``X``, ``Y``, and ``Z`` fields. Returns an
            empty list if there are no contours to sample. Sample locations
            are in the catchment horizontal CRS.

        Notes
        -----
        The stored ``Z`` values are multiplied by ``-1`` whether they come
        from 3D contour coordinates or from ``z_label``. Returned values
        define elevation in metres.
        """

        assert (
            resolution > 0
        ), f"The sampling resolution must be greater than 0, but is {resolution}."

        points_df = self._contour.geometry.apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    row.interpolate(i * resolution)
                    for i in range(int(numpy.ceil(row.length / resolution)))
                ]
            )
        )
        if len(points_df) == 0:
            # No data, return an empty array
            return []

        points = numpy.empty(
            [points_df.apply(lambda row: len(row.geoms)).sum()],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points["X"] = numpy.concatenate(
            points_df.apply(lambda row: [row_i.x for row_i in row.geoms]).to_list()
        )
        points["Y"] = numpy.concatenate(
            points_df.apply(lambda row: [row_i.y for row_i in row.geoms]).to_list()
        )
        if self.z_label is None:
            points["Z"] = (
                numpy.concatenate(
                    points_df.apply(
                        lambda row: [row_i.z for row_i in row.geoms]
                    ).to_list()
                )
                * -1
            )
        else:
            points["Z"] = (
                numpy.concatenate(
                    [
                        numpy.ones(len(points_df.loc[i].geoms))
                        * self._contour[self.z_label].loc[i]
                        for i in range(len(points_df))
                    ]
                )
                * -1
            )
        return points


class OceanPoints:
    """Access marine bathymetry points for interpolating offshore elevations."""

    def __init__(
        self,
        points_file: str,
        catchment_geometry: CatchmentGeometry,
        is_depth: bool = False,
        z_label: str = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._points = geopandas.read_file(points_file)
        self.catchment_geometry = catchment_geometry
        self.is_depth = is_depth
        self.z_label = z_label

        self._set_up()

    def _set_up(self):
        """Set CRS and clip to > 2x catchment area."""

        self._points = self._points.to_crs(self.catchment_geometry.crs["horizontal"])
        self._points = self._points.explode(ignore_index=True)
        self._points = self._points.clip(
            self.catchment_geometry.catchment.buffer(
                numpy.sqrt(self.catchment_geometry.catchment.area.max())
            )
        )

    @property
    def boundary(self):
        """Return the bounding box containing data"""

        bounds = self._points.total_bounds
        boundary = geopandas.GeoDataFrame(
            geometry=[
                shapely.geometry.Polygon(
                    [
                        [bounds[0], bounds[1]],
                        [bounds[2], bounds[1]],
                        [bounds[2], bounds[3]],
                        [bounds[0], bounds[3]],
                    ]
                )
            ],
            crs=self._points.crs,
        )
        return boundary

    @property
    def points(self):
        """Return the points"""

        return self._points

    @property
    def x(self):
        """The x values"""

        if self._x is None:
            self._x = self._points.apply(
                lambda row: row["geometry"].x, axis=1
            ).to_numpy()
        return self._x

    @property
    def y(self):
        """The y values"""

        if self._y is None:
            self._y = self._points.apply(
                lambda row: row["geometry"].y, axis=1
            ).to_numpy()
        return self._y

    @property
    def z(self):
        """The z values"""

        if self._z is None:
            # map depth to elevation
            multiplier = 1 if not self.is_depth else -1
            if self.z_label is None:
                self._z = (
                    self._points.apply(lambda row: row["geometry"].z, axis=1).to_numpy()
                    * multiplier
                )
            else:
                self._z = self._points[self.z_label].to_numpy()
        return self._z

    def sample(self) -> numpy.ndarray:
        """Return points - rename in future."""
        points = numpy.empty(
            [len(self._points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points["X"] = self._points.apply(
            lambda row: row["geometry"].x, axis=1
        ).to_numpy()
        points["Y"] = self._points.apply(
            lambda row: row["geometry"].y, axis=1
        ).to_numpy()
        if self.is_depth:
            points["Z"] = (
                self._points.apply(lambda row: row["geometry"].z, axis=1).to_numpy()
                * -1
            )
        else:
            points["Z"] = self._points.apply(
                lambda row: row["geometry"].z, axis=1
            ).to_numpy()

        return points


class ElevationPoints:
    """A class for accessing estimated or measured river, mouth and waterway
    elevations as points. Paired elevation and polygon files are expected.
    The elevations are used to interpolate elevations within the polygons.
    """

    Z_LABEL = "z"
    BANK_HEIGHT_LABEL = "bank_height"
    WIDTH_LABEL = "width"
    OSM_ID = "OSM_id"

    def __init__(
        self,
        points_files: list,
        polygon_files: list,
        catchment_geometry: CatchmentGeometry,
        filter_osm_ids: list = [],
        z_labels: list = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.catchment_geometry = catchment_geometry

        self.z_label = z_labels is not None
        self._points = None
        self._polygon = None

        self._set_up(points_files, polygon_files, z_labels, filter_osm_ids)

    def _set_up(
        self,
        points_files: list,
        polygon_files: list,
        z_labels: list,
        filter_osm_ids: list,
    ):
        """Load, concatenate, and clip point and polygon files to the catchment."""

        # Validate and normalize the parallel point, polygon, and Z-label inputs.
        if len(points_files) != len(polygon_files):
            raise ValueError(
                "The polygon and point lists should all be the same length. Instead there "
                f"are {len(points_files)} points files and {len(polygon_files)} polygon "
                "files"
            )

        if z_labels is not None and type(z_labels) is str:
            z_labels = [z_labels for i in range(len(points_files))]

        if z_labels is not None and len(z_labels) != len(points_files):
            raise ValueError(
                "There is a mismatch in length between the provided z_labels "
                f"and the point files: {z_labels} {points_files}"
            )

        # Read each point file and retain only the supported attributes.
        points_list = []
        polygon_list = []
        for i in range(0, len(points_files)):
            point_frame = geopandas.read_file(points_files[i])
            if z_labels is not None and z_labels[i] != self.Z_LABEL:
                point_frame = point_frame.rename(columns={z_labels[i]: self.Z_LABEL})
            point_columns = [self.Z_LABEL, "geometry"]
            if self.BANK_HEIGHT_LABEL in point_frame.columns:
                point_columns.append(self.BANK_HEIGHT_LABEL)
            if self.WIDTH_LABEL in point_frame.columns:
                point_columns.append(self.WIDTH_LABEL)
            if self.OSM_ID in point_frame.columns:
                point_columns.append(self.OSM_ID)
            # Keep paired polygons only when their point file contains data.
            if len(point_frame) > 0:
                point_frame = point_frame[point_columns]
                points_list.append(point_frame)
                polygon_frame = geopandas.read_file(polygon_files[i])
                polygon_list.append(polygon_frame)

        # Combine files, align their CRS, and clip to the catchment.
        if len(points_list) == 0:
            self.logger.warning("No elevations. Ignoring.")
            self._points = []
            self._polygon = []
            return
        points = geopandas.GeoDataFrame(
            pandas.concat(points_list, ignore_index=True),
            crs=points_list[0].crs,
        )
        points = points.to_crs(self.catchment_geometry.crs["horizontal"])
        polygon = geopandas.GeoDataFrame(
            pandas.concat(polygon_list, ignore_index=True),
            crs=polygon_list[0].crs,
        )
        polygon = polygon.to_crs(self.catchment_geometry.crs["horizontal"])
        polygon.set_geometry(polygon.buffer(0), inplace=True)
        polygon = polygon.clip(self.catchment_geometry.catchment, keep_geom_type=True)
        points = points.clip(polygon.buffer(0), keep_geom_type=True)
        points = points.clip(self.catchment_geometry.catchment, keep_geom_type=True)
        points = points.reset_index(drop=True)

        # Filter if there are OSM IDs specified to filter
        if len(filter_osm_ids) > 0:
            for osm_id in filter_osm_ids:
                points = points[points[self.OSM_ID] != osm_id]
                polygon = polygon[polygon[self.OSM_ID] != osm_id]

        # Set to class members
        self._points = points
        self._polygon = polygon

    @property
    def polygons(self) -> geopandas.GeoDataFrame:
        """Return the polygons."""
        return self._polygon

    @property
    def points_array(self) -> numpy.ndarray:
        """Return the points as a single array."""

        points = self._points

        points_array = numpy.empty(
            [len(points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        if len(points_array) == 0:
            return points_array
        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points_array["X"] = points.apply(lambda row: row.geometry.x, axis=1).to_list()
        points_array["Y"] = points.apply(lambda row: row.geometry.y, axis=1).to_list()
        if self.z_label:
            points_array["Z"] = points.apply(
                lambda row: row[self.Z_LABEL], axis=1
            ).to_list()
        else:
            points_array["Z"] = points.apply(
                lambda row: row.geometry.z, axis=1
            ).to_list()
        return points_array

    @property
    def points(self):
        """Return the points"""

        return self._points

    def bank_heights_exist(self) -> bool:
        """Return true if the bank heights are defined in array"""
        return (
            self.BANK_HEIGHT_LABEL in self._points.columns
            and self._points[self.BANK_HEIGHT_LABEL].any()
        )

    def bank_height_points(self) -> numpy.ndarray:
        """Return the points with 'Z' being the estimated bank height as a single array."""

        # Filter the points by the type label
        points = self._points

        points_array = numpy.empty(
            [len(points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points_array["X"] = points.apply(lambda row: row.geometry.x, axis=1).to_list()
        points_array["Y"] = points.apply(lambda row: row.geometry.y, axis=1).to_list()

        # Pull out the bank heights
        points_array["Z"] = points.apply(
            lambda row: row[self.BANK_HEIGHT_LABEL], axis=1
        ).to_list()
        return points_array

    @property
    def x(self):
        """The x values"""

        if self._x is None:
            self._x = self._points.points.apply(
                lambda row: row["geometry"][0].x, axis=1
            ).to_numpy()
        return self._x

    @property
    def y(self):
        """The y values"""

        if self._y is None:
            self._y = self._points.points.apply(
                lambda row: row["geometry"][0].y, axis=1
            ).to_numpy()
        return self._y

    @property
    def z(self):
        """The z values"""

        if self.z_label:
            self._z = self._points.apply(
                lambda row: row[self.Z_LABEL], axis=1
            ).to_list()
        else:
            self._z = self._points.apply(lambda row: row.geometry.z, axis=1).to_list()
        return self._z


class ElevationContours(ElevationPoints):
    """Resample at spatial resolution at points"""

    def __init__(
        self,
        points_files: list,
        polygon_files: list,
        catchment_geometry: CatchmentGeometry,
        z_labels: list = None,
    ):
        super(ElevationContours, self).__init__(
            points_files=points_files,
            polygon_files=polygon_files,
            catchment_geometry=catchment_geometry,
            filter_osm_ids=[],
            z_labels=z_labels,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # convert contoursto samples points at resolution
        self.sample_contours(self.catchment_geometry.resolution)

    def sample_contours(self, resolution: float) -> numpy.ndarray:
        """Sample the contours at the specified resolution."""

        # convert contours to multipoints
        self._points.loc[:, "geometry"] = self._points.geometry.apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    row.interpolate(i * resolution)
                    for i in range(int(numpy.ceil(row.length / resolution)))
                ]
            )
        )
        self._points = self._points.explode(index_parts=True, ignore_index=True)


class TileInfo:
    """A class for working with tiling information."""

    def __init__(self, tile_file: str, catchment_geometry: CatchmentGeometry):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._tile_info = geopandas.read_file(tile_file)
        self.catchment_geometry = catchment_geometry

        self.file_name = None

        self._set_up()

    def _set_up(self):
        """Set CRS and select all tiles partially within the catchment, and look up the
        file column name"""

        self._tile_info = self._tile_info.to_crs(
            self.catchment_geometry.crs["horizontal"]
        )
        self._tile_info = geopandas.sjoin(
            self._tile_info, self.catchment_geometry.catchment
        )
        self._tile_info = self._tile_info.reset_index(drop=True)

        column_names = self._tile_info.columns

        column_name_matches = [
            name for name in column_names if "filename" == name.lower()
        ]
        column_name_matches.extend(
            [name for name in column_names if "file_name" == name.lower()]
        )

        assert len(column_name_matches) == 1, (
            "No single `file name` column detected in the tile file with"
            + f" columns: {column_names}"
        )
        self.file_name = column_name_matches[0]

    @property
    def tile_names(self):
        """Return the names of all tiles within the catchment"""

        return self._tile_info[self.file_name]


class RiverMouthFan:
    """Create a fan-shaped transition from river-bed estimates or measurements
    to ocean depths.

    The fan extends from the downstream-most river-width values to the first
    ocean contour deeper than twice the mouth depth. It provides a transition
    region slightly deeper than the surrounding bathymetry. A fan is calculated
    for each river-bed elevation approach. If the mouth width is unavailable,
    use the nearest upstream location with a valid width.

    TODO: Consider defining the fan width as ten times the mouth width.

    Attributes
    ----------
    crs
        Horizontal CRS, such as EPSG:2193.
    cross_section_spacing
        Spacing between sampled cross sections, in metres.
    aligned_channel_file
        Path to the aligned river channel file.
    river_bathymetry_file
        Path to the river bathymetry values.
    ocean_contour_file
        Path to ocean contours; depths are positive.
    ocean_points_file
        Path to ocean points; elevations are negative.
    ocean_contour_depth_label
        Column containing ocean contour depths.
    """

    FAN_ANGLE = 30
    FAN_MAX_LENGTH = 10_000

    def __init__(
        self,
        aligned_channel_file: str,
        river_bathymetry_file: str,
        river_polygon_file: str,
        ocean_contour_file: str,
        ocean_points_file: str,
        crs: int,
        cross_section_spacing: float,
        elevation_labels: list,
        ocean_elevation_label: str = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.crs = crs
        self.cross_section_spacing = cross_section_spacing
        self.aligned_channel_file = aligned_channel_file
        self.river_polygon_file = river_polygon_file
        self.river_bathymetry_file = river_bathymetry_file
        self.ocean_contour_file = ocean_contour_file
        self.ocean_elevation_label = ocean_elevation_label
        self.elevation_labels = elevation_labels
        self.ocean_points_file = ocean_points_file

    def _get_mouth_alignment(self):
        """Get the location and alignment of the river mouth."""

        # Get the river alignment and keep from the most bathymetry point
        aligned_channel = geopandas.read_file(self.aligned_channel_file)
        river_bathymetry = geopandas.read_file(self.river_bathymetry_file)
        river_polygon = geopandas.read_file(self.river_polygon_file).make_valid()

        # Ensure the river alignment is taken from the most downstream bathymetry point
        first_bathymetry_index = (
            river_bathymetry[river_bathymetry.geometry.isna() == False].index.min()
        )
        channel_split_distance = first_bathymetry_index * self.cross_section_spacing

        if channel_split_distance > self.cross_section_spacing:
            split_point = aligned_channel.interpolate(channel_split_distance)
            aligned_channel = shapely.ops.snap(
                aligned_channel.loc[0].geometry, split_point.loc[0], tolerance=0.1
            )
            aligned_channel = geopandas.GeoDataFrame(
                geometry=(
                    list(shapely.ops.split(aligned_channel, split_point.loc[0]).geoms)
                ),
                crs=river_polygon.crs,
            )
            if len(aligned_channel) > 1:
                aligned_channel = aligned_channel.iloc[[1]]

        # Explode in case clipping produced a MultiLineString.
        channel_x, channel_y = (
            aligned_channel.explode(index_parts=True).iloc[0].geometry.xy
        )

        # Calculate the normal and tangent to the channel segment at the mouth
        segment_dx = channel_x[0] - channel_x[1]
        segment_dy = channel_y[0] - channel_y[1]
        segment_length = numpy.sqrt(segment_dx**2 + segment_dy**2)
        mouth_tangent = shapely.geometry.Point(
            [segment_dx / segment_length, segment_dy / segment_length]
        )
        mouth_normal = shapely.geometry.Point([-mouth_tangent.y, mouth_tangent.x])

        # Get the midpoint of the river mouth from the river bathymetry
        river_bathymetry = river_bathymetry.clip(river_polygon).sort_index(
            ascending=True
        )
        mouth_point = river_bathymetry.iloc[0].geometry

        return mouth_point, mouth_tangent, mouth_normal

    def _get_mouth_bathymetry(self):
        """Get the width and depth at the river mouth."""

        # Get river bathymetry values and clip to the river polygon
        river_bathymetry = geopandas.read_file(self.river_bathymetry_file)
        river_polygon = geopandas.read_file(self.river_polygon_file)
        river_bathymetry = river_bathymetry.clip(
            river_polygon.buffer(self.cross_section_spacing / 2)
        ).sort_index(ascending=True)

        river_mouth_elevations = [
            river_bathymetry[elevation_label].iloc[0]
            for elevation_label in self.elevation_labels
        ]
        river_mouth_width = river_bathymetry["width"].iloc[0]
        return river_mouth_width, river_mouth_elevations

    def _get_ocean_contours(
        self,
        river_mouth_depth,
        depth_sign: int = -1,
        depth_multiplier: int = 2,
    ) -> geopandas.GeoDataFrame:
        """Load ocean contours and select those suitable for the fan boundary.

        Contours close to shore are excluded because they would create an
        excessively steep transition.

        Parameters
        ----------
        river_mouth_depth
            River-mouth depth, in metres.
        depth_sign
            Sign convention used for the depth values.
        depth_multiplier
            Multiple of the river-mouth depth used to select the end contour.
        """

        # Load in the ocean contours and find the contours to terminate against
        ocean_contours = geopandas.read_file(self.ocean_contour_file).to_crs(self.crs)
        elevations = self._get_elevations(ocean_contours)

        # Determine the end depth and filter the contours to include only these contours
        ocean_contours = ocean_contours[
            elevations > depth_multiplier * river_mouth_depth * depth_sign
        ].reset_index(drop=True)

        assert (
            len(ocean_contours) > 0
        ), f"No contours exist with a depth {depth_multiplier}x the river mouth depth. "

        return ocean_contours

    def _get_elevations(self, ocean_geometry: geopandas.GeoDataFrame):
        """Return the an array of depths using either the geometry or a column"""
        if self.ocean_elevation_label is None:
            elevations = ocean_geometry.apply(lambda row: row.geometry.z, axis=1)
        else:
            elevations = ocean_geometry[self.ocean_elevation_label]
        return elevations

    def _get_distance_to_nearest_ocean_point_in_fan(
        self,
        river_mouth_z,
        fan,
        mouth_point,
        depth_multiplier,
    ) -> geopandas.GeoDataFrame:
        """Find the nearest ocean point in the fan that is deeper than the river mouth.

        Parameters
        ----------
        river_mouth_z
            River-mouth elevation used for the depth comparison.
        fan
            Fan polygon in which to search for ocean points.
        mouth_point
            Point marking the centre of the river mouth.
        depth_multiplier
            Multiplier of the river-mouth depth used to define the minimum depth.
        """

        # Load in the ocean contours and find the contours to terminate against
        ocean_points = geopandas.read_file(self.ocean_points_file).to_crs(self.crs)

        # Determine the end depth and filter the contours to include only these contours
        ocean_points = ocean_points.clip(fan).reset_index(drop=True)
        if river_mouth_z > 0:
            river_mouth_z = 0
        ocean_points = ocean_points[
            self._get_elevations(ocean_points) < depth_multiplier * river_mouth_z
        ].reset_index(drop=True)

        if len(ocean_points) == 0:
            raise ValueError(
                f"No points exist within fan at a depth {depth_multiplier}x the river mouth depth. "
            )

        distance = ocean_points.distance(mouth_point).min()
        elevation = self._get_elevations(ocean_points).iloc[
            ocean_points.distance(mouth_point).idxmin()
        ]

        return distance, elevation

    def _bathymetry(
        self,
        intersection_line: shapely.geometry.LineString,
        river_mouth_elevations: list,
        end_depth: float,
        mouth_point: shapely.geometry.Point,
        mouth_tangent: shapely.geometry.Point,
    ):
        """Calculate fan bathymetry for both river-bed estimation approaches.

        Parameters
        ----------
        intersection_line
            Contour line defining the end of the fan.
        river_mouth_elevations
            River-mouth elevations used to define the fan's starting edge.
        end_depth
            Depth of the fan's end contour, in metres.
        mouth_point
            Centre point of the river mouth.
        mouth_tangent
            Tangent vector along the channel axis at the river mouth.
        """

        # Get the length of the fan centreline
        fan_centre = shapely.geometry.LineString(
            [
                mouth_point,
                [
                    mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                    mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y,
                ],
            ]
        )
        distance = fan_centre.intersection(intersection_line).distance(mouth_point)

        # Setup the fan data values
        fan_depths = {
            **{"geometry": []},
            **dict((elevation_label, []) for elevation_label in self.elevation_labels),
        }
        number_of_samples = int(distance / self.cross_section_spacing)
        depth_increments = [
            (-1 * end_depth - river_mouth_elevation) / number_of_samples
            for river_mouth_elevation in river_mouth_elevations
        ]

        # Iterate through creating fan bathymetry
        for i in range(1, number_of_samples):
            fan_depths["geometry"].append(
                shapely.geometry.Point(
                    [
                        mouth_point.x
                        + mouth_tangent.x * i * self.cross_section_spacing,
                        mouth_point.y
                        + mouth_tangent.y * i * self.cross_section_spacing,
                    ]
                )
            )
            for elevation_label, river_mouth_elevation, depth_increment in zip(
                self.elevation_labels, river_mouth_elevations, depth_increments
            ):
                fan_depths[elevation_label].append(
                    river_mouth_elevation + i * depth_increment
                )
        fan_depths = geopandas.GeoDataFrame(fan_depths, crs=self.crs)
        return fan_depths

    def _fixed_length_fan_polygon(
        self,
        river_mouth_width: float,
        mouth_point: float,
        mouth_tangent: float,
        mouth_normal: float,
        length: float,
    ):
        """Return a maximum-length fan polygon.

        The polygon provides coverage to the first contour at least twice as
        deep as the river mouth.

        Parameters
        ----------
        river_mouth_width
            Width of the river mouth.
        mouth_point
            Centre point of the river mouth.
        mouth_tangent
            Tangent vector along the channel axis at the river mouth.
        mouth_normal
            Normal vector across the channel at the river mouth.
        length
            Maximum fan length, in metres.
        """

        end_width = river_mouth_width + 2 * length * numpy.tan(
            numpy.pi / 180 * self.FAN_ANGLE / 2
        )
        fan_end_point = shapely.geometry.Point(
            [
                mouth_point.x + length * mouth_tangent.x,
                mouth_point.y + length * mouth_tangent.y,
            ]
        )
        fan_polygon = shapely.geometry.Polygon(
            [
                [
                    mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                    mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                ],
                [
                    mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                    mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                ],
                [
                    fan_end_point.x + mouth_normal.x * end_width / 2,
                    fan_end_point.y + mouth_normal.y * end_width / 2,
                ],
                [
                    fan_end_point.x - mouth_normal.x * end_width / 2,
                    fan_end_point.y - mouth_normal.y * end_width / 2,
                ],
            ]
        )
        return fan_polygon

    def _fan_from_ocean_points(
        self,
        river_mouth_width,
        river_mouth_elevations,
        mouth_point,
        mouth_tangent,
        mouth_normal,
        fan_polygon,
    ):
        """Return the fan boundary, end line, and depth using ocean points."""
        length, end_elevation = self._get_distance_to_nearest_ocean_point_in_fan(
            river_mouth_z=max(river_mouth_elevations),
            fan=fan_polygon,
            mouth_point=mouth_point,
            depth_multiplier=2,
        )
        end_depth = -1 * end_elevation
        fan_polygon = self._fixed_length_fan_polygon(
            river_mouth_width=river_mouth_width,
            mouth_point=mouth_point,
            mouth_tangent=mouth_tangent,
            mouth_normal=mouth_normal,
            length=length,
        )
        end_width = river_mouth_width + 2 * length * numpy.tan(
            numpy.pi / 180 * self.FAN_ANGLE / 2
        )
        fan_end_point = shapely.geometry.Point(
            [
                mouth_point.x + length * mouth_tangent.x,
                mouth_point.y + length * mouth_tangent.y,
            ]
        )
        intersection_line = shapely.geometry.LineString(
            [
                [
                    fan_end_point.x + mouth_normal.x * end_width / 2,
                    fan_end_point.y + mouth_normal.y * end_width / 2,
                ],
                [
                    fan_end_point.x - mouth_normal.x * end_width / 2,
                    fan_end_point.y - mouth_normal.y * end_width / 2,
                ],
            ]
        )
        return fan_polygon, intersection_line, end_depth

    def _fan_from_ocean_contours(
        self,
        river_mouth_width,
        river_mouth_elevations,
        mouth_point,
        mouth_normal,
        fan_centre,
        fan_polygon,
    ):
        """Return the fan boundary, end line, and depth using ocean contours."""
        ocean_contours = self._get_ocean_contours(max(river_mouth_elevations))
        end_depth = self._get_elevations(ocean_contours).min()
        ocean_contours = ocean_contours.explode(ignore_index=True)
        ocean_contours = ocean_contours[
            ~ocean_contours.intersection(fan_centre).is_empty
        ]

        # Keep the closest contour intersecting the fan centreline.
        if len(ocean_contours) > 0:
            ocean_contours = ocean_contours.clip(fan_polygon).explode(
                ignore_index=True
            )
            min_index = ocean_contours.distance(mouth_point).idxmin()
            intersection_line = ocean_contours.loc[min_index].geometry
            elevations = self._get_elevations(ocean_contours)
            end_depth = elevations.loc[min_index]
        else:
            self.logger.warning(
                "No ocean contour intersected. Instaed assumed fan geoemtry"
            )
            intersection_line = shapely.geometry.LineString(
                [
                    [fan_polygon.exterior.xy[0][2], fan_polygon.exterior.xy[1][2]],
                    [fan_polygon.exterior.xy[0][3], fan_polygon.exterior.xy[1][3]],
                ]
            )
            end_depth = max(river_mouth_elevations) * 10

        # Construct a fan ending at the selected contour or fallback boundary.
        contour_x, contour_y = intersection_line.xy
        polygon_points = [
            [x_coordinate, y_coordinate]
            for x_coordinate, y_coordinate in zip(contour_x, contour_y)
        ]

        contour_direction = numpy.array(
            [contour_x[-1] - contour_x[0], contour_y[-1] - contour_y[0]]
        )
        contour_direction = contour_direction / numpy.linalg.norm(contour_direction)
        mouth_normal_direction = numpy.array([mouth_normal.x, mouth_normal.y])

        if (
            numpy.arccos(numpy.dot(contour_direction, mouth_normal_direction))
            > numpy.pi / 2
        ):
            polygon_points.extend(
                [
                    [
                        mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                        mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                    ],
                    [
                        mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                        mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                    ],
                ]
            )
        else:
            polygon_points.extend(
                [
                    [
                        mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                        mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                    ],
                    [
                        mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                        mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                    ],
                ]
            )
        fan_polygon = shapely.geometry.Polygon(polygon_points)
        return fan_polygon, intersection_line, end_depth

    def polygon_and_bathymetry(self):
        """Calculate and return the fan polygon values. Perform calculations for both
        river bed estimation approaches, so both have a smooth transition to offshore
        bathyemtry."""

        # Load in river mouth alignment and bathymetry
        mouth_point, mouth_tangent, mouth_normal = self._get_mouth_alignment()
        (
            river_mouth_width,
            river_mouth_elevations,
        ) = self._get_mouth_bathymetry()

        # Create maximum fan polygon
        fan_polygon = self._fixed_length_fan_polygon(
            river_mouth_width=river_mouth_width,
            mouth_point=mouth_point,
            mouth_tangent=mouth_tangent,
            mouth_normal=mouth_normal,
            length=self.FAN_MAX_LENGTH,
        )

        # Define the fan centreline.
        fan_centre = shapely.geometry.LineString(
            [
                mouth_point,
                [
                    mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                    mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y,
                ],
            ]
        )
        if self.ocean_points_file is not None:
            (
                fan_polygon,
                intersection_line,
                end_depth,
            ) = self._fan_from_ocean_points(
                river_mouth_width=river_mouth_width,
                river_mouth_elevations=river_mouth_elevations,
                mouth_point=mouth_point,
                mouth_tangent=mouth_tangent,
                mouth_normal=mouth_normal,
                fan_polygon=fan_polygon,
            )
        else:
            (
                fan_polygon,
                intersection_line,
                end_depth,
            ) = self._fan_from_ocean_contours(
                river_mouth_width=river_mouth_width,
                river_mouth_elevations=river_mouth_elevations,
                mouth_point=mouth_point,
                mouth_normal=mouth_normal,
                fan_centre=fan_centre,
                fan_polygon=fan_polygon,
            )
        fan_polygon = geopandas.GeoDataFrame(geometry=[fan_polygon], crs=self.crs)

        # Get bathymetry values
        bathymetry = self._bathymetry(
            intersection_line=intersection_line,
            end_depth=end_depth,
            river_mouth_elevations=river_mouth_elevations,
            mouth_point=mouth_point,
            mouth_tangent=mouth_tangent,
        )
        return fan_polygon, bathymetry
