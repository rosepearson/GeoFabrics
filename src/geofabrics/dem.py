# -*- coding: utf-8 -*-
"""
This module contains classes associated with loading, generating, and combining
DEMs.
"""
import rioxarray
import rioxarray.merge
import rasterio
import xarray
import numpy
import math
import typing
import pathlib
import geopandas
import shapely
import dask
import dask.array
import pdal
import json
import abc
import logging
import scipy.interpolate
import scipy.spatial
from . import geometry


class ReferenceDem:
    """A class to manage reference or background DEMs in the catchment context

    Specifically, clip within the catchment land and foreshore. There is the option to
    clip outside any LiDAR using the
    optional 'exclusion_extent' input.

    If set_foreshore is True all positive DEM values in the foreshore are set to zero.
    """

    def __init__(
        self,
        dem_file,
        catchment_geometry: geometry.CatchmentGeometry,
        set_foreshore: bool = True,
        exclusion_extent: geopandas.GeoDataFrame = None,
    ):
        """Load in the reference DEM, clip and extract points"""

        self.catchment_geometry = catchment_geometry
        self.set_foreshore = set_foreshore
        # Drop the band coordinate added by rasterio.open()
        self._dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True).squeeze(
            "band", drop=True
        )

        self._extents = None
        self._points = None

        self._set_up(exclusion_extent)

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The overall DEM
        if self._dem is not None:
            self._dem.close()
            del self._dem

    def _set_up(self, exclusion_extent):
        """Set DEM CRS and trim the DEM to size"""

        self._dem.rio.set_crs(self.catchment_geometry.crs["horizontal"])

        if exclusion_extent is not None:
            # Clip within land & foreshore and remove any sub-pixel polygons
            exclusion_extent = exclusion_extent.clip(
                self.catchment_geometry.land_and_foreshore, keep_geom_type=True
            )
            exclusion_extent = exclusion_extent[
                exclusion_extent.area
                > self.catchment_geometry.resolution
                * self.catchment_geometry.resolution
            ]
            self._extents = geopandas.overlay(
                self.catchment_geometry.land_and_foreshore,
                exclusion_extent,
                how="difference",
            )
        else:
            self._extents = self.catchment_geometry.land_and_foreshore
        self._dem = self._dem.rio.clip(self._extents.geometry)
        self._extract_points()

    def _extract_points(self):
        """Create a points list from the DEM"""

        if self.catchment_geometry.land.area.sum() > 0:
            land_dem = self._dem.rio.clip(self.catchment_geometry.land.geometry)
            # get reference DEM points on land
            land_flat_z = land_dem.data.flatten()
            land_mask_z = ~numpy.isnan(land_flat_z)
            land_grid_x, land_grid_y = numpy.meshgrid(land_dem.x, land_dem.y)

            land_x = land_grid_x.flatten()[land_mask_z]
            land_y = land_grid_y.flatten()[land_mask_z]
            land_z = land_flat_z[land_mask_z]
        else:  # If there is no DEM outside LiDAR/exclusion_extent and on land
            land_x = []
            land_y = []
            land_z = []
        if self.catchment_geometry.foreshore.area.sum() > 0:
            foreshore_dem = self._dem.rio.clip(
                self.catchment_geometry.foreshore.geometry
            )

            # get reference DEM points on the foreshore
            if self.set_foreshore:
                foreshore_dem.data[0][foreshore_dem.data[0] > 0] = 0
            foreshore_flat_z = foreshore_dem.data[0].flatten()
            foreshore_mask_z = ~numpy.isnan(foreshore_flat_z)
            foreshore_grid_x, foreshore_grid_y = numpy.meshgrid(
                foreshore_dem.x, foreshore_dem.y
            )

            foreshore_x = foreshore_grid_x.flatten()[foreshore_mask_z]
            foreshore_y = foreshore_grid_y.flatten()[foreshore_mask_z]
            foreshore_z = foreshore_flat_z[foreshore_mask_z]
        else:  # If there is no DEM outside LiDAR/exclusion_extent and on foreshore
            foreshore_x = []
            foreshore_y = []
            foreshore_z = []
        assert (
            len(land_x) + len(foreshore_x) > 0
        ), "The reference DEM has no values on the land or foreshore"

        # combine in an single array
        self._points = numpy.empty(
            [len(land_x) + len(foreshore_x)],
            dtype=[("X", numpy.float64), ("Y", numpy.float64), ("Z", numpy.float64)],
        )
        self._points["X"][: len(land_x)] = land_x
        self._points["Y"][: len(land_x)] = land_y
        self._points["Z"][: len(land_x)] = land_z

        self._points["X"][len(land_x) :] = foreshore_x
        self._points["Y"][len(land_x) :] = foreshore_y
        self._points["Z"][len(land_x) :] = foreshore_z

    @property
    def points(self) -> numpy.ndarray:
        """The reference DEM points after any extent or foreshore value
        filtering."""

        return self._points

    @property
    def extents(self) -> geopandas.GeoDataFrame:
        """The extents for the reference DEM"""

        return self._extents


class DenseDem(abc.ABC):
    """An abstract class to manage the dense DEM in a catchment context.

    The dense DEM is made up of a dense DEM that is loaded in, and an offshore DEM that
    is interpolated from bathymetry contours offshore and outside all LiDAR tiles.

    Parameters
    ----------

    catchment_geometry
        Defines the spatial extents of the catchment, land, foreshore, and offshore
        regions
    extents
        Defines the extents of any dense (LiDAR or refernence DEM) values already added.
    dense_dem
        The dense portion of the DEM
    interpolation_method
        If not None, interpolate using that method. Valid options are 'linear',
        'nearest', and 'cubic'
    """

    CACHE_SIZE = 10000  # The maximum RBF input without performance issues
    SOURCE_CLASSIFICATION = {
        "LiDAR": 1,
        "ocean bathymetry": 2,
        "river bathymetry": 3,
        "reference DEM": 4,
        "interpolated": 0,
        "no data": -1,
    }

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        extents: geopandas.GeoDataFrame,
        dense_dem: xarray.core.dataarray.DataArray,
        interpolation_method: str,
    ):
        """Setup base DEM to add future tiles too"""

        self.catchment_geometry = catchment_geometry
        self._dense_dem = dense_dem
        self._extents = extents

        self.interpolation_method = interpolation_method

        self._offshore_dem = None
        self._river_dem = None

        self._dem = None

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The dense DEM - may be opened from memory
        if self._dense_dem is not None:
            self._dense_dem.close()
            del self._dense_dem
        # The offshore DEM
        if self._offshore_dem is not None:
            self._offshore_dem.close()
            del self._offshore_dem
        # The river DEM
        if self._river_dem is not None:
            self._river_dem.close()
            del self._river_dem
        # The overall DEM
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def extents(self):
        """The combined extents for all added LiDAR tiles"""

        if self._extents is None:
            logging.warning(
                "Warning in DenseDem.extents: No tiles with extents have been added yet"
            )
        return self._extents

    @property
    def dense_dem(self):
        """Return the dense DEM from tiles and any interpolated offshore values"""
        return self._dense_dem

    @property
    def dem(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""

        if self._dem is None:
            self._dem = self.combine_dem_parts()

            # Ensure valid name and increasing dimension indexing for the dem
            if (
                self.interpolation_method is not None
            ):  # methods are 'nearest', 'linear' and 'cubic'
                self._dem.source_class.data[
                    numpy.isnan(self._dem.z.data)
                ] = self.SOURCE_CLASSIFICATION["interpolated"]
                self._dem["z"] = self._dem.z.rio.interpolate_na(
                    method=self.interpolation_method
                )
            self._dem = self._dem.rio.clip(self.catchment_geometry.catchment.geometry)
            self._dem = self._ensure_positive_indexing(
                self._dem
            )  # Some programs require positively increasing indices
        return self._dem

    def combine_dem_parts(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""

        if self._offshore_dem is None and self._river_dem is None:
            combined_dem = self.dense_dem
        elif self._river_dem is None:
            # method='first' or 'last'; use method='first' as
            # `DenseDemFromFiles.dense_dem` clipped to extents
            combined_dem = rioxarray.merge.merge_datasets(
                [self.dense_dem, self._offshore_dem], method="first"
            )
        elif self._offshore_dem is None:
            # method='first' or 'last'; use method='first' as
            # `DenseDemFromFiles.dense_dem` clipped to extents
            combined_dem = rioxarray.merge.merge_datasets(
                [self._river_dem, self.dense_dem], method="first"
            )
        else:
            combined_dem = rioxarray.merge.merge_datasets(
                [self._river_dem, self.dense_dem, self._offshore_dem], method="first"
            )
        return combined_dem

    @staticmethod
    def _ensure_positive_indexing(
        dem: xarray.core.dataarray.DataArray,
    ) -> xarray.core.dataarray.DataArray:
        """A routine to check an xarray has positive dimension indexing and to reindex
        if needed."""

        x = dem.x
        y = dem.y
        if x[0] > x[-1]:
            x = x[::-1]
        if y[0] > y[-1]:
            y = y[::-1]
        dem = dem.reindex(x=x, y=y)
        return dem

    @staticmethod
    def _write_netcdf_conventions_in_place(
        dem: xarray.core.dataarray.DataArray, crs_dict: dict
    ):
        """Write the CRS and transform associated with a netCDF file such that it is CF
        complient and meets the GDAL
        expectations for transform information.

        Parameters
        ----------

        dem
            The dataset to have its spatial data written in place.
        crs_dict
            A dict with horizontal and vertical CRS information.
        """

        dem.rio.write_crs(crs_dict["horizontal"], inplace=True)
        dem.z.rio.write_crs(crs_dict["horizontal"], inplace=True)
        dem.source_class.rio.write_crs(crs_dict["horizontal"], inplace=True)
        dem.rio.write_transform(inplace=True)
        dem.z.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
        dem.source_class.rio.write_nodata(numpy.nan, encoded=True, inplace=True)

    def _sample_offshore_edge(self, resolution) -> numpy.ndarray:
        """Return the pixel values of the offshore edge to be used for offshore
        interpolation"""

        assert resolution >= self.catchment_geometry.resolution, (
            "_sample_offshore_edge only supports downsampling"
            f" and not  up-samping. The requested sampling resolution of {resolution} "
            "must be equal to or larger than the catchment resolution of "
            f" {self.catchment_geometry.resolution}"
        )

        offshore_dense_data_edge = self.catchment_geometry.offshore_dense_data_edge(
            self._extents
        )
        offshore_edge_dem = self.dense_dem.rio.clip(offshore_dense_data_edge.geometry)

        # If the sampling resolution is coaser than the catchment_geometry resolution
        # resample the DEM
        if resolution > self.catchment_geometry.resolution:
            x = numpy.arange(
                offshore_edge_dem.x.min(),
                offshore_edge_dem.x.max() + resolution / 2,
                resolution,
            )
            y = numpy.arange(
                offshore_edge_dem.y.min(),
                offshore_edge_dem.y.max() + resolution / 2,
                resolution,
            )
            offshore_edge_dem = offshore_edge_dem.interp(x=x, y=y, method="nearest")
            offshore_edge_dem = offshore_edge_dem.rio.clip(
                offshore_dense_data_edge.geometry
            )  # Reclip to inbounds
        offshore_grid_x, offshore_grid_y = numpy.meshgrid(
            offshore_edge_dem.x, offshore_edge_dem.y
        )
        offshore_flat_z = offshore_edge_dem.z.data.flatten()
        offshore_mask_z = ~numpy.isnan(offshore_flat_z)

        offshore_edge = numpy.empty(
            [offshore_mask_z.sum().sum()],
            dtype=[("X", numpy.float64), ("Y", numpy.float64), ("Z", numpy.float64)],
        )

        offshore_edge["X"] = offshore_grid_x.flatten()[offshore_mask_z]
        offshore_edge["Y"] = offshore_grid_y.flatten()[offshore_mask_z]
        offshore_edge["Z"] = offshore_flat_z[offshore_mask_z]

        return offshore_edge

    def interpolate_offshore(self, bathy_contours):
        """Performs interpolation offshore outside LiDAR extents using the SciPy RBF
        function."""

        offshore_edge_points = self._sample_offshore_edge(
            self.catchment_geometry.resolution
        )
        bathy_points = bathy_contours.sample_contours(
            self.catchment_geometry.resolution
        )
        offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])

        # Resample at a lower resolution if too many offshore points
        if len(offshore_points) > self.CACHE_SIZE:
            reduced_resolution = (
                self.catchment_geometry.resolution
                * len(offshore_points)
                / self.CACHE_SIZE
            )
            logging.info(
                "Reducing the number of 'offshore_points' used to create the RBF "
                "function by increasing the resolution from "
                f" {self.catchment_geometry.resolution} to {reduced_resolution}"
            )
            offshore_edge_points = self._sample_offshore_edge(reduced_resolution)
            bathy_points = bathy_contours.sample_contours(reduced_resolution)
            offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])
        # Set up the interpolation function
        logging.info("Creating offshore interpolant")
        rbf_function = scipy.interpolate.Rbf(
            offshore_points["X"],
            offshore_points["Y"],
            offshore_points["Z"],
            function="linear",
        )

        # Setup the empty offshore area ready for interpolation
        offshore_no_dense_data = self.catchment_geometry.offshore_no_dense_data(
            self._extents
        )
        self._offshore_dem = self.dense_dem.rio.clip(
            self.catchment_geometry.offshore.geometry
        )

        # set all zero (or to ocean bathy classification) then clip out dense region
        # where we don't need to interpolate
        self._offshore_dem.z.data[:] = 0
        self._offshore_dem.source_class.data[:] = self.SOURCE_CLASSIFICATION[
            "ocean bathymetry"
        ]
        self._offshore_dem = self._offshore_dem.rio.clip(
            offshore_no_dense_data.geometry
        )

        grid_x, grid_y = numpy.meshgrid(self._offshore_dem.x, self._offshore_dem.y)
        flat_z = self._offshore_dem.z.data.flatten()
        mask_z = ~numpy.isnan(flat_z)

        flat_x_masked = grid_x.flatten()[mask_z]
        flat_y_masked = grid_y.flatten()[mask_z]
        flat_z_masked = flat_z[mask_z]

        # Tile offshore area - this limits the maximum memory required at any one time
        number_offshore_tiles = math.ceil(len(flat_x_masked) / self.CACHE_SIZE)
        for i in range(number_offshore_tiles):
            logging.info(f"Offshore intepolant tile {i+1} of {number_offshore_tiles}")
            start_index = int(i * self.CACHE_SIZE)
            end_index = (
                int((i + 1) * self.CACHE_SIZE)
                if i + 1 != number_offshore_tiles
                else len(flat_x_masked)
            )

            flat_z_masked[start_index:end_index] = rbf_function(
                flat_x_masked[start_index:end_index],
                flat_y_masked[start_index:end_index],
            )
        flat_z[mask_z] = flat_z_masked
        self._offshore_dem.z.data = flat_z.reshape(self._offshore_dem.z.data.shape)

        # Ensure the DEM is recalculated to include the interpolated offshore region
        self._dem = None

    def interpolate_river_bathymetry(self, river_bathymetry):
        """Performs interpolation with a river polygon using the SciPy RBF function."""

        # Reset the river and overall DEM
        self._river_dem = None
        self._dem = None

        # combined DEM
        combined_dem = self.combine_dem_parts()

        # Get edge points
        edge_dem = combined_dem.rio.clip(
            river_bathymetry.polygon.buffer(self.catchment_geometry.resolution),
            drop=True,
        )
        edge_dem = edge_dem.rio.clip(
            river_bathymetry.polygon.geometry, invert=True, drop=True
        )
        grid_x, grid_y = numpy.meshgrid(edge_dem.x, edge_dem.y)
        flat_z = edge_dem.z.data.flatten()
        mask_z = ~numpy.isnan(flat_z)
        edge_points = numpy.empty(
            [mask_z.sum().sum()],
            dtype=[("X", numpy.float64), ("Y", numpy.float64), ("Z", numpy.float64)],
        )

        edge_points["X"] = grid_x.flatten()[mask_z]
        edge_points["Y"] = grid_y.flatten()[mask_z]
        edge_points["Z"] = flat_z[mask_z]

        # Get Bathy Points then concatenate
        bathy_points = river_bathymetry.points_array()
        river_points = numpy.concatenate([edge_points, bathy_points])

        # Setup the empty river area ready for interpolation
        self._river_dem = combined_dem.rio.clip(river_bathymetry.polygon.geometry)
        # set all zero (or to ocean bathy classification) then clip out dense region
        # where we don't need to interpolate
        self._river_dem.z.data[:] = 0
        self._river_dem.source_class.data[:] = self.SOURCE_CLASSIFICATION[
            "river bathymetry"
        ]
        self._river_dem = self._river_dem.rio.clip(river_bathymetry.polygon.geometry)

        grid_x, grid_y = numpy.meshgrid(self._river_dem.x, self._river_dem.y)
        flat_z = self._river_dem.z.data[:].flatten()
        mask_z = ~numpy.isnan(flat_z)

        flat_x_masked = grid_x.flatten()[mask_z]
        flat_y_masked = grid_y.flatten()[mask_z]
        flat_z_masked = flat_z[mask_z]

        # check there are actually pixels in the river
        logging.info(f"There are {len(flat_z_masked)} pixels in the river")

        # Interpolate river area - use cubic or linear interpolation
        flat_z_masked = scipy.interpolate.griddata(
            points=(river_points["X"], river_points["Y"]),
            values=river_points["Z"],
            xi=(flat_x_masked, flat_y_masked),
            method="cubic",  # cubic, linear
        )
        # Set the interpolated value in the DEM
        flat_z[mask_z] = flat_z_masked
        self._river_dem.z.data = flat_z.reshape(self._river_dem.z.data.shape)

        # Ensure the DEM will be recalculated to include the newly interpolated region
        self._dem = None


class DenseDemFromFiles(DenseDem):
    """A class to manage loading in an already created and saved dense DEM that has yet
    to have an offshore DEM associated with it.

    Parameters
    ----------

    Logic controlling behaviour
        interpolation_method
            If not None, interpolate using that method. Valid options are 'linear', 'nearest', and 'cubic'
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        dense_dem_path: typing.Union[str, pathlib.Path],
        extents_path: typing.Union[str, pathlib.Path],
        interpolation_method: str,
    ):
        """Load in the extents and dense DEM. Ensure the dense DEM is clipped within the extents"""

        extents = geopandas.read_file(pathlib.Path(extents_path))

        # Read in the dense DEM raster - and free up file by performing a deep copy.
        dense_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(dense_dem_path), masked=True, parse_coordinates=True
        )

        # Deep copy to ensure the opened file is properly unlocked; Squeeze as rasterio.open() adds band coordinate
        dense_dem = dense_dem.squeeze("band", drop=True)
        self._write_netcdf_conventions_in_place(dense_dem, catchment_geometry.crs)

        # Ensure all values outside the exents are nan as that defines the dense extents
        # and clip the dense dem to the catchment extents to ensure performance
        dense_dem = dense_dem.rio.clip(extents.geometry, drop=True)
        dense_dem = dense_dem.rio.clip(catchment_geometry.catchment.geometry, drop=True)

        # Setup the DenseDem class
        super(DenseDemFromFiles, self).__init__(
            catchment_geometry=catchment_geometry,
            dense_dem=dense_dem,
            extents=extents,
            interpolation_method=interpolation_method,
        )


class DenseDemFromTiles(DenseDem):
    """A class to manage the population of the DenseDem's dense_dem from LiDAR tiles, and/or a reference DEM.

    The dense DEM is made up of tiles created from dense point data - Either LiDAR point clouds, or a reference DEM.

    DenseDemFromTiles logic can be controlled by the constructor inputs.

    Parameters
    ----------

    drop_offshore_lidar
        If True only keep LiDAR values within the foreshore and land regions defined by the catchment_geometry.
        If False keep all LiDAR values.
    interpolation_method
        If not None, interpolate using that method. Valid options are 'linear', 'nearest', and 'cubic'.
    lidar_interpolation_method
        The interpolation method to apply to point clouds. Options are: mean, median, IDW
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        interpolation_method: str,
        lidar_interpolation_method: str,
        drop_offshore_lidar: bool = True,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        self.drop_offshore_lidar = drop_offshore_lidar
        self.elevation_range = elevation_range
        assert elevation_range is None or (
            type(elevation_range) == list and len(elevation_range) == 2
        ), "Error the 'elevation_range' must either be none, or a two entry list"

        self.raster_type = numpy.float64

        self.lidar_interpolation_method = lidar_interpolation_method

        super(DenseDemFromTiles, self).__init__(
            catchment_geometry=catchment_geometry,
            dense_dem=None,
            extents=None,
            interpolation_method=interpolation_method,
        )

    def _set_up_chunks(self, chunk_size: int) -> (list, list):
        """Define the chunked coordinates to cover the catchment"""

        catchment_bounds = self.catchment_geometry.catchment.loc[0].geometry.bounds
        resolution = self.catchment_geometry.resolution

        # Determine the number of chunks
        if chunk_size is None or chunk_size <= 0:
            # Determine x and y coordinates for no chunks
            dim_x = [
                numpy.arange(
                    catchment_bounds[0] + resolution / 2,
                    catchment_bounds[2],
                    resolution,
                    dtype=self.raster_type,
                )
            ]
            dim_y = [
                numpy.arange(
                    catchment_bounds[3] - resolution / 2,
                    catchment_bounds[1],
                    -resolution,
                    dtype=self.raster_type,
                )
            ]
        else:
            n_chunks_x = int(
                numpy.ceil(
                    (catchment_bounds[2] - catchment_bounds[0])
                    / (chunk_size * resolution)
                )
            )
            n_chunks_y = int(
                numpy.ceil(
                    (catchment_bounds[3] - catchment_bounds[1])
                    / (chunk_size * resolution)
                )
            )

            # Determine x and y coordinates rounded up to the nearest chunk
            dim_x = [
                numpy.arange(
                    catchment_bounds[0] + resolution / 2 + i * chunk_size * resolution,
                    catchment_bounds[0]
                    + resolution / 2
                    + (i + 1) * chunk_size * resolution,
                    resolution,
                    dtype=self.raster_type,
                )
                for i in range(n_chunks_x)
            ]
            dim_y = [
                numpy.arange(
                    catchment_bounds[3] - resolution / 2 - i * chunk_size * resolution,
                    catchment_bounds[3]
                    - resolution / 2
                    - (i + 1) * chunk_size * resolution,
                    -resolution,
                    dtype=self.raster_type,
                )
                for i in range(n_chunks_y)
            ]
        return dim_x, dim_y

    def _define_chunk_region(
        self,
        region_to_rasterise: geopandas.GeoDataFrame,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        radius: float,
    ):
        """Define the region to rasterise within a single chunk."""
        # Define the region to tile
        chunk_geometry = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.Polygon(
                        [
                            (numpy.min(dim_x) - radius, numpy.min(dim_y) - radius),
                            (numpy.max(dim_x) + radius, numpy.min(dim_y) - radius),
                            (numpy.max(dim_x) + radius, numpy.max(dim_y) + radius),
                            (numpy.min(dim_x) - radius, numpy.max(dim_y) + radius),
                        ]
                    )
                ]
            },
            crs=self.catchment_geometry.crs["horizontal"],
        )

        # Define region to rasterise inside the chunk area - remove any subpixel polygons
        chunk_region_to_tile = geopandas.GeoDataFrame(
            geometry=region_to_rasterise.buffer(radius).clip(
                chunk_geometry, keep_geom_type=True
            )
        )
        chunk_region_to_tile = chunk_region_to_tile[
            chunk_region_to_tile.area
            > self.catchment_geometry.resolution * self.catchment_geometry.resolution
        ]

        return chunk_region_to_tile

    def _calculate_dense_extents(self):
        """Calculate the extents of the current dense DEM. Remove holes as these can
        cause self intersection warnings."""

        dense_extents = [
            shapely.geometry.shape(polygon[0])
            for polygon in rasterio.features.shapes(
                numpy.uint8(numpy.logical_not(numpy.isnan(self.dense_dem.z.data)))
            )
            if polygon[1] == 1.0
        ]
        dense_extents = shapely.ops.unary_union(dense_extents)

        # Remove any internal holes for select types as these may cause self intersection errors
        if type(dense_extents) is shapely.geometry.Polygon:
            dense_extents = shapely.geometry.Polygon(dense_extents.exterior)
        elif type(dense_extents) is shapely.geometry.MultiPolygon:
            dense_extents = shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.Polygon(polygon.exterior)
                    for polygon in dense_extents
                ]
            )
        # Convert into a Geopandas dataframe
        dense_extents = geopandas.GeoDataFrame(
            {"geometry": [dense_extents]}, crs=self.catchment_geometry.crs["horizontal"]
        )

        # Apply a transform so in the same space as the dense DEM - buffer(0) to reduce self intersection warnings
        dense_dem_affine = self.dense_dem.z.rio.transform()
        dense_extents = dense_extents.affine_transform(
            [
                dense_dem_affine.a,
                dense_dem_affine.b,
                dense_dem_affine.d,
                dense_dem_affine.e,
                dense_dem_affine.xoff,
                dense_dem_affine.yoff,
            ]
        ).buffer(0)

        # And make our GeoSeries into a GeoDataFrame
        dense_extents = geopandas.GeoDataFrame(geometry=dense_extents)

        return dense_extents

    def _tile_index_column_name(
        self, tile_index_file: typing.Union[str, pathlib.Path] = None
    ):
        """Read in tile index file and determine the column name of the tile geometries"""
        # Check to see if a extents file was added
        tile_index_extents = (
            geopandas.read_file(tile_index_file)
            if tile_index_file is not None
            else None
        )
        tile_index_name_column = None

        # If there is a tile_index_file - remove tiles outside the catchment & get the 'file name' column
        if tile_index_extents is not None:
            tile_index_extents = tile_index_extents.to_crs(
                self.catchment_geometry.crs["horizontal"]
            )
            tile_index_extents = geopandas.sjoin(
                tile_index_extents, self.catchment_geometry.catchment
            )
            tile_index_extents = tile_index_extents.reset_index(drop=True)

            column_names = tile_index_extents.columns
            tile_index_name_column = column_names[
                [
                    "filename" == name.lower() or "file_name" == name.lower()
                    for name in column_names
                ]
            ][0]
        return tile_index_extents, tile_index_name_column

    def _rasterise_tile(
        self,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        tile_points: numpy.ndarray,
        options: dict,
    ):
        """Rasterise all points within a tile."""

        # keep only the selected classification points for averaging calculations
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

        if len(tile_points) == 0:
            logging.warning(
                "In DenseDem._rasterise_tile the tile has no data and is being ignored."
            )
            return
        # Get the indicies overwhich to perform IDW
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()

        # Perform the specified averaging over the dense DEM within the extents of this point cloud tile
        z_flat = rasterise_points(
            point_cloud=tile_points, xy_out=xy_out, options=options
        )
        grid_z = z_flat.reshape(grid_x.shape)

        # TODO - add roughness calculation

        return grid_z

    def add_lidar(
        self,
        lidar_files: typing.List[typing.Union[str, pathlib.Path]],
        tile_index_file: typing.Union[str, pathlib.Path],
        chunk_size: int,
        lidar_classifications_to_keep: list,
        source_crs: dict,
        drop_offshore_lidar: bool,
        metadata: dict,
    ):
        """Read in all LiDAR files and use to create a dense DEM.

        Parameters
        ----------

        source_crs
            Specify if the CRS encoded in the LiDAR files are incorrect/only partially defined
            (i.e. missing vertical CRS) and need to be overwritten.
        drop_offshore_lidar
            If True, trim any LiDAR values that are offshore as specified by the catchment_geometry
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water. See
            https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for standard list
        tile_index_file
            Must exist if there are many LiDAR files. This is used to determine chunking.
        """

        if source_crs is not None:
            assert "horizontal" in source_crs, (
                "The horizontal component of the source CRS is not specified. "
                + f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            )
            assert "vertical" in source_crs, (
                "The vertical component of the source CRS is not specified. "
                + f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            )
        if drop_offshore_lidar:
            region_to_rasterise = self.catchment_geometry.land_and_foreshore
        else:
            region_to_rasterise = self.catchment_geometry.catchment
        # create dictionary defining raster options
        raster_options = {
            "lidar_classifications_to_keep": lidar_classifications_to_keep,
            "raster_type": self.raster_type,
            "elevation_range": self.elevation_range,
            "radius": self.catchment_geometry.resolution * numpy.sqrt(2),
            "method": self.lidar_interpolation_method,
        }

        # Determine if adding a single file or tiles
        if len(lidar_files) == 1:  # If one file it's ok if there is no tile_index
            self._dense_dem = self._add_file(
                lidar_file=lidar_files[0],
                region_to_rasterise=region_to_rasterise,
                source_crs=source_crs,
                options=raster_options,
                metadata=metadata,
            )
        else:
            assert (
                tile_index_file is not None
            ), "A tile index file is required for multiple tile files added together"
            assert chunk_size > 0 and chunk_size is not None, (
                "The chunk size should be set when reading in tiled LiDAR "
                "files. Ideally it should include as many tiles can easily be read in by on core. You will have to equate"
                " The tile extents with chunk size by extents / resolution. "
            )
            assert len(lidar_files) > 1, "There are no LiDAR files specified"
            self._dense_dem = self._add_tiled_lidar_chunked(
                lidar_files=lidar_files,
                tile_index_file=tile_index_file,
                source_crs=source_crs,
                raster_options=raster_options,
                region_to_rasterise=region_to_rasterise,
                chunk_size=chunk_size,
                metadata=metadata,
            )
        # Set any values outside the region_to_rasterise to NaN
        self._dense_dem = self.dense_dem.rio.clip(
            region_to_rasterise.geometry, drop=False
        )

        # Create a polygon defining the region where there are dense DEM values
        self._extents = self._calculate_dense_extents()

        # Ensure the dem will be recalculated as another tile has been added
        self._dem = None

    def _add_tiled_lidar_chunked(
        self,
        lidar_files: typing.List[typing.Union[str, pathlib.Path]],
        tile_index_file: typing.Union[str, pathlib.Path],
        source_crs: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        chunk_size: int,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create a dense DEM from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        # Remove all tiles entirely outside the region to raserise
        tile_index_extents, tile_index_name_column = self._tile_index_column_name(
            tile_index_file
        )

        # get chunking information - if negative, 0 or None chunk_size then default to a single chunk
        chunked_dim_x, chunked_dim_y = self._set_up_chunks(chunk_size)

        # cycle through index chunks - and collect in a delayed array
        logging.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")
        delayed_chunked_matrix = []
        for i, dim_y in enumerate(chunked_dim_y):
            delayed_chunked_x = []
            for j, dim_x in enumerate(chunked_dim_x):
                logging.info(f"\tChunk {[i, j]}")

                # Define the region to tile
                chunk_region_to_tile = self._define_chunk_region(
                    region_to_rasterise=region_to_rasterise,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    radius=raster_options["radius"],
                )

                # Load in files and rasterise
                chunk_points = delayed_load_tiles_in_chunk(
                    dim_x=dim_x,
                    dim_y=dim_y,
                    tile_index_extents=tile_index_extents,
                    tile_index_name_column=tile_index_name_column,
                    lidar_files=lidar_files,
                    source_crs=source_crs,
                    chunk_region_to_tile=chunk_region_to_tile,
                    catchment_geometry=self.catchment_geometry,
                )
                delayed_chunked_x.append(
                    dask.array.from_delayed(
                        delayed_rasterise_chunk(
                            dim_x=dim_x,
                            dim_y=dim_y,
                            tile_points=chunk_points,
                            options=raster_options,
                        ),
                        shape=(chunk_size, chunk_size),
                        dtype=numpy.float32,
                    )
                )
            delayed_chunked_matrix.append(delayed_chunked_x)
        # Combine chunks into a dataset
        elevation = dask.array.block(delayed_chunked_matrix)
        x = numpy.concatenate(chunked_dim_x)
        y = numpy.concatenate(chunked_dim_y)
        chunked_dem = self._create_data_set(x=x, y=y, z=elevation, metadata=metadata)
        logging.info("Computing chunks")
        chunked_dem = chunked_dem.compute()

        # Clip result to within the catchment - removing NaN filled chunked areas outside the catchment
        logging.debug("Chunked DEM computed and ready to be cut")
        dense_dem = chunked_dem.rio.clip(self.catchment_geometry.catchment.geometry)
        return dense_dem

    def _add_file(
        self,
        lidar_file: typing.Union[str, pathlib.Path],
        region_to_rasterise: geopandas.GeoDataFrame,
        options: dict,
        source_crs: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create the dense DEM region from a single LiDAR file."""

        logging.info(f"On LiDAR tile 1 of 1: {lidar_file}")

        # Use PDAL to load in file
        pdal_pipeline = read_file_with_pdal(
            lidar_file,
            source_crs=source_crs,
            region_to_tile=region_to_rasterise,
            get_extents=True,
            catchment_geometry=self.catchment_geometry,
        )

        # Load LiDAR points from pipeline
        tile_array = pdal_pipeline.arrays[0]

        # Get the raster indicies
        dim_x, dim_y = self._set_up_chunks(chunk_size=None)
        dim_x = dim_x[0]
        dim_y = dim_y[0]

        raster_values = self._rasterise_tile(
            dim_x=dim_x, dim_y=dim_y, tile_points=tile_array, options=options
        )
        elevation = raster_values.reshape((len(dim_y), len(dim_x)))

        # Create xarray
        dense_dem = self._create_data_set(
            x=dim_x, y=dim_y, z=elevation, metadata=metadata
        )

        return dense_dem

    def _create_data_set(
        self, x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray, metadata: dict
    ) -> xarray.Dataset:
        """A function to create a new dataset from x, y and z arrays.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            z
                Elevations over the x, and y coordiantes.
        """

        # Create source variable - assume all values are defined from LiDAR
        source_class = numpy.ones_like(z) * self.SOURCE_CLASSIFICATION["no data"]
        source_class[numpy.logical_not(numpy.isnan(z))] = self.SOURCE_CLASSIFICATION[
            "LiDAR"
        ]
        dem = xarray.Dataset(
            data_vars=dict(
                z=(
                    ["y", "x"],
                    z,
                    {
                        "units": "m",
                        "long_name": "ground elevation",
                        "vertical_datum": f"EPSG:{self.catchment_geometry.crs['vertical']}",
                    },
                ),
                source_class=(
                    ["y", "x"],
                    source_class,
                    {
                        "units": "",
                        "long_name": "source data classification",
                        "classifications": f"{self.SOURCE_CLASSIFICATION}",
                    },
                ),
            ),
            coords=dict(x=(["x"], x), y=(["y"], y)),
            attrs={
                "title": "Geofabric representing elevation and roughness",
                "source": f"{metadata['library_name']} version {metadata['library_version']}",
                "description": f"{metadata['library_name']}:{metadata['class_name']} resolution"
                + f" {self.catchment_geometry.resolution}",
                "history": f"{metadata['utc_time']}: {metadata['library_name']}:{metadata['class_name']} "
                + f"resolution {self.catchment_geometry.resolution};",
                "geofabrics_instructions": f"{metadata['instructions']}",
            },
        )

        # ensure the expected CF conventions are followed
        self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)
        return dem

    def add_reference_dem(
        self, tile_points: numpy.ndarray, tile_extent: geopandas.GeoDataFrame
    ):
        """Update gaps in dense DEM from areas with no LiDAR with the reference DEM."""

        # Areas not covered by LiDAR values
        mask = numpy.isnan(self.dense_dem.z.data)

        if len(tile_points) == 0:
            logging.warning(
                "DenseDem.add_tile: the latest reference DEM has no data and is being ignored."
            )
            return
        elif mask.sum() == 0:
            logging.warning(
                "DenseDem.add_tile: LiDAR covers all raster values so the reference DEM is being ignored."
            )
            return
        # create dictionary defining raster options
        raster_options = {
            "raster_type": self.raster_type,
            "radius": self.catchment_geometry.resolution * numpy.sqrt(2),
            "method": self.lidar_interpolation_method,
        }

        # Get the indicies overwhich to perform averaging
        grid_x, grid_y = numpy.meshgrid(self.dense_dem.x, self.dense_dem.y)

        xy_out = numpy.empty((mask.sum(), 2))
        xy_out[:, 0] = grid_x[mask]
        xy_out[:, 1] = grid_y[mask]

        # Perform the specified averaging over the dense DEM within the extents of this point cloud tile
        z_flat = rasterise_points(
            point_cloud=tile_points, xy_out=xy_out, options=raster_options
        )
        self.dense_dem.z.data[mask] = z_flat
        self.dense_dem.source_class.data[mask] = self.SOURCE_CLASSIFICATION[
            "reference DEM"
        ]

        # Update the dense DEM extents
        self._extents = self._calculate_dense_extents()

        # Ensure the dem will be recalculated as another tile has been added
        self._dem = None


def read_file_with_pdal(
    lidar_file: typing.Union[str, pathlib.Path],
    region_to_tile: geopandas.GeoDataFrame,
    catchment_geometry: geometry.CatchmentGeometry,
    source_crs: dict = None,
    get_extents: bool = False,
):
    """Read a tile file in using PDAL"""

    # Define instructions for loading in LiDAR
    pdal_pipeline_instructions = [{"type": "readers.las", "filename": str(lidar_file)}]

    # Specify reprojection - if a source_crs is specified use this to define the 'in_srs'
    if source_crs is None:
        pdal_pipeline_instructions.append(
            {
                "type": "filters.reprojection",
                "out_srs": f"EPSG:{catchment_geometry.crs['horizontal']}+"
                + f"{catchment_geometry.crs['vertical']}",
            }
        )
    else:
        pdal_pipeline_instructions.append(
            {
                "type": "filters.reprojection",
                "in_srs": f"EPSG:{source_crs['horizontal']}+{source_crs['vertical']}",
                "out_srs": f"EPSG:{catchment_geometry.crs['horizontal']}+"
                + f"{catchment_geometry.crs['vertical']}",
            }
        )
    # Add instructions for clip within either the catchment, or the land and foreshore
    pdal_pipeline_instructions.append(
        {"type": "filters.crop", "polygon": str(region_to_tile.loc[0].geometry)}
    )

    # Add instructions for creating a polygon extents of the remaining point cloud
    if get_extents:
        pdal_pipeline_instructions.append({"type": "filters.hexbin"})
    # Load in LiDAR and perform operations
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
    pdal_pipeline.execute()
    return pdal_pipeline


def rasterise_points(
    point_cloud: numpy.ndarray,
    xy_out,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
):
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud["X"]
    xy_in[:, 1] = point_cloud["Y"]

    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    tree_index_list = tree.query_ball_point(
        xy_out, r=options["radius"], eps=eps
    )  # , eps=0.2)
    z_out = numpy.zeros(len(xy_out), dtype=options["raster_type"])

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            if options["method"] == "mean":
                z_out[i] = numpy.mean(point_cloud["Z"][near_indicies])
            elif options["method"] == "median":
                z_out[i] = numpy.median(point_cloud["Z"][near_indicies])
            elif options["method"] == "idw":
                z_out[i] = calculate_idw(
                    near_indicies=near_indicies,
                    point=point,
                    tree=tree,
                    point_cloud=point_cloud,
                )
            elif options["method"] == "min":
                z_out[i] = numpy.min(point_cloud["Z"][near_indicies])
            elif options["method"] == "max":
                z_out[i] = numpy.max(point_cloud["Z"][near_indicies])
            elif options["method"] == "std":
                z_out[i] = numpy.std(point_cloud["Z"][near_indicies])
            elif options["method"] == "count":
                z_out[i] = numpy.len(point_cloud["Z"][near_indicies])
            else:
                assert (
                    False
                ), f"An invalid lidar_interpolation_method of '{options['method']}' was"
                " provided"
    return z_out


def calculate_idw(
    near_indicies: list,
    point: numpy.ndarray,
    tree: scipy.spatial.KDTree,
    point_cloud: numpy.ndarray,
    smoothing: float = 0,
    power: int = 2,
):
    """Calculate DEM elevation values at the specified locations by
    calculating the mean. This implementation is based on the
    scipy.spatial.KDTree"""

    distance_vectors = point - tree.data[near_indicies]
    smoothed_distances = numpy.sqrt(
        ((distance_vectors**2).sum(axis=1) + smoothing**2)
    )
    if smoothed_distances.min() == 0:  # in the case of an exact match
        idw = point_cloud["Z"][tree.query(point, k=1)[1]]
    else:
        idw = (point_cloud["Z"][near_indicies] / (smoothed_distances**power)).sum(
            axis=0
        ) / (1 / (smoothed_distances**power)).sum(axis=0)
    return idw


def load_tiles_in_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_index_extents: geopandas.GeoDataFrame,
    tile_index_name_column: str,
    lidar_files: typing.List[typing.Union[str, pathlib.Path]],
    source_crs: dict,
    chunk_region_to_tile: geopandas.GeoDataFrame,
    catchment_geometry: geometry.CatchmentGeometry,
):
    """Read in all LiDAR files within the chunked region - clipped to within the region
    within which to rasterise."""

    # Clip the tile indices to only include those within the chunk region
    chunk_tile_index_extents = tile_index_extents.drop(columns=["index_right"])
    chunk_tile_index_extents = geopandas.sjoin(
        chunk_tile_index_extents, chunk_region_to_tile
    )
    chunk_tile_index_extents = chunk_tile_index_extents.reset_index(drop=True)

    logging.info(
        f"Reading all {len(chunk_tile_index_extents[tile_index_name_column])} files in"
        " chunk."
    )

    # Initialise LiDAR points
    lidar_points = []

    # Cycle through each file loading it in an adding it to a numpy array
    for tile_index_name in chunk_tile_index_extents[tile_index_name_column]:
        logging.info(f"\t Loading in file {tile_index_name}")
        # get the LiDAR file with the tile_index_name
        lidar_file = [
            lidar_file
            for lidar_file in lidar_files
            if lidar_file.name == tile_index_name
        ]
        assert (
            len(lidar_file) == 1
        ), f"Error no single LiDAR file matches the tile name. {lidar_file}"

        # read in the LiDAR file
        pdal_pipeline = read_file_with_pdal(
            lidar_file=lidar_file[0],
            region_to_tile=chunk_region_to_tile,
            source_crs=source_crs,
            catchment_geometry=catchment_geometry,
            get_extents=False,
        )
        lidar_points.append(pdal_pipeline.arrays[0])
    if len(lidar_points) > 0:
        lidar_points = numpy.concatenate(lidar_points)
    return lidar_points


def rasterise_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_points: numpy.ndarray,
    options: dict,
):
    """Rasterise all points within a chunk."""

    # Get the indicies overwhich to perform IDW
    grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
    xy_out = numpy.concatenate(
        [[grid_x.flatten()], [grid_y.flatten()]], axis=0
    ).transpose()
    grid_z = numpy.ones(grid_x.shape) * numpy.nan

    # If no points return an array of NaN
    if len(tile_points) == 0:
        logging.warning(
            "In dem.rasterise_chunk the latest chunk has no data and is being ignored."
        )
        return grid_z
    # keep only the selected classification points for averaging calculations
    classification_mask = numpy.zeros_like(tile_points["Classification"], dtype=bool)
    for classification in options["lidar_classifications_to_keep"]:
        classification_mask[tile_points["Classification"] == classification] = True
    tile_points = tile_points[classification_mask]

    # optionally filter to within the specified elevation range
    elevation_range = options["elevation_range"]
    if elevation_range is not None:
        tile_points = tile_points[tile_points["Z"] >= elevation_range[0]]
        tile_points = tile_points[tile_points["Z"] <= elevation_range[1]]
    # Check again - if no points return an array of NaN
    if len(tile_points) == 0:
        return grid_z
    # Perform the specified averaging method over the dense DEM within the extents of this point cloud tile
    z_flat = rasterise_points(point_cloud=tile_points, xy_out=xy_out, options=options)
    grid_z = z_flat.reshape(grid_x.shape)

    # TODO - add roughness calculation

    return grid_z


""" Wrap the `rasterise_chunk` routine in dask.delayed """
delayed_rasterise_chunk = dask.delayed(rasterise_chunk)


""" Wrap the `load_tiles_in_chunk` routine in dask.delayed """
delayed_load_tiles_in_chunk = dask.delayed(load_tiles_in_chunk)
