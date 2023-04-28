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
import pandas
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


class CoarseDem:
    """A class to manage coarse or background DEMs in the catchment context

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
        """Load in the coarse DEM, clip and extract points"""

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

    @property
    def dem(self) -> xarray.Dataset:
        """Return the DEM over the catchment region"""
        return self._dem

    @property
    def resolution(self) -> float:
        """Return the largest dimension of the coarse DEM resolution"""

        resolution = self._dem.rio.resolution()
        resolution = max(abs(resolution[0]), abs(resolution[1]))
        return resolution

    def _set_up(self, exclusion_extent):
        """Set DEM CRS and trim the DEM to size"""

        self._dem.rio.set_crs(self.catchment_geometry.crs["horizontal"])

        # Define a buffered land & foreshore
        buffered_land_and_foreshore = geopandas.GeoDataFrame(
            geometry=self.catchment_geometry.land_and_foreshore.buffer(
                self.resolution * numpy.sqrt(2)
            )
        )

        if exclusion_extent is not None:
            # Remove any sub-pixel polygons
            exclusion_extent = exclusion_extent[
                exclusion_extent.area
                > self.catchment_geometry.resolution
                * self.catchment_geometry.resolution
            ]
            # Keep the coarse DEM where there's no LiDAR & trim outside buffered area
            self._extents = buffered_land_and_foreshore.overlay(
                exclusion_extent,
                how="difference",
            )
        else:
            # If no LiDAR - only use the coarse DEM on land
            self._extents = buffered_land_and_foreshore
        self._dem = self._dem.rio.clip(self._extents.geometry, drop=True)
        self._extract_points()

    def _extract_points(self):
        """Create a points list from the DEM"""

        # Get the DEM bounding box
        dem_bounds = self._dem.rio.bounds()
        dem_bounds = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.Polygon(
                        [
                            [dem_bounds[0], dem_bounds[1]],
                            [dem_bounds[2], dem_bounds[1]],
                            [dem_bounds[2], dem_bounds[3]],
                            [dem_bounds[0], dem_bounds[3]],
                        ]
                    )
                ]
            },
            crs=self.catchment_geometry.crs["horizontal"],
        )

        # Take the values on land only - separately consider the buffered foreshore area
        if (
            self.catchment_geometry.land.area.sum() > 0
            and dem_bounds.overlay(
                self.catchment_geometry.land, how="intersection"
            ).area.sum()
            > 0
        ):
            # Define buffered land region - buffer to avoid edge effects
            buffered_land = geopandas.GeoDataFrame(
                geometry=self.catchment_geometry.land.buffer(
                    self.resolution * numpy.sqrt(2)
                )
            )
            buffered_land = buffered_land.overlay(
                self.catchment_geometry.foreshore_and_offshore, how="difference"
            )
            buffered_land = buffered_land.overlay(
                self.catchment_geometry.full_land,
                how="intersection",
                keep_geom_type=True,
            )
            # Clip DEM to buffered lanzd
            land_dem = self._dem.rio.clip(buffered_land.geometry, drop=True)
            # get coarse DEM points on land
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
        # Take the values on foreshore only - separately consider the buffered land area
        if (
            self.catchment_geometry.foreshore.area.sum() > 0
            and dem_bounds.overlay(
                self.catchment_geometry.foreshore, how="intersection"
            ).area.sum()
            > 0
        ):
            # Define buffered foreshore region - buffer to avoid edge effects
            buffered_foreshore = geopandas.GeoDataFrame(
                geometry=self.catchment_geometry.foreshore.buffer(
                    self.resolution * numpy.sqrt(2)
                )
            )
            buffered_foreshore = buffered_foreshore.overlay(
                self.catchment_geometry.land, how="difference"
            )
            buffered_foreshore = buffered_foreshore.overlay(
                self.catchment_geometry.full_land, how="difference", keep_geom_type=True
            )
            # Clip DEM to buffered foreshore
            foreshore_dem = self._dem.rio.clip(buffered_foreshore.geometry, drop=True)

            # get coarse DEM points on the foreshore - with any positive set to zero
            if self.set_foreshore:
                foreshore_dem.data[foreshore_dem.data > 0] = 0
            foreshore_flat_z = foreshore_dem.data.flatten()
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
        ), "The coarse DEM has no values on the land or foreshore"

        # combine in an single array
        self._points = numpy.empty(
            [len(land_x) + len(foreshore_x)],
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )
        self._points["X"][: len(land_x)] = land_x
        self._points["Y"][: len(land_x)] = land_y
        self._points["Z"][: len(land_x)] = land_z

        self._points["X"][len(land_x) :] = foreshore_x
        self._points["Y"][len(land_x) :] = foreshore_y
        self._points["Z"][len(land_x) :] = foreshore_z

    @property
    def points(self) -> numpy.ndarray:
        """The coarse DEM points after any extent or foreshore value
        filtering."""

        return self._points

    @property
    def extents(self) -> geopandas.GeoDataFrame:
        """The extents for the coarse DEM"""

        return self._extents


class DemBase(abc.ABC):
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
    """

    CACHE_SIZE = 10000  # The maximum RBF input without performance issues
    SOURCE_CLASSIFICATION = {
        "LiDAR": 1,
        "ocean bathymetry": 2,
        "rivers and fans": 3,
        "waterways": 4,
        "coarse DEM": 5,
        "interpolated": 0,
        "no data": -1,
    }

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        extents: geopandas.GeoDataFrame,
    ):
        """Setup base DEM to add future tiles too"""

        self.catchment_geometry = catchment_geometry
        self._extents = extents

    @property
    def extents(self):
        """The combined extents for all added LiDAR tiles"""

        if self._extents is None:
            logging.warning(
                "Warning in DenseDem.extents: No tiles with extents have been added yet"
            )
        return self._extents

    @property
    def dem(self) -> xarray.Dataset:
        """Return the DEM over the catchment region"""
        raise NotImplementedError("dem must be instantiated in the child class")

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
        dem.rio.write_transform(inplace=True)
        if "z" in dem:
            dem.z.rio.write_crs(crs_dict["horizontal"], inplace=True)
            dem.z.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
        if "source_class" in dem:
            dem.source_class.rio.write_crs(crs_dict["horizontal"], inplace=True)
            dem.source_class.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
        if "zo" in dem:
            dem.zo.rio.write_crs(crs_dict["horizontal"], inplace=True)
            dem.zo.rio.write_nodata(numpy.nan, encoded=True, inplace=True)


class HydrologicallyConditionedDem(DemBase):
    """A class to manage loading in an already created and saved dense DEM that has yet
    to have an offshore DEM associated with it.

    Parameters
    ----------

    Logic controlling behaviour
        interpolation_method
            If not None, interpolate using that method. Valid options are 'linear',
            'nearest', and 'cubic'
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        raw_dem_path: typing.Union[str, pathlib.Path],
        extents_path: typing.Union[str, pathlib.Path],
        interpolation_method: str,
    ):
        """Load in the extents and dense DEM. Ensure the dense DEM is clipped within the
        extents"""

        # Load in dense DEM and extents
        extents = geopandas.read_file(pathlib.Path(extents_path))

        # Read in the dense DEM raster - and free up file by performing a deep copy.
        raw_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(raw_dem_path), masked=True, parse_coordinates=True
        )

        # Deep copy to ensure the opened file is properly unlocked; Squeeze as
        # rasterio.open() adds band coordinate
        raw_dem = raw_dem.squeeze("band", drop=True)
        self._write_netcdf_conventions_in_place(raw_dem, catchment_geometry.crs)

        # Ensure all values outside the exents are nan as that defines the dense extents
        # and clip the dense dem to the catchment extents to ensure performance
        raw_dem = raw_dem.rio.clip(catchment_geometry.catchment.geometry, drop=True)
        raw_dem = raw_dem.rio.clip(extents.geometry, drop=False)

        # Setup the DenseDemBase class
        super(HydrologicallyConditionedDem, self).__init__(
            catchment_geometry=catchment_geometry,
            extents=extents,
        )

        # Set attributes
        self._raw_dem = raw_dem
        self.interpolation_method = interpolation_method

        # DEMs for hydrologically conditioning
        self._offshore_dem = None
        self._river_dem = None
        self._drain_dem = None
        self._dem = None

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The dense DEM - may be opened from memory
        if self._raw_dem is not None:
            self._raw_dem.close()
            del self._raw_dem
        # The offshore DEM
        if self._offshore_dem is not None:
            self._offshore_dem.close()
            del self._offshore_dem
        # The river DEM
        if self._river_dem is not None:
            self._river_dem.close()
            del self._river_dem
        if self._drain_dem is not None:
            self._drain_dem.close()
            del self._drain_dem
        # The overall DEM
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def dem(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""

        self._dem = self.combine_dem_parts()

        # Ensure valid name and increasing dimension indexing for the dem
        if (
            self.interpolation_method is not None
        ):  # methods are 'nearest', 'linear' and 'cubic'
            interpolation_mask = numpy.isnan(self._dem.z.data)
            self._dem["z"] = self._dem.z.rio.interpolate_na(
                method=self.interpolation_method
            )
            # If any NaN remain apply nearest neighbour interpolation
            if numpy.isnan(self._dem.z.data).any():
                self._dem["z"] = self._dem.z.rio.interpolate_na(method="nearest")
            # Only set areas with successful interpolation as interpolated
            interpolation_mask &= numpy.logical_not(numpy.isnan(self._dem.z.data))
            self._dem.source_class.data[
                interpolation_mask
            ] = self.SOURCE_CLASSIFICATION["interpolated"]
        # Ensure all area's with NaN values are marked as no-data
        self._dem.source_class.data[
            numpy.isnan(self._dem.z.data)
        ] = self.SOURCE_CLASSIFICATION["no data"]
        self._dem = self._dem.rio.clip(
            self.catchment_geometry.catchment.geometry, drop=True
        )
        # Some programs require positively increasing indices
        self._dem = self._ensure_positive_indexing(self._dem)
        return self._dem

    def combine_dem_parts(self):
        """Return the combined DEM from all layers that exist."""

        if (
            self._offshore_dem is None
            and self._river_dem is None
            and self._drain_dem is None
        ):
            combined_dem = self._raw_dem
        else:
            # Combine DEM layers that have been created
            dems = []
            if self._river_dem is not None:
                dems.append(self._river_dem)
            if self._drain_dem is not None:
                dems.append(self._drain_dem)
            dems.append(self._raw_dem)
            if self._offshore_dem is not None:
                dems.append(self._offshore_dem)
            # combine the merged DEMs
            combined_dem = rioxarray.merge.merge_datasets(
                dems,
                method="first",
            )
        return combined_dem

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
        offshore_edge_dem = self._raw_dem.rio.clip(offshore_dense_data_edge.geometry)

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
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )

        offshore_edge["X"] = offshore_grid_x.flatten()[offshore_mask_z]
        offshore_edge["Y"] = offshore_grid_y.flatten()[offshore_mask_z]
        offshore_edge["Z"] = offshore_flat_z[offshore_mask_z]

        return offshore_edge

    def interpolate_bathymetry_points(
        self,
        bathymetry_points: numpy.ndarray,
        flat_x_array: numpy.ndarray,
        flat_y_array: numpy.ndarray,
        method: str,
    ) -> numpy.ndarray:
        """Interpolate the bathymetry points at the specified locations using the
        specified method."""

        if method == "rbf":
            # Ensure the number of points is not too great for RBF interpolation
            if len(bathymetry_points) < self.CACHE_SIZE:
                logging.warning(
                    "The number of points to fit and RBF interpolant to is"
                    f" {len(bathymetry_points)}. We recommend using fewer "
                    f" than {self.CACHE_SIZE} for best performance and to. "
                    "avoid errors in the `scipy.interpolate.Rbf` function"
                )
            # Create RBF function
            logging.info("Creating RBF interpolant")
            rbf_function = scipy.interpolate.Rbf(
                bathymetry_points["X"],
                bathymetry_points["Y"],
                bathymetry_points["Z"],
                function="linear",
            )
            # Tile area - this limits the maximum memory required at any one time
            flat_z_array = numpy.ones_like(flat_x_array) * numpy.nan
            number_offshore_tiles = math.ceil(len(flat_x_array) / self.CACHE_SIZE)
            for i in range(number_offshore_tiles):
                logging.info(
                    f"Offshore intepolant tile {i+1} of {number_offshore_tiles}"
                )
                start_index = int(i * self.CACHE_SIZE)
                end_index = (
                    int((i + 1) * self.CACHE_SIZE)
                    if i + 1 != number_offshore_tiles
                    else len(flat_x_array)
                )

                flat_z_array[start_index:end_index] = rbf_function(
                    flat_x_array[start_index:end_index],
                    flat_y_array[start_index:end_index],
                )
        elif method == "linear" or method == "cubic":
            # Interpolate river area - use cubic or linear interpolation
            flat_z_array = scipy.interpolate.griddata(
                points=(bathymetry_points["X"], bathymetry_points["Y"]),
                values=bathymetry_points["Z"],
                xi=(flat_x_array, flat_y_array),
                method=method,  # linear or cubic
            )
        else:
            raise ValueError("method must be rbf, linear or cubic")
        return flat_z_array

    def interpolate_ocean_bathymetry(self, bathy_contours):
        """Performs interpolation offshore outside LiDAR extents using the SciPy RBF
        function."""

        # Reset the offshore DEM
        self._offshore_dem = None

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
        # Setup the empty offshore area ready for interpolation
        offshore_no_dense_data = self.catchment_geometry.offshore_no_dense_data(
            self._extents
        )
        self._offshore_dem = self._raw_dem.rio.clip(
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

        # Set up the interpolation function
        logging.info("Offshore interpolation")
        flat_z_masked = self.interpolate_bathymetry_points(
            bathymetry_points=offshore_points,
            flat_x_array=flat_x_masked,
            flat_y_array=flat_y_masked,
            method="linear",
        )
        flat_z[mask_z] = flat_z_masked
        self._offshore_dem.z.data = flat_z.reshape(self._offshore_dem.z.data.shape)

    def interpolate_waterbed_elevations(
        self,
        estimated_bathymetry: geometry.EstimatedBathymetryPoints,
    ):
        """Performs interpolation over drains, culverts, rivers and river fan
        polygons using various interpolation techniques."""

        # First interpolated over open and closed drains using linear interpolation
        self._drain_dem = None
        if (estimated_bathymetry.points["type"] == "waterways").any():
            self._drain_dem = self._interpolate_estimated_waterways(
                estimated_bathymetry=estimated_bathymetry,
                method="cubic",
            )
        # Reset the river DEM
        self._river_dem = None
        if (estimated_bathymetry.points["type"] == "rivers").any():
            self._river_dem = self._interpolate_estimated_rivers_and_fans(
                estimated_bathymetry=estimated_bathymetry,
                method="rbf",
            )
        elif (estimated_bathymetry.points["type"] == "fans").any():
            logging.warning(
                "'fans' included but no 'rivers'. GeoFabrics currently "
                "requires both to include for hydrological conditioning"
            )

    def _interpolate_estimated_waterways(
        self,
        estimated_bathymetry: geometry.EstimatedBathymetryPoints,
        method: str,
    ) -> xarray.Dataset:
        """Performs interpolation of the estimated bed elevations with the waterways
        type_label within a polygon using the specified interpolation approach. The
        type_label also determines the source classification."""

        type_label = "waterways"

        # extract points and polygon
        estimated_points = estimated_bathymetry.filtered_points(type_label=type_label)
        estimated_polygons = estimated_bathymetry.filtered_polygons(
            type_label=type_label
        )

        # combined DEM
        combined_dem = self.combine_dem_parts()

        # Get edge points - from DEM
        edge_dem = combined_dem.rio.clip(
            estimated_polygons.dissolve().buffer(self.catchment_geometry.resolution),
            drop=True,
        )
        edge_dem = edge_dem.rio.clip(
            estimated_polygons.dissolve().geometry,
            invert=True,
            drop=True,
        )
        # Define the edge points
        grid_x, grid_y = numpy.meshgrid(edge_dem.x, edge_dem.y)
        flat_z = edge_dem.z.data.flatten()
        mask_z = ~numpy.isnan(flat_z)
        # Define edge points and heights
        edge_points = numpy.empty(
            [mask_z.sum().sum()],
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )
        edge_points["X"] = grid_x.flatten()[mask_z]
        edge_points["Y"] = grid_y.flatten()[mask_z]
        edge_points["Z"] = flat_z[mask_z]

        # Combine the estimated and edge points
        bathy_points = numpy.concatenate([edge_points, estimated_points])

        # Setup the empty area ready for interpolation
        estimated_dem = combined_dem.rio.clip(estimated_polygons.geometry)
        # Set value for all, then use clip to set regions outside polygon to NaN
        estimated_dem.z.data[:] = 0
        estimated_dem.source_class.data[:] = self.SOURCE_CLASSIFICATION[type_label]
        estimated_dem = estimated_dem.rio.clip(estimated_polygons.geometry)

        grid_x, grid_y = numpy.meshgrid(estimated_dem.x, estimated_dem.y)
        flat_z = estimated_dem.z.data[:].flatten()
        mask_z = ~numpy.isnan(flat_z)

        flat_x_masked = grid_x.flatten()[mask_z]
        flat_y_masked = grid_y.flatten()[mask_z]
        flat_z_masked = flat_z[mask_z]

        # check there are actually pixels in the river
        logging.info(f"There are {len(flat_z_masked)} estimated points")

        # Interpolate river area - use cubic or linear interpolation
        logging.info("Offshore interpolation")
        flat_z_masked = self.interpolate_bathymetry_points(
            bathymetry_points=bathy_points,
            flat_x_array=flat_x_masked,
            flat_y_array=flat_y_masked,
            method=method,
        )

        # Set the interpolated value in the DEM
        flat_z[mask_z] = flat_z_masked
        estimated_dem.z.data = flat_z.reshape(estimated_dem.z.data.shape)

        return estimated_dem

    def _interpolate_estimated_rivers_and_fans(
        self,
        estimated_bathymetry: geometry.EstimatedBathymetryPoints,
        method: str,
    ) -> xarray.Dataset:
        """Performs interpolation from estimated bathymetry points within a polygon
        using the specified interpolation approach after filtering the points based
        on the type label. The type_label also determines the source classification."""

        # Extract river points and polygon
        river_points = estimated_bathymetry.filtered_points(type_label="rivers")
        river_polygons = estimated_bathymetry.filtered_polygons(type_label="rivers")
        # Combine with fan data if provided
        if (estimated_bathymetry.points["type"] == "fans").any():
            fan_points = estimated_bathymetry.filtered_points(type_label="fans")
            fan_polygons = estimated_bathymetry.filtered_polygons(type_label="fans")
            estimated_points = numpy.concatenate([river_points, fan_points])
            estimated_polygons = geopandas.GeoDataFrame(
                pandas.concat([river_polygons, fan_polygons], ignore_index=True),
                crs=river_polygons.crs,
            )
        else:
            estimated_points = river_points
            estimated_polygons = river_polygons

        # Combined DEM
        combined_dem = self.combine_dem_parts()

        # Get the river and fan edge points - from DEM
        edge_dem = combined_dem.rio.clip(
            estimated_polygons.dissolve().buffer(self.catchment_geometry.resolution),
            drop=True,
        )
        edge_dem = edge_dem.rio.clip(
            estimated_polygons.dissolve().geometry,
            invert=True,
            drop=True,
        )
        # Define the river and mouth edge points
        grid_x, grid_y = numpy.meshgrid(edge_dem.x, edge_dem.y)
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        flat_z = edge_dem.z.data.flatten()
        mask_z = ~numpy.isnan(flat_z)

        # Interpolate the estimated river bank heights along only the river
        if estimated_bathymetry.bank_heights_exist(type_label="rivers"):
            # TODO consider interpolation along the banks if not bank heights
            # Create a mask defining the river points within the edge_points
            edge_dem = edge_dem.rio.clip(
                river_polygons.dissolve().buffer(self.catchment_geometry.resolution),
                drop=False,  # Don't drop as mask is the same size as mask_z
            )
            mask_z_river = ~numpy.isnan(edge_dem.z.data.flatten())

            # Get the estimated river bank heights
            river_bank_points = estimated_bathymetry.filtered_bank_height_points(
                type_label="rivers"
            )
            # Interpolate the estimated river bank heights along the river
            xy_out = numpy.concatenate(
                [[flat_x[mask_z_river]], [flat_y[mask_z_river]]], axis=0
            ).transpose()
            options = {
                "radius": estimated_bathymetry.points["width"].max(),
                "raster_type": geometry.RASTER_TYPE,
                "method": "linear",
            }
            estimated_river_edge_z = elevation_from_points(
                point_cloud=river_bank_points, xy_out=xy_out, options=options
            )

            # Take the estimated bank heights where lower than the DEM edge values
            river_edge_z = estimated_river_edge_z.copy()
            river_edge_z[flat_z[mask_z_river] < estimated_river_edge_z] = flat_z[
                mask_z_river
            ][flat_z[mask_z_river] < estimated_river_edge_z]
            flat_z[mask_z_river] = river_edge_z

        # Use the flat_x/y/z to define edge points and heights
        edge_points = numpy.empty(
            [mask_z.sum().sum()],
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )
        edge_points["X"] = flat_x[mask_z]
        edge_points["Y"] = flat_y[mask_z]
        edge_points["Z"] = flat_z[mask_z]
        # Combine the estimated and edge points
        bathy_points = numpy.concatenate([edge_points, estimated_points])

        # Setup the empty river (& fan) area ready for interpolation
        estimated_dem = combined_dem.rio.clip(estimated_polygons.geometry)
        # Set value for all, then use clip to set regions outside polygon to NaN
        estimated_dem.z.data[:] = 0
        estimated_dem.source_class.data[:] = self.SOURCE_CLASSIFICATION[
            "rivers and fans"
        ]
        estimated_dem = estimated_dem.rio.clip(estimated_polygons.geometry)

        grid_x, grid_y = numpy.meshgrid(estimated_dem.x, estimated_dem.y)
        flat_z = estimated_dem.z.data[:].flatten()
        mask_z = ~numpy.isnan(flat_z)

        flat_x_masked = grid_x.flatten()[mask_z]
        flat_y_masked = grid_y.flatten()[mask_z]
        flat_z_masked = flat_z[mask_z]

        # check there are actually pixels in the river
        logging.info(f"There are {len(flat_z_masked)} estimated points")

        # Interpolate river area - use specified interpolation
        logging.info("Offshore interpolation")
        flat_z_masked = self.interpolate_bathymetry_points(
            bathymetry_points=bathy_points,
            flat_x_array=flat_x_masked,
            flat_y_array=flat_y_masked,
            method=method,
        )

        # Set the interpolated value in the DEM
        flat_z[mask_z] = flat_z_masked
        estimated_dem.z.data = flat_z.reshape(estimated_dem.z.data.shape)

        return estimated_dem


class LidarBase(DemBase):
    """A class with some base methods for reading in LiDAR data.

    Parameters
    ----------

    catchment_geometry
        Defines the geometry of the catchment
    elevation_range
        The range of valid LiDAR elevations. i.e. define elevation filtering to apply.
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        self.elevation_range = elevation_range
        assert elevation_range is None or (
            type(elevation_range) == list and len(elevation_range) == 2
        ), "Error the 'elevation_range' must either be none, or a two entry list"

        self._dem = None

        super(LidarBase, self).__init__(
            catchment_geometry=catchment_geometry,
            extents=None,
        )

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The dense DEM - may be opened from memory
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def dem(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""

        # Ensure positively increasing indices as required by some programs
        self._dem = self._ensure_positive_indexing(self._dem)
        return self._dem

    def _extents_from_mask(self, mask: numpy.ndarray, dem: xarray.Dataset):
        """Define the spatial extents of the pixels in the DEM as defined by the mask
        (i.e. what are the spatial extents of pixels in the DEM that are marked True in
         the mask).

         Remove holes as these can cause self intersection warnings."""

        dense_extents = [
            shapely.geometry.shape(polygon[0])
            for polygon in rasterio.features.shapes(numpy.uint8(mask))
            if polygon[1] == 1.0
        ]
        dense_extents = shapely.ops.unary_union(dense_extents)

        # Remove internal holes for select types as these may cause self-intersections
        if type(dense_extents) is shapely.geometry.Polygon:
            dense_extents = shapely.geometry.Polygon(dense_extents.exterior)
        elif type(dense_extents) is shapely.geometry.MultiPolygon:
            dense_extents = shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.Polygon(polygon.exterior)
                    for polygon in dense_extents.geoms
                ]
            )
        # Convert into a Geopandas dataframe
        dense_extents = geopandas.GeoDataFrame(
            {"geometry": [dense_extents]}, crs=self.catchment_geometry.crs["horizontal"]
        )

        # Move from image to the dem space & buffer(0) to reduce self-intersections
        dense_dem_affine = dem.z.rio.transform()
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

    def _set_up_chunks(self, chunk_size: int) -> (list, list):
        """Define the chunked coordinates to cover the catchment"""

        raise NotImplementedError(
            "_set_up_chunks must be instantiated in the child " "class"
        )

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
                            (dim_x.min(), dim_y.min()),
                            (dim_x.max(), dim_y.min()),
                            (dim_x.max(), dim_y.max()),
                            (dim_x.min(), dim_y.max()),
                        ]
                    )
                ]
            },
            crs=self.catchment_geometry.crs["horizontal"],
        )

        # Define region to rasterise inside the chunk area
        chunk_region_to_tile = geopandas.GeoDataFrame(
            geometry=region_to_rasterise.buffer(radius).clip(
                chunk_geometry.buffer(radius), keep_geom_type=True
            )
        )
        # remove any subpixel polygons
        chunk_region_to_tile = chunk_region_to_tile[
            chunk_region_to_tile.area
            > self.catchment_geometry.resolution * self.catchment_geometry.resolution
        ]

        return chunk_region_to_tile

    def _tile_index_column_name(
        self, tile_index_file: typing.Union[str, pathlib.Path] = None
    ):
        """Read in tile index file and determine the column name of the tile
        geometries"""
        # Check to see if a extents file was added
        tile_index_extents = (
            geopandas.read_file(tile_index_file)
            if tile_index_file is not None
            else None
        )
        tile_index_name_column = None

        # If there is a tile_index_file - remove tiles outside the catchment & get the
        # 'file name' column
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

    def _check_valid_inputs(self, lidar_datasets_info, chunk_size):
        """Check the combination of inputs for adding LiDAR is valid.

        Parameters
        ----------

        lidar_datasets_info
            A dictionary of dictionaties of LiDAR dataset information. The CRS, list of
            LAS files and tile index file are included for each dataset.
        chunk_size
            The chunk size in pixels for parallel/staged processing
        """

        for dataset_name in lidar_datasets_info:
            # Check the source_crs is valid
            source_crs = lidar_datasets_info[dataset_name]["crs"]
            if source_crs is not None:
                assert "horizontal" in source_crs, (
                    "The horizontal component of the source CRS is not specified. Both "
                    "horizontal and vertical CRS need to be defined. The source_crs "
                    f"specified is: {source_crs} for {dataset_name}"
                )
                assert "vertical" in source_crs, (
                    "The vertical component of the source CRS is not specified. Both "
                    "horizontal and vertical CRS need to be defined. The source_crs "
                    f"specified is: {self.source_crs} for {dataset_name}"
                )
            # Check some LiDAR files are soecified
            lidar_files = lidar_datasets_info[dataset_name]["file_paths"]
            assert len(lidar_files) >= 1, "There are no LiDAR files specified"
            # Check for valid combination of chunk_size, lidar_files and tile_index_file
            if chunk_size is None:
                assert (
                    len(lidar_files) == 1
                ), "If there is no chunking there must be only one LiDAR file"
            else:
                assert (
                    chunk_size > 0 and type(chunk_size) is int
                ), "chunk_size must be a positive integer"
                tile_index_file = lidar_datasets_info[dataset_name]["tile_index_file"]
                assert tile_index_file is not None, (
                    "A tile index file must be provided if chunking is defined for"
                    f"for {dataset_name}"
                )
        # There should only be one dataset if there is no chunking information
        if chunk_size is None:
            assert len(lidar_datasets_info) == 1, (
                "If there is no chunking there must only be one LiDAR dataset. Instead"
                "there is {len(lidar_file_info)} with keys {lidar_file_info.keys()}"
            )

    def add_lidar(
        self,
        lidar_files: typing.List[typing.Union[str, pathlib.Path]],
        tile_index_file: typing.Union[str, pathlib.Path],
        chunk_size: int,
        lidar_classifications_to_keep: list,
        source_crs: dict,
        metadata: dict,
    ):
        """Read in all LiDAR files and use to create a dense DEM.

        Parameters
        ----------

        source_crs
            Coordinate reference system information
        chunk_size
            The chunk size in pixels for parallel/staged processing
        lidar_files
            The list of LiDAR files to read in
        tile_index_file
            A file specifying the spatial extents of the LiDAR files.
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM.
        """

        raise NotImplementedError("add_lidar must be instantiated in the child class")

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
        """Create/Update dataset from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        raise NotImplementedError(
            "_add_tiled_lidar_chunked must be instantiated in the" " child class"
        )

    def _add_lidar_no_chunking(
        self,
        lidar_file: typing.Union[str, pathlib.Path],
        region_to_rasterise: geopandas.GeoDataFrame,
        options: dict,
        source_crs: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create/Update dataset from a single LiDAR file."""

        raise NotImplementedError(
            "_add_lidar_no_chunking must be instantiated in the " "child class"
        )


class RawDem(LidarBase):
    """A class to manage the creation of a 'raw' DEM from LiDAR tiles, and/or a
    coarse DEM.

    Parameters
    ----------

    drop_offshore_lidar
        If True only keep LiDAR values within the foreshore and land regions defined by
        the catchment_geometry. If False keep all LiDAR values.
    elevation_range
        Optitionally specify a range of valid elevations. Any LiDAR points with
        elevations outside this range will be filtered out.
    lidar_interpolation_method
        The interpolation method to apply to LiDAR during downsampling/averaging.
        Options are: mean, median, IDW, max, min, STD.
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        lidar_interpolation_method: str,
        drop_offshore_lidar: bool = True,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(RawDem, self).__init__(
            catchment_geometry=catchment_geometry,
            elevation_range=elevation_range,
        )

        self.drop_offshore_lidar = drop_offshore_lidar
        self.lidar_interpolation_method = lidar_interpolation_method
        self._dem = None

    def _set_up_chunks(self, chunk_size: int) -> (list, list):
        """Define the chunks to break the catchment into when reading in and
        downsampling LiDAR.

        Parameters
        ----------

        chunk_size
            The size in pixels of each chunk.
        """

        bounds = self.catchment_geometry.catchment.geometry.bounds
        resolution = self.catchment_geometry.resolution

        # Determine the number of chunks
        minx = bounds.minx.min()
        maxy = bounds.maxy.max()
        n_chunks_x = int(
            numpy.ceil((bounds.maxx.max() - minx) / (chunk_size * resolution))
        )
        n_chunks_y = int(
            numpy.ceil((maxy - bounds.miny.min()) / (chunk_size * resolution))
        )

        # The x coordinates rounded up to the nearest chunk
        dim_x = numpy.arange(
            minx + resolution / 2,
            minx + resolution / 2 + n_chunks_x * chunk_size * resolution,
            resolution,
            dtype=geometry.RASTER_TYPE,
        )
        dim_x = dim_x.reshape((n_chunks_x, chunk_size))
        # The y coordinates rounded up to the nearest chunk
        dim_y = numpy.arange(
            maxy - resolution / 2,
            maxy - resolution / 2 - n_chunks_y * chunk_size * resolution,
            -resolution,
            dtype=geometry.RASTER_TYPE,
        )
        dim_y = dim_y.reshape((n_chunks_y, chunk_size))
        return dim_x, dim_y

    def _calculate_raw_extents(self):
        """Define the extents of the DEM with values (i.e. what are the spatial extents
        of pixels in the DEM that are defined from LiDAR or a coarse DEM)."""

        # Defines extents where raw DEM values exist
        mask = numpy.logical_not(numpy.isnan(self._dem.z.data))
        extents = self._extents_from_mask(mask, self._dem)
        return extents

    def add_lidar(
        self,
        lidar_datasets_info: dict,
        chunk_size: int,
        lidar_classifications_to_keep: list,
        metadata: dict,
    ):
        """Read in all LiDAR files and use to define a 'raw' DEM with elevations in
        pixels where there is LiDAR or coarse DEM information.

        Parameters
        ----------

        lidar_datasets_info
            One of more dictionaries of LiDAR dataset information - including a list of
            LAS files, CRS and tile index file for each.
        chunk_size
            The chunk size in pixels for parallel/staged processing
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM.
        """

        # Check valid inputs
        self._check_valid_inputs(
            lidar_datasets_info=lidar_datasets_info, chunk_size=chunk_size
        )
        # Define the region to rasterise over
        region_to_rasterise = (
            self.catchment_geometry.land_and_foreshore
            if self.drop_offshore_lidar
            else self.catchment_geometry.catchment
        )
        # create dictionary defining raster options
        raster_options = {
            "lidar_classifications_to_keep": lidar_classifications_to_keep,
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": self.elevation_range,
            "radius": self.catchment_geometry.resolution / numpy.sqrt(2),
            "method": self.lidar_interpolation_method,
        }

        # Don't use dask delayed if there is no chunking
        if chunk_size is None:
            dem = self._add_lidar_no_chunking(
                lidar_dataset_info=next(iter(lidar_datasets_info.values())),
                region_to_rasterise=region_to_rasterise,
                options=raster_options,
                metadata=metadata,
            )
        else:
            dem = self._add_tiled_lidar_chunked(
                lidar_datasets_info=lidar_datasets_info,
                raster_options=raster_options,
                region_to_rasterise=region_to_rasterise,
                chunk_size=chunk_size,
                metadata=metadata,
            )
        # Clip DEM to Catchment and ensure NaN outside region to rasterise
        dem = dem.rio.clip(self.catchment_geometry.catchment.geometry, drop=True)
        dem = dem.rio.clip(region_to_rasterise.geometry, drop=False)

        # If drop offshrore LiDAR ensure the foreshore values are 0 or negative
        if (
            self.drop_offshore_lidar
            and self.catchment_geometry.foreshore.area.sum() > 0
        ):
            buffered_foreshore = geopandas.GeoDataFrame(
                geometry=self.catchment_geometry.foreshore.buffer(
                    self.catchment_geometry.resolution * numpy.sqrt(2)
                )
            )
            buffered_foreshore = buffered_foreshore.overlay(
                self.catchment_geometry.full_land, how="difference", keep_geom_type=True
            )
            # Clip DEM to buffered foreshore
            mask = numpy.logical_not(
                numpy.isnan(
                    dem.z.rio.clip(buffered_foreshore.geometry, drop=False).data
                )
            )

            # get coarse DEM points on the foreshore - with any positive set to zero
            dem.z.data[mask & (dem.z.data > 0)] = 0
        self._dem = dem
        # Create a polygon defining the region where there are dense DEM values
        self._extents = self._calculate_raw_extents()

    def _add_tiled_lidar_chunked(
        self,
        lidar_datasets_info: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        chunk_size: int,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create a 'raw'' DEM from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        # get chunking information
        chunked_dim_x, chunked_dim_y = self._set_up_chunks(chunk_size)
        elevations = []

        logging.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")
        for dataset_name in lidar_datasets_info.keys():
            # Pull out the dataset information
            lidar_files = lidar_datasets_info[dataset_name]["file_paths"]
            tile_index_file = lidar_datasets_info[dataset_name]["tile_index_file"]
            source_crs = lidar_datasets_info[dataset_name]["crs"]

            # create a map from tile name to tile file name
            lidar_files_map = {
                lidar_file.name: lidar_file for lidar_file in lidar_files
            }

            # remove all tiles entirely outside the region to raserise
            tile_index_extents, tile_index_name_column = self._tile_index_column_name(
                tile_index_file
            )

            # cycle through index chunks - and collect in a delayed array
            logging.info(f"Running over dataset {dataset_name}")
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

                    # Load in files into tiles
                    chunk_lidar_files = select_lidar_files(
                        tile_index_extents=tile_index_extents,
                        tile_index_name_column=tile_index_name_column,
                        chunk_region_to_tile=chunk_region_to_tile,
                        lidar_files_map=lidar_files_map,
                    )
                    chunk_points = delayed_load_tiles_in_chunk(
                        lidar_files=chunk_lidar_files,
                        source_crs=source_crs,
                        chunk_region_to_tile=chunk_region_to_tile,
                        catchment_geometry=self.catchment_geometry,
                    )
                    # Rasterise tiles
                    delayed_chunked_x.append(
                        dask.array.from_delayed(
                            delayed_elevation_over_chunk(
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
            elevations.append(dask.array.block(delayed_chunked_matrix))
        chunked_dem = self._create_data_set(
            x=numpy.concatenate(chunked_dim_x),
            y=numpy.concatenate(chunked_dim_y),
            elevations=elevations,
            metadata=metadata,
        )
        logging.info("Computing chunks")
        chunked_dem = chunked_dem.compute()
        logging.debug("Chunked DEM computed")

        return chunked_dem

    def _add_lidar_no_chunking(
        self,
        lidar_dataset_info: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        options: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create a 'raw' DEM from a single LiDAR file with no chunking."""

        lidar_file = lidar_dataset_info["file_paths"][0]
        source_crs = lidar_dataset_info["crs"]
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
        tile_points = pdal_pipeline.arrays[0]

        # define the raster/DEM dimensions
        bounds = self.catchment_geometry.catchment.geometry.bounds
        resolution = self.catchment_geometry.resolution
        dim_x = numpy.arange(
            bounds.minx.min() + resolution / 2,
            bounds.maxx.max(),
            resolution,
            dtype=geometry.RASTER_TYPE,
        )
        dim_y = numpy.arange(
            bounds.maxy.max() - resolution / 2,
            bounds.miny.min(),
            -resolution,
            dtype=geometry.RASTER_TYPE,
        )

        # Create elevation raster
        raster_values = self._elevation_over_tile(
            dim_x=dim_x, dim_y=dim_y, tile_points=tile_points, options=options
        )
        elevation = raster_values.reshape((len(dim_y), len(dim_x)))

        # Create xarray
        dem = self._create_data_set(
            x=dim_x, y=dim_y, elevations=[elevation], metadata=metadata
        )

        return dem

    def _elevation_over_tile(
        self,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        tile_points: numpy.ndarray,
        options: dict,
    ):
        """Rasterise all points within a tile to give elevation. Does not require a tile
        index file."""

        # keep only the specified classifications (should be ground / water)
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

        if len(tile_points) == 0:
            logging.warning(
                "In RawDem._elevation_over_tile the tile has no data and is being "
                "ignored."
            )
            return
        # Get the grided locations overwhich to perform IDW
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()

        # Perform the specified rasterisation over the grid locations
        z_flat = elevation_from_points(
            point_cloud=tile_points, xy_out=xy_out, options=options
        )
        grid_z = z_flat.reshape(grid_x.shape)

        return grid_z

    def _create_data_set(
        self, x: numpy.ndarray, y: numpy.ndarray, elevations: list, metadata: dict
    ) -> xarray.Dataset:
        """A function to create a new dataset from x, y and z arrays.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            elevations
                A list of elevations over the x, and y coordiantes.One for each dataset
        """

        # Lood over each dataset and add data to the DEM
        dems = []
        for z in elevations:
            # Create source variable - assume all values are defined from LiDAR
            source_class = numpy.ones_like(z) * numpy.nan
            source_class[
                numpy.logical_not(numpy.isnan(z))
            ] = self.SOURCE_CLASSIFICATION["LiDAR"]
            dem = xarray.Dataset(
                data_vars=dict(
                    z=(
                        ["y", "x"],
                        z,
                        {
                            "units": "m",
                            "long_name": "ground elevation",
                            "vertical_datum": "EPSG:"
                            f"{self.catchment_geometry.crs['vertical']}",
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
                    "source": f"{metadata['library_name']} version "
                    f"{metadata['library_version']}",
                    "description": f"{metadata['library_name']}:"
                    f"{metadata['class_name']} resolution "
                    f"{self.catchment_geometry.resolution}",
                    "history": f"{metadata['utc_time']}: {metadata['library_name']}"
                    f":{metadata['class_name']} resolution "
                    f"{self.catchment_geometry.resolution};",
                    "geofabrics_instructions": f"{metadata['instructions']}",
                },
            )
            # ensure the expected CF conventions are followed
            self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)
            dems.append(dem)
        if len(dems) == 1:
            dem = dems[0]
        else:
            dem = rioxarray.merge.merge_datasets(dems, method="first")
        # Set areas with no values to No Data
        dem.source_class.data[
            numpy.isnan(dem.source_class.data)
        ] = self.SOURCE_CLASSIFICATION["no data"]

        # set any offshre values to ocean assuming drop offshore is selected
        if (
            self.catchment_geometry.foreshore_and_offshore.area.sum() > 0
            and self.drop_offshore_lidar
        ):
            ocean_mask = numpy.logical_not(
                numpy.isnan(
                    dem.z.rio.clip(
                        self.catchment_geometry.foreshore_and_offshore.geometry,
                        drop=False,
                    ).data
                )
            )
            dem.source_class.data[ocean_mask] = self.SOURCE_CLASSIFICATION[
                "ocean bathymetry"
            ]
        return dem

    def add_coarse_dem(self, coarse_dem: CoarseDem):
        """Fill gaps in dense DEM from areas with no LiDAR with the coarse DEM.
        Perform linear interpolation.

        Currently doesn't use chunking - this may be required if a large area is covered
        by the coarse DEM."""

        logging.info("Add a coarse DEM to fill areas outside the LiDAR extents")
        # Only rasterise on land/foreshore and outside where there is LiDAR
        region_to_rasterise = self.catchment_geometry.land_and_foreshore.overlay(
            self._extents, how="difference"
        )

        # Create a mask of area not covered by LiDAR (excludes holes in LiDAR tiles)
        z = self._dem.z.copy(deep=True)
        z.data[:] = 0
        mask = numpy.logical_not(
            numpy.isnan(z.rio.clip(region_to_rasterise.geometry, drop=False).data)
        )

        if mask.sum() == 0:
            logging.warning(
                "RawDem.add_coarse_dem: LiDAR covers all raster values so the "
                "coarse DEM is being ignored."
            )
            return
        # create dictionary defining raster options
        # Set search radius to the diagonal cell length to ensure corners covered
        raster_options = {
            "raster_type": geometry.RASTER_TYPE,
            "radius": coarse_dem.resolution * numpy.sqrt(2),
            "method": "linear",
        }
        # Get the grid locations overwhich to perform averaging
        grid_x, grid_y = numpy.meshgrid(self._dem.x, self._dem.y)

        # Mask to only rasterise where there aren't already LiDAR derived values
        xy_out = numpy.empty((mask.sum(), 2))
        xy_out[:, 0] = grid_x[mask]
        xy_out[:, 1] = grid_y[mask]

        # Perform specified averaging from the coarse DEM where there is no data
        z_flat = elevation_from_points(
            point_cloud=coarse_dem.points, xy_out=xy_out, options=raster_options
        )

        # Update the DEM
        self._dem.z.data[mask] = z_flat
        # Update the source layer - where defined by the coarse DEM and set foreshore
        self._dem.source_class.data[
            mask & numpy.logical_not(numpy.isnan(self._dem.z.data))
        ] = self.SOURCE_CLASSIFICATION["coarse DEM"]
        if (
            self.catchment_geometry.foreshore.area.sum() > 0
            and coarse_dem.set_foreshore
        ):
            foreshore_mask = numpy.logical_not(
                numpy.isnan(
                    self._dem.z.rio.clip(
                        self.catchment_geometry.foreshore.geometry,
                        drop=False,
                    ).data
                )
            )
            self._dem.source_class.data[
                mask & foreshore_mask
            ] = self.SOURCE_CLASSIFICATION["ocean bathymetry"]
        # Update the dense DEM extents
        self._extents = self._calculate_raw_extents()


class RoughnessDem(LidarBase):
    """A class to add a roughness (zo) layer to a hydrologically conditioned DEM.

    They STD and mean height of ground cover classified points are calculated from the
    LiDAR data and z (elevation) layer of the hydrologically conditioned DEM, and used
    to estimate roughness emperically.

    RoughnessDem logic can be controlled by the constructor inputs.

    Parameters
    ----------

    catchment_geometry
        Defines the geometry of the catchment
    hydrological_dem_path
        The path to the hydrologically conditioned DEM.
    interpolation_method
        If not None, interpolate using that method. Valid options are 'linear',
        'nearest', and 'cubic'.
    lidar_interpolation_method
        The interpolation method to apply to LiDAR. Options are: mean, median, IDW.
    """

    ROUGHNESS_DEFAULTS = {
        "land": 0.014,
        "water": 0.004,
        "minimum": 0.00001
    }

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        hydrological_dem_path: typing.Union[str, pathlib.Path],
        interpolation_method: str,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(RoughnessDem, self).__init__(
            catchment_geometry=catchment_geometry,
            elevation_range=elevation_range,
        )

        # Load hyrdological DEM. Squeeze as rasterio.open() adds band coordinate.
        hydrological_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(hydrological_dem_path), masked=True, parse_coordinates=True
        )
        hydrological_dem = hydrological_dem.squeeze("band", drop=True)
        self._write_netcdf_conventions_in_place(
            hydrological_dem, catchment_geometry.crs
        )

        # Ensure the resolution of the hydrological DEM matches the input DEM
        assert (
            abs(float(hydrological_dem.x[1] - hydrological_dem.x[0]))
            == self.catchment_geometry.resolution
        ), (
            "The input DEM resolution doesn't match the input resolution. These must "
            "match"
        )

        # Clip to the catchment extents to ensure performance
        hydrological_dem = hydrological_dem.rio.clip(
            self.catchment_geometry.catchment.geometry, drop=True
        )

        self.interpolation_method = interpolation_method
        self._dem = hydrological_dem

    def _calculate_lidar_extents(self):
        """Calculate the extents of the LiDAR data."""

        # Defines extents where raw DEM values exist
        mask = self._dem.source_class.data == self.SOURCE_CLASSIFICATION["LiDAR"]
        extents = self._extents_from_mask(mask, self._dem)
        return extents

    def _set_up_chunks(self, chunk_size: int) -> (list, list):
        """Define the chunks to break the catchment into when reading in and
        downsampling LiDAR.

        Parameters
        ----------

        chunk_size
            The size in pixels of each chunk.
        """

        dim_x_all = self._dem.x.data
        dim_y_all = self._dem.y.data
        resolution = self.catchment_geometry.resolution

        # Check dims x and y are ordered in expected direction
        assert dim_x_all[-1] > dim_x_all[0], "dim_x should be increasing along length"
        assert dim_y_all[-1] < dim_y_all[0], "dim_y should be decreasing along length"

        # Determine the number of chunks
        n_chunks_x = int(
            numpy.ceil((dim_x_all[-1] - dim_x_all[0]) / (chunk_size * resolution))
        )
        n_chunks_y = int(
            numpy.ceil((dim_y_all[0] - dim_y_all[-1]) / (chunk_size * resolution))
        )

        # Determine x coordinates rounded up to the nearest chunk
        dim_x = [
            numpy.arange(
                dim_x_all[0] + i * chunk_size * resolution,
                dim_x_all[0] + (i + 1) * chunk_size * resolution,
                resolution,
                dtype=geometry.RASTER_TYPE,
            )
            for i in range(n_chunks_x)
        ]
        # Determine y coordinates rounded up to the nearest chunk
        dim_y = [
            numpy.arange(
                dim_y_all[0] - i * chunk_size * resolution,
                dim_y_all[0] - (i + 1) * chunk_size * resolution,
                -resolution,
                dtype=geometry.RASTER_TYPE,
            )
            for i in range(n_chunks_y)
        ]
        return dim_x, dim_y

    def add_lidar(
        self,
        lidar_datasets_info: dict,
        chunk_size: int,
        lidar_classifications_to_keep: list,
        metadata: dict,
    ):
        """Read in all LiDAR files and use the point cloud distribution, source_class
        information, and hydrologiaclly conditioned elevations to estimate the roughness
        across the DEM.

        Parameters
        ----------

        lidar_datasets_info
            One of more dictionaries of LiDAR dataset information - including a list of
            LAS files, CRS and tile index file for each.
        chunk_size
            The chunk size in pixels for parallel/staged processing
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM.
        """

        # Check valid inputs
        self._check_valid_inputs(
            lidar_datasets_info=lidar_datasets_info, chunk_size=chunk_size
        )

        # Calculate roughness from LiDAR
        region_to_rasterise = self._calculate_lidar_extents()

        # create dictionary defining raster options
        raster_options = {
            "lidar_classifications_to_keep": lidar_classifications_to_keep,
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": self.elevation_range,
            "radius": self.catchment_geometry.resolution / numpy.sqrt(2),
        }

        # Set roughness where LiDAR
        if chunk_size is None:  # If one file it's ok if there is no tile_index
            self._dem = self._add_lidar_no_chunking(
                lidar_dataset_info=next(iter(lidar_datasets_info.values())),
                region_to_rasterise=region_to_rasterise,
                options=raster_options,
                metadata=metadata,
            )
        else:
            self._dem = self._add_tiled_lidar_chunked(
                lidar_datasets_info=lidar_datasets_info,
                raster_options=raster_options,
                region_to_rasterise=region_to_rasterise,
                chunk_size=chunk_size,
                metadata=metadata,
            )
        # Set roughness where water
        self._dem.zo.data[
            (
                self._dem.source_class.data
                == self.SOURCE_CLASSIFICATION["ocean bathymetry"]
            )
            | (
                self._dem.source_class.data
                == self.SOURCE_CLASSIFICATION["rivers and fans"]
            )
            | (self._dem.source_class.data == self.SOURCE_CLASSIFICATION["waterways"])
        ] = self.ROUGHNESS_DEFAULTS["water"]
        # Set roughness where land and no LiDAR
        self._dem.zo.data[
            self._dem.source_class.data == self.SOURCE_CLASSIFICATION["coarse DEM"]
        ] = self.ROUGHNESS_DEFAULTS[
            "land"
        ]  # or LiDAR with no roughness estimate
        # Interpolate any missing roughness values
        if self.interpolation_method is not None:
            self._dem["zo"] = self._dem.zo.rio.interpolate_na(
                method=self.interpolation_method
            )
            # If any NaN remain apply nearest neighbour interpolation
            if numpy.isnan(self._dem.z.data).any():
                self._dem["zo"] = self._dem.zo.rio.interpolate_na(method="nearest")
        self._dem = self._dem.rio.clip(
            self.catchment_geometry.catchment.geometry, drop=True
        )

    def _add_tiled_lidar_chunked(
        self,
        lidar_datasets_info: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        chunk_size: int,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create a roughness layer with estimates where there is LiDAR from a set of
        tiled LiDAR files. Read these in over non-overlapping chunks and then combine.
        """

        # get chunks to tile over
        chunked_dim_x, chunked_dim_y = self._set_up_chunks(chunk_size)

        roughnesses = []

        logging.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")
        for dataset_name in lidar_datasets_info.keys():
            # Pull out the dataset information
            lidar_files = lidar_datasets_info[dataset_name]["file_paths"]
            tile_index_file = lidar_datasets_info[dataset_name]["tile_index_file"]
            source_crs = lidar_datasets_info[dataset_name]["crs"]

            # create a map from tile name to tile file name
            lidar_files_map = {
                lidar_file.name: lidar_file for lidar_file in lidar_files
            }

            # Remove all tiles entirely outside the region to raserise
            tile_index_extents, tile_index_name_column = self._tile_index_column_name(
                tile_index_file
            )

            # cycle through chunks - and collect in a delayed array
            logging.info(f"Running over dataset {dataset_name}")
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

                    # Load in files into tiles
                    chunk_lidar_files = select_lidar_files(
                        tile_index_extents=tile_index_extents,
                        tile_index_name_column=tile_index_name_column,
                        chunk_region_to_tile=chunk_region_to_tile,
                        lidar_files_map=lidar_files_map,
                    )
                    chunk_points = delayed_load_tiles_in_chunk(
                        lidar_files=chunk_lidar_files,
                        source_crs=source_crs,
                        chunk_region_to_tile=chunk_region_to_tile,
                        catchment_geometry=self.catchment_geometry,
                    )
                    # Rasterise tiles
                    xy_ground = self._dem.z.sel(
                        x=dim_x, y=dim_y, method="nearest"
                    ).data.flatten()
                    delayed_chunked_x.append(
                        dask.array.from_delayed(
                            delayed_roughness_over_chunk(
                                dim_x=dim_x,
                                dim_y=dim_y,
                                tile_points=chunk_points,
                                xy_ground=xy_ground,
                                options=raster_options,
                            ),
                            shape=(chunk_size, chunk_size),
                            dtype=numpy.float32,
                        )
                    )
                delayed_chunked_matrix.append(delayed_chunked_x)
            # Combine chunks and add to dataset
            roughnesses.append(dask.array.block(delayed_chunked_matrix))
        chunked_dem = self._add_roughness_to_data_set(
            x=numpy.concatenate(chunked_dim_x),
            y=numpy.concatenate(chunked_dim_y),
            roughnesses=roughnesses,
            metadata=metadata,
            region_to_rasterise=region_to_rasterise,
        )
        logging.info("Computing chunks")
        chunked_dem = chunked_dem.compute()
        logging.debug("Chunked DEM computed")

        return chunked_dem

    def _add_lidar_no_chunking(
        self,
        lidar_dataset_info: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        options: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create a roughness layer with estimates where there is LiDAR from a single
        LiDAR file with no chunking."""

        lidar_file = lidar_dataset_info["file_paths"][0]
        source_crs = lidar_dataset_info["crs"]
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

        # Get the locations to rasterise
        dim_x = self._dem.x.data
        dim_y = self._dem.y.data

        # Estimate roughness over the region
        raster_values = self._roughness_over_tile(
            dim_x=dim_x,
            dim_y=dim_y,
            tile_points=tile_array,
            xy_ground=self._dem.z.data.flatten(),
            options=options,
        )
        roughness = raster_values.reshape((len(dim_y), len(dim_x)))

        # Create xarray
        dem = self._add_roughness_to_data_set(
            x=dim_x,
            y=dim_y,
            roughnesses=[roughness],
            metadata=metadata,
            region_to_rasterise=region_to_rasterise,
        )

        return dem

    def _roughness_over_tile(
        self,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        xy_ground: numpy.ndarray,
        tile_points: numpy.ndarray,
        options: dict,
    ):
        """Rasterise all points within a tile to give roughness. Does not require a tile
        index file."""

        # keep only the specified classifications (should be ground cover)
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

        if len(tile_points) == 0:
            logging.warning(
                "In RoughnessDem._roughness_over_tile the tile has no data and is being"
                " ignored."
            )
            return
        # Get the locations overwhich to estimate roughness
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()

        # Perform roughness estimation within the extents of the tile
        z_flat = roughness_from_points(
            point_cloud=tile_points,
            xy_out=xy_out,
            xy_ground=xy_ground,
            options=options,
        )
        grid_z = z_flat.reshape(grid_x.shape)

        return grid_z

    def _add_roughness_to_data_set(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        roughnesses: list,
        metadata: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
    ) -> xarray.Dataset:
        """A function to add zo to the existing DEM as a new variable.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            roughnesses
                A list of roughnesses over the x, and y coordiantes for each dataset.
        """

        # Create a DataArray of zo
        zos = []
        for zo_array in roughnesses:
            zo = xarray.DataArray(
                data=zo_array,
                dims=["y", "x"],
                coords=dict(x=(["x"], x), y=(["y"], y)),
                attrs=dict(
                    long_name="ground roughness",
                    units="",
                ),
            )
            zo.rio.write_crs(self.catchment_geometry.crs["horizontal"], inplace=True)
            zo.rio.write_transform(inplace=True)
            zo.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
            zos.append(zo)
        if len(zos) == 1:
            zo = zos[0]
        else:
            zo = rioxarray.merge.merge_arrays(zos, method="first")
        # Resize zo to share the same dimensions at the DEM
        self._dem["zo"] = zo.sel(x=self._dem.x, y=self._dem.y, method="nearest")
        # Ensure no negative roughnesses
        self._dem.zo.data[self._dem.zo.data < 0] = self.ROUGHNESS_DEFAULTS["minimum"]

        # ensure the expected CF conventions are followed
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

        """# Ensure roughness is NaN where there is no LiDAR information
        dem.zo.data = dem.zo.rio.clip(region_to_rasterise.geometry, drop=False)"""
        return self._dem


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

    # Specify reprojection - if a source_crs is specified use this to define the
    # 'in_srs'
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


def roughness_from_points(
    point_cloud: numpy.ndarray,
    xy_out: numpy.ndarray,
    xy_ground: numpy.ndarray,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
) -> numpy.ndarray:
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    assert len(xy_out) == len(xy_ground), (
        f"xy_out and xy_ground arrays differ in length: {len(xy_out)} vs "
        "{len(xy_ground)}"
    )

    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud["X"]
    xy_in[:, 1] = point_cloud["Y"]

    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=options["radius"], eps=eps)
    z_out = numpy.zeros(len(xy_out), dtype=options["raster_type"])

    for i, (near_indicies, ground) in enumerate(zip(tree_index_list, xy_ground)):
        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            height = numpy.mean(point_cloud["Z"][near_indicies]) - ground
            std = numpy.std(point_cloud["Z"][near_indicies])

            # if building/plantation - set value based on classification
            # Emperical relationship between mean and std above the ground
            z_out[i] = max(std / 3, height / 6) / 10
    return z_out


def elevation_from_points(
    point_cloud: numpy.ndarray,
    xy_out,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
) -> numpy.ndarray:
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

    for i, (near_indices, point) in enumerate(zip(tree_index_list, xy_out)):
        if len(near_indices) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            if options["method"] == "mean":
                z_out[i] = numpy.mean(point_cloud["Z"][near_indices])
            elif options["method"] == "median":
                z_out[i] = numpy.median(point_cloud["Z"][near_indices])
            elif options["method"] == "idw":
                z_out[i] = calculate_idw(
                    near_indices=near_indices,
                    point=point,
                    tree=tree,
                    point_cloud=point_cloud,
                )
            elif options["method"] == "linear":
                z_out[i] = calculate_linear(
                    near_indices=near_indices,
                    point=point,
                    tree=tree,
                    point_cloud=point_cloud,
                )
            elif options["method"] == "min":
                z_out[i] = numpy.min(point_cloud["Z"][near_indices])
            elif options["method"] == "max":
                z_out[i] = numpy.max(point_cloud["Z"][near_indices])
            elif options["method"] == "std":
                z_out[i] = numpy.std(point_cloud["Z"][near_indices])
            elif options["method"] == "count":
                z_out[i] = numpy.len(point_cloud["Z"][near_indices])
            else:
                assert (
                    False
                ), f"An invalid lidar_interpolation_method of '{options['method']}' was"
                " provided"
    return z_out


def calculate_idw(
    near_indices: list,
    point: numpy.ndarray,
    tree: scipy.spatial.KDTree,
    point_cloud: numpy.ndarray,
    smoothing: float = 0,
    power: int = 2,
):
    """Calculate the IDW mean of the 'near_indices' points. This implementation is based
    on the scipy.spatial.KDTree"""

    distance_vectors = point - tree.data[near_indices]
    smoothed_distances = numpy.sqrt(
        ((distance_vectors**2).sum(axis=1) + smoothing**2)
    )
    if smoothed_distances.min() == 0:  # in the case of an exact match
        idw = point_cloud["Z"][tree.query(point, k=1)[1]]
    else:
        idw = (point_cloud["Z"][near_indices] / (smoothed_distances**power)).sum(
            axis=0
        ) / (1 / (smoothed_distances**power)).sum(axis=0)
    return idw


def calculate_linear(
    near_indices: list,
    point: numpy.ndarray,
    tree: scipy.spatial.KDTree,
    point_cloud: numpy.ndarray,
):
    """Calculate linear interpolation of the 'near_indices' points. Take the straight
    mean if the points are co-linear or too few for linear interpolation."""

    if len(near_indices) > 3:  # There are enough points for a linear interpolation
        linear = scipy.interpolate.griddata(
            points=tree.data[near_indices],
            values=point_cloud["Z"][near_indices],
            xi=point,
            method="linear",
        )[0]
    elif len(near_indices) == 1:
        linear = point_cloud["Z"][near_indices][0]
    else:
        linear = numpy.nan
    # NaN will have occured if colinear points - replace with straight mean
    if numpy.isnan(linear) and len(near_indices) > 0:
        linear = numpy.mean(point_cloud["Z"][near_indices])
    return linear


def select_lidar_files(
    tile_index_extents: geopandas.GeoDataFrame,
    tile_index_name_column: str,
    chunk_region_to_tile: geopandas.GeoDataFrame,
    lidar_files_map: typing.Dict[str, pathlib.Path],
) -> typing.List[pathlib.Path]:
    if chunk_region_to_tile.empty:
        return []
    # clip the tile indices to only include those within the chunk region
    chunk_tile_index_extents = geopandas.sjoin(
        chunk_region_to_tile, tile_index_extents.drop(columns=["index_right"])
    )

    # get the LiDAR file with the tile_index_name
    filtered_lidar_files = [
        lidar_files_map[tile_index_name]
        for tile_index_name in chunk_tile_index_extents[tile_index_name_column]
    ]

    return filtered_lidar_files


def load_tiles_in_chunk(
    lidar_files: typing.List[pathlib.Path],
    source_crs: dict,
    chunk_region_to_tile: geopandas.GeoDataFrame,
    catchment_geometry: geometry.CatchmentGeometry,
):
    """Read in all LiDAR files within the chunked region - clipped to within the region
    within which to rasterise."""

    logging.info(f"Reading all {len(lidar_files)} files in chunk.")

    # Initialise LiDAR points
    lidar_points = []

    # Cycle through each file loading it in an adding it to a numpy array
    for lidar_file in lidar_files:
        logging.info(f"\t Loading in file {lidar_file}")

        # read in the LiDAR file
        pdal_pipeline = read_file_with_pdal(
            lidar_file=lidar_file,
            region_to_tile=chunk_region_to_tile,
            source_crs=source_crs,
            catchment_geometry=catchment_geometry,
            get_extents=False,
        )
        lidar_points.append(pdal_pipeline.arrays[0])
    if len(lidar_points) > 0:
        lidar_points = numpy.concatenate(lidar_points)
    return lidar_points


def roughness_over_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_points: numpy.ndarray,
    xy_ground: numpy.ndarray,
    options: dict,
) -> numpy.ndarray:
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
            "In dem.roughness_over_chunk the latest chunk has no data and is being "
            "ignored."
        )
        return grid_z
    # keep only the specified classifications (should be ground cover)
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
    # Perform the point cloud roughness estimation method over chunk
    z_flat = roughness_from_points(
        point_cloud=tile_points,
        xy_out=xy_out,
        xy_ground=xy_ground,
        options=options,
    )
    grid_z = z_flat.reshape(grid_x.shape)

    return grid_z


def elevation_over_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_points: numpy.ndarray,
    options: dict,
) -> numpy.ndarray:
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
            "In dem.elevation_over_chunk the latest chunk has no data and is being "
            "ignored."
        )
        return grid_z
    # keep only the specified classifications (should be ground / water)
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
    # Perform the specified averaging method over the dense DEM within the extents of
    # this point cloud tile
    z_flat = elevation_from_points(
        point_cloud=tile_points, xy_out=xy_out, options=options
    )
    grid_z = z_flat.reshape(grid_x.shape)

    return grid_z


""" Wrap the `roughness_over_chunk` routine in dask.delayed """
delayed_roughness_over_chunk = dask.delayed(roughness_over_chunk)

""" Wrap the `rasterise_chunk` routine in dask.delayed """
delayed_elevation_over_chunk = dask.delayed(elevation_over_chunk)


""" Wrap the `load_tiles_in_chunk` routine in dask.delayed """
delayed_load_tiles_in_chunk = dask.delayed(load_tiles_in_chunk)
