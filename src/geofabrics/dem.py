# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
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
    """ A class to manage the reference DEM in a catchment context

    Specifically, clip within the catchment land and foreshore. There is the option to clip outside any LiDAR using the
    optional 'exclusion_extent' input.

    If set_foreshore is True all positive DEM values in the foreshore are set to zero. """

    def __init__(self, dem_file, catchment_geometry: geometry.CatchmentGeometry, set_foreshore: bool = True,
                 exclusion_extent: geopandas.GeoDataFrame = None):
        """ Load in the reference DEM, clip and extract points """

        self.catchment_geometry = catchment_geometry
        self.set_foreshore = set_foreshore
        with rioxarray.rioxarray.open_rasterio(dem_file, masked=True) as self._dem:
            self._dem.load()

        self._extents = None
        self._points = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """ Set DEM CRS and trim the DEM to size """

        self._dem.rio.set_crs(self.catchment_geometry.crs['horizontal'])

        if exclusion_extent is not None:
            exclusion_extent = geopandas.clip(exclusion_extent, self.catchment_geometry.land_and_foreshore)
            self._extents = geopandas.overlay(self.catchment_geometry.land_and_foreshore,
                                              exclusion_extent, how="difference")
        else:
            self._extents = self.catchment_geometry.land_and_foreshore

        self._dem = self._dem.rio.clip(self._extents.geometry)
        self._extract_points()

    def _extract_points(self):
        """ Create a points list from the DEM """

        if self.catchment_geometry.land.area.sum() > 0:
            land_dem = self._dem.rio.clip(self.catchment_geometry.land.geometry)
            # get reference DEM points on land
            land_flat_z = land_dem.data[0].flatten()
            land_mask_z = ~numpy.isnan(land_flat_z)
            land_grid_x, land_grid_y = numpy.meshgrid(land_dem.x, land_dem.y)

            land_x = land_grid_x.flatten()[land_mask_z]
            land_y = land_grid_y.flatten()[land_mask_z]
            land_z = land_flat_z[land_mask_z]
        else:  # In the case that there is no DEM outside LiDAR/exclusion_extent and on land
            land_x = []
            land_y = []
            land_z = []

        if self.catchment_geometry.foreshore.area.sum() > 0:
            foreshore_dem = self._dem.rio.clip(self.catchment_geometry.foreshore.geometry)

            # get reference DEM points on the foreshore
            if self.set_foreshore:
                foreshore_dem.data[0][foreshore_dem.data[0] > 0] = 0
            foreshore_flat_z = foreshore_dem.data[0].flatten()
            foreshore_mask_z = ~numpy.isnan(foreshore_flat_z)
            foreshore_grid_x, foreshore_grid_y = numpy.meshgrid(foreshore_dem.x, foreshore_dem.y)

            foreshore_x = foreshore_grid_x.flatten()[foreshore_mask_z]
            foreshore_y = foreshore_grid_y.flatten()[foreshore_mask_z]
            foreshore_z = foreshore_flat_z[foreshore_mask_z]
        else:  # In the case that there is no DEM outside LiDAR/exclusion_extent and on foreshore
            foreshore_x = []
            foreshore_y = []
            foreshore_z = []

        assert len(land_x) + len(foreshore_x) > 0, "The reference DEM has no values on the land or foreshore"

        # combine in an single array
        self._points = numpy.empty([len(land_x) + len(foreshore_x)],
                                   dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])
        self._points['X'][:len(land_x)] = land_x
        self._points['Y'][:len(land_x)] = land_y
        self._points['Z'][:len(land_x)] = land_z

        self._points['X'][len(land_x):] = foreshore_x
        self._points['Y'][len(land_x):] = foreshore_y
        self._points['Z'][len(land_x):] = foreshore_z

    @property
    def points(self) -> numpy.ndarray:
        """ The reference DEM points after any extent or foreshore value
        filtering. """

        return self._points

    @property
    def extents(self) -> geopandas.GeoDataFrame:
        """ The extents for the reference DEM """

        return self._extents


class DenseDem(abc.ABC):
    """ An abstract class class to manage the dense DEM in a catchment context.

    The dense DEM is made up of a dense DEM that is loaded in, and an offshore DEM that is interpolated from bathymetry
    contours offshore and outside all LiDAR tiles.
    Logical controlling behaviour:
        * interpolate_missing_values - If True any missing values at the end of the rasterisation process will be
          populated using nearest neighbour interpolation.

    """

    DENSE_BINNING = "idw"
    CACHE_SIZE = 10000  # The max number of points to create the offshore RBF and to evaluate in the RBF at one time

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, extents: geopandas.GeoDataFrame,
                 dense_dem: xarray.core.dataarray.DataArray, interpolate_missing_values: bool):
        """ Setup base DEM to add future tiles too """

        self.catchment_geometry = catchment_geometry
        self._dense_dem = dense_dem
        self._extents = extents

        self.interpolate_missing_values = interpolate_missing_values

        self._offshore_dem = None

        self._dem = None

    @property
    def extents(self):
        """ The combined extents for all added LiDAR tiles """

        if self._extents is None:
            logging.warning("Warning in DenseDem.extents: No tiles with extents have been added yet")

        return self._extents

    @property
    def dense_dem(self):
        """ Return the dense DEM from tiles and any interpolated offshore values """
        return self._dense_dem

    @property
    def dem(self):
        """ Return the combined DEM from tiles and any interpolated offshore values """

        if self._dem is None:
            if self._offshore_dem is None:
                self._dem = self._dense_dem
            else:
                # method='first' or 'last'; use method='first' as `DenseDemFromFiles._dense_dem` clipped to extents
                self._dem = rioxarray.merge.merge_arrays([self._dense_dem, self._offshore_dem], method='first')

        # Ensure valid name and increasing dimension indexing for the dem
        self._dem = self._dem.rename(self.DENSE_BINNING)
        if self.interpolate_missing_values:
            self._dem = self._dem.rio.interpolate_na(method='nearest')  # other methods are 'linear' and 'cubic'
        self._dem = self._dem.rio.clip(self.catchment_geometry.catchment.geometry)
        self._dem = self._ensure_positive_indexing(self._dem)  # Some programs require positively increasing indices
        return self._dem

    @staticmethod
    def _ensure_positive_indexing(dem: xarray.core.dataarray.DataArray) -> xarray.core.dataarray.DataArray:
        """ A routine to check an xarray has positive dimension indexing and to reindex if needed. """

        x = dem.x
        y = dem.y
        if x[0] > x[-1]:
            x = x[::-1]
        if y[0] > y[-1]:
            y = y[::-1]
        dem = dem.reindex(x=x, y=y)
        return dem

    def _sample_offshore_edge(self, resolution) -> numpy.ndarray:
        """ Return the pixel values of the offshore edge to be used for offshore interpolation """

        assert resolution >= self.catchment_geometry.resolution, "_sample_offshore_edge only supports downsampling" + \
            f" and not  up-samping. The requested sampling resolution of {resolution} must be equal to or larger than" + \
            f" the catchment resolution of {self.catchment_geometry.resolution}"

        offshore_dense_data_edge = self.catchment_geometry.offshore_dense_data_edge(self._extents)
        offshore_edge_dem = self._dense_dem.rio.clip(offshore_dense_data_edge.geometry)

        # If the sampling resolution is coaser than the catchment_geometry resolution resample the DEM
        if resolution > self.catchment_geometry.resolution:
            x = numpy.arange(offshore_edge_dem.x.min(), offshore_edge_dem.x.max() + resolution / 2, resolution)
            y = numpy.arange(offshore_edge_dem.y.min(), offshore_edge_dem.y.max() + resolution / 2, resolution)
            offshore_edge_dem = offshore_edge_dem.interp(x=x, y=y, method="nearest")
            offshore_edge_dem = offshore_edge_dem.rio.clip(offshore_dense_data_edge.geometry)  # Reclip to inbounds

        offshore_grid_x, offshore_grid_y = numpy.meshgrid(offshore_edge_dem.x, offshore_edge_dem.y)
        offshore_flat_z = offshore_edge_dem.data[0].flatten()
        offshore_mask_z = ~numpy.isnan(offshore_flat_z)

        offshore_edge = numpy.empty([offshore_mask_z.sum().sum()],
                                    dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])

        offshore_edge['X'] = offshore_grid_x.flatten()[offshore_mask_z]
        offshore_edge['Y'] = offshore_grid_y.flatten()[offshore_mask_z]
        offshore_edge['Z'] = offshore_flat_z[offshore_mask_z]

        return offshore_edge

    def interpolate_offshore(self, bathy_contours):
        """ Performs interpolation offshore outside LiDAR extents using the SciPy RBF function. """

        offshore_edge_points = self._sample_offshore_edge(self.catchment_geometry.resolution)
        bathy_points = bathy_contours.sample_contours(self.catchment_geometry.resolution)
        offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])

        # Resample at a lower resolution if too many offshore points
        if len(offshore_points) > self.CACHE_SIZE:
            reduced_resolution = self.catchment_geometry.resolution * len(offshore_points) / self.CACHE_SIZE
            logging.info("Reducing the number of 'offshore_points' used to create the RBF function by increasing the "
                         f"resolution from {self.catchment_geometry.resolution} to {reduced_resolution}")
            offshore_edge_points = self._sample_offshore_edge(reduced_resolution)
            bathy_points = bathy_contours.sample_contours(reduced_resolution)
            offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])

        # Set up the interpolation function
        logging.info("Creating offshore interpolant")
        rbf_function = scipy.interpolate.Rbf(offshore_points['X'], offshore_points['Y'], offshore_points['Z'],
                                             function='linear')

        # Setup the empty offshore area ready for interpolation
        offshore_no_dense_data = self.catchment_geometry.offshore_no_dense_data(self._extents)
        self._offshore_dem = self._dense_dem.rio.clip(self.catchment_geometry.offshore.geometry)
        self._offshore_dem.data[0] = 0  # set all to zero then clip out dense region where we don't need to interpolate
        self._offshore_dem = self._offshore_dem.rio.clip(offshore_no_dense_data.geometry)

        grid_x, grid_y = numpy.meshgrid(self._offshore_dem.x, self._offshore_dem.y)
        flat_z = self._offshore_dem.data[0].flatten()
        mask_z = ~numpy.isnan(flat_z)

        flat_x_masked = grid_x.flatten()[mask_z]
        flat_y_masked = grid_y.flatten()[mask_z]
        flat_z_masked = flat_z[mask_z]

        # Tile offshore area - this limits the maximum memory required at any one time
        number_offshore_tiles = math.ceil(len(flat_x_masked)/self.CACHE_SIZE)
        for i in range(number_offshore_tiles):
            logging.info(f"Offshore intepolant tile {i+1} of {number_offshore_tiles}")
            start_index = int(i*self.CACHE_SIZE)
            end_index = int((i+1)*self.CACHE_SIZE) if i + 1 != number_offshore_tiles else len(flat_x_masked)

            flat_z_masked[start_index:end_index] = rbf_function(flat_x_masked[start_index:end_index],
                                                                flat_y_masked[start_index:end_index])
        flat_z[mask_z] = flat_z_masked
        self._offshore_dem.data[0] = flat_z.reshape(self._offshore_dem.data[0].shape)

        # Ensure the DEM will be recalculated to include the interpolated offshore region
        self._dem = None


class DenseDemFromFiles(DenseDem):
    """ A class to manage loading in an already created and saved dense DEM that has yet to have an offshore DEM
    associated with it.
    Logic controlling behaviour:
        * interpolate_missing_values - If True any missing values at the end of the rasterisation process will be
          populated using nearest neighbour interpolation.
    """

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry,
                 dense_dem_path: typing.Union[str, pathlib.Path], extents_path: typing.Union[str, pathlib.Path],
                 interpolate_missing_values: bool = True):
        """ Load in the extents and dense DEM. Ensure the dense DEM is clipped within the extents """

        extents = geopandas.read_file(pathlib.Path(extents_path))

        # Read in the dense DEM raster - and free up file by performing a deep copy.
        with rioxarray.rioxarray.open_rasterio(pathlib.Path(dense_dem_path), masked=True) as dense_dem:
            dense_dem.load()
        dense_dem = dense_dem.copy(deep=True)  # Deep copy is required to ensure the opened file is properly unlocked
        dense_dem.rio.set_crs(catchment_geometry.crs['horizontal'])

        # Ensure all values outside the exents are nan as that defines the dense extents
        dense_dem_inside_extents = dense_dem.rio.clip(extents.geometry)
        dense_dem.data[0] = numpy.nan
        dense_dem = rioxarray.merge.merge_arrays([dense_dem, dense_dem_inside_extents], method='first')

        # Setup the DenseDem class
        super(DenseDemFromFiles, self).__init__(catchment_geometry=catchment_geometry, dense_dem=dense_dem,
                                                extents=extents, interpolate_missing_values=interpolate_missing_values)


class DenseDemFromTiles(DenseDem):
    """ A class to manage the population of the DenseDem's dense_dem from LiDAR tiles, and/or a reference DEM.

    The dense DEM is made up of tiles created from dense point data - Either LiDAR point clouds, or a reference DEM

    DenseDemFromTiles logic can be controlled by the constructor inputs:
        * drop_offshore_lidar - If True only keep LiDAR values within the foreshore and land regions defined by
          the catchment_geometry. If False keep all LiDAR values.
        * interpolate_missing_values - If True any missing values at the end of the rasterisation process will be
          populated using nearest neighbour interpolation.
        * idw_power - the power to apply when performing IDW
        * idw_radius - the radius to apply IDW over
    """

    LAS_GROUND = 2  # As specified in the LAS/LAZ format

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, idw_power: int, idw_radius: float,
                 drop_offshore_lidar: bool = True, interpolate_missing_values: bool = True):
        """ Setup base DEM to add future tiles too """

        self.drop_offshore_lidar = drop_offshore_lidar

        self.raster_type = numpy.float64

        self.idw_power = idw_power
        self.idw_radius = idw_radius

        super(DenseDemFromTiles, self).__init__(catchment_geometry=catchment_geometry, dense_dem=None,
                                                extents=None, interpolate_missing_values=interpolate_missing_values)

    def _set_up_chunks(self, chunk_size: int) -> (list, list):
        """ Define the chunked coordinates to cover the catchment """

        catchment_bounds = self.catchment_geometry.catchment.loc[0].geometry.bounds
        resolution = self.catchment_geometry.resolution

        # Determine the number of chunks
        if chunk_size is None or chunk_size <= 0:
            # Determine x and y coordinates for no chunks
            dim_x = [numpy.arange(catchment_bounds[0] + resolution / 2, catchment_bounds[2],
                                  resolution, dtype=self.raster_type)]
            dim_y = [numpy.arange(catchment_bounds[3] - resolution / 2, catchment_bounds[1],
                                  -resolution, dtype=self.raster_type)]
        else:
            n_chunks_x = int(numpy.ceil((catchment_bounds[2] - catchment_bounds[0]) / (chunk_size * resolution)))
            n_chunks_y = int(numpy.ceil((catchment_bounds[3] - catchment_bounds[1]) / (chunk_size * resolution)))

            # Determine x and y coordinates rounded up to the nearest chunk
            dim_x = [numpy.arange(catchment_bounds[0] + resolution / 2 + i * chunk_size * resolution,
                                  catchment_bounds[0] + resolution / 2 + (i + 1) * chunk_size * resolution,
                                  resolution, dtype=self.raster_type) for i in range(n_chunks_x)]
            dim_y = [numpy.arange(catchment_bounds[3] - resolution / 2 - i * chunk_size * resolution,
                                  catchment_bounds[3] - resolution / 2 - (i + 1) * chunk_size * resolution,
                                  -resolution, dtype=self.raster_type) for i in range(n_chunks_y)]

        return dim_x, dim_y

    def _calculate_dense_extents(self):
        """ Calculate the extents of the current dense DEM. Remove holes as these can cause self intersection
        warnings. """

        dense_extents = [shapely.geometry.shape(polygon[0]) for polygon in
                         rasterio.features.shapes(numpy.uint8(numpy.isnan(self._dense_dem.data) == False))
                         if polygon[1] == 1.0]
        dense_extents = shapely.ops.unary_union(dense_extents)

        # Remove any internal holes for select types as these may cause self intersection errors
        if type(dense_extents) is shapely.geometry.Polygon:
            dense_extents = shapely.geometry.Polygon(dense_extents.exterior)
        elif type(dense_extents) is shapely.geometry.MultiPolygon:
            dense_extents = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(polygon.exterior)
                                                           for polygon in dense_extents])

        # Convert into a Geopandas dataframe
        dense_extents = geopandas.GeoDataFrame({'geometry': [dense_extents]},
                                               crs=self.catchment_geometry.crs['horizontal'])

        # Apply a transform so in the same space as the dense DEM - buffer(0) to reduce self intersection warnings
        dense_dem_affine = self._dense_dem.rio.transform()
        dense_extents = dense_extents.affine_transform([dense_dem_affine.a, dense_dem_affine.b,
                                                        dense_dem_affine.d, dense_dem_affine.e,
                                                        dense_dem_affine.xoff, dense_dem_affine.yoff]).buffer(0)

        # And make our GeoSeries into a GeoDataFrame
        dense_extents = geopandas.GeoDataFrame(geometry=dense_extents)

        return dense_extents

    def _tile_index_column_name(self, tile_index_file: typing.Union[str, pathlib.Path] = None):
        """ Read in tile index file and determine the column name of the tile geometries """
        # Check to see if a extents file was added
        tile_index_extents = geopandas.read_file(tile_index_file) if tile_index_file is not None else None
        tile_index_name_column = None

        # If there is a tile_index_file - remove tiles outside the catchment & get the 'file name' column
        if tile_index_extents is not None:
            tile_index_extents = tile_index_extents.to_crs(self.catchment_geometry.crs['horizontal'])
            tile_index_extents = geopandas.sjoin(tile_index_extents, self.catchment_geometry.catchment)
            tile_index_extents = tile_index_extents.reset_index(drop=True)

            column_names = tile_index_extents.columns
            tile_index_name_column = column_names[["filename" == name.lower() or "file_name" == name.lower()
                                                   for name in column_names]][0]
        return tile_index_extents, tile_index_name_column

    def _rasterise_tile(self, dim_x: numpy.ndarray, dim_y: numpy.ndarray, tile_points: numpy.ndarray,
                        keep_only_ground_lidar: bool):
        """ Rasterise all points within a tile. """

        # filter out only ground points for idw ground calculations
        if keep_only_ground_lidar:
            tile_points = tile_points[tile_points['Classification'] == self.LAS_GROUND]

        if len(tile_points) == 0:
            logging.warning("In DenseDem._rasterise_tile the tile has no data and is being ignored.")
            return

        # Get the indicies overwhich to perform IDW
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate([[grid_x.flatten()], [grid_y.flatten()]], axis=0).transpose()

        # Perform IDW over the dense DEM within the extents of this point cloud tile
        z_idw = rasterise_with_idw(point_cloud=tile_points, xy_out=xy_out, idw_radius=self.idw_radius,
                                   idw_power=self.idw_power, raster_type=self.raster_type, smoothing=0,
                                   eps=0, leaf_size=10)
        grid_z = z_idw.reshape(grid_x.shape)

        # TODO - add roughness calculation

        return grid_z

    def add_lidar(self, lidar_files: typing.List[typing.Union[str, pathlib.Path]],
                  tile_index_file: typing.Union[str, pathlib.Path], chunk_size: int,
                  source_crs: dict = None, keep_only_ground_lidar: bool = True, drop_offshore_lidar: bool = True):
        """ Read in all LiDAR files and use to create a dense DEM.

            source_crs - specify if the CRS encoded in the LiDAR files are incorrect/only partially defined
                (i.e. missing vertical CRS) and need to be overwritten.
            drop_offshore_lidar - if True, trim any LiDAR values that are offshore as specified by the
                catchment_geometry
            keep_only_ground_lidar - if True, only keep LiDAR values that are coded '2' of ground
            tile_index_file - must exist if there are many LiDAR files. This is used to determine chunking. """

        if source_crs is not None:
            assert 'horizontal' in source_crs, "The horizontal component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"
            assert 'vertical' in source_crs, "The vertical component of the source CRS is not specified. " + \
                f"Both horizontal and vertical CRS need to be defined. The source_crs specified is: {self.source_crs}"

        if drop_offshore_lidar:
            region_to_rasterise = self.catchment_geometry.land_and_foreshore
        else:
            region_to_rasterise = self.catchment_geometry.catchment

        # Determine if adding a single file or tiles
        if len(lidar_files) == 1:  # If one file it's ok if there is no tile_index
            self._dense_dem = self._add_file(lidar_file=lidar_files[0], region_to_rasterise=region_to_rasterise,
                                             source_crs=source_crs, keep_only_ground_lidar=keep_only_ground_lidar)
        else:
            assert tile_index_file is not None, "A tile index file is required for multiple tile files added together"
            assert chunk_size > 0 and chunk_size is not None, "The chunk size should be set when reading in tiled LiDAR " \
                "files. Ideally it should include as many tiles can easily be read in by on core. You will have to equate" \
                " The tile extents with chunk size by extents / resolution. "
            self._dense_dem = self._add_tiled_lidar_chunked(lidar_files=lidar_files, tile_index_file=tile_index_file,
                                                            source_crs=source_crs,
                                                            keep_only_ground_lidar=keep_only_ground_lidar,
                                                            region_to_rasterise=region_to_rasterise,
                                                            chunk_size=chunk_size)

        # Set any values outside the region_to_rasterise to NaN
        self._dense_dem = self._dense_dem.rio.clip(region_to_rasterise.geometry, drop=False)

        # Create a polygon defining the region where there are dense DEM values
        self._extents = self._calculate_dense_extents()

        # Ensure the dem will be recalculated as another tile has been added
        self._dem = None

    def _add_tiled_lidar_chunked(self, lidar_files: typing.List[typing.Union[str, pathlib.Path]],
                                 tile_index_file: typing.Union[str, pathlib.Path], source_crs: dict,
                                 keep_only_ground_lidar: bool, region_to_rasterise: geopandas.GeoDataFrame,
                                 chunk_size: int) -> xarray.DataArray:
        """ Create a dense DEM from a set of tiled LiDAR files. Read these in over non-overlapping chunks and
        then combine """

        # Remove all tiles entirely outside the region to raserise
        tile_index_extents, tile_index_name_column = self._tile_index_column_name(tile_index_file)

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
                chunk_geometry = geopandas.GeoDataFrame(
                     {'geometry': [shapely.geometry.Polygon(
                         [(numpy.min(dim_x) - self.idw_radius, numpy.min(dim_y) - self.idw_radius),
                          (numpy.max(dim_x) + self.idw_radius, numpy.min(dim_y) - self.idw_radius),
                          (numpy.max(dim_x) + self.idw_radius, numpy.max(dim_y) + self.idw_radius),
                          (numpy.min(dim_x) - self.idw_radius, numpy.max(dim_y) + self.idw_radius)])]},
                     crs=self.catchment_geometry.crs['horizontal'])

                # Ensure edge pixels will have a full set of values to perform IDW over
                chunk_region_to_tile = geopandas.GeoDataFrame(
                     geometry=region_to_rasterise.buffer(self.idw_radius).clip(chunk_geometry))

                # Load in files and rasterise
                chunk_points = delayed_load_tiles_in_chunk(dim_x=dim_x,
                                                           dim_y=dim_y,
                                                           tile_index_extents=tile_index_extents,
                                                           tile_index_name_column=tile_index_name_column,
                                                           lidar_files=lidar_files,
                                                           source_crs=source_crs,
                                                           chunk_region_to_tile=chunk_region_to_tile,
                                                           catchment_geometry=self.catchment_geometry)
                delayed_chunked_x.append(dask.array.from_delayed(
                    delayed_rasterise_chunk(dim_x=dim_x,
                                            dim_y=dim_y,
                                            tile_points=chunk_points,
                                            keep_only_ground_lidar=keep_only_ground_lidar,
                                            ground_code=self.LAS_GROUND,
                                            idw_radius=self.idw_radius,
                                            idw_power=self.idw_power,
                                            raster_type=self.raster_type),
                    shape=(chunk_size, chunk_size), dtype=numpy.float32))
            delayed_chunked_matrix.append(delayed_chunked_x)
        chunked_dem = xarray.DataArray(dask.array.block([delayed_chunked_matrix]),
                                       coords={'band': [1], 'y': numpy.concatenate(chunked_dim_y),
                                               'x': numpy.concatenate(chunked_dim_x)},
                                       dims=['band', 'y', 'x'],
                                       attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'idw'})
        chunked_dem.rio.write_crs(self.catchment_geometry.crs['horizontal'], inplace=True)
        chunked_dem.name = 'z'
        chunked_dem = chunked_dem.rio.write_nodata(numpy.nan)
        logging.info("Computing chunks")
        chunked_dem = chunked_dem.compute()

        # Clip result to within the catchment - removing NaN filled chunked areas outside the catchment
        logging.debug("Chunked DEM computed and ready to be cut")
        dense_dem = chunked_dem.rio.clip(self.catchment_geometry.catchment.geometry)
        return dense_dem

    def _add_file(self, lidar_file: typing.Union[str, pathlib.Path], region_to_rasterise: geopandas.GeoDataFrame,
                  source_crs: dict = None, keep_only_ground_lidar: bool = True) -> xarray.DataArray:
        """ Create the dense DEM region from a single LiDAR file. """

        logging.info(f"On LiDAR tile 1 of 1: {lidar_file}")

        # Use PDAL to load in file
        pdal_pipeline = read_file_with_pdal(lidar_file, source_crs=source_crs, region_to_tile=region_to_rasterise,
                                            get_extents=True, catchment_geometry=self.catchment_geometry)

        # Load LiDAR points from pipeline
        tile_array = pdal_pipeline.arrays[0]

        # Get the raster indicies
        dim_x, dim_y = self._set_up_chunks(chunk_size=None)
        dim_x = dim_x[0]
        dim_y = dim_y[0]

        raster_values = self._rasterise_tile(dim_x=dim_x, dim_y=dim_y, tile_points=tile_array,
                                             keep_only_ground_lidar=keep_only_ground_lidar)

        # Create xarray
        dense_dem = xarray.DataArray(raster_values.reshape((1, len(dim_y), len(dim_x))),
                                     coords={'band': [1], 'y': dim_y, 'x': dim_x}, dims=['band', 'y', 'x'],
                                     attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'idw'})
        dense_dem.rio.write_crs(self.catchment_geometry.crs['horizontal'], inplace=True)
        dense_dem.name = 'z'
        dense_dem = dense_dem.rio.write_nodata(numpy.nan)

        return dense_dem

    def add_reference_dem(self, tile_points: numpy.ndarray, tile_extent: geopandas.GeoDataFrame):
        """ Update gaps in dense DEM from areas with no LiDAR with the reference DEM. """

        # Areas not covered by LiDAR values
        mask = numpy.isnan(self._dense_dem.data[0])

        if len(tile_points) == 0:
            logging.warning("DenseDem.add_tile: the latest reference DEM has no data and is being ignored.")
            return
        elif mask.sum() == 0:
            logging.warning("DenseDem.add_tile: LiDAR covers all raster values so the reference DEM is being ignored.")
            return

        # Get the indicies overwhich to perform IDW
        grid_x, grid_y = numpy.meshgrid(self._dense_dem.x, self._dense_dem.y)

        xy_out = numpy.empty((mask.sum(), 2))
        xy_out[:, 0] = grid_x[mask]
        xy_out[:, 1] = grid_y[mask]

        # Perform IDW over the dense DEM within the extents of this point cloud tile
        z_idw = rasterise_with_idw(point_cloud=tile_points, xy_out=xy_out, idw_radius=self.idw_radius,
                                   idw_power=self.idw_power,
                                   raster_type=self.raster_type, smoothing=0, eps=0, leaf_size=10)
        self._dense_dem.data[0, mask] = z_idw

        # Update the dense DEM extents
        self._extents = self._calculate_dense_extents()

        # Ensure the dem will be recalculated as another tile has been added
        self._dem = None


def read_file_with_pdal(lidar_file: typing.Union[str, pathlib.Path], region_to_tile: geopandas.GeoDataFrame,
                        catchment_geometry: geometry.CatchmentGeometry, source_crs: dict = None,
                        get_extents: bool = False):
    """ Read a tile file in using PDAL """

    # Define instructions for loading in LiDAR
    pdal_pipeline_instructions = [{"type":  "readers.las", "filename": str(lidar_file)}]

    # Specify reprojection - if a source_crs is specified use this to define the 'in_srs'
    if source_crs is None:
        pdal_pipeline_instructions.append(
            {"type": "filters.reprojection",
             "out_srs": f"EPSG:{catchment_geometry.crs['horizontal']}+" +
             f"{catchment_geometry.crs['vertical']}"})
    else:
        pdal_pipeline_instructions.append(
            {"type": "filters.reprojection",
             "in_srs": f"EPSG:{source_crs['horizontal']}+{source_crs['vertical']}",
             "out_srs": f"EPSG:{catchment_geometry.crs['horizontal']}+" +
             f"{catchment_geometry.crs['vertical']}"})

    # Add instructions for clip within either the catchment, or the land and foreshore
    pdal_pipeline_instructions.append(
        {"type": "filters.crop", "polygon": str(region_to_tile.loc[0].geometry)})

    # Add instructions for creating a polygon extents of the remaining point cloud
    if get_extents:
        pdal_pipeline_instructions.append({"type": "filters.hexbin"})

    # Load in LiDAR and perform operations
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
    pdal_pipeline.execute()
    return pdal_pipeline


def rasterise_with_idw(point_cloud: numpy.ndarray, xy_out, idw_radius: float, idw_power: int, raster_type,
                       smoothing: float = 0, eps: float = 0, leaf_size: int = 10):
    """ Calculate DEM elevation values at the specified locations using the inverse distance weighing (IDW)
    approach. This implementation is based on the scipy.spatial.KDTree """
    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud['X']
    xy_in[:, 1] = point_cloud['Y']

    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=idw_radius, eps=eps)  # , eps=0.2)
    z_out = numpy.zeros(len(xy_out), dtype=raster_type)

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            distance_vectors = point - tree.data[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # in the case of an exact match
                z_out[i] = point_cloud['Z'][tree.query(point, k=1)[1]]
            else:
                z_out[i] = (point_cloud['Z'][near_indicies] / (smoothed_distances**idw_power)).sum(axis=0) \
                    / (1 / (smoothed_distances**idw_power)).sum(axis=0)

    return z_out


def load_tiles_in_chunk(dim_x: numpy.ndarray, dim_y: numpy.ndarray, tile_index_extents: geopandas.GeoDataFrame,
                        tile_index_name_column: str, lidar_files: typing.List[typing.Union[str, pathlib.Path]],
                        source_crs: dict, chunk_region_to_tile: geopandas.GeoDataFrame,
                        catchment_geometry: geometry.CatchmentGeometry):
    """ Read in all LiDAR files within the chunked region - clipped to within the region within which to rasterise.
    """

    # Clip the tile indices to only include those within the chunk region
    chunk_tile_index_extents = tile_index_extents.drop(columns=['index_right'])
    chunk_tile_index_extents = geopandas.sjoin(chunk_tile_index_extents, chunk_region_to_tile)
    chunk_tile_index_extents = chunk_tile_index_extents.reset_index(drop=True)

    logging.info(f"Reading all {len(chunk_tile_index_extents[tile_index_name_column])} files in chunk.")

    # Initialise LiDAR points
    lidar_points = []

    # Cycle through each file loading it in an adding it to a numpy array
    for tile_index_name in chunk_tile_index_extents[tile_index_name_column]:
        logging.info(f"\t Loading in file {tile_index_name}")
        # get the LiDAR file with the tile_index_name
        lidar_file = [lidar_file for lidar_file in lidar_files if lidar_file.name == tile_index_name]
        assert len(lidar_file) == 1, f"Error no single LiDAR file matches the tile name. {lidar_file}"

        # read in the LiDAR file
        pdal_pipeline = read_file_with_pdal(lidar_file=lidar_file[0], region_to_tile=chunk_region_to_tile,
                                            source_crs=source_crs, catchment_geometry=catchment_geometry,
                                            get_extents=False)
        lidar_points.append(pdal_pipeline.arrays[0])

    if len(lidar_points) > 0:
        lidar_points = numpy.concatenate(lidar_points)
    return lidar_points


def rasterise_chunk(dim_x: numpy.ndarray, dim_y: numpy.ndarray, tile_points: numpy.ndarray, raster_type,
                    keep_only_ground_lidar: bool, ground_code: int, idw_radius: float, idw_power: int):
    """ Rasterise all points within a chunk. """

    # Get the indicies overwhich to perform IDW
    grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
    xy_out = numpy.concatenate([[grid_x.flatten()], [grid_y.flatten()]], axis=0).transpose()
    grid_z = numpy.ones(grid_x.shape) * numpy.nan

    # If no points return an array of NaN
    if len(tile_points) == 0:
        logging.warning("In dem.rasterise_chunk the latest chunk has no data and is being ignored.")
        return grid_z

    # use only ground points for idw ground calculations - note the code works even if for empty input tile_points
    if keep_only_ground_lidar:
        tile_points = tile_points[tile_points['Classification'] == ground_code]

    # Check again - if no points return an array of NaN
    if len(tile_points) == 0:
        return grid_z

    # Perform IDW over the dense DEM within the extents of this point cloud tile
    z_idw = rasterise_with_idw(point_cloud=tile_points, xy_out=xy_out,
                               idw_radius=idw_radius, idw_power=idw_power,
                               smoothing=0, eps=0, leaf_size=10, raster_type=raster_type)
    grid_z = z_idw.reshape(grid_x.shape)

    # TODO - add roughness calculation

    return grid_z


""" Wrap the `rasterise_chunk` routine in dask.delayed """
delayed_rasterise_chunk = dask.delayed(rasterise_chunk)


""" Wrap the `load_tiles_in_chunk` routine in dask.delayed """
delayed_load_tiles_in_chunk = dask.delayed(load_tiles_in_chunk)
