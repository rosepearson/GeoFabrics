# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
import rioxarray
import rioxarray.merge
import xarray
import numpy
import math
import pdal
import json
import typing
import pathlib
import geopandas
import shapely
import abc
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

    """

    DENSE_BINNING = "idw"
    CACHE_SIZE = 10000  # The max number of points to create the offshore RBF and to evaluate in the RBF at one time

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, extents: geopandas.GeoDataFrame,
                 dense_dem: xarray.core.dataarray.DataArray, verbose: bool = True):
        """ Setup base DEM to add future tiles too """

        self.catchment_geometry = catchment_geometry
        self._dense_dem = dense_dem
        self._extents = extents

        self.verbose = verbose

        self._offshore_dem = None

        self._dem = None

    @property
    def extents(self):
        """ The combined extents for all added LiDAR tiles """

        if self.verbose and self._extents is None:
            print("Warning in DenseDem.extents: No tiles with extents have been added yet")

        return self._extents

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
        self._dem = self._dem.rio.interpolate_na()
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
            print("Reducing the number of 'offshore_points' used to create the RBF function by increasing the " +
                  f"resolution from {self.catchment_geometry.resolution} to {reduced_resolution}")
            offshore_edge_points = self._sample_offshore_edge(reduced_resolution)
            bathy_points = bathy_contours.sample_contours(reduced_resolution)
            offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])

        # Set up the interpolation function
        if self.verbose:
            print("Creating offshore interpolant")
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
            if self.verbose:
                print(f"Offshore intepolant tile {i+1} of {number_offshore_tiles}")
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
    """

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry,
                 dense_dem_path: typing.Union[str, pathlib.Path], extents_path: typing.Union[str, pathlib.Path],
                 verbose: bool = True):
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
                                                extents=extents, verbose=verbose)


class DenseDemFromTiles(DenseDem):
    """ A class to manage the population of the DenseDem's dense_dem from LiDAR tiles, and/or a reference DEM.

    The dense DEM is made up of tiles created from dense point data - Either LiDAR point clouds, or a reference DEM

    DenseDemFromTiles logic can be controlled by the constructor inputs:
        * area_to_drop - If '> 0' this defines the size of any holes in the LiDAR coverage to ignore.
        * drop_offshore_lidar - If True only keep LiDAR values within the foreshore and land regions defined by
          the catchment_geometry. If False keep all LiDAR values.
    """

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry,
                 temp_raster_path: typing.Union[str, pathlib.Path], drop_offshore_lidar: bool = True,
                 area_to_drop: float = None, verbose: bool = True):
        """ Setup base DEM to add future tiles too """

        self._temp_dem_file = pathlib.Path(temp_raster_path)

        self.area_to_drop = area_to_drop
        self.drop_offshore_lidar = drop_offshore_lidar

        self.raster_origin = None
        self.raster_size = None
        self.raster_type = numpy.float64

        empty_dem = self._set_up(catchment_geometry)

        super(DenseDemFromTiles, self).__init__(catchment_geometry=catchment_geometry, dense_dem=empty_dem,
                                                extents=None, verbose=verbose)

    def _set_up(self, catchment_geometry):
        """ Create the empty dense DEM to fill and define the raster size and origin """

        catchment_bounds = catchment_geometry.catchment.loc[0].geometry.bounds
        resolution = catchment_geometry.resolution

        # Create an empty xarray to store results in
        dim_x = numpy.arange(catchment_bounds[0] + resolution / 2, catchment_bounds[2], resolution,
                             dtype=self.raster_type)
        dim_y = numpy.arange(catchment_bounds[3] - resolution / 2, catchment_bounds[1], -resolution,
                             dtype=self.raster_type)
        grid_dem_z = numpy.empty((1, len(dim_y), len(dim_x)), dtype=self.raster_type)
        dem = xarray.DataArray(grid_dem_z, coords={'band': [1], 'y': dim_y, 'x': dim_x}, dims=['band', 'y', 'x'],
                               attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'idw'})
        dem.rio.write_crs(catchment_geometry.crs['horizontal'], inplace=True)
        dem.name = 'z'
        dem = dem.rio.write_nodata(numpy.nan)

        # Ensure the DEM is empty - all NaN - and clipped to size
        dem.data[0] = numpy.nan
        dem = dem.rio.clip(catchment_geometry.catchment.geometry)
        return dem

    def _set_up_pdal(self, catchment_geometry):
        """ Create the empty dense DEM to fill and define the raster size and origin """

        catchment_bounds = catchment_geometry.catchment.loc[0].geometry.bounds
        self.raster_origin = [catchment_bounds[0],
                              catchment_bounds[1]]

        self.raster_size = [int((catchment_bounds[2] -
                                 catchment_bounds[0]) / catchment_geometry.resolution),
                            int((catchment_bounds[3] -
                                 catchment_bounds[1]) / catchment_geometry.resolution)]

        # Create a dummy DEM for updated origin and size
        empty_points = numpy.zeros([1], dtype=[('X', numpy.float64), ('Y', numpy.float64), ('Z', numpy.float64)])
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": catchment_geometry.resolution,
             "gdalopts": "a_srs=EPSG:" + str(catchment_geometry.crs['horizontal']),
             "output_type": ["idw"], "filename": str(self._temp_dem_file),
             "origin_x": self.raster_origin[0], "origin_y": self.raster_origin[1],
             "width": self.raster_size[0], "height": self.raster_size[1]}
        ]
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [empty_points])
        pdal_pipeline.execute()
        metadata = json.loads(pdal_pipeline.get_metadata())
        assert metadata['metadata']['writers.gdal']['filename'][0] == str(self._temp_dem_file), "The specified  file" \
            + f"file location: {self._temp_dem_file} and written file location: " + \
            f"{metadata['metadata']['writers.gdal']['filename'][0]} do not match."

        with rioxarray.rioxarray.open_rasterio(str(self._temp_dem_file), masked=True) as empty_dem:
            empty_dem.load()
            empty_dem.rio.set_crs(catchment_geometry.crs['horizontal'])

        # Check if the raster origin has been moved by PDAL writers.gdal and update if it has
        raster_origin = [empty_dem.x.data.min() - catchment_geometry.resolution/2,
                         empty_dem.y.data.min() - catchment_geometry.resolution/2]
        if self.raster_origin != raster_origin:
            print("In process: The generated dense DEM has an origin differing from the one specified. Updating the " +
                  f"catchment geometry raster origin from {self.raster_origin} to {raster_origin}")
            self.raster_origin = raster_origin

        # Ensure the DEM is empty - all NaN
        empty_dem.data[0] = numpy.nan
        empty_dem = empty_dem.rio.clip(catchment_geometry.catchment.geometry)
        return empty_dem

    def _point_cloud_to_raster_idw(self, point_cloud: numpy.ndarray, xy_out, power: int, search_radius: float,
                                   smoothing: float = 0, eps: float = 0, leaf_size: int = 10):
        """ Create a DEM tile from a LiDAR tile over a specified region.
        Currently PDAL writers.gdal is used and a temporary file is written out. In future another approach may be used.
        """
        xy_in = numpy.empty((len(point_cloud), 2))
        xy_in[:, 0] = point_cloud['X']
        xy_in[:, 1] = point_cloud['Y']

        tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
        tree_index_list = tree.query_ball_point(xy_out, r=search_radius, eps=eps)  # , eps=0.2)
        z_out = numpy.zeros(len(xy_out))

        for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

            if len(near_indicies) == 0:  # Set NaN if no values in search region
                z_out[i] = numpy.nan
            else:
                distance_vectors = point - tree.data[near_indicies]
                smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
                if smoothed_distances.min() == 0:  # incase the of an exact match
                    z_out[i] = point_cloud['Z'][tree.query(point, k=1)[1]]
                else:
                    z_out[i] = (point_cloud['Z'][near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                        / (1 / (smoothed_distances**power)).sum(axis=0)

        return z_out

    def add_tile(self, tile_points: numpy.ndarray, tile_extent: geopandas.GeoDataFrame, window_size: int,
                 idw_power: int, radius: float, method: str = 'first'):
        """ Create the DEM tile and then update the overall DEM with the tile. Only perform IDW within the tile
        extents. """

        if len(tile_points) == 0:
            if self.verbose:
                print("Warning in DenseDem.add_tile the latest tile has no data and is being ignored.")
            return

        # Update the tile extents with the new tile, then clip tile by extents (remove bleeding outside LiDAR area)
        tile_extent = self._update_extents(tile_extent)

        # Get the indicies overwhich to perform IDW
        tile = self._dense_dem.rio.clip(tile_extent.geometry)
        tile.data[0] = 0  # set all to zero then clip outside tile again - setting it to NaN
        tile = tile.rio.clip(tile_extent.geometry)

        grid_x, grid_y = numpy.meshgrid(tile.x, tile.y)
        flat_z = tile.data[0].flatten()
        mask_z = ~numpy.isnan(flat_z)

        xy_out = numpy.empty((mask_z.sum(), 2))
        xy_out[:, 0] = grid_x.flatten()[mask_z]
        xy_out[:, 1] = grid_y.flatten()[mask_z]

        # Perform IDW over the dense DEM within the extents of this point cloud tile
        z_idw = self._point_cloud_to_raster_idw(tile_points, xy_out, idw_power, radius,
                                                smoothing=0, eps=0, leaf_size=10)
        flat_z[mask_z] = z_idw
        tile.data[0] = flat_z.reshape(grid_x.shape)

        # Add the tile to the dense DEM
        self._dense_dem = rioxarray.merge.merge_arrays([self._dense_dem, tile], method=method)

        # Ensure the dem will be recalculated as another tile has been added
        self._dem = None

    def _update_extents(self, tile_extent: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        """ Update the extents of all LiDAR tiles updated. If 'drop_offshore_lidar' is True ensure extents are
        limited to the land and foreshore of the catchment. If 'area_to_drop' is True extend the extents to fill
        any holes around the catchment boundary smaller than the specified 'area_to_drop'.

        Return the updated tile_extent after filling any holes and any gaps around the edge and triming offshore."""

        assert len(tile_extent) == 1, "The tile_extent is expected to be contained in one shape. Instead " + \
            f"tile_extent: {tile_extent} is of length {len(tile_extent)}."

        if tile_extent.geometry.area.sum() > 0:  # check polygon isn't empty

            if self._extents is None:
                updated_extents = tile_extent
            else:
                updated_extents = geopandas.GeoDataFrame(
                    {'geometry': [shapely.ops.cascaded_union([self._extents.loc[0].geometry,
                                                              tile_extent.loc[0].geometry])]},
                    crs=self.catchment_geometry.crs['horizontal'])

            # Fill any holes smaller than the 'area_to_drop' internal to the extents or between it and catchment edge
            if self.area_to_drop is not None and self.area_to_drop >= 0:
                updated_extents = self._filter_holes_inside_polygon(self.area_to_drop, updated_extents)
                updated_extents = self._filter_holes_around_polygon(self.area_to_drop, updated_extents)

            if self.drop_offshore_lidar:
                updated_extents = geopandas.clip(self.catchment_geometry.land_and_foreshore, updated_extents)
            else:
                updated_extents = geopandas.clip(self.catchment_geometry.catchment, updated_extents)

            # Update the tile extents based on the updated overall cumlative tiles extents
            if self._extents is None:
                filtered_tile_extents = updated_extents
            else:
                filtered_tile_extents = geopandas.overlay(updated_extents, self._extents, how="difference")

            # Update the cumlative extents
            self._extents = updated_extents
            return filtered_tile_extents
        else:
            return tile_extent

    def _filter_holes_inside_polygon(self, area_to_filter: float,
                                     polygon_in: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        """ Check through the input polygon geometry and remove any holes less than the specified area. """

        if area_to_filter is None and area_to_filter <= 0:
            return polygon_in
        else:
            # Check through the extents geometry and remove any internal holes with an area less than the 'area_to_drop'
            polygon = polygon_in.loc[0].geometry

            # Remove any holes internal to the extents that are less than the area_to_drop
            if polygon.geometryType() == "Polygon":
                polygon = shapely.geometry.Polygon(
                    polygon.exterior.coords, [interior for interior in polygon.interiors if
                                              shapely.geometry.Polygon(interior).area > area_to_filter])
                polygon_out = geopandas.GeoDataFrame({'geometry': [polygon]},
                                                     crs=self.catchment_geometry.crs['horizontal'])
                polygon_out = geopandas.clip(self.catchment_geometry.catchment, polygon_out)
            elif polygon.geometryType() == "MultiPolygon":
                polygons = []
                for polygon_i in polygon:
                    polygons.append(shapely.geometry.Polygon(
                        polygon_i.exterior.coords, [interior for interior in polygon_i.interiors if
                                                    shapely.geometry.Polygon(interior).area > area_to_filter]))
                polygon = shapely.geometry.MultiPolygon(polygons)
                polygon_out = geopandas.GeoDataFrame({'geometry': [polygon]},
                                                     crs=self.catchment_geometry.crs['horizontal'])
                polygon_out = geopandas.clip(self.catchment_geometry.catchment, polygon_out)
            else:
                if self.verbose:
                    print("Warning filtering holes in CatchmentLidar using filter_lidar_extents_for_holes is not yet "
                          + f"supported for {polygon.geometryType()}")
            return polygon_out

    def _filter_holes_around_polygon(self, area_to_filter: float,
                                     polygon_in: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        """ Check around the input polygon geometry and remove any holes less than the specified area between it and the
        catchment polygon geometry. """

        if area_to_filter is None and area_to_filter <= 0:
            return polygon_in
        else:
            # Check through the extents geometry and remove any internal holes with an area less than the 'area_to_drop'
            polygon = geopandas.overlay(self.catchment_geometry.catchment, polygon_in, how="difference")

            if len(polygon) > 0:
                assert len(polygon) == 1, f"Expected the difference polygon {polygon} to have length 1 as both " + \
                    "input polygon and the catchment polygons are expected to have length 1."
                polygon = polygon.loc[0].geometry
                if polygon.geometryType() == "MultiPolygon":
                    # Run through each polygon element checking if smaller than the 'area_to_drop' - add if so
                    polygon_list = [polygon_i for polygon_i in polygon if polygon_i.area < area_to_filter]
                    polygon_list.append(polygon_in.loc[0].geometry)
                    polygon_out = geopandas.GeoDataFrame(
                        {'geometry': [shapely.ops.cascaded_union(polygon_list)]},
                        crs=self.catchment_geometry.crs['horizontal'])

                elif polygon.geometryType() == "Polygon":
                    if polygon.area < area_to_filter:
                        # Check if the polygon is smaller than the 'area_to_drop' - add if so
                        polygon_out = geopandas.GeoDataFrame(
                            {'geometry': [shapely.ops.cascaded_union([polygon_in.loc[0].geometry, polygon])]},
                            crs=self.catchment_geometry.crs['horizontal'])
                else:
                    if self.verbose:
                        print("Warning filtering holes in DenseDem using _update_extents is not yet "
                              + f"supported for {polygon.geometryType()}")
            return polygon_out
