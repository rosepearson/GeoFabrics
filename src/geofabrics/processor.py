# -*- coding: utf-8 -*-
"""
This module contains classes associated with generating hydrologically conditioned DEMs from
LiDAR and bathymetry contours based on the instructions contained in a JSON file.
"""
import numpy
import json
import pathlib
import shutil
import abc
import logging
import distributed
import rioxarray
import pandas
import geopandas
from . import geometry
from . import bathymetry_estimation
import geoapis.lidar
import geoapis.vector
from . import dem


class BaseProcessor(abc.ABC):
    """ An abstract class with general methods for accessing elements in instruction files including populating default
    values. Also contains functions for downloading remote data using geopais, and constructing data file lists.
    """

    def __init__(self, json_instructions: json):
        self.instructions = json_instructions

        self.catchment_geometry = None

    def get_instruction_path(self, key: str) -> str:
        """ Return the file path from the instruction file, or default if there is a default value and the local cache
        is specified. Raise an error if the key is not in the instructions. """

        defaults = {'result_dem': "generated_dem.nc", 'dense_dem': "dense_dem.nc",
                    'dense_dem_extents': "dense_extents.geojson"}

        if key in self.instructions['instructions']['data_paths']:
            return self.instructions['instructions']['data_paths'][key]
        elif "local_cache" in self.instructions['instructions']['data_paths'] and key in defaults.keys():
            return pathlib.Path(self.instructions['instructions']['data_paths']['local_cache']) / defaults[key]
        else:
            assert False, f"Either the key `{key}` missing from data paths, or either the `local_cache` is not " + \
                f"specified in the data paths or the key is not specified in the defaults: {defaults}"

    def check_instruction_path(self, key: str) -> bool:
        """ Return True if the file path exists in the instruction file, or True if there is a default value and the
        local cache is specified. """

        defaults = ['result_dem', 'dense_dem_extents', 'dense_dem']

        if key in self.instructions['instructions']['data_paths']:
            return True
        else:
            return 'local_cache' in self.instructions['instructions']['data_paths'] and key in defaults

    def get_resolution(self) -> float:
        """ Return the resolution from the instruction file. Raise an error if not in the instructions. """

        assert 'output' in self.instructions['instructions'], "'output' is not a key-word in the instructions. It " + \
            "should exist and is where the resolution and optionally the CRS of the output DEM is defined."
        assert 'resolution' in self.instructions['instructions']['output']['grid_params'], \
            "'resolution' is not a key-word in the instructions"
        return self.instructions['instructions']['output']['grid_params']['resolution']

    def get_crs(self) -> dict:
        """ Return the CRS projection information (horiztonal and vertical) from the instruction file. Raise an error
        if 'output' is not in the instructions. If no 'crs' or 'horizontal' or 'vertical' values are specified then use
        the default value for each one missing from the instructions."""

        defaults = {'horizontal': 2193, 'vertical': 7839}

        assert 'output' in self.instructions['instructions'], "'output' is not a key-word in the instructions. It " + \
            "should exist and is where the resolution and optionally the CRS of the output DEM is defined."

        if 'crs' not in self.instructions['instructions']['output']:
            logging.warning("No output the coordinate system EPSG values specified. We will instead be using the " +
                            f"defaults: {defaults}.")
            return defaults
        else:
            crs_instruction = self.instructions['instructions']['output']['crs']
            crs_dict = {}
            crs_dict['horizontal'] = crs_instruction['horizontal'] if 'horizontal' in crs_instruction else \
                defaults['horizontal']
            crs_dict['vertical'] = crs_instruction['vertical'] if 'vertical' in crs_instruction else \
                defaults['vertical']
            logging.info(f"The output the coordinate system EPSG values of {crs_dict} will be used. If these are not "
                         "as expected. Check both the 'horizontal' and 'vertical' values are specified.")
            return crs_dict

    def get_instruction_general(self, key: str):
        """ Return the general instruction from the instruction file or return the default value if not specified in
        the instruction file. Raise an error if the key is not in the instructions and there is no default value. """

        defaults = {'set_dem_shoreline': True,
                    'bathymetry_contours_z_label': None, 'drop_offshore_lidar': True,
                    'lidar_classifications_to_keep': [2], 'interpolate_missing_values': True}

        assert key in defaults or key in self.instructions['instructions']['general'], f"The key: {key} is missing " \
            + "from the general instructions, and does not have a default value"
        if 'general' in self.instructions['instructions'] and key in self.instructions['instructions']['general']:
            return self.instructions['instructions']['general'][key]
        else:
            return defaults[key]

    def get_processing_instructions(self, key: str):
        """ Return the processing instruction from the instruction file or return the default value if not specified in
        the instruction file. """

        defaults = {'number_of_cores': 1, "chunk_size": None}

        assert key in defaults or key in self.instructions['instructions']['processing'], f"The key: {key} is missing " \
            + "from the general instructions, and does not have a default value"
        if 'processing' in self.instructions['instructions'] and key in self.instructions['instructions']['processing']:
            return self.instructions['instructions']['processing'][key]
        else:
            return defaults[key]

    def check_apis(self, key) -> bool:
        """ Check to see if APIs are included in the instructions and if the key is included in specified apis """

        if "apis" in self.instructions['instructions']:
            # 'apis' included instructions and Key included in the APIs
            return key in self.instructions['instructions']['apis']
        else:
            return False

    def check_vector(self, key) -> bool:
        """ Check to see if vector key (i.e. land, bathymetry_contours, etc) is included either as a file path, or
        within any of the vector API's (i.e. LINZ or LRIS). """

        data_services = ["linz", "lris"]  # This list will increase as geopais is extended to support more vector APIs

        if "data_paths" in self.instructions['instructions'] and key in self.instructions['instructions']['data_paths']:
            # Key included in the data paths
            return True
        elif "apis" in self.instructions['instructions']:
            for data_service in data_services:
                if data_service in self.instructions['instructions']['apis'] \
                        and key in self.instructions['instructions']['apis'][data_service]:
                    # Key is included in one or more of the data_service's APIs
                    return True
        else:
            return False

    def get_vector_paths(self, key) -> list:
        """ Get the path to the vector key data included either as a file path or as a LINZ API. Return all paths
        where the vector key is specified. In the case that an API is specified ensure the data is fetched as well. """

        paths = []

        # Check the instructions for vector data specified as a data_paths
        if "data_paths" in self.instructions['instructions'] and key in self.instructions['instructions']['data_paths']:
            # Key included in the data paths - add - either list or individual path
            if type(self.instructions['instructions']['data_paths'][key]) == list:
                paths.extend(self.instructions['instructions']['data_paths'][key])
            else:
                paths.append(self.instructions['instructions']['data_paths'][key])

        # Define the supported vector 'apis' keywords and the geoapis class for accessing that data service
        data_services = {"linz": geoapis.vector.Linz, "lris": geoapis.vector.Lris}

        # Check the instructions for vector data hosted in the supported vector data services: LINZ and LRIS
        for data_service in data_services.keys():
            if self.check_apis(data_service) and key in self.instructions['instructions']['apis'][data_service]:

                # Get the location to cache vector data downloaded from data services
                assert self.check_instruction_path('local_cache'), "Local cache file path must exist to specify the" + \
                    f" location to download vector data from the vector APIs: {data_services}"
                cache_dir = pathlib.Path(self.get_instruction_path('local_cache'))

                # Get the API key for the data_serive being checked
                assert 'key' in self.instructions['instructions']['apis'][data_service], "A 'key' must be specified" + \
                    f" for the {data_service} data service instead the instruction only includes: " + \
                    f"{self.instructions['instructions']['apis'][data_service]}"
                api_key = self.instructions['instructions']['apis'][data_service]['key']

                assert self.catchment_geometry is not None, "The `self.catchment_geometry` object must exist " + \
                    "before a vector is downloaded using `vector.Linz`"

                # Instantiate the geoapis object for downloading vectors from the data service.
                vector_fetcher = data_services[data_service](api_key,
                                                             bounding_polygon=self.catchment_geometry.catchment,
                                                             verbose=True)

                vector_instruction = self.instructions['instructions']['apis'][data_service][key]
                geometry_type = vector_instruction['geometry_name '] if 'geometry_name ' in vector_instruction else None

                logging.info(f"Downloading vector layers {vector_instruction['layers']} from the {data_service} data" +
                             "service")

                # Cycle through all layers specified - save each and add to the path list
                for layer in vector_instruction['layers']:
                    # Use the run method to download each layer in turn
                    vector = vector_fetcher.run(layer, geometry_type)

                    # Ensure directory for layer and save vector file
                    layer_dir = cache_dir / str(layer)
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    vector_dir = layer_dir / key
                    vector.to_file(vector_dir)
                    shutil.make_archive(base_name=vector_dir, format='zip', root_dir=vector_dir)
                    shutil.rmtree(vector_dir)
                    paths.append(layer_dir / f"{key}.zip")
        return paths

    def get_lidar_dataset_crs(self, data_service, dataset_name) -> dict:
        """ Checks to see if source CRS of an associated LiDAR datasets has be specified in the instruction file. If it
        has been specified, this CRS is returned, and will later be used to override the CRS encoded in the LAS files.
        """

        if self.check_apis(data_service) and type(self.instructions['instructions']['apis'][data_service]) is dict \
                and dataset_name in self.instructions['instructions']['apis'][data_service] \
                and type(self.instructions['instructions']['apis'][data_service][dataset_name]) is dict:
            dataset_instruction = self.instructions['instructions']['apis'][data_service][dataset_name]

            if 'crs' in dataset_instruction and 'horizontal' in dataset_instruction['crs'] and \
                    'vertical' in dataset_instruction['crs']:
                dataset_crs = {'horizontal': dataset_instruction['crs']['horizontal'],
                               'vertical': dataset_instruction['crs']['vertical']}
                logging.info(f"The LiDAR dataset {dataset_name} is assumed to have the source coordinate system EPSG: "
                             f"{dataset_crs} as defined in the instruction file")
                return dataset_crs
            else:
                logging.info(f"The LiDAR dataset {dataset_name} will use the source the coordinate system EPSG defined"
                             " in its LAZ files")
                return None
        else:
            logging.info(f"The LiDAR dataset {dataset_name} will use the source coordinate system EPSG from its LAZ "
                         "files")
            return None

    def get_lidar_file_list(self, data_service) -> dict:
        """ Return a dictionary with three enties 'file_paths', 'crs' and 'tile_index_file'. The 'file_paths' contains a
        list of LiDAR tiles to process.

        The 'crs' (or coordinate system of the LiDAR data as defined by an EPSG code) is only optionally set (if unset
        the value is None). The 'crs' should only be set if the CRS information is not correctly encoded in the LAZ/LAS
        files. Currently this is only supported for OpenTopography LiDAR.

        The 'tile_index_file' is also optional (if unset the value is None). The 'tile_index_file' should be given if a
        tile index file exists for the LiDAR files specifying the extents of each tile. This is currently only supported
        for OpenTopography files.

        If a LiDAR API is specified this is checked and all files within the catchment area are downloaded and used to
        construct the file list. If none is specified, the instruction 'data_paths' is checked for 'lidars' and these
        are returned.
        """

        lidar_dataset_index = 0  # currently only support one LiDAR dataset

        lidar_dataset_info = {}

        # See if 'OpenTopography' or another data_service has been specified as an area to look first
        if self.check_apis(data_service):

            assert self.check_instruction_path('local_cache'), "A 'local_cache' must be specified under the " + \
                "'file_paths' in the instruction file if you are going to use an API - like 'open_topography'"

            # download the specified datasets from the data service - then get the local file path
            self.lidar_fetcher = geoapis.lidar.OpenTopography(cache_path=self.get_instruction_path('local_cache'),
                                                              search_polygon=self.catchment_geometry.catchment,
                                                              verbose=True)
            # Loop through each specified dataset and download it
            for dataset_name in self.instructions['instructions']['apis'][data_service].keys():
                logging.info(f"Fetching dataset: {dataset_name}")
                self.lidar_fetcher.run(dataset_name)
            assert len(self.lidar_fetcher.dataset_prefixes) == 1, "geofabrics currently only supports creating a DEM" \
                " from only one LiDAR dataset at a time. Please create an issue if you want support for mutliple " \
                f"datasets. Error as the following datasets were specified: {self.lidar_fetcher.dataset_prefixes}"
            dataset_prefix = self.lidar_fetcher.dataset_prefixes[lidar_dataset_index]
            lidar_dataset_info['file_paths'] = sorted(
                pathlib.Path(self.lidar_fetcher.cache_path / dataset_prefix).glob('*.laz'))
            lidar_dataset_info['crs'] = self.get_lidar_dataset_crs(data_service, dataset_prefix)
            lidar_dataset_info['tile_index_file'] = self.lidar_fetcher.cache_path / dataset_prefix / \
                f"{dataset_prefix}_TileIndex.zip"
        else:
            # get the specified file paths from the instructions
            lidar_dataset_info['file_paths'] = self.get_instruction_path('lidars')
            lidar_dataset_info['crs'] = None
            lidar_dataset_info['tile_index_file'] = None

        return lidar_dataset_info

    @abc.abstractmethod
    def run(self):
        """ This method controls the processor execution and code-flow. """

        raise NotImplementedError("NETLOC_API must be instantiated in the child class")


class DemGenerator(BaseProcessor):
    """ DemGenerator executes a pipeline for creating a hydrologically conditioned DEM from LiDAR and optionally a
    reference DEM and/or bathymetry contours. The data and pipeline logic is defined in the json_instructions file.

    The `DemGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the generation of a DEM as polygons.
     * dense_dem - Defines the hydrologically conditioned DEM as a combination of tiles from LiDAR and interpolated from
       bathymetry.
     * reference_dem - This optional object defines a background DEM that may be used to fill on land gaps in the LiDAR.
     * bathy_contours - This optional object defines the bathymetry vectors used by the dense_dem to define the DEM
       offshore.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and an instruction file
    """

    def __init__(self, json_instructions: json):

        super(DemGenerator, self).__init__(json_instructions=json_instructions)

        self.dense_dem = None
        self.reference_dem = None
        self.bathy_contours = None

    def run(self):
        """ This method executes the geofabrics generation pipeline to produce geofabric derivatives.

        Note it currently only considers one LiDAR dataset that can have many tiles.
        See 'get_lidar_file_list' for where to change this. """

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = 10.0/100  # Used to decide if a background DEM or bathymetry should be included

        # create the catchment geometry object
        catchment_dirs = self.get_instruction_path('catchment_boundary')
        assert type(catchment_dirs) is not list, f"A list of catchment_boundary's is provided: {catchment_dirs}, " + \
            "where only one is supported."
        self.catchment_geometry = geometry.CatchmentGeometry(catchment_dirs, self.get_crs(),
                                                             self.get_resolution(), foreshore_buffer=2)
        land_dirs = self.get_vector_paths('land')
        assert len(land_dirs) == 1, f"{len(land_dirs)} catchment_boundary's provided, where only one is supported." + \
            f" Specficially land_dirs = {land_dirs}."
        self.catchment_geometry.land = land_dirs[0]

        # Define PDAL/GDAL griding parameter values
        idw_radius = self.catchment_geometry.resolution * numpy.sqrt(2)
        idw_power = 2

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_dataset_info = self.get_lidar_file_list('open_topography')

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDemFromTiles(
            catchment_geometry=self.catchment_geometry,
            drop_offshore_lidar=self.get_instruction_general('drop_offshore_lidar'),
            interpolate_missing_values=self.get_instruction_general('interpolate_missing_values'),
            idw_power=idw_power, idw_radius=idw_radius)

        # Setup Dask cluster and client
        cluster_kwargs = {'n_workers': self.get_processing_instructions('number_of_cores'),
                          'threads_per_worker': 1,
                          'processes': True}
        with distributed.LocalCluster(**cluster_kwargs) as cluster, distributed.Client(cluster) as client:
            print(client)
            # Load in LiDAR tiles
            self.dense_dem.add_lidar(lidar_files=lidar_dataset_info['file_paths'], source_crs=lidar_dataset_info['crs'],
                                     drop_offshore_lidar=self.get_instruction_general('drop_offshore_lidar'),
                                     lidar_classifications_to_keep=self.get_instruction_general('lidar_classifications_to_keep'),
                                     tile_index_file=lidar_dataset_info['tile_index_file'],
                                     chunk_size=self.get_processing_instructions('chunk_size'))

        # save dense DEM results
        self.dense_dem.dense_dem.to_netcdf(self.get_instruction_path('dense_dem'))

        # Load in reference DEM if any significant land/foreshore not covered by LiDAR
        if self.check_instruction_path('reference_dems'):
            area_without_lidar = \
                self.catchment_geometry.land_and_foreshore_without_lidar(self.dense_dem.extents).geometry.area.sum()
            if area_without_lidar > self.catchment_geometry.land_and_foreshore.area.sum() * area_threshold:

                assert len(self.get_instruction_path('reference_dems')) == 1, \
                    f"{len(self.get_instruction_path('reference_dems'))} reference_dems specified, but only one supported" \
                    + f" currently. reference_dems: {self.get_instruction_path('reference_dems')}"

                logging.info(f"Incorporating background DEM: {self.get_instruction_path('reference_dems')}")

                # Load in background DEM - cut away within the LiDAR extents
                self.reference_dem = dem.ReferenceDem(dem_file=self.get_instruction_path('reference_dems')[0],
                                                      catchment_geometry=self.catchment_geometry,
                                                      set_foreshore=self.get_instruction_general('set_dem_shoreline'),
                                                      exclusion_extent=self.dense_dem.extents)

                # Add the reference DEM patch where there's no LiDAR to the dense DEM without updting the extents
                self.dense_dem.add_reference_dem(tile_points=self.reference_dem.points,
                                                 tile_extent=self.reference_dem.extents)

        if self.dense_dem.extents is not None:  # Save ou the extents of the LiDAR - before reference DEM
            self.dense_dem.extents.to_file(self.get_instruction_path('dense_dem_extents'))
        else:
            logging.warning("In processor.DemGenerator - no LiDAR extents exist so no extents file written")

        # Load in bathymetry and interpolate offshore if significant offshore is not covered by LiDAR
        if self.check_vector('bathymetry_contours'):
            area_without_lidar = \
                self.catchment_geometry.offshore_without_lidar(self.dense_dem.extents).geometry.area.sum()
            if area_without_lidar > self.catchment_geometry.offshore.area.sum() * area_threshold:

                # Get the bathymetry data directory
                bathy_contour_dirs = self.get_vector_paths('bathymetry_contours')
                assert len(bathy_contour_dirs) == 1, f"{len(bathy_contour_dirs)} bathymetry_contours's provided. " + \
                    f"Specficially {bathy_contour_dirs}. Support has not yet been added for multiple datasets."

                logging.info(f"Incorporating offshore Bathymetry: {bathy_contour_dirs}")

                # Load in bathymetry
                self.bathy_contours = geometry.BathymetryContours(
                    bathy_contour_dirs[0], self.catchment_geometry,
                    z_label=self.get_instruction_general('bathymetry_contours_z_label'),
                    exclusion_extent=self.dense_dem.extents)

                # interpolate
                self.dense_dem.interpolate_offshore(self.bathy_contours)

        # Load in river bathymetry and incorporate where decernable at the resolution
        if self.check_vector('river_polygons') and self.check_vector('river_bathymetry'):

            # Get the polygons and bathymetry and check only one of each
            bathy_dirs = self.get_vector_paths('river_bathymetry')
            assert len(bathy_dirs) == 1, f"{len(bathy_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {bathy_dirs}. Support has not yet been added for multiple datasets."
            poly_dirs = self.get_vector_paths('river_polygons')
            assert len(poly_dirs) == 1, f"{len(poly_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {poly_dirs}. Support has not yet been added for multiple datasets."

            logging.info(f"Incorporating river Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.river_bathy = geometry.RiverBathymetryPoints(
                points_file=bathy_dirs[0],
                polygon_file=poly_dirs[0],
                catchment_geometry=self.catchment_geometry,
                z_label=self.get_instruction_general("river_bathy_z_label"))

            # Call interpolate river on the DEM - note the DEM checks to see if any pixels actually fall inside the polygon
            self.dense_dem.interpolate_river_bathymetry(river_bathymetry=self.river_bathy)

        # fill combined dem - save results
        self.dense_dem.dem.to_netcdf(self.get_instruction_path('result_dem'))


class OffshoreDemGenerator(BaseProcessor):
    """ OffshoreDemGenerator executes a pipeline for loading in a Dense DEM and extents before interpolating offshore
    DEM values. The data and pipeline logic is defined in the json_instructions file.

    The `DemGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the generation of a DEM as polygons.
     * dense_dem - Defines the hydrologically conditioned DEM as a combination of tiles from LiDAR and interpolated from
       bathymetry.
     * bathy_contours - This object defines the bathymetry vectors used by the dense_dem to define the DEM
       offshore.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and an instruction file
    """

    def __init__(self, json_instructions: json):

        super(OffshoreDemGenerator, self).__init__(json_instructions=json_instructions)

        self.dense_dem = None
        self.bathy_contours = None

    def run(self):
        """ This method executes the geofabrics generation pipeline to produce geofabric derivatives.

        Note it currently only considers one LiDAR dataset that may have many tiles.
        See 'get_lidar_file_list' for where to change this. """

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = 10.0/100  # Used to decide if bathymetry should be included

        # create the catchment geometry object
        catchment_dirs = self.get_instruction_path('catchment_boundary')
        assert type(catchment_dirs) is not list, f"A list of catchment_boundary's is provided: {catchment_dirs}, " + \
            "where only one is supported."
        self.catchment_geometry = geometry.CatchmentGeometry(catchment_dirs, self.get_crs(),
                                                             self.get_resolution(), foreshore_buffer=2)
        land_dirs = self.get_vector_paths('land')
        assert len(land_dirs) == 1, f"{len(land_dirs)} catchment_boundary's provided, where only one is supported." + \
            f" Specficially land_dirs = {land_dirs}."
        self.catchment_geometry.land = land_dirs[0]

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDemFromFiles(catchment_geometry=self.catchment_geometry,
                                               dense_dem_path=self.get_instruction_path('dense_dem'),
                                               extents_path=self.get_instruction_path('dense_dem_extents'),
                                               interpolate_missing_values=self.get_instruction_general('interpolate_missing_values'))

        # Load in bathymetry and interpolate offshore if significant offshore is not covered by LiDAR
        area_without_lidar = \
            self.catchment_geometry.offshore_without_lidar(self.dense_dem.extents).geometry.area.sum()
        if (self.check_vector('bathymetry_contours') and
                area_without_lidar > self.catchment_geometry.offshore.area.sum() * area_threshold):

            # Get the bathymetry data directory
            bathy_contour_dirs = self.get_vector_paths('bathymetry_contours')
            assert len(bathy_contour_dirs) == 1, f"{len(bathy_contour_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {catchment_dirs}. Support has not yet been added for multiple datasets."

            logging.info(f"Incorporating Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.bathy_contours = geometry.BathymetryContours(
                bathy_contour_dirs[0], self.catchment_geometry,
                z_label=self.get_instruction_general('bathymetry_contours_z_label'),
                exclusion_extent=self.dense_dem.extents)

            # interpolate
            self.dense_dem.interpolate_offshore(self.bathy_contours)

        # Load in river bathymetry and incorporate where decernable at the resolution
        if self.check_vector('river_polygons') and self.check_vector('river_bathymetry'):

            # Get the polygons and bathymetry and check only one of each
            bathy_dirs = self.get_vector_paths('river_bathymetry')
            assert len(bathy_dirs) == 1, f"{len(bathy_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {bathy_dirs}. Support has not yet been added for multiple datasets."
            poly_dirs = self.get_vector_paths('river_polygons')
            assert len(poly_dirs) == 1, f"{len(poly_dirs)} bathymetry_contours's provided. " + \
                f"Specficially {poly_dirs}. Support has not yet been added for multiple datasets."

            logging.info(f"Incorporating river Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.river_bathy = geometry.RiverBathymetryPoints(
                points_file=bathy_dirs[0],
                polygon_file=poly_dirs[0],
                catchment_geometry=self.catchment_geometry,
                z_label=self.get_instruction_general("river_bathy_z_label"))

            # Call interpolate river on the DEM - note the DEM checks to see if any pixels actually fall inside the polygon
            self.dense_dem.interpolate_river_bathymetry(river_bathymetry=self.river_bathy)

        # fill combined dem - save results
        self.dense_dem.dem.to_netcdf(self.get_instruction_path('result_dem'))


class RiverBathymetryGenerator(BaseProcessor):
    """ RiverbathymetryGenerator executes a pipeline to estimate river
    bathymetry depths from flows, slopes, friction and widths along a main
    channel. This is dones by first creating a hydrologically conditioned DEM
    of the channel. A json_instructions file defines the pipeline logic and
    data.

    Attributes:
        channel_polyline  The main channel along which to estimate depth. This
            is a polyline.
        dem  The ground DEM generated along the main channel. This is a
            raster.
        veg_dem  The vegetation DEM generated along the main channel. This is a
            raster.
        aligned_channel_polyline  The main channel after its alignment has been
            updated based on the DEM. Width and slope are estimated on this.
        transects  Transect polylines perpindicular to the aligned channel with
            samples of the DEM values
    """

    def __init__(self, json_instructions: json):

        self.instructions = json_instructions

        self.channel_bathymetry = None
        self.dem = None
        self.veg_dem = None
        self.aligned_channel_plyline = None
        self.transects = None

    def run(self, instruction_parameters):
        """ This method extracts a main channel then executes the DemGeneration
        pipeline to produce a DEM before sampling this to extimate width, slope
        and eventually depth. """

        # Read ininstructions
        crs = self.instructions["instructions"]["output"]["crs"]["horizontal"]
        rec = geopandas.read_file(self.instructions['instructions']['channel_bathymetry']['rec_file'])
        flow = pandas.read_csv(self.instructions['instructions']['channel_bathymetry']['flow_file'])
        channel_rec_id = self.instructions['instructions']['channel_bathymetry']['channel_rec_id']
        area_threshold = self.instructions['instructions']['channel_bathymetry']['channel_area_threshold']

        buffer = 50
        max_channel_width = self.instructions['instructions']['channel_bathymetry']['max_channel_width']
        min_channel_width = self.instructions['instructions']['channel_bathymetry']['min_channel_width']
        rec_alignment_tolerance = self.instructions['instructions']['channel_bathymetry']['rec_alignment_tolerance']

        resolution = self.instructions['instructions']['output']['grid_params']['resolution']
        transect_spacing = self.instructions['instructions']['channel_bathymetry']['transect_spacing']

        bank_threshold = self.instructions['instructions']['channel_bathymetry']['bank_threshold']
        max_bank_height = self.instructions['instructions']['channel_bathymetry']['max_bank_height']

        width_centre_smoothing_multiplier = \
            self.instructions['instructions']['channel_bathymetry']['width_centre_smoothing']

        # Define paths for generated files
        local_cache = pathlib.Path(self.instructions['instructions']['data_paths']['local_cache'])
        catchment_file = local_cache / f"channel_catchment_{area_threshold}.geojson"
        self.instructions['instructions']['data_paths']['catchment_boundary'] = str(catchment_file)
        aligned_channel_file = local_cache / f"aligned_channel_{area_threshold}.geojson"
        channel_file = local_cache / "rec_main_channel.geojson"

        # Identify the main channel, fit a spline
        channel = bathymetry_estimation.Channel.from_rec(rec_network=rec,
                                                         reach_id=channel_rec_id,
                                                         resolution=transect_spacing,
                                                         area_threshold=area_threshold)
        if not channel_file.is_file():
            channel.channel.to_file(channel_file)
        if not (local_cache / "smoothed_main_channel.geojson").is_file():
            channel.get_sampled_spline_fit().to_file(local_cache / "smoothed_main_channel.geojson")

        # Generate the DEM
        dem_file = local_cache / f"channel_dem_{area_threshold}.nc"
        self.instructions['instructions']['data_paths']['result_dem'] = str(dem_file)
        if not dem_file.is_file():
            # Create the ground DEM file if this has not be created yet!
            print("No gnd DEM along the channel. Generating a DEM.")
            if not catchment_file.is_file():
                corridor_radius = max_channel_width / 2 + rec_alignment_tolerance + buffer
                channel_catchment = channel.get_channel_catchment(corridor_radius=corridor_radius)
                channel_catchment.to_file(self.instructions['instructions']['data_paths']['catchment_boundary'])
            runner = DemGenerator(self.instructions)
            runner.run()
            self.dem = runner.dense_dem.dem
        else:
            print("gnd DEM along the channel exists. Loading the DEM.")
            with rioxarray.rioxarray.open_rasterio(dem_file, masked=True) as dem:
                dem.load()
            self.dem = dem.copy(deep=True)

        # Generate a DEM of vegetration with LiDAR
        dem_file = local_cache / f"channel_veg_dem_{area_threshold}.nc"
        self.instructions['instructions']['data_paths']['result_dem'] = str(dem_file)
        self.instructions['instructions']['general']['lidar_classifications_to_keep'] = \
            self.instructions['instructions']['channel_bathymetry']['veg_lidar_classifications_to_keep']
        if not dem_file.is_file():
            # Create the catchment file if this has not be created yet!
            print("No veg DEM along the channel. Generating a DEM.")
            if not pathlib.Path(self.instructions['instructions']['data_paths']['catchment_boundary']).is_file():
                corridor_radius = max_channel_width / 2 + rec_alignment_tolerance + buffer
                channel_catchment = channel.get_channel_catchment(corridor_radius=corridor_radius)
                channel_catchment.to_file(self.instructions['instructions']['data_paths']['catchment_boundary'])
            runner = DemGenerator(self.instructions)
            runner.run()
            self.veg_dem = runner.dense_dem.dem
        else:
            print("veg DEM along the channel exists. Loading the DEM.")
            with rioxarray.rioxarray.open_rasterio(dem_file, masked=True) as dem:
                dem.load()
            self.veg_dem = dem.copy(deep=True)

        # Create channel bathymetry estimator
        self.channel_bathymetry = bathymetry_estimation.ChannelBathymetry(
            channel=channel,
            dem=self.dem,
            veg_dem=self.veg_dem,
            transect_spacing=transect_spacing,
            resolution=resolution)

        # Get/Generate aligned channel
        if not aligned_channel_file.is_file():
            print("No aligned channel provided. Aligning the channel.")

            corridor_radius = max_channel_width / 2 + rec_alignment_tolerance + buffer

            aligned_channel, transects = self.channel_bathymetry.align_channel(
                threshold=bank_threshold,
                min_channel_width=min_channel_width,
                search_radius=rec_alignment_tolerance,
                width_centre_smoothing_multiplier=width_centre_smoothing_multiplier,
                cross_section_radius=corridor_radius)

            aligned_channel.to_file(aligned_channel_file)
            transects[['width_line', 'valid', 'channel_count']].set_geometry(
                'width_line').to_file(local_cache / "intial_widths.geojson")
            transects[['geometry', 'channel_count', 'valid']].to_file(local_cache / "transects.geojson")
        else:
            print("Channel already aligned and loaded in.")
            aligned_channel = geopandas.read_file(aligned_channel_file)

        # Estimate width and slope of channel
        if not (local_cache / "final_values.geojson").is_file():
            print("Calculating the final widths.")

            corridor_radius = max_channel_width / 2 + buffer
            transects, river_polygon = self.channel_bathymetry.estimate_width_and_slope(
                aligned_channel=aligned_channel,
                threshold=bank_threshold,
                cross_section_radius=corridor_radius,
                search_radius=rec_alignment_tolerance,
                min_channel_width=min_channel_width,
                max_threshold=max_bank_height,
                river_polygon_smoothing_multiplier=width_centre_smoothing_multiplier)

            river_polygon.to_file(local_cache / ("river_polygon.geojson"))
            transects[['geometry', 'channel_count']].to_file(local_cache / "final_transects.geojson")
            columns = ['geometry']
            columns.extend([column_name for column_name in transects.columns
                            if 'slope' in column_name or 'widths' in column_name
                            or 'min_z' in column_name or 'threshold' in column_name
                            or 'valid' in column_name])
            transects[columns].to_file(local_cache / "final_transect_values.geojson")
            transects.set_geometry('width_line', drop=True)[['geometry', 'valid']].to_file(local_cache / "final_widths.geojson")
            transects.set_geometry('flat_midpoint', drop=True)[columns].to_file(local_cache / "final_values.geojson")
            transects.set_geometry('river_polygon_midpoint', drop=True)[columns].to_file(local_cache / "river_polygon_midpoint.geojson")
        else:
            print("The final widths have already been generated")

        # Read in the flow file and calcaulate the depths - write out the results
        width_values = geopandas.read_file(local_cache / "river_polygon_midpoint.geojson")

        # Match each channel midpoint to a nzsegment id - based on what channel reach is closest
        width_values['nzsegment'] = numpy.zeros(len(width_values['widths']), dtype=int)
        for i, row in width_values.iterrows():
            distances = channel.channel.distance(width_values.loc[i].geometry)
            width_values.loc[i, ('nzsegment')] = channel.channel[distances == distances.min()]['nzsegment'].min()

        # Add the friction and flow values to the widths and slopes
        width_values['mannings_n'] = numpy.zeros(len(width_values['nzsegment']), dtype=int)
        width_values['flow'] = numpy.zeros(len(width_values['nzsegment']), dtype=int)
        for nzsegment in width_values['nzsegment'].unique():
            width_values.loc[width_values['nzsegment'] == nzsegment,
                             ('mannings_n')] = flow[flow['nzsegment'] == nzsegment]['n'].unique()[0]
            width_values.loc[width_values['nzsegment'] == nzsegment,
                             ('flow')] = flow[flow['nzsegment'] == nzsegment]['flow'].unique()[0]

        # Calculate the depths using various approaches
        slope_name = 'slope_mean_2.0km'
        min_z_name = 'min_z_centre_unimodal'
        width_name = 'widths_mean_0.25km'
        threshold_name = 'thresholds_mean_0.25km'
        width_values['depth_Neal_et_al'] = \
            (width_values['mannings_n'] * width_values['flow']
             / (numpy.sqrt(width_values[slope_name]) * width_values[width_name])) ** (3/5) - width_values[threshold_name]
        a = 0.745
        b = 0.305
        K_0 = 6.16
        width_values['depth_Smart_et_al'] = \
            (width_values['flow'] / (K_0 * width_values[width_name]
                                     * width_values[slope_name] ** b)) ** (1 / (1+a)) - width_values[threshold_name]

        # Calculate the bed elevation
        width_values['bed_elevation_Neal_et_al'] = width_values[min_z_name] - width_values['depth_Neal_et_al']
        width_values['bed_elevation_Smart_et_al'] = width_values[min_z_name] - width_values['depth_Smart_et_al']

        # Save the bed elevations
        width_values[['geometry', 'bed_elevation_Neal_et_al', 'bed_elevation_Smart_et_al']].to_file(local_cache / "river_bathymetry.geojson")

        ## Ocean fan calculations
        # Add some bathymetry transitioning from river to coast and a polygon
        aligned_channel = geopandas.read_file(aligned_channel_file)

        # Calculate the tangent and normal to the last segment
        import shapely
        (x,y) = aligned_channel.loc[0].geometry.xy
        mouth_point = shapely.geometry.Point([x[0], y[0]])
        segment_dx = x[0] - x[1]
        segment_dy = y[0] - y[1]
        segment_length = numpy.sqrt(segment_dx**2 + segment_dy**2)
        tangent_x = segment_dx / segment_length
        tangent_y = segment_dy / segment_length
        normal_x = -tangent_y
        normal_y = tangent_x

        # create fan centreline
        fan_max_length = 10000
        extended_line = shapely.geometry.LineString(
            [mouth_point,
             [mouth_point.x + fan_max_length * tangent_x, mouth_point.y + fan_max_length * tangent_y]])

        # Load in the depth and width at the river mouth
        river_mouth_depth = geopandas.read_file(local_cache / "river_bathymetry.geojson")['bed_elevation_Smart_et_al'].iloc[0]
        river_mouth_width = geopandas.read_file(local_cache / "final_values.geojson")['widths'].iloc[0]

        # Load in the ocean contours and find the contours to terminate against
        ocean_contours = geopandas.read_file(self.get_vector_paths('bathymetry_contours')[0]).to_crs(crs)
        depth_label = self.instructions['instructions']['general']['bathymetry_contours_z_label']
        depth_sign = -1
        depth_multiplier = 2
        end_depth = ocean_contours[depth_label][ocean_contours[depth_label] > depth_multiplier * river_mouth_depth * depth_sign ].min()
        ocean_contours = ocean_contours[ocean_contours[depth_label] == end_depth].reset_index(drop=True)

        # Cycle through contours finding the first 'deep enough' contour to cross
        distance = numpy.inf
        end_point = shapely.geometry.Point()

        for i, row in ocean_contours.iterrows():
            if row.geometry.intersects(extended_line):
                intersection_point = row.geometry.intersection(extended_line)
                if intersection_point.distance(mouth_point) < distance:
                    distance = intersection_point.distance(mouth_point)
                    end_point = intersection_point

        # Define and save the fan polygon
        fan_angle = 15
        end_width = river_mouth_width + 2 * distance * numpy.tan(numpy.pi/180 * fan_angle)
        fan_polygon = shapely.geometry.Polygon([[mouth_point.x - normal_x * river_mouth_width / 2,
                                                 mouth_point.y - normal_y * river_mouth_width / 2],
                                                [mouth_point.x + normal_x * river_mouth_width / 2,
                                                 mouth_point.y + normal_y * river_mouth_width / 2],
                                                [end_point.x + normal_x * end_width / 2,
                                                 end_point.y + normal_y * end_width / 2],
                                                [end_point.x - normal_x * end_width / 2,
                                                 end_point.y - normal_y * end_width / 2]])
        geopandas.GeoDataFrame(geometry=[fan_polygon], crs=crs).to_file(local_cache / "river_mouth_fan_polygon.geojson")

        # Define and save the fan depths
        fan_depths = {'geometry': [], 'depths': []}
        number_of_samples = int(distance / transect_spacing)
        depth_increment = (-1 * end_depth - river_mouth_depth) / number_of_samples
        
        for i in range(1, number_of_samples):
            fan_depths['geometry'].append(shapely.geometry.Point([mouth_point.x + tangent_x * i * transect_spacing,
                                                                  mouth_point.y + tangent_y * i * transect_spacing]))
            fan_depths['depths'].append(river_mouth_depth + i * depth_increment)
        geopandas.GeoDataFrame(fan_depths, crs=crs).to_file(local_cache / "river_mouth_fan_depths.geojson")

        # Update parameter file - in time only update the bits that have been re-run
        with open(instruction_parameters, 'w') as file_pointer:
            json.dump(self.instructions, file_pointer)
