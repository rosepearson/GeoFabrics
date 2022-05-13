# -*- coding: utf-8 -*-
"""
This module contains classes associated with generating GeoFabric layers from
LiDAR and bathymetry contours based on the instructions contained in a JSON file.

GeoFabric layers include hydrologically conditioned DEMs.
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
import datetime
import shapely
import xarray
import OSMPythonTools.overpass
from . import geometry
from . import bathymetry_estimation
from . import version
import geoapis.lidar
import geoapis.vector
from . import dem


class BaseProcessor(abc.ABC):
    """An abstract class with general methods for accessing elements in
    instruction files including populating default values. Also contains
    functions for downloading remote data using geopais, and constructing data
    file lists.
    """

    def __init__(self, json_instructions: json):
        self.instructions = json_instructions

        self.catchment_geometry = None

    def get_instruction_path(self, key: str) -> str:
        """Return the file path from the instruction file, or default if there
        is a default value and the local cache is specified. Raise an error if
        the key is not in the instructions."""

        defaults = {
            "result_dem": "generated_dem.nc",
            "dense_dem": "dense_dem.nc",
            "dense_dem_extents": "dense_extents.geojson",
        }

        path_instructions = self.instructions["instructions"]["data_paths"]
        local_cache = pathlib.Path(path_instructions["local_cache"])

        if key == "local_cache":
            return local_cache
        elif key in path_instructions:
            # check if a list or single path
            if type(path_instructions[key]) == list:
                absolute_file_paths = []
                file_paths = path_instructions[key]
                for file_path in file_paths:
                    file_path = pathlib.Path(file_path)
                    file_path = (
                        file_path
                        if file_path.is_absolute()
                        else local_cache / file_path
                    )
                    absolute_file_paths.append(str(file_path))
                return absolute_file_paths
            else:
                file_path = pathlib.Path(path_instructions[key])
                file_path = (
                    file_path if file_path.is_absolute() else local_cache / file_path
                )
                return file_path
        elif key in defaults.keys():
            return local_cache.absolute() / defaults[key]
        else:
            assert False, (
                f"The key `{key}` is either missing from data "
                "paths, not specified in the defaults: {defaults}"
            )

    def check_instruction_path(self, key: str) -> bool:
        """Return True if the file path exists in the instruction file, or True
        if there is a default value and the local cache is specified."""

        assert (
            "local_cache" in self.instructions["instructions"]["data_paths"]
        ), "local_cache is a required 'data_paths' entry"

        defaults = ["result_dem", "dense_dem_extents", "dense_dem"]

        if key in self.instructions["instructions"]["data_paths"]:
            return True
        else:
            return key in defaults

    def get_resolution(self) -> float:
        """Return the resolution from the instruction file. Raise an error if
        not in the instructions."""

        assert "output" in self.instructions["instructions"], (
            "'output' is not a key-word in the instructions. It should exist"
            " and is where the resolution and optionally the CRS of the output"
            " DEM is defined."
        )
        assert (
            "resolution" in self.instructions["instructions"]["output"]["grid_params"]
        ), "'resolution' is not a key-word in the instructions"
        return self.instructions["instructions"]["output"]["grid_params"]["resolution"]

    def get_crs(self) -> dict:
        """Return the CRS projection information (horiztonal and vertical) from
        the instruction file. Raise an error if 'output' is not in the instructions. If
        no 'crs' or 'horizontal' or 'vertical' values are specified then use the default
        value for each one missing from the instructions. If the default is used it is
        added to the instructions.
        """

        defaults = {"horizontal": 2193, "vertical": 7839}

        assert "output" in self.instructions["instructions"], (
            "'output' is not a key-word in the instructions. It should exist "
            "and is where the resolution and optionally the CRS of the output "
            "DEM is defined."
        )

        if "crs" not in self.instructions["instructions"]["output"]:
            logging.warning(
                "No output the coordinate system EPSG values specified. We "
                f"will instead be using the defaults: {defaults}."
            )
            self.instructions["instructions"]["output"]["crs"] = defaults
            return defaults
        else:
            crs_instruction = self.instructions["instructions"]["output"]["crs"]
            crs_dict = {}
            crs_dict["horizontal"] = (
                crs_instruction["horizontal"]
                if "horizontal" in crs_instruction
                else defaults["horizontal"]
            )
            crs_dict["vertical"] = (
                crs_instruction["vertical"]
                if "vertical" in crs_instruction
                else defaults["vertical"]
            )
            logging.info(
                f"The output the coordinate system EPSG values of {crs_dict} "
                "will be used. If these are not as expected. Check both the "
                "'horizontal' and 'vertical' values are specified."
            )
            # Update the CRS just incase this includes any default values
            self.instructions["instructions"]["output"]["crs"] = crs_dict
            return crs_dict

    def get_instruction_general(self, key: str):
        """Return the general instruction from the instruction file or return
        the default value if not specified in the instruction file. Raise an
        error if the key is not in the instructions and there is no default
        value. If the default is used it is added to the instructions."""

        defaults = {
            "set_dem_shoreline": True,
            "bathymetry_contours_z_label": None,
            "drop_offshore_lidar": True,
            "lidar_classifications_to_keep": [2],
            "interpolation_method": None,
            "elevation_range": None,
            "lidar_interpolation_method": "idw",
        }

        assert key in defaults or key in self.instructions["instructions"]["general"], (
            f"The key: {key} is missing from the general instructions, and"
            " does not have a default value"
        )
        if (
            "general" in self.instructions["instructions"]
            and key in self.instructions["instructions"]["general"]
        ):
            return self.instructions["instructions"]["general"][key]
        else:
            self.instructions["instructions"]["general"][key] = defaults[key]
            return defaults[key]

    def get_processing_instructions(self, key: str):
        """Return the processing instruction from the instruction file or
        return the default value if not specified in the instruction file. If
        the default is used it is added to the instructions."""

        defaults = {"number_of_cores": 1, "chunk_size": None}

        assert (
            key in defaults or key in self.instructions["instructions"]["processing"]
        ), (
            f"The key: {key} is missing "
            + "from the general instructions, and does not have a default value"
        )
        if (
            "processing" in self.instructions["instructions"]
            and key in self.instructions["instructions"]["processing"]
        ):
            return self.instructions["instructions"]["processing"][key]
        else:
            if "processing" not in self.instructions["instructions"]:
                self.instructions["instructions"]["processing"] = {}
            self.instructions["instructions"]["processing"][key] = defaults[key]
            return defaults[key]

    def check_apis(self, key) -> bool:
        """Check to see if APIs are included in the instructions and if the key is
        included in specified apis"""

        if "apis" in self.instructions["instructions"]:
            # 'apis' included instructions and Key included in the APIs
            return key in self.instructions["instructions"]["apis"]
        else:
            return False

    def check_vector(self, key) -> bool:
        """Check to see if vector key (i.e. land, bathymetry_contours, etc) is included
        either as a file path, or
        within any of the vector API's (i.e. LINZ or LRIS)."""

        data_services = [
            "linz",
            "lris",
        ]  # This list will increase as geopais is extended to support more vector APIs

        if (
            "data_paths" in self.instructions["instructions"]
            and key in self.instructions["instructions"]["data_paths"]
        ):
            # Key included in the data paths
            return True
        elif "apis" in self.instructions["instructions"]:
            for data_service in data_services:
                if (
                    data_service in self.instructions["instructions"]["apis"]
                    and key in self.instructions["instructions"]["apis"][data_service]
                ):
                    # Key is included in one or more of the data_service's APIs
                    return True
        else:
            return False

    def get_vector_paths(self, key) -> list:
        """Get the path to the vector key data included either as a file path or as a
        LINZ API. Return all paths where the vector key is specified. In the case that
        an API is specified ensure the data is fetched as well."""

        paths = []

        # Check the instructions for vector data specified as a data_paths
        if (
            "data_paths" in self.instructions["instructions"]
            and key in self.instructions["instructions"]["data_paths"]
        ):
            # Key included in the data paths - add - either list or individual path
            data_paths = self.get_instruction_path(key)
            if type(data_paths) is list:
                paths.extend(data_paths)
            else:
                paths.append(data_paths)
        # Define the supported vector 'apis' keywords and the geoapis class for
        # accessing that data service
        data_services = {"linz": geoapis.vector.Linz, "lris": geoapis.vector.Lris}

        # Check the instructions for vector data hosted in the supported vector data
        # services: LINZ and LRIS
        for data_service in data_services.keys():
            if (
                self.check_apis(data_service)
                and key in self.instructions["instructions"]["apis"][data_service]
            ):

                # Get the location to cache vector data downloaded from data services
                assert self.check_instruction_path("local_cache"), (
                    "Local cache file path must exist to specify thelocation to"
                    + f" download vector data from the vector APIs: {data_services}"
                )
                cache_dir = pathlib.Path(self.get_instruction_path("local_cache"))

                # Get the API key for the data_serive being checked
                assert (
                    "key" in self.instructions["instructions"]["apis"][data_service]
                ), (
                    f"A 'key' must be specified for the {data_service} data"
                    "  service instead the instruction only includes: "
                    f"{self.instructions['instructions']['apis'][data_service]}"
                )
                api_key = self.instructions["instructions"]["apis"][data_service]["key"]

                # Instantiate the geoapis object for downloading vectors from the data
                # service.
                bounding_polygon = (
                    self.catchment_geometry.catchment
                    if self.catchment_geometry is not None
                    else None
                )
                vector_fetcher = data_services[data_service](
                    api_key, bounding_polygon=bounding_polygon, verbose=True
                )

                vector_instruction = self.instructions["instructions"]["apis"][
                    data_service
                ][key]
                geometry_type = (
                    vector_instruction["geometry_name "]
                    if "geometry_name " in vector_instruction
                    else None
                )

                logging.info(
                    f"Downloading vector layers {vector_instruction['layers']} from the"
                    " {data_service} data service"
                )

                # Cycle through all layers specified - save each & add to the path list
                for layer in vector_instruction["layers"]:
                    # Use the run method to download each layer in turn
                    vector = vector_fetcher.run(layer, geometry_type)

                    # Ensure directory for layer and save vector file
                    layer_dir = cache_dir / str(layer)
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    vector_dir = layer_dir / key
                    vector.to_file(vector_dir)
                    shutil.make_archive(
                        base_name=vector_dir, format="zip", root_dir=vector_dir
                    )
                    shutil.rmtree(vector_dir)
                    paths.append(layer_dir / f"{key}.zip")
        return paths

    def get_lidar_dataset_crs(self, data_service, dataset_name) -> dict:
        """Checks to see if source CRS of an associated LiDAR datasets has be specified
        in the instruction file. If it has been specified, this CRS is returned, and
        will later be used to override the CRS encoded in the LAS files.
        """

        apis_instructions = self.instructions["instructions"]["apis"]

        if (
            self.check_apis(data_service)
            and type(apis_instructions[data_service]) is dict
            and dataset_name in apis_instructions[data_service]
            and type(apis_instructions[data_service][dataset_name]) is dict
        ):
            dataset_instruction = apis_instructions[data_service][dataset_name]

            if (
                "crs" in dataset_instruction
                and "horizontal" in dataset_instruction["crs"]
                and "vertical" in dataset_instruction["crs"]
            ):
                dataset_crs = {
                    "horizontal": dataset_instruction["crs"]["horizontal"],
                    "vertical": dataset_instruction["crs"]["vertical"],
                }
                logging.info(
                    f"The LiDAR dataset {dataset_name} is assumed to have the source "
                    f"coordinate system EPSG: {dataset_crs} as defined in the "
                    "instruction file"
                )
                return dataset_crs
            else:
                logging.info(
                    f"The LiDAR dataset {dataset_name} will use the source the "
                    "coordinate system EPSG defined in its LAZ files"
                )
                return None
        else:
            logging.info(
                f"The LiDAR dataset {dataset_name} will use the source coordinate "
                "system EPSG from its LAZ files"
            )
            return None

    def get_lidar_file_list(self, data_service) -> dict:
        """Return a dictionary with three enties 'file_paths', 'crs' and
        'tile_index_file'. The 'file_paths' contains a list of LiDAR tiles to process.

        The 'crs' (or coordinate system of the LiDAR data as defined by an EPSG code) is
        only optionally set (if unset the value is None). The 'crs' should only be set
        if the CRS information is not correctly encoded in the LAZ/LAS files. Currently
        this is only supported for OpenTopography LiDAR.

        The 'tile_index_file' is also optional (if unset the value is None). The
        'tile_index_file' should be given if a tile index file exists for the LiDAR
        files specifying the extents of each tile. This is currently only supported for
        OpenTopography files.

        If a LiDAR API is specified this is checked and all files within the catchment
        area are downloaded and used to construct the file list. If none is specified,
        the instruction 'data_paths' is checked for 'lidars' and these are returned.
        """

        lidar_dataset_index = 0  # currently only support one LiDAR dataset

        lidar_dataset_info = {}

        # See if 'OpenTopography' or another data_service has been specified as an area
        # to look first
        if self.check_apis(data_service):

            assert self.check_instruction_path("local_cache"), (
                "A 'local_cache' must be specified under the 'file_paths' in the "
                "instruction file if you are going to use an API - like "
                "'open_topography'"
            )

            # download the specified datasets from the data service - then get the
            # local file path
            search_polygon = (
                self.catchment_geometry.catchment
                if self.catchment_geometry is not None
                else None
            )
            self.lidar_fetcher = geoapis.lidar.OpenTopography(
                cache_path=self.get_instruction_path("local_cache"),
                search_polygon=search_polygon,
                verbose=True,
            )
            # Loop through each specified dataset and download it
            for dataset_name in self.instructions["instructions"]["apis"][
                data_service
            ].keys():
                logging.info(f"Fetching dataset: {dataset_name}")
                self.lidar_fetcher.run(dataset_name)
            assert len(self.lidar_fetcher.dataset_prefixes) == 1, (
                "geofabrics currently only supports creating a DEM from only one LiDAR "
                "dataset at a time. Please create an issue if you want support for "
                "mutliple datasets. Error as the following datasets were specified: "
                f"{self.lidar_fetcher.dataset_prefixes}"
            )
            dataset_prefix = self.lidar_fetcher.dataset_prefixes[lidar_dataset_index]
            lidar_dataset_info["file_paths"] = sorted(
                pathlib.Path(self.lidar_fetcher.cache_path / dataset_prefix).glob(
                    "*.laz"
                )
            )
            lidar_dataset_info["crs"] = self.get_lidar_dataset_crs(
                data_service, dataset_prefix
            )
            lidar_dataset_info["tile_index_file"] = (
                self.lidar_fetcher.cache_path
                / dataset_prefix
                / f"{dataset_prefix}_TileIndex.zip"
            )
        else:
            # get the specified file paths from the instructions
            lidar_dataset_info["file_paths"] = self.get_instruction_path("lidars")
            lidar_dataset_info["crs"] = None
            lidar_dataset_info["tile_index_file"] = None
        return lidar_dataset_info

    def create_catchment(self) -> geometry.CatchmentGeometry:
        # create the catchment geometry object
        catchment_dirs = self.get_instruction_path("catchment_boundary")
        assert type(catchment_dirs) is not list, (
            f"A list of catchment_boundary's is provided: {catchment_dirs}, "
            + "where only one is supported."
        )
        catchment_geometry = geometry.CatchmentGeometry(
            catchment_dirs, self.get_crs(), self.get_resolution(), foreshore_buffer=2
        )
        land_dirs = self.get_vector_paths("land")
        assert len(land_dirs) == 1, (
            f"{len(land_dirs)} catchment_boundary's provided, where only one is "
            f"supported. Specficially land_dirs = {land_dirs}."
        )
        catchment_geometry.land = land_dirs[0]
        return catchment_geometry

    @abc.abstractmethod
    def run(self):
        """This method controls the processor execution and code-flow."""

        raise NotImplementedError("NETLOC_API must be instantiated in the child class")


class BathymetryDemGenerator(BaseProcessor):
    """BathymetryDemGenerator executes a pipeline for loading in a Dense DEM and extents
    before interpolating offshore DEM values. The data and pipeline logic is defined in
    the json_instructions file.

    The `BathymetryDemGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the
       generation of a DEM as polygons.
     * dense_dem - Defines the hydrologically conditioned DEM as a combination of tiles
       from LiDAR and interpolated from bathymetry.
     * bathy_contours - This object defines the bathymetry vectors used by the dense_dem
       to define the DEM offshore.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and
    an instruction file
    """

    def __init__(self, json_instructions: json):

        super(BathymetryDemGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.dense_dem = None
        self.bathy_contours = None

    def add_bathymetry(self, area_threshold: float, catchment_dirs: pathlib.Path):
        """Add in any bathymetry data - ocean or river"""

        # Load in bathymetry and interpolate offshore if significant offshore is not
        # covered by LiDAR
        area_without_lidar = self.catchment_geometry.offshore_without_lidar(
            self.dense_dem.extents
        ).geometry.area.sum()
        if (
            self.check_vector("bathymetry_contours")
            and area_without_lidar
            > self.catchment_geometry.offshore.area.sum() * area_threshold
        ):

            # Get the bathymetry data directory
            bathy_contour_dirs = self.get_vector_paths("bathymetry_contours")
            assert len(bathy_contour_dirs) == 1, (
                f"{len(bathy_contour_dirs)} bathymetry_contours's provided. "
                f"Specficially {catchment_dirs}. Support has not yet been added for "
                "multiple datasets."
            )

            logging.info(f"Incorporating Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.bathy_contours = geometry.BathymetryContours(
                bathy_contour_dirs[0],
                self.catchment_geometry,
                z_label=self.get_instruction_general("bathymetry_contours_z_label"),
                exclusion_extent=self.dense_dem.extents,
            )

            # interpolate
            self.dense_dem.interpolate_offshore(self.bathy_contours)
        # Load in river bathymetry and incorporate where discernable at the resolution
        if self.check_vector("river_polygons") and self.check_vector(
            "river_bathymetry"
        ):

            # Get the polygons and bathymetry and can be multiple
            bathy_dirs = self.get_vector_paths("river_bathymetry")
            poly_dirs = self.get_vector_paths("river_polygons")

            logging.info(f"Incorporating river Bathymetry: {bathy_dirs}")

            # Load in bathymetry
            self.river_bathy = geometry.RiverBathymetryPoints(
                points_files=bathy_dirs,
                polygon_files=poly_dirs,
                catchment_geometry=self.catchment_geometry,
                z_labels=self.get_instruction_general("river_bathy_z_label"),
            )

            # Call interpolate river on the DEM - the class checks to see if any pixels
            # actually fall inside the polygon
            self.dense_dem.interpolate_river_bathymetry(
                river_bathymetry=self.river_bathy
            )

    def run(self):
        """This method executes the geofabrics generation pipeline to produce geofabric
        derivatives."""

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = 10.0 / 100  # Used to decide if bathymetry should be included

        # create the catchment geometry object
        self.catchment_geometry = self.create_catchment()

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDemFromFiles(
            catchment_geometry=self.catchment_geometry,
            dense_dem_path=self.get_instruction_path("dense_dem"),
            extents_path=self.get_instruction_path("dense_dem_extents"),
            interpolation_method=self.get_instruction_general("interpolation_method"),
        )

        # Check for and add any bathymetry information
        self.add_bathymetry(
            area_threshold=area_threshold,
            catchment_dirs=self.get_instruction_path("catchment_boundary"),
        )

        # fill combined dem - save results
        self.dense_dem.dem.to_netcdf(
            self.get_instruction_path("result_dem"), format="NETCDF4", engine="netcdf4"
        )


class LidarDemGenerator(BathymetryDemGenerator):
    """LidarDemGenerator executes a pipeline for creating a hydrologically conditioned
    DEM from LiDAR and optionally a reference DEM and/or bathymetry contours. The data
    and pipeline logic is defined in the json_instructions file.

    The `DemGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the
       generation of a DEM as polygons.
     * dense_dem - Defines the hydrologically conditioned DEM as a combination of tiles
       from LiDAR and interpolated from bathymetry.
     * reference_dem - This optional object defines a background DEM that may be used to
       fill on land gaps in the LiDAR.
     * bathy_contours - This optional object defines the bathymetry vectors used by the
       dense_dem to define the DEM offshore.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and
    an instruction file.
    """

    def __init__(self, json_instructions: json):

        super(LidarDemGenerator, self).__init__(json_instructions=json_instructions)

        self.dense_dem = None
        self.reference_dem = None
        self.bathy_contours = None

    def create_metadata(self) -> dict:
        """A clase to create metadata to be added as netCDF attributes."""
        metadata = {
            "library_name": "GeoFabrics",
            "library_version": version.__version__,
            "class_name": self.__class__.__name__,
            "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "instructions": self.instructions,
        }
        return metadata

    def run(self):
        """This method executes the geofabrics generation pipeline to produce geofabric
        derivatives.

        Note it currently only considers one LiDAR dataset that can have many tiles.
        See 'get_lidar_file_list' for where to change this."""

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = 10.0 / 100  # Used to decide if bathymetry should be included

        # create the catchment geometry object
        self.catchment_geometry = self.create_catchment()

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_dataset_info = self.get_lidar_file_list("open_topography")

        # setup dense DEM and catchment LiDAR objects
        self.dense_dem = dem.DenseDemFromTiles(
            catchment_geometry=self.catchment_geometry,
            drop_offshore_lidar=self.get_instruction_general("drop_offshore_lidar"),
            interpolation_method=self.get_instruction_general("interpolation_method"),
            lidar_interpolation_method=self.get_instruction_general(
                "lidar_interpolation_method"
            ),
            elevation_range=self.get_instruction_general("elevation_range"),
        )

        # Setup Dask cluster and client
        cluster_kwargs = {
            "n_workers": self.get_processing_instructions("number_of_cores"),
            "threads_per_worker": 1,
            "processes": True,
        }
        with distributed.LocalCluster(**cluster_kwargs) as cluster, distributed.Client(
            cluster
        ) as client:
            print(client)
            # Load in LiDAR tiles
            self.dense_dem.add_lidar(
                lidar_files=lidar_dataset_info["file_paths"],
                source_crs=lidar_dataset_info["crs"],
                drop_offshore_lidar=self.get_instruction_general("drop_offshore_lidar"),
                lidar_classifications_to_keep=self.get_instruction_general(
                    "lidar_classifications_to_keep"
                ),
                tile_index_file=lidar_dataset_info["tile_index_file"],
                chunk_size=self.get_processing_instructions("chunk_size"),
                metadata=self.create_metadata(),
            )  # Note must be called after all others if it is to be complete
        # Load in reference DEM if any significant land/foreshore not covered by LiDAR
        if self.check_instruction_path("reference_dems"):
            area_without_lidar = (
                self.catchment_geometry.land_and_foreshore_without_lidar(
                    self.dense_dem.extents
                ).geometry.area.sum()
            )
            if (
                area_without_lidar
                > self.catchment_geometry.land_and_foreshore.area.sum() * area_threshold
            ):

                assert len(self.get_instruction_path("reference_dems")) == 1, (
                    f"{len(self.get_instruction_path('reference_dems'))} reference_dems"
                    " specified, but only one supported currently. reference_dems: "
                    f"{self.get_instruction_path('reference_dems')}"
                )

                logging.info(
                    "Incorporating background DEM: "
                    f"{self.get_instruction_path('reference_dems')}"
                )

                # Load in background DEM - cut away within the LiDAR extents
                self.reference_dem = dem.ReferenceDem(
                    dem_file=self.get_instruction_path("reference_dems")[0],
                    catchment_geometry=self.catchment_geometry,
                    set_foreshore=self.get_instruction_general("set_dem_shoreline"),
                    exclusion_extent=self.dense_dem.extents,
                )

                # Add the reference DEM patch where there's no LiDAR to the dense DEM
                # without updting the extents
                self.dense_dem.add_reference_dem(
                    tile_points=self.reference_dem.points,
                    tile_extent=self.reference_dem.extents,
                )
        # save dense DEM and extents
        self.dense_dem.dense_dem.to_netcdf(
            self.get_instruction_path("dense_dem"), format="NETCDF4", engine="netcdf4"
        )
        if self.dense_dem.extents is not None:
            self.dense_dem.extents.to_file(
                self.get_instruction_path("dense_dem_extents")
            )
        else:
            logging.warning(
                "In processor.DemGenerator - no LiDAR extents exist so no extents file "
                "written"
            )
        # Check for and add any bathymetry information
        self.add_bathymetry(
            area_threshold=area_threshold,
            catchment_dirs=self.get_instruction_path("catchment_boundary"),
        )

        # fill combined dem - save results
        self.dense_dem.dem.to_netcdf(
            self.get_instruction_path("result_dem"), format="NETCDF4", engine="netcdf4"
        )


class RiverBathymetryGenerator(BaseProcessor):
    """RiverbathymetryGenerator executes a pipeline to estimate river
    bathymetry depths from flows, slopes, friction and widths along a main
    channel. This is dones by first creating a hydrologically conditioned DEM
    of the channel. A json_instructions file defines the pipeline logic and
    data.

    Attributes:
        channel_polyline  The main channel along which to estimate depth. This
            is a polyline.
        gen_dem  The ground DEM generated along the main channel. This is a
            raster.
        veg_dem  The vegetation DEM generated along the main channel. This is a
            raster.
        aligned_channel_polyline  The main channel after its alignment has been
            updated based on the DEM. Width and slope are estimated on this.
        transects  Transect polylines perpindicular to the aligned channel with
            samples of the DEM values
    """

    def __init__(self, json_instructions: json, debug: bool = True):

        super(RiverBathymetryGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.debug = debug

    def channel_characteristics_exist(self) -> bool:
        """Return true if the DEMs are required for later processing"""

        # Check if channel needs to be aligned or widths calculated
        river_characteristics_file = self.get_result_file_path(
            key="river_characteristics"
        )
        river_polygon_file = self.get_result_file_path(key="river_polygon")

        return river_characteristics_file.is_file() and river_polygon_file.is_file()

    def channel_bathymetry_exist(self) -> bool:
        """Return true if the river channel and bathymetry files exist."""

        # Check if the expected bathymetry and polygon files exist
        river_bathymetry_file = self.get_result_file_path(key="river_bathymetry")
        river_polygon_file = self.get_result_file_path(key="river_polygon")
        fan_bathymetry_file = self.get_result_file_path(key="fan_bathymetry")
        fan_polygon_file = self.get_result_file_path(key="fan_polygon")
        return (
            river_bathymetry_file.is_file()
            and river_polygon_file.is_file()
            and fan_bathymetry_file.is_file()
            and fan_polygon_file.is_file()
        )

    def alignment_exists(self) -> bool:
        """Return true if the DEMs are required for later processing


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        # Check if channel needs to be aligned or widths calculated
        aligned_channel_file = self.get_result_file_path(key="aligned")

        return aligned_channel_file.is_file()

    def get_result_file_name(self, key: str = None, name: str = None) -> str:
        """Return the file name of the file to save.


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        if key is not None and name is not None:
            assert False, "Only either a key or a name can be provided"
        if key is None and name is None:
            assert False, "Either a key or a name must be provided"
        if key is not None:
            area_threshold = self.get_bathymetry_instruction("channel_area_threshold")

            # key to output name mapping
            name_dictionary = {
                "aligned": f"aligned_channel_{area_threshold}.geojson",
                "river_characteristics": "river_characteristics.geojson",
                "river_polygon": "river_polygon.geojson",
                "river_bathymetry": "river_bathymetry.geojson",
                "fan_bathymetry": "fan_bathymetry.geojson",
                "fan_polygon": "fan_polygon.geojson",
                "gnd_dem": f"channel_dem_{area_threshold}.nc",
                "veg_dem": f"channel_veg_dem_{area_threshold}.nc",
                "catchment": f"channel_catchment_{area_threshold}.geojson",
                "rec_channel": f"rec_channel_{area_threshold}.geojson",
                "rec_channel_smoothed": f"rec_channel_{area_threshold}_smoothed.geojson",
            }
            return name_dictionary[key]
        else:
            return name

    def get_result_file_path(self, key: str = None, name: str = None) -> pathlib.Path:
        """Return the file name of the file to save with the local cache path.


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        local_cache = pathlib.Path(
            self.instructions["instructions"]["data_paths"]["local_cache"]
        )

        name = self.get_result_file_name(key=key, name=name)

        return local_cache / name

    def get_bathymetry_instruction(self, key: str):
        """Return true if the DEMs are required for later processing


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        return self.instructions["instructions"]["channel_bathymetry"][key]

    def get_rec_channel(self) -> bathymetry_estimation.Channel:
        """Read in or create a rec channel."""

        # Get instructions
        crs = self.get_crs()["horizontal"]
        area_threshold = self.get_bathymetry_instruction("channel_area_threshold")
        channel_rec_id = self.get_bathymetry_instruction("channel_rec_id")
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")

        # Check if file exists
        rec_name = self.get_result_file_path(key="rec_channel")
        if rec_name.is_file():
            channel = bathymetry_estimation.Channel(
                channel=geopandas.read_file(rec_name),
                resolution=cross_section_spacing,
                sampling_direction=-1,
            )
        else:
            # Else, create if it doesn't exist
            rec_network = geopandas.read_file(
                self.get_bathymetry_instruction("rec_file")
            ).to_crs(crs)
            channel = bathymetry_estimation.Channel.from_rec(
                rec_network=rec_network,
                reach_id=channel_rec_id,
                resolution=cross_section_spacing,
                area_threshold=area_threshold,
            )

            if self.debug:
                # Save the REC channel and smoothed REC channel if not already
                rec_name = self.get_result_file_path(key="rec_channel")
                if not rec_name.is_file():
                    channel.channel.to_file(rec_name)
                smoothed_rec_name = self.get_result_file_path(
                    key="rec_channel_smoothed"
                )
                if not smoothed_rec_name.is_file():
                    channel.get_sampled_spline_fit().to_file(smoothed_rec_name)
        return channel

    def get_dems(self, buffer: float, channel: geometry.CatchmentGeometry) -> tuple:
        """Allow selection of the ground or vegetation DEM, and either create
        or load it.


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        instruction_paths = self.instructions["instructions"]["data_paths"]

        # Extract instructions from JSON
        max_channel_width = self.get_bathymetry_instruction("max_channel_width")
        rec_alignment_tolerance = self.get_bathymetry_instruction(
            "rec_alignment_tolerance"
        )

        # Define ground and veg files
        gnd_file = self.get_result_file_path(key="gnd_dem")
        veg_file = self.get_result_file_path(key="veg_dem")

        # Ensure channel catchment exists and is up to date if needed
        if not gnd_file.is_file() or not veg_file.is_file():
            catchment_file = self.get_result_file_path(key="catchment")
            instruction_paths["catchment_boundary"] = str(
                self.get_result_file_name(key="catchment")
            )
            corridor_radius = max_channel_width / 2 + rec_alignment_tolerance + buffer
            channel_catchment = channel.get_channel_catchment(
                corridor_radius=corridor_radius
            )
            channel_catchment.to_file(catchment_file)
        # Remove bathymetry contour information if it exists while creating DEMs
        bathy_data_paths = None
        bathy_apis = None
        if "bathymetry_contours" in instruction_paths:
            bathy_data_paths = instruction_paths.pop("bathymetry_contours")
        if "bathymetry_contours" in self.instructions["instructions"]["apis"]["linz"]:
            bathy_apis = self.instructions["instructions"]["apis"]["linz"].pop(
                "bathymetry_contours"
            )
        # Get the ground DEM
        if not gnd_file.is_file():
            # Create the ground DEM file if this has not be created yet!
            print("Generating ground DEM.")
            instruction_paths["result_dem"] = str(
                self.get_result_file_name(key="gnd_dem")
            )
            instruction_paths["dense_dem"] = "dense_gnd_dem.nc"
            instruction_paths["dense_dem_extents"] = "dense_gnd_extents.geojson"
            runner = LidarDemGenerator(self.instructions)
            runner.run()
            gnd_dem = runner.dense_dem.dem
            instruction_paths.pop("dense_dem")
            instruction_paths.pop("dense_dem_extents")
            instruction_paths.pop("result_dem")
        else:
            print("Loading ground DEM.")  # drop band added by rasterio.open()
            gnd_dem = rioxarray.rioxarray.open_rasterio(gnd_file, masked=True).squeeze(
                "band", drop=True
            )
        # Get the vegetation DEM
        if not veg_file.is_file():
            # Create the catchment file if this has not be created yet!
            print("Generating vegetation DEM.")
            instruction_paths["result_dem"] = str(
                self.get_result_file_name(key="veg_dem")
            )
            self.instructions["instructions"]["general"][
                "lidar_classifications_to_keep"
            ] = self.get_bathymetry_instruction("veg_lidar_classifications_to_keep")
            instruction_paths["dense_dem"] = "dense_veg_dem.nc"
            instruction_paths["dense_dem_extents"] = "dense_veg_extents.geojson"
            runner = LidarDemGenerator(self.instructions)
            runner.run()
            veg_dem = runner.dense_dem.dem
            instruction_paths.pop("dense_dem")
            instruction_paths.pop("dense_dem_extents")
            instruction_paths.pop("result_dem")
        else:
            print("Loading the vegetation DEM.")  # drop band added by rasterio.open()
            veg_dem = dem.rioxarray.rioxarray.open_rasterio(
                veg_file, masked=True
            ).squeeze("band", drop=True)
        # Replace bathymetry contour information if it exists
        if bathy_data_paths is not None:
            instruction_paths["bathymetry_contours"] = bathy_data_paths
        if bathy_apis is not None:
            self.instructions["instructions"]["apis"]["linz"][
                "bathymetry_contours"
            ] = bathy_apis
        return gnd_dem, veg_dem

    def align_channel(
        self,
        channel_width: bathymetry_estimation.ChannelCharacteristics,
        channel: bathymetry_estimation.Channel,
        buffer: float,
    ) -> geopandas.GeoDataFrame:
        """Align the REC defined channel based on LiDAR and save the aligned
        channel.


        Parameters:
            channel_width  The class for characterising channel width and other
                properties
            channel  The REC defined channel alignment
        """

        # Get instruciton parameters
        max_channel_width = self.get_bathymetry_instruction("max_channel_width")
        min_channel_width = self.get_bathymetry_instruction("min_channel_width")
        rec_alignment_tolerance = self.get_bathymetry_instruction(
            "rec_alignment_tolerance"
        )

        bank_threshold = self.get_bathymetry_instruction("min_bank_height")
        width_centre_smoothing_multiplier = self.get_bathymetry_instruction(
            "width_centre_smoothing"
        )

        # The width of cross sections to sample
        corridor_radius = max_channel_width / 2 + rec_alignment_tolerance + buffer

        aligned_channel, sampled_cross_sections = channel_width.align_channel(
            threshold=bank_threshold,
            min_channel_width=min_channel_width,
            initial_channel=channel,
            search_radius=rec_alignment_tolerance,
            width_centre_smoothing_multiplier=width_centre_smoothing_multiplier,
            cross_section_radius=corridor_radius,
        )

        # Save out results
        aligned_channel_file = self.get_result_file_path(key="aligned")
        aligned_channel.to_file(aligned_channel_file)
        if self.debug:
            sampled_cross_sections[
                ["width_line", "valid", "channel_count"]
            ].set_geometry("width_line").to_file(
                self.get_result_file_path(name="initial_widths.geojson")
            )
            sampled_cross_sections[["geometry", "channel_count", "valid"]].to_file(
                self.get_result_file_path(name="initial_cross_sections.geojson")
            )
        return aligned_channel

    def calculate_channel_characteristics(
        self,
        channel_width: bathymetry_estimation.ChannelCharacteristics,
        aligned_channel: geopandas.GeoDataFrame,
        buffer: float,
    ) -> tuple:
        """Align the REC defined channel based on LiDAR and save the aligned
        channel.


        Parameters:
            channel_width  The class for characterising channel width and other
                properties
            channel  The REC defined channel alignment
        """

        # Get instruciton parameters
        max_channel_width = self.get_bathymetry_instruction("max_channel_width")
        min_channel_width = self.get_bathymetry_instruction("min_channel_width")
        bank_threshold = self.get_bathymetry_instruction("min_bank_height")
        max_bank_height = self.get_bathymetry_instruction("max_bank_height")
        width_centre_smoothing_multiplier = self.get_bathymetry_instruction(
            "width_centre_smoothing"
        )
        rec_alignment_tolerance = self.get_bathymetry_instruction(
            "rec_alignment_tolerance"
        )

        corridor_radius = max_channel_width / 2 + buffer

        sampled_cross_sections, river_polygon = channel_width.estimate_width_and_slope(
            aligned_channel=aligned_channel,
            threshold=bank_threshold,
            cross_section_radius=corridor_radius,
            search_radius=rec_alignment_tolerance,
            min_channel_width=min_channel_width,
            max_threshold=max_bank_height,
            river_polygon_smoothing_multiplier=width_centre_smoothing_multiplier,
        )

        river_polygon.to_file(self.get_result_file_path("river_polygon"))
        columns = ["geometry"]
        columns.extend(
            [
                column_name
                for column_name in sampled_cross_sections.columns
                if "slope" in column_name
                or "widths" in column_name
                or "min_z" in column_name
                or "threshold" in column_name
                or "valid" in column_name
                or "channel_count" in column_name
            ]
        )
        sampled_cross_sections.set_geometry("river_polygon_midpoint", drop=True)[
            columns
        ].to_file(self.get_result_file_path(key="river_characteristics"))

        if self.debug:
            # Write out optional outputs
            sampled_cross_sections[columns].to_file(
                self.get_result_file_path(name="final_cross_sections.geojson")
            )
            sampled_cross_sections.set_geometry("width_line", drop=True)[
                ["geometry", "valid"]
            ].to_file(self.get_result_file_path(name="final_widths.geojson"))
            sampled_cross_sections.set_geometry("flat_midpoint", drop=True)[
                columns
            ].to_file(self.get_result_file_path(name="final_flat_midpoints.geojson"))

    def characterise_channel(
        self, buffer: float
    ) -> bathymetry_estimation.ChannelCharacteristics:
        """Calculate the channel width, slope and other characteristics. This requires a
        ground and vegetation DEM. This also may require alignment of the channel
        centreline.


        Parameters:
            buffer  The amount of extra space to create around the river catchment
        """

        logging.info("The channel hasn't been characerised. Charactreising now.")

        # Extract instructions
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        resolution = self.get_resolution()

        # Create REC defined channel
        channel = self.get_rec_channel()

        # Get DEMs - create and save if don't exist
        gnd_dem, veg_dem = self.get_dems(buffer=buffer, channel=channel)

        # Create the channel width object
        channel_width = bathymetry_estimation.ChannelCharacteristics(
            gnd_dem=gnd_dem,
            veg_dem=veg_dem,
            cross_section_spacing=cross_section_spacing,
            resolution=resolution,
            debug=self.debug,
        )

        # Align channel if required
        if not self.alignment_exists():
            print("No aligned channel provided. Aligning the channel.")

            # Align and save the REC defined channel
            aligned_channel = self.align_channel(
                channel_width=channel_width, channel=channel, buffer=buffer
            )
        else:
            aligned_channel_file = self.get_result_file_path(key="aligned")
            aligned_channel = geopandas.read_file(aligned_channel_file)
        # calculate the channel width and save results
        print("Characterising the aligned channel.")
        self.calculate_channel_characteristics(
            channel_width=channel_width, aligned_channel=aligned_channel, buffer=buffer
        )

    def calculate_river_bed_elevations(self):
        """Calculate and save depth estimates along the channel using various
        approaches.

        """

        # Read in the flow file and calcaulate the depths - write out the results
        width_values = geopandas.read_file(
            self.get_result_file_path(key="river_characteristics")
        )
        flow = pandas.read_csv(self.get_bathymetry_instruction("flow_file"))
        channel = self.get_rec_channel()

        # Match each channel midpoint to a nzsegment ID - based on what channel reach is
        # closest
        width_values["nzsegment"] = (
            numpy.ones(len(width_values["widths"]), dtype=float) * numpy.nan
        )
        for i, row in width_values.iterrows():
            if row.geometry is not None and not row.geometry.is_empty:
                distances = channel.channel.distance(width_values.loc[i].geometry)
                width_values.loc[i, ("nzsegment")] = channel.channel[
                    distances == distances.min()
                ]["nzsegment"].min()
        # Fill in any missing values
        width_values["nzsegment"] = (
            width_values["nzsegment"].fillna(method="ffill").fillna(method="bfill")
        )
        width_values["nzsegment"] = width_values["nzsegment"].astype("int")

        # Add the friction and flow values to the widths and slopes
        width_values["mannings_n"] = numpy.zeros(
            len(width_values["nzsegment"]), dtype=int
        )
        width_values["flow"] = numpy.zeros(len(width_values["nzsegment"]), dtype=int)
        for nzsegment in width_values["nzsegment"].unique():
            width_values.loc[
                width_values["nzsegment"] == nzsegment, ("mannings_n")
            ] = flow[flow["nzsegment"] == nzsegment]["n"].unique()[0]
            width_values.loc[width_values["nzsegment"] == nzsegment, ("flow")] = flow[
                flow["nzsegment"] == nzsegment
            ]["flow"].unique()[0]
        # Names of values to use
        slope_name = "slope_mean_2.0km"
        min_z_name = "min_z_centre_unimodal"
        width_name = "widths_mean_0.25km"
        flat_width_name = "flat_widths_mean_0.25km"
        threshold_name = "thresholds_mean_0.25km"

        # Calculate depths and bed elevation using the Neal et al approach (Uniform flow
        # theory)
        full_bank_depth = self._calculate_neal_et_al_depth(
            width_values=width_values,
            width_name=width_name,
            slope_name=slope_name,
            threshold_name=threshold_name,
        )
        active_channel_bank_depth = self._convert_full_bank_to_channel_depth(
            full_bank_depth=full_bank_depth,
            threshold_name=threshold_name,
            flat_width_name=flat_width_name,
            full_bank_width_name=width_name,
            width_values=width_values,
        )
        width_values["bed_elevation_Neal_et_al"] = (
            width_values[min_z_name] - active_channel_bank_depth
        )
        if self.debug:
            # Optionally write out additional depth information
            width_values["depth_Neal_et_al"] = active_channel_bank_depth
            width_values["flood_depth_Neal_et_al"] = full_bank_depth
        # Calculate depths and bed elevation using the Rupp & Smart approach (Hydrologic
        # geometry)
        full_bank_depth = self._calculate_rupp_and_smart_depth(
            width_values=width_values,
            width_name=width_name,
            slope_name=slope_name,
            threshold_name=threshold_name,
        )
        active_channel_bank_depth = self._convert_full_bank_to_channel_depth(
            full_bank_depth=full_bank_depth,
            threshold_name=threshold_name,
            flat_width_name=flat_width_name,
            full_bank_width_name=width_name,
            width_values=width_values,
        )
        width_values["bed_elevation_Rupp_and_Smart"] = (
            width_values[min_z_name] - active_channel_bank_depth
        )
        if self.debug:
            # Optionally write out additional depth information
            width_values["depth_Rupp_and_Smart"] = active_channel_bank_depth
            width_values["flood_depth_Rupp_and_Smart"] = full_bank_depth
        # Save the bed elevations
        values_to_save = [
            "geometry",
            "bed_elevation_Neal_et_al",
            "bed_elevation_Rupp_and_Smart",
            "widths",
            width_name,
            flat_width_name,
        ]
        if self.debug:
            # Optionally write out additional depth information
            values_to_save.extend(
                [
                    "depth_Neal_et_al",
                    "depth_Rupp_and_Smart",
                    "flood_depth_Neal_et_al",
                    "flood_depth_Rupp_and_Smart",
                ]
            )
        # Save the widths and depths
        width_values[values_to_save].to_file(
            self.get_result_file_path(key="river_bathymetry")
        )

    def _calculate_neal_et_al_depth(
        self, width_values, width_name, slope_name, threshold_name
    ):
        """Calculate the uniform flow theory depth estimate as laid out in Neal
        et al.

        Parameters:
            width_values  A dataframe of channel charateristics.
            width_name  The name of the channel width column.
            slope_name  The name of the down-river channel slope column.
            threshold_name The name of the bank height threshold column."""

        full_bank_depth = (
            width_values["mannings_n"]
            * width_values["flow"]
            / (numpy.sqrt(width_values[slope_name]) * width_values[width_name])
        ) ** (3 / 5)
        return full_bank_depth

    def _calculate_rupp_and_smart_depth(
        self, width_values, width_name, slope_name, threshold_name
    ):
        """Calculate the hydrolic geometry depth estimate as laid out in Rupp
        and Smart.

        Parameters:
            width_values  A dataframe of channel charateristics.
            width_name  The name of the channel width column.
            slope_name  The name of the down-river channel slope column.
            threshold_name The name of the bank height threshold column."""

        a = 0.745
        b = 0.305
        K_0 = 6.16
        full_bank_depth = (
            width_values["flow"]
            / (K_0 * width_values[width_name] * width_values[slope_name] ** b)
        ) ** (1 / (1 + a))
        return full_bank_depth

    def _convert_full_bank_to_channel_depth(
        self,
        full_bank_depth,
        threshold_name,
        flat_width_name,
        full_bank_width_name,
        width_values,
    ):
        """Calculate the depth of the channel as detected in the LiDAR derived
        DEM. Remove the above water area, then calculate the depth for the
        'flat water' width as derived in the LiDAR.

        Parameters:
            full_bank_depth  The flood depth from the flood water height
            width_values  A dataframe of channel charateristics.
            full_bank_width_name  The name of the full bank width column.
            flat_width_name  The name of the down-river channel slope column.
            threshold_name The name of the bank height threshold column."""

        # The depth of flood water above the water surface
        flood_depth = width_values[threshold_name]

        # Calculate the area estimated for full bank width flow
        full_flood_area = full_bank_depth * width_values[full_bank_width_name]

        # Calculate the flood waters (i.e. flowing above the water surface), but with a
        # correction for the exposed river banks
        above_water_area = flood_depth * width_values[full_bank_width_name]
        exposed_bank_area = (
            flood_depth - self.get_bathymetry_instruction("min_bank_height")
        ) * (width_values[full_bank_width_name] - width_values[flat_width_name])
        assert (exposed_bank_area >= 0).all(), "The exposed bank area must be postive"
        extra_flood_area = above_water_area - exposed_bank_area

        # The area to convert to the active 'flat' channel depth
        flat_flow_area = full_flood_area - extra_flood_area

        # Calculate the depth from the area
        flat_flow_depth = flat_flow_area / width_values[flat_width_name]

        return flat_flow_depth

    def estimate_river_mouth_fan(self):
        """Calculate and save depth estimates along the river mouth fan."""

        # Required inputs
        crs = self.get_crs()["horizontal"]
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        river_bathymetry_file = self.get_result_file_path(key="river_bathymetry")
        ocean_contour_file = self.get_vector_paths("bathymetry_contours")[0]
        aligned_channel_file = self.get_result_file_path(key="aligned")
        ocean_contour_depth_label = self.get_instruction_general(
            "bathymetry_contours_z_label"
        )

        # Create fan object
        fan = geometry.RiverMouthFan(
            aligned_channel_file=aligned_channel_file,
            river_bathymetry_file=river_bathymetry_file,
            ocean_contour_file=ocean_contour_file,
            crs=crs,
            cross_section_spacing=cross_section_spacing,
            ocean_contour_depth_label=ocean_contour_depth_label,
        )

        # Estimate the fan extents and bathymetry
        fan_polygon, fan_bathymetry = fan.polygon_and_bathymetry()
        fan_polygon.to_file(self.get_result_file_path(key="fan_polygon"))
        fan_bathymetry.to_file(self.get_result_file_path(key="fan_bathymetry"))

    def run(self, instruction_parameters):
        """This method extracts a main channel then executes the DemGeneration
        pipeline to produce a DEM before sampling this to extimate width, slope
        and eventually depth."""

        logging.info("Adding river and fan bathymetry if it doesn't already" "exist.")

        # Characterise river channel if not already done - may generate DEMs
        if not self.channel_characteristics_exist():
            buffer = 50
            self.characterise_channel(buffer=buffer)
        # Estimate channel and fan depths if not already done
        if not self.channel_bathymetry_exist():
            logging.info("Estimating the channel and fan bathymetry.")
            print("Estimating the channel and fan bathymetry.")

            # Calculate and save river bathymetry depths
            self.calculate_river_bed_elevations()
            self.estimate_river_mouth_fan()
        # Update parameter file - in time only update the bits that have been re-run
        with open(instruction_parameters, "w") as file_pointer:
            json.dump(self.instructions, file_pointer)


class DrainBathymetryGenerator(BaseProcessor):
    """DrainBathymetryGenerator executes a pipeline to pull in OpenStreetMap drain and
    tunnel information. A DEM is generated of the surrounding area and this used to
    unblock drains and tunnels.

    """

    OSM_CRS = "EPSG:4326"

    def __init__(self, json_instructions: json, debug: bool = True):

        super(DrainBathymetryGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.debug = debug

    def get_result_file_name(self, key: str) -> str:
        """Return the name of the file to save."""

        drain_width = self.instructions["instructions"]["drains"]["width"]
        tag = f"{drain_width}m_width"

        # key to output name mapping
        name_dictionary = {
            "dem": f"dem_{tag}.nc",
            "open_polygon": f"open_drain_polygon_{tag}.geojson",
            "open_elevation": f"open_drain_elevation_{tag}.geojson",
            "closed_polygon": f"closed_drain_polygon_{tag}.geojson",
            "closed_elevation": f"closed_drain_elevation_{tag}.geojson",
            "drain_polygon": f"drain_polygon_{tag}.geojson",
        }
        return name_dictionary[key]

    def get_result_file_path(self, key: str) -> pathlib.Path:
        """Return the file name of the file to save with the local cache path.

        Parameters:
            instructions  The json instructions defining the behaviour
        """

        local_cache = pathlib.Path(
            self.instructions["instructions"]["data_paths"]["local_cache"]
        )

        name = self.get_result_file_name(key=key)

        return local_cache / name

    # Get the min in each polygon
    def minimum_elevation_in_polygon(
        self, geometry: shapely.geometry.Polygon, dem: xarray.Dataset
    ):
        """Select only coordinates within the polygon bounding box before clipping
        to the bounding box and then returning the minimum elevation."""

        # Index in polygon bbox
        bbox = geometry.bounds

        # Select only DEM within the geometry bounding box
        small_z = dem.z.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))

        # clip to polygon and return minimum elevation
        return float(small_z.rio.clip([geometry]).min())

    def estimate_closed_bathymetry(
        self, drains: geopandas.GeoDataFrame, dem: xarray.Dataset
    ):
        """Sample the DEM around the tunnels to estimate the bed elevation."""

        # Check if already generated
        polygon_file = self.get_result_file_path(key="closed_polygon")
        elevation_file = self.get_result_file_path(key="closed_elevation")
        if polygon_file.is_file() and elevation_file.is_file():
            print("Closed drains already recorded. ")
            logging.info(
                "Estimating closed drain and tunnel bed elevation from OpenStreetMap."
            )
            return
        drain_width = self.instructions["instructions"]["drains"]["width"]

        closed_drains = drains[drains["tunnel"]]
        closed_drains = closed_drains.clip(self.catchment_geometry.catchment)
        closed_drains["polygon"] = closed_drains.buffer(drain_width)

        # save out the polygons
        closed_drains.set_geometry("polygon", drop=True)[["geometry"]].to_file(
            polygon_file
        )

        elevations = closed_drains.apply(
            lambda row: self.minimum_elevation_in_polygon(
                geometry=row["polygon"], dem=dem
            ),
            axis=1,
        )

        # Create sampled points
        points = closed_drains["geometry"].apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    # Ensure even spacing across the length of the drain
                    row.interpolate(
                        i * row.length / int(numpy.ceil(row.length / drain_width))
                    )
                    for i in range(int(numpy.ceil(row.length / drain_width)) + 1)
                ]
            )
        )
        points = geopandas.GeoDataFrame(
            {
                "elevation": elevations,
                "geometry": points,
            },
            crs=2193,
        )

        # Save bathymetry
        points.explode(ignore_index=True).to_file(elevation_file)

    def estimate_open_bathymetry(
        self, drains: geopandas.GeoDataFrame, dem: xarray.Dataset
    ):
        """Sample the DEM along the open waterways to enforce a decreasing elevation."""

        # Check if already generated
        polygon_file = self.get_result_file_path(key="open_polygon")
        elevation_file = self.get_result_file_path(key="open_elevation")
        if polygon_file.is_file() and elevation_file.is_file():
            print("Open drains already recorded. ")
            logging.info(
                "Estimating open drain and tunnel bed elevation from OpenStreetMap."
            )
            return
        drain_width = self.instructions["instructions"]["drains"]["width"]

        open_drains = drains[numpy.logical_not(drains["tunnel"])]
        open_drains = open_drains.clip(self.catchment_geometry.catchment)

        # save out the polygons
        open_drains.buffer(drain_width).to_file(polygon_file)

        # sample the ends of the drain - sample over a polygon at each end
        polygons = open_drains.interpolate(0).buffer(drain_width)
        open_drains["start_elevation"] = polygons.apply(
            lambda geometry: self.minimum_elevation_in_polygon(
                geometry=geometry, dem=dem
            )
        )
        polygons = open_drains.interpolate(open_drains.length).buffer(drain_width)
        open_drains["end_elevation"] = polygons.geometry.apply(
            lambda geometry: self.minimum_elevation_in_polygon(
                geometry=geometry, dem=dem
            )
        )

        # Sample down-slope location along each line
        def sample_location_down_slope(row, drain_width):
            """Sample evenly space poinst along polylines in the downslope direction"""

            if row["start_elevation"] > row["end_elevation"]:
                sample_range = range(
                    int(numpy.ceil(row.geometry.length / drain_width)) + 1
                )
            else:
                sample_range = range(
                    int(numpy.ceil(row.geometry.length / drain_width)), -1, -1
                )
            sampled_multipoints = shapely.geometry.MultiPoint(
                [
                    # Ensure even spacing across the length of the drain
                    row.geometry.interpolate(
                        i
                        * row.geometry.length
                        / int(numpy.ceil(row.geometry.length / drain_width))
                    )
                    for i in sample_range
                ]
            )

            return sampled_multipoints

        points = open_drains.apply(
            lambda row: sample_location_down_slope(row=row, drain_width=drain_width),
            axis=1,
        )

        # Sample elevation enforcing no local elevation gain
        bathymetries = []
        for drain_index, row in enumerate(points):
            row_bathymetries = [
                max(
                    open_drains.iloc[drain_index]["start_elevation"],
                    open_drains.iloc[drain_index]["end_elevation"],
                )
            ]
            for point in row.geoms[1:]:
                elevation = float(dem.z.sel(x=point.x, y=point.y, method="nearest"))
                row_bathymetries.append(
                    elevation
                    if elevation < row_bathymetries[-1]
                    else row_bathymetries[-1]
                )
            bathymetries.extend(row_bathymetries)
        points = geopandas.GeoDataFrame(
            {
                "elevation": bathymetries,
                "geometry": points.explode(ignore_index=True),
            },
            crs=open_drains.crs,
        )

        # Save bathymetry
        points.to_file(elevation_file)

    def create_dem(self, drains: geopandas.GeoDataFrame) -> xarray.Dataset:
        """Create and return a DEM at a resolution 1.5x the drain width."""

        dem_file = self.get_result_file_path(key="dem")

        # Load already created DEM file in
        if dem_file.is_file():
            dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True).squeeze(
                "band", drop=True
            )
        else:  # Create DEM over the drain region

            drain_width = self.instructions["instructions"]["drains"]["width"]

            # Save out the drain polygons as a file with a single multipolygon
            drain_polygon_file = self.get_result_file_path(key="drain_polygon")
            drain_polygon = drains.buffer(drain_width)
            drain_polygon = geopandas.GeoDataFrame(
                geometry=[shapely.ops.unary_union(drain_polygon.geometry.array)],
                crs=drain_polygon.crs,
            )
            drain_polygon = drain_polygon.clip(self.catchment_geometry.catchment)
            drain_polygon.to_file(
                drain_polygon_file
            )  # gpd.overlay(g1, g1, how='union')

            # Create DEM generation instructions
            dem_instructions = self.instructions
            dem_instruction_paths = dem_instructions["instructions"]["data_paths"]
            dem_instruction_paths["catchment_boundary"] = str(drain_polygon_file)
            dem_instruction_paths["result_dem"] = str(dem_file)
            dem_instruction_paths["dense_dem"] = f"dense_dem_{drain_width}m_width.nc"
            dem_instruction_paths[
                "dense_dem_extents"
            ] = "dense_extents_{drain_width}m_width.geojson"

            # Create the ground DEM file if this has not be created yet!
            print("Generating drain DEM.")
            runner = LidarDemGenerator(self.instructions)
            runner.run()
            dem = runner.dense_dem.dem
        return dem

    def download_osm_values(self) -> bool:
        """Download OpenStreetMap drains and tunnels within the catchment BBox."""

        # Create area to query within
        self.catchment_geometry = self.create_catchment()
        bbox_lat_long = self.catchment_geometry.catchment.to_crs(self.OSM_CRS)

        # Construct query
        query = OSMPythonTools.overpass.overpassQueryBuilder(
            bbox=[
                bbox_lat_long.bounds.miny[0],
                bbox_lat_long.bounds.minx[0],
                bbox_lat_long.bounds.maxy[0],
                bbox_lat_long.bounds.maxx[0],
            ],
            elementType="way",
            selector="waterway",
            out="body",
            includeGeometry=True,
        )

        # Perform query
        overpass = OSMPythonTools.overpass.Overpass()
        rivers = overpass.query(query)

        # Extract information
        element_dict = {
            "geometry": [],
            "OSM_id": [],
            "waterway": [],
            "tunnel": [],
        }

        for element in rivers.elements():
            element_dict["geometry"].append(element.geometry())
            element_dict["OSM_id"].append(element.id())
            element_dict["waterway"].append(element.tags()["waterway"])
            element_dict["tunnel"].append("tunnel" in element.tags().keys())
        drains = geopandas.GeoDataFrame(element_dict, crs=self.OSM_CRS).to_crs(
            self.catchment_geometry.crs["horizontal"]
        )

        # Remove rivers and polygons
        drains = drains[drains["waterway"] != "river"]
        drains = drains[drains.geometry.type == "LineString"]

        return drains

    def run(self, instruction_parameters):
        """This method runs a pipeline that:
        * downloads all tunnels and drains within a catchment.
        * creates and samples a DEM around each feature to estimate the bed
          elevation.
        * saves out extents and bed elevations of the drain and tunnel network"""

        logging.info("Estimating drain and tunnel bed elevation from OpenStreetMap.")

        # Download drains and tunnels from OSM
        drains = self.download_osm_values()

        # Create a DEM where the drains and tunnels are
        dem = self.create_dem(drains)

        # Estimate the drain and tunnel bed elevations from the DEM
        self.estimate_closed_bathymetry(drains=drains, dem=dem)
        self.estimate_open_bathymetry(drains=drains, dem=dem)
