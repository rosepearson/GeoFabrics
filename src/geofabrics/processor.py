# -*- coding: utf-8 -*-
"""
This module contains classes associated with generating GeoFabric layers from
LiDAR and bathymetry contours based on the instructions contained in a JSON file.

GeoFabric layers include hydrologically conditioned DEMs.
"""
import numpy
import json
import pathlib
import abc
import logging
import distributed
import rioxarray
import copy
import geopandas
import pandas
import datetime
import shapely
import xarray
import OSMPythonTools.overpass
from . import geometry
from . import bathymetry_estimation
from . import version
import geoapis.lidar
import geoapis.vector
import geoapis.raster
from . import dem


class BaseProcessor(abc.ABC):
    """An abstract class with general methods for accessing elements in
    instruction files including populating default values. Also contains
    functions for downloading remote data using geopais, and constructing data
    file lists.
    """

    OSM_CRS = "EPSG:4326"

    def __init__(self, json_instructions: json):
        self.instructions = copy.deepcopy(json_instructions)

        self.catchment_geometry = None

    def create_metadata(self) -> dict:
        """A clase to create metadata to be added as netCDF attributes."""

        # Ensure no senstive key information is printed out as part of the instructions
        cleaned_instructions = copy.deepcopy(self.instructions)
        if "datasets" in cleaned_instructions:
            for data_type in cleaned_instructions["datasets"].keys():
                for data_service in cleaned_instructions["datasets"][data_type].keys():
                    if "key" in data_service:
                        cleaned_instructions["datasets"][data_type]["key"].pop()
        metadata = {
            "library_name": "GeoFabrics",
            "library_version": version.__version__,
            "class_name": self.__class__.__name__,
            "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "instructions": cleaned_instructions,
        }
        return metadata

    def get_instruction_path(self, key: str, defaults: dict = {}) -> pathlib.Path:
        """Return the file path from the instruction file, or default if there
        is a default value and the local cache is specified. Raise an error if
        the key is not in the instructions."""

        defaults = dict(
            **defaults,
            **{
                "result_dem": "generated_dem.nc",
                "result_geofabric": "generated_geofabric.nc",
                "raw_dem": "raw_dem.nc",
                "subfolder": "results",
                "downloads": "downloads",
            },
        )

        path_instructions = self.instructions["data_paths"]

        # Return the local cache - for downloaded input data
        local_cache = pathlib.Path(path_instructions["local_cache"])
        if key == "local_cache":
            return local_cache
        # Return the subfolder path - where results are stored
        subfolder = (
            path_instructions["subfolder"]
            if "subfolder" in path_instructions
            else defaults["subfolder"]
        )
        # Return the subfolder outputs folder
        if key == "subfolder":
            return local_cache / subfolder
        # return the downloads cache folder
        if key == "downloads":
            downloads = (
                path_instructions["downloads"]
                if "downloads" in path_instructions
                else defaults["downloads"]
            )
            return local_cache / downloads
        # return the full path of the specified key
        if key in path_instructions:
            # check if a list or single path
            if type(path_instructions[key]) == list:
                absolute_file_paths = []
                file_paths = path_instructions[key]
                for file_path in file_paths:
                    file_path = pathlib.Path(file_path)
                    file_path = (
                        file_path
                        if file_path.is_absolute()
                        else local_cache / subfolder / file_path
                    )
                    absolute_file_paths.append(str(file_path))
                return absolute_file_paths
            else:
                file_path = pathlib.Path(path_instructions[key])
                file_path = (
                    file_path
                    if file_path.is_absolute()
                    else local_cache / subfolder / file_path
                )
                return file_path
        elif key in defaults.keys():
            return local_cache.absolute() / subfolder / defaults[key]
        else:
            assert False, (
                f"The key `{key}` is either missing from data "
                f"paths, not specified in the defaults: {defaults}"
            )

    def check_instruction_path(self, key: str) -> bool:
        """Return True if the file path exists in the instruction file, or True
        if there is a default value and the local cache is specified."""

        assert (
            "local_cache" in self.instructions["data_paths"]
        ), "local_cache is a required 'data_paths' entry"

        defaults = [
            "result_dem",
            "raw_dem",
            "subfolder",
            "result_geofabric",
            "downloads",
        ]

        if key in self.instructions["data_paths"]:
            return True
        else:
            return key in defaults

    def create_results_folder(self):
        """Ensure the results folder has been created."""

        results_folder = self.get_instruction_path("subfolder")
        results_folder.mkdir(parents=True, exist_ok=True)

    def get_resolution(self) -> float:
        """Return the resolution from the instruction file. Raise an error if
        not in the instructions."""

        assert "output" in self.instructions, (
            "'output' is not a key-word in the instructions. It should exist"
            " and is where the resolution and optionally the CRS of the output"
            " DEM is defined."
        )
        assert (
            "resolution" in self.instructions["output"]["grid_params"]
        ), "'resolution' is not a key-word in the instructions"
        return self.instructions["output"]["grid_params"]["resolution"]

    def get_crs(self) -> dict:
        """Return the CRS projection information (horiztonal and vertical) from
        the instruction file. Raise an error if 'output' is not in the instructions. If
        no 'crs' or 'horizontal' or 'vertical' values are specified then use the default
        value for each one missing from the instructions. If the default is used it is
        added to the instructions.
        """

        defaults = {"horizontal": 2193, "vertical": 7839}

        assert "output" in self.instructions, (
            "'output' is not a key-word in the instructions. It should exist "
            "and is where the resolution and optionally the CRS of the output "
            "DEM is defined."
        )

        if "crs" not in self.instructions["output"]:
            logging.warning(
                "No output the coordinate system EPSG values specified. We "
                f"will instead be using the defaults: {defaults}."
            )
            self.instructions["output"]["crs"] = defaults
            return defaults
        else:
            crs_instruction = self.instructions["output"]["crs"]
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
            self.instructions["output"]["crs"] = crs_dict
            return crs_dict

    def get_instruction_general(self, key: str, subkey: str = None):
        """Return the general instruction from the instruction file or return
        the default value if not specified in the instruction file. Raise an
        error if the key is not in the instructions and there is no default
        value. If the default is used it is added to the instructions."""

        defaults = {
            "bathymetry_points_type": None,
            "drop_offshore_lidar": True,
            "lidar_classifications_to_keep": [2],
            "elevation_range": None,
            "download_limit_gbytes": 100,
            "lidar_buffer": 0,
            "interpolation": {
                "rivers": "rbf",
                "waterways": "cubic",
                "ocean": "linear",
                "lidar": "idw",
                "no_data": None,
            },
            "z_labels": {
                "waterways": "elevations",
                "rivers": "bed_elevation_Rupp_and_Smart",
                "ocean": None,
            },
        }

        if key not in defaults and key not in self.instructions["general"]:
            raise ValueError(
                f"The key: {key} is missing from the general instructions, and"
                " does not have a default value"
            )
        if "general" not in self.instructions:
            # Ensure general is a dictionary in the instructions
            self.instructions["general"] = {}
        if key not in self.instructions["general"]:
            # Add the default to the instructions
            self.instructions["general"][key] = defaults[key]
        if subkey is None:
            # Return the key value
            return self.instructions["general"][key]
        else:
            # Return the subkey value as specified
            if subkey not in self.instructions["general"][key]:
                # Add the default to the instructions
                self.instructions["general"][key][subkey] = defaults[key][subkey]
            return self.instructions["general"][key][subkey]

    def get_processing_instructions(self, key: str):
        """Return the processing instruction from the instruction file or
        return the default value if not specified in the instruction file. If
        the default is used it is added to the instructions.

        Parameters
        ----------

        key
            The string identifying the instruction
        """

        defaults = {
            "number_of_cores": 1,
            "chunk_size": None,
            "memory_limit": "10GiB",
        }

        assert key in defaults or key in self.instructions["processing"], (
            f"The key: {key} is missing "
            + "from the general instructions, and does not have a default value"
        )
        if "processing" in self.instructions and key in self.instructions["processing"]:
            return self.instructions["processing"][key]
        else:
            if "processing" not in self.instructions:
                self.instructions["processing"] = {}
            self.instructions["processing"][key] = defaults[key]
            return defaults[key]

    def check_datasets(self, key: str, data_type: str) -> bool:
        """Check to see if the dataset is included in the instructions:
        key = dataservice (i.e. local, opentogaphy, linxz, lris)

        Parameters
        ----------

        key
            The string identifying how the data is accessed
            (e.g. local, opentopography, linz, etc)
        data_type
            The string identifying the data type (e.g. raster, lidar, vector)
        """

        if (
            "datasets" in self.instructions
            and data_type in self.instructions["datasets"]
        ):
            # 'apis' included instructions and Key included in the APIs
            return key in self.instructions["datasets"][data_type]
        else:
            return False

    def check_vector_or_raster(self, key: str, api_type: str) -> bool:
        """Check to see if vector or raster key (i.e. land, ocean_contours, etc) is
        included either as a file path, or within any of API's (i.e. LINZ or LRIS).

        Parameters
        ----------

        key
            The string identifying the vector/raster
        api_type
            The string identifying if the key is a vector or raster
        """

        data_services = [
            "linz",
            "lris",
            "statsnz",
        ]  # This list will increase as geopais is extended to support more vector APIs

        if "data_paths" in self.instructions and key in self.instructions["data_paths"]:
            # Key included in the data paths
            return True
        elif (
            "datasets" in self.instructions
            and api_type in self.instructions["datasets"]
        ):
            for data_service in data_services:
                if (
                    data_service in self.instructions["datasets"][api_type]
                    and key in self.instructions["datasets"][api_type][data_service]
                ):
                    # Key is included in one or more of the data_service's APIs
                    return True
        else:
            return False

    def get_vector_or_raster_paths(
        self, key: str, data_type: str, required: bool = True
    ) -> list:
        """Get the path to the vector/raster key data included either as a file path or
        as an API. Return all paths where the vector key is specified. In the case that
        an API is specified ensure the data is fetched as well.

        Parameters
        ----------

        key
            The string identifying the vector/raster
        data_type
            The string identifying if the key is a vector or raster
        required
            If `True`, an exception will be raised if no path(s) are found.
        """

        paths = []

        # Check the instructions for vector data specified as a data_paths
        if "data_paths" in self.instructions and key in self.instructions["data_paths"]:
            # Key included in the data paths - add - either list or individual path
            data_paths = self.get_instruction_path(key)
            if type(data_paths) is list:
                paths.extend([pathlib.Path(data_path) for data_path in data_paths])
            else:
                paths.append(pathlib.Path(data_paths))
        if data_type == "vector":
            # Define the supported vector 'datasets' keywords and the geoapis class for
            # accessing that data service
            data_services = {
                "linz": geoapis.vector.Linz,
                "lris": geoapis.vector.Lris,
                "statsnz": geoapis.vector.StatsNz,
            }
        elif data_type == "raster":
            data_services = {
                "linz": geoapis.raster.Linz,
                "lris": geoapis.raster.Lris,
                "statsnz": geoapis.raster.StatsNz,
            }
        else:
            logging.warning(f"Unsupported API type specified: {data_type}. Ignored.")
            return
        # Check the instructions for vector data hosted in the supported vector data
        # services: LINZ and LRIS
        base_dir = pathlib.Path(self.get_instruction_path("local_cache"))
        subfolder = self.get_instruction_path("subfolder").relative_to(base_dir)
        cache_dir = pathlib.Path(self.get_instruction_path("downloads"))
        bounding_polygon = (
            self.catchment_geometry.catchment
            if self.catchment_geometry is not None
            else None
        )
        for data_service in data_services.keys():
            if (
                self.check_datasets(data_service, data_type=data_type)
                and key in self.instructions["datasets"][data_type][data_service]
            ):
                # Get the location to cache vector data downloaded from data services
                assert self.check_instruction_path("local_cache"), (
                    "Local cache file path must exist to specify thelocation to"
                    + f" download vector data from the vector APIs: {data_services}"
                )

                # Get the API key for the data_serive being checked
                assert (
                    "key" in self.instructions["datasets"][data_type][data_service]
                ), (
                    f"A 'key' must be specified for the {data_type}:{data_service} data"
                    "  service instead the instruction only includes: "
                    f"{self.instructions['datasets'][data_type][data_service]}"
                )
                api_key = self.instructions["datasets"][data_type][data_service]["key"]

                # Instantiate the geoapis object for downloading vectors from the data
                # service.
                if data_type == "vector":
                    fetcher = data_services[data_service](
                        api_key,
                        bounding_polygon=bounding_polygon,
                        verbose=True,
                    )

                    api_instruction = self.instructions["datasets"][data_type][
                        data_service
                    ][key]
                    geometry_type = (
                        api_instruction["geometry_name "]
                        if "geometry_name " in api_instruction
                        else None
                    )

                    logging.info(
                        f"Downloading vector layers {api_instruction['layers']} from"
                        f" the {data_service} data service"
                    )

                    # Cycle through all layers specified - save each & add to the path
                    # list
                    for layer in api_instruction["layers"]:
                        # Use the run method to download each layer in turn
                        vector = fetcher.run(layer, geometry_type)

                        # Write out file if not already recorded
                        layer_file = (
                            cache_dir / "vector" / subfolder / f"{layer}.geojson"
                        )
                        if not layer_file.exists():
                            layer_file.parent.mkdir(parents=True, exist_ok=True)
                            vector.to_file(layer_file)
                        paths.append(layer_file)
                elif data_type == "raster":
                    # simplify the bounding_polygon geometry
                    if bounding_polygon is not None:
                        bounds = bounding_polygon.bounds
                        bounds = shapely.geometry.Polygon(
                            [
                                (bounds["minx"].min(), bounds["miny"].min()),
                                (bounds["maxx"].max(), bounds["miny"].min()),
                                (bounds["maxx"].max(), bounds["maxy"].max()),
                                (bounds["minx"].min(), bounds["maxy"].max()),
                            ]
                        )
                        bounding_polygon = geopandas.GeoDataFrame(
                            geometry=[bounds], crs=bounding_polygon.crs
                        )
                    raster_dir = cache_dir / "raster" / subfolder
                    raster_dir.parent.mkdir(parents=True, exist_ok=True)
                    fetcher = data_services[data_service](
                        key=api_key,
                        bounding_polygon=bounding_polygon,
                        cache_path=raster_dir,
                    )

                    api_instruction = self.instructions["datasets"][data_type][
                        data_service
                    ][key]

                    logging.info(
                        f"Downloading rater layers {api_instruction['layers']} from"
                        f" the {data_service} data service"
                    )

                    # Cycle through all layers specified - save each & add to the path
                    # list
                    for layer in api_instruction["layers"]:
                        # Use the run method to download each layer in turn
                        raster_paths = fetcher.run(layer)

                        # Add the downloaded to paths
                        for raster_path in raster_paths:
                            paths.append(raster_path)

        if required and len(paths) == 0:
            raise Exception(
                f"Error the expected {data_type} key '{key}' "
                "is not included in either the `data_paths` "
                "or in the `datasets. Please add then re-run."
            )
        return paths

    def get_lidar_dataset_crs(self, data_service, dataset_name) -> dict:
        """Checks to see if source CRS of an associated LiDAR dataset has be specified
        in the instruction file. If it has been specified, this CRS is returned, and
        will later be used to override the CRS encoded in the LAS files.
        """

        dataset_type = "lidar"
        apis_instructions = self.instructions["datasets"]

        if (
            self.check_datasets(data_service, data_type=dataset_type)
            and type(apis_instructions[dataset_type][data_service]) is dict
            and dataset_name in apis_instructions[dataset_type][data_service]
            and type(apis_instructions[dataset_type][data_service][dataset_name])
            is dict
        ):
            dataset_instruction = apis_instructions[dataset_type][data_service][
                dataset_name
            ]

            if (
                "crs" in dataset_instruction
                and "horizontal" in dataset_instruction["crs"]
                and "vertical" in dataset_instruction["crs"]
            ):
                dataset_crs = dataset_instruction["crs"]
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

    def get_lidar_datasets_info(self) -> dict:
        """Return a dictionary with three enties 'file_paths', 'crs' and
        'tile_index_file'. The 'file_paths' contains a list of LiDAR tiles to
        process.

        The 'crs' (or coordinate system of the LiDAR data as defined by an EPSG
        code) is only optionally set (if unset the value is None). The 'crs'
        should only be set if the CRS information is not correctly encoded in
        this is only supported for OpenTopography LiDAR.

        The 'tile_index_file' is also optional (if unset the value is None).
        The 'tile_index_file' should be given if a tile index file exists for
        the LiDAR files specifying the extents of each tile. This is currently
        only supported for OpenTopography files.

        If a LiDAR dataset (either through an API or locally) is specified this
        is checked and all files within the catchment area are downloaded and
        used to construct the file list. If none is specified, the instruction
        'data_paths' is checked for 'lidars' and these are returned.
        """

        # Store dataset information in a dictionary of dictionaries
        lidar_datasets_info = {}
        data_type = "lidar"

        # See if 'OpenTopography' or another data_service has been specified as an area
        # to look first
        data_service = "open_topography"
        if self.check_datasets(data_service, data_type="lidar"):
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
                cache_path=self.get_instruction_path("downloads") / "lidar",
                search_polygon=search_polygon,
                verbose=True,
                download_limit_gbytes=self.get_instruction_general(
                    "download_limit_gbytes"
                ),
            )
            # Loop through each specified dataset and download it
            for dataset_name in self.instructions["datasets"][data_type][
                data_service
            ].keys():
                logging.info(f"Fetching dataset: {dataset_name}")
                self.lidar_fetcher.run(dataset_name)
                lidar_datasets_info[dataset_name] = {}
                lidar_datasets_info[dataset_name]["file_paths"] = sorted(
                    pathlib.Path(self.lidar_fetcher.cache_path / dataset_name).rglob(
                        "*.laz"
                    )
                )
                lidar_datasets_info[dataset_name]["crs"] = self.get_lidar_dataset_crs(
                    data_service, dataset_name
                )
                lidar_datasets_info[dataset_name]["tile_index_file"] = (
                    self.lidar_fetcher.cache_path
                    / dataset_name
                    / f"{dataset_name}_TileIndex.zip"
                )
        # Next check for any additional local LiDAR datasets.
        # for multiple local lidar datasets - must be in separate folders
        data_service = "local"
        if self.check_datasets(data_service, data_type="lidar"):
            local_datasets = copy.deepcopy(
                self.instructions["datasets"][data_type][data_service]
            )
            for dataset_name, dataset in local_datasets.items():
                # Ensure the file_paths (LAZ, and LAS files) are specified
                if "file_paths" not in dataset and "folder_path" in dataset:
                    # Compressed file type
                    dataset["file_paths"] = sorted(
                        pathlib.Path(dataset["folder_path"]).rglob("*.laz")
                    )
                    # uncompressed file type
                    dataset["file_paths"].extend(
                        sorted(pathlib.Path(dataset["folder_path"]).rglob("*.las"))
                    )
                elif "file_paths" not in dataset and "folder_path" not in dataset:
                    raise Exception(
                        "Local datasets must have either a `folder_path` or "
                        "file_paths specified. Both are missing for dataset:"
                        f"{dataset_name}."
                    )
                # Ensure the tile_index file is specified
                if "tile_index_file" not in dataset and "folder_path" in dataset:
                    dataset["tile_index_file"] = (
                        pathlib.Path(dataset["folder_path"])
                        / f"{dataset_name}_TileIndex.zip"
                    )
                    if not dataset["tile_index_file"].exists():
                        raise Exception(
                            f"{dataset['tile_index_file']} does not exist. If "
                            "the tile index file has a different name it must "
                            "be specified with the key `tile_index_file`. If "
                            "there is not tile index file the lidar files "
                            "need to be included as `cache_path:lidar_files`."
                        )
                elif "tile_index_file" not in dataset and "folder_path" not in dataset:
                    raise Exception(
                        "Local datasets must have either a `folder_path` or "
                        "file_paths specified. Both are missing for dataset:"
                        f"{dataset_name}."
                    )
            # Check no overlap between local and remote (API) keys
            if len(lidar_datasets_info.keys() & local_datasets.keys()) > 0:
                raise Exception(
                    "The local and API (remote) LiDAR dataset names must be "
                    "unique. Both contain the dataset name(s): "
                    "{lidar_datasets_info.keys() & local_datasets.keys()}."
                )
            # Add the local datasets to any remote (API) datasets
            lidar_datasets_info.update(local_datasets)
        # Finally if there are no LiDAR datasets see if invidividual LiDAR
        # files have been specified.
        if len(lidar_datasets_info) == 0 and self.check_instruction_path("lidar_files"):
            # get the specified file paths from the instructions,
            lidar_datasets_info["local_files"] = {}
            lidar_datasets_info["local_files"][
                "file_paths"
            ] = self.get_instruction_path("lidar_files")
            lidar_datasets_info["local_files"]["crs"] = None
            lidar_datasets_info["local_files"]["tile_index_file"] = None
            # Ensure this is added to the LiDAR mapping - add if missing
            if (
                "dataset_mapping" not in self.instructions
                or "lidar" not in self.instructions["dataset_mapping"]
            ):
                if "dataset_mapping" not in self.instructions:
                    self.instructions["dataset_mapping"] = {}
                self.instructions["dataset_mapping"]["lidar"] = {"local_files": 1}
        elif len(lidar_datasets_info) == 0 and not self.check_instruction_path(
            "lidar_files"
        ):
            logging.warning(
                "No LiDAR datasets or `lidar_files` have been "
                "specified. Please check your instruction file/"
                "dict if this is unexpected."
            )
        elif self.check_instruction_path("lidar_files"):
            logging.warning(
                "Full LiDAR datasets have been specified (either "
                "through the APIs or locally) as well as "
                "`lidar_files`. These will be ignored."
            )
        # Ensure the data_mapping exists and matches the datasets
        if len(lidar_datasets_info) > 0:
            if (
                "dataset_mapping" not in self.instructions
                or "lidar" not in self.instructions["dataset_mapping"]
            ):
                # Either create if only one dataset or raise and error if many
                if len(lidar_datasets_info) == 1:
                    # only one dataset so can unabiguously create a dataset mapping
                    if "dataset_mapping" not in self.instructions:
                        self.instructions["dataset_mapping"] = {}
                    self.instructions["dataset_mapping"]["lidar"] = {
                        list(lidar_datasets_info.keys())[0]: 1
                    }
                else:
                    raise Exception(
                        "A lidar dataset mapping mut be specified in "
                        "the instructions if there are mutliple LiDAR "
                        "datasets. See the GitHub wiki."
                    )
            else:
                # Ensure all lidar dataset names are included in the mapping
                lidar_dataset_mapping = self.instructions["dataset_mapping"]["lidar"]
                if len(lidar_datasets_info) < len(
                    lidar_datasets_info.keys() & lidar_dataset_mapping.keys()
                ):
                    raise Exception(
                        "One of the LiDAR dataset names is missing"
                        "from the LiDAR dataset mapping. Dataset "
                        f"name are: {lidar_datasets_info.keys()}, "
                        "and the mappings are: "
                        f"{lidar_dataset_mapping.keys()}"
                    )
            # Check the reserved '-1' code for 'no LiDAR' isn't already used
            lidar_dataset_mapping = self.instructions["dataset_mapping"]["lidar"]
            if -1 in lidar_dataset_mapping.values():
                raise Exception(
                    "The mapping value of -1 is reserved for "
                    "no lidar data. Please select a different "
                    f"mapping value. {lidar_dataset_mapping}"
                )
            # Add a no LiDAR mapping value
            self.instructions["dataset_mapping"]["lidar"][
                "no LiDAR"
            ] = dem.DemBase.SOURCE_CLASSIFICATION["no data"]
            # Sort the lidar_dataset_info order by lidar_dataset_mapping values
            # First sort the lidar_dataset_mapping by value
            lidar_dataset_mapping = self.instructions["dataset_mapping"]["lidar"]
            lidar_dataset_mapping = dict(
                sorted(lidar_dataset_mapping.items(), key=lambda item: item[1])
            )
            # Next sort the lidar_datasets_info by the lidar_dataset_mapping value
            lidar_datasets_info = {
                key: lidar_datasets_info[key]
                for key in lidar_dataset_mapping.keys()
                if key in lidar_datasets_info.keys()
            }

        return lidar_datasets_info

    def create_catchment(self) -> geometry.CatchmentGeometry:
        # create the catchment geometry object
        catchment_dirs = self.get_instruction_path("extents")
        assert type(catchment_dirs) is not list, (
            f"A list of `extents`s is provided: {catchment_dirs}, "
            + "where only one is supported."
        )
        catchment_geometry = geometry.CatchmentGeometry(
            catchment_dirs,
            self.get_crs(),
            self.get_resolution(),
            foreshore_buffer=2,
        )
        land_dirs = self.get_vector_or_raster_paths(
            key="land", data_type="vector", required=False
        )
        # Use the catchment outline as the land outline if the land is not specified
        if len(land_dirs) == 0:
            catchment_geometry.land = catchment_dirs
        elif len(land_dirs) == 1:
            catchment_geometry.land = land_dirs[0]
        else:
            raise Exception(
                f"{len(land_dirs)} land_dir's provided, where only one is "
                f"supported. Specficially land_dirs = {land_dirs}."
            )

        return catchment_geometry

    @abc.abstractmethod
    def run(self):
        """This method controls the processor execution and code-flow."""

        raise NotImplementedError("NETLOC_API must be instantiated in the child class")


class RawLidarDemGenerator(BaseProcessor):
    """RawLidarDemGenerator executes a pipeline for creating a DEM from LiDAR and
    optionally a coarse DEM. The data sources and pipeline logic is defined in the
    json_instructions file.

    The `DemGenerator` class contains several important class members:
     * instructions - Defines the pipeline execution instructions
     * catchment_geometry - Defines all relevant regions in a catchment required in the
       generation of a DEM as polygons.
     * raw_dem - A combination of LiDAR tiles and any referecnce DEM
       from LiDAR and interpolated from bathymetry.
     * coarse_dem - This optional object defines a background DEM that may be used to
       fill on land gaps in the LiDAR.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and
    an instruction file.
    """

    def __init__(self, json_instructions: json, debug: bool = True):
        super(RawLidarDemGenerator, self).__init__(json_instructions=json_instructions)

        self.raw_dem = None
        self.debug = debug

    def run(self):
        """This method executes the geofabrics generation pipeline to produce geofabric
        derivatives.

        Note it currently only considers one LiDAR dataset that can have many tiles.
        See 'get_lidar_datasets_info' for where to change this."""

        logging.info("Create a raw DEM layer from LiDAR.")

        # Ensure the results folder has been created
        self.create_results_folder()

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = (
            10.0 / 100
        )  # Used to decide if non-LiDAR data should be included

        # create the catchment geometry object
        self.catchment_geometry = self.create_catchment()

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_datasets_info = self.get_lidar_datasets_info()

        # setup the raw DEM generator
        self.raw_dem = dem.RawDem(
            catchment_geometry=self.catchment_geometry,
            drop_offshore_lidar=self.get_instruction_general("drop_offshore_lidar"),
            lidar_interpolation_method=self.get_instruction_general(
                key="interpolation", subkey="lidar"
            ),
            elevation_range=self.get_instruction_general("elevation_range"),
        )

        # Load in LiDAR tiles
        self.raw_dem.add_lidar(
            lidar_datasets_info=lidar_datasets_info,
            lidar_classifications_to_keep=self.get_instruction_general(
                "lidar_classifications_to_keep"
            ),
            chunk_size=self.get_processing_instructions("chunk_size"),
            metadata=self.create_metadata(),
        )  # Note must be called after all others if it is to be complete

        # Add a coarse DEM if significant area without LiDAR and a coarse DEM
        if self.check_vector_or_raster(key="coarse_dems", api_type="raster"):
            coarse_dem_paths = self.get_vector_or_raster_paths(
                key="coarse_dems", data_type="raster"
            )

            # Add coarse DEMs if there are any and if area
            self.raw_dem.add_coarse_dems(
                coarse_dem_paths=coarse_dem_paths,
                area_threshold=area_threshold,
                buffer_cells=self.get_instruction_general("lidar_buffer"),
                chunk_size=self.get_processing_instructions("chunk_size"),
            )

        # Setup Dask cluster and client - LAZY SAVE LIDAR DEM
        cluster_kwargs = {
            "n_workers": self.get_processing_instructions("number_of_cores"),
            "threads_per_worker": 1,
            "processes": True,
            "memory_limit": self.get_processing_instructions("memory_limit"),
        }
        cluster = distributed.LocalCluster(**cluster_kwargs)
        with cluster, distributed.Client(cluster) as client:
            print("Dask client:", client)
            print("Dask dashboard:", client.dashboard_link)

            # compute and save raw DEM
            logging.info(
                "In processor.DemGenerator - write out the raw DEM " "from LiDAR"
            )
            try:
                self.raw_dem.dem.to_netcdf(
                    self.get_instruction_path("raw_dem"),
                    format="NETCDF4",
                    engine="netcdf4",
                )
            except (Exception, KeyboardInterrupt) as caught_exception:
                pathlib.Path(self.get_instruction_path("raw_dem")).unlink()
                logging.info(
                    f"Caught error {caught_exception} and deleting"
                    "partially created netCDF output "
                    f"{self.get_instruction_path('raw_dem')}"
                    " before re-raising error."
                )
                raise caught_exception

        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "dem_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)


class HydrologicDemGenerator(BaseProcessor):
    """HydrologicDemGenerator executes a pipeline for loading in a raw DEM and extents
    before incorporating bathymetry (offshore, rivers and waterways) to produce a
    hydrologically conditioned DEM. The data and pipeline logic is defined in
    the json_instructions file.

    The `HydrologicDemGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the
       generation of a DEM as polygons.
     * hydrologic_dem - Defines the hydrologically conditioned DEM as a combination of
       tiles from LiDAR and interpolated from bathymetry.
     * bathy_contours - This object defines the bathymetry vectors used to define the
       DEM values offshore.

    See the README.md for usage examples or GeoFabrics/tests/ for examples of usage and
    an instruction file
    """

    def __init__(self, json_instructions: json, debug: bool = True):
        super(HydrologicDemGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.hydrologic_dem = None
        self.bathy_contours = None
        self.debug = debug

    def add_bathymetry(self, area_threshold: float, catchment_dirs: pathlib.Path):
        """Add in any bathymetry data - ocean or river"""

        # Check for ocean bathymetry. Interpolate offshore if significant
        # offshore area is not covered by LiDAR
        area_without_lidar = self.catchment_geometry.offshore_without_lidar(
            self.hydrologic_dem.raw_extents
        ).geometry.area.sum()
        if (
            self.check_vector_or_raster(key="ocean_contours", api_type="vector")
            and area_without_lidar
            > self.catchment_geometry.offshore.area.sum() * area_threshold
        ):
            # Get the bathymetry data directory
            bathy_contour_dirs = self.get_vector_or_raster_paths(
                key="ocean_contours", data_type="vector"
            )
            assert len(bathy_contour_dirs) == 1, (
                f"{len(bathy_contour_dirs)} ocean_contours's provided. "
                f"Specficially {catchment_dirs}. Support has not yet been added for "
                "multiple datasets."
            )

            logging.info(f"Incorporating Bathymetry: {bathy_contour_dirs}")

            # Load in bathymetry
            self.bathy_contours = geometry.BathymetryContours(
                bathy_contour_dirs[0],
                self.catchment_geometry,
                z_label=self.get_instruction_general(key="z_labels", subkey="ocean"),
                exclusion_extent=self.hydrologic_dem.raw_extents,
            )

            # interpolate
            self.hydrologic_dem.interpolate_ocean_bathymetry(self.bathy_contours)
        # Check for waterways and interpolate if they exist
        if "waterways" in self.instructions["data_paths"]:
            # Load in all open and closed waterway elevation and extents in one go
            # Get the polygons and bathymetry and can be multiple
            subfolder = self.get_instruction_path(key="subfolder")
            bathy_dirs = [
                pathlib.Path(waterway_dict["elevations"])
                for waterway_dict in self.instructions["data_paths"]["waterways"]
            ]
            poly_dirs = [
                pathlib.Path(waterway_dict["extents"])
                for waterway_dict in self.instructions["data_paths"]["waterways"]
            ]
            for index, (bathy_dir, poly_dir) in enumerate(zip(bathy_dirs, poly_dirs)):
                if not bathy_dir.is_absolute():
                    bathy_dirs[index] = subfolder / bathy_dir
                if not poly_dir.is_absolute():
                    poly_dirs[index] = subfolder / poly_dir

            logging.info(f"Incorporating waterways: {bathy_dirs}")

            # Load in bathymetry
            self.estimated_bathymetry_points = geometry.EstimatedBathymetryPoints(
                points_files=bathy_dirs,
                polygon_files=poly_dirs,
                catchment_geometry=self.catchment_geometry,
                z_labels=self.get_instruction_general(
                    key="z_labels", subkey="waterways"
                ),
            )

            # Call interpolate river on the DEM - the class checks to see if any pixels
            # actually fall inside the polygon
            self.hydrologic_dem.interpolate_waterways(
                estimated_bathymetry=self.estimated_bathymetry_points,
                method=self.get_instruction_general(
                    key="interpolation", subkey="waterways"
                ),
            )
        # Load in river bathymetry and incorporate where discernable at the resolution
        if "rivers" in self.instructions["data_paths"]:
            # Loop through each river in turn adding individually
            subfolder = self.get_instruction_path(key="subfolder")
            for river_dict in self.instructions["data_paths"]["rivers"]:
                bathy_dir = pathlib.Path(river_dict["elevations"])
                poly_dir = pathlib.Path(river_dict["extents"])
                if not bathy_dir.is_absolute():
                    bathy_dir = subfolder / bathy_dir
                if not poly_dir.is_absolute():
                    poly_dir = subfolder / poly_dir

                logging.info(f"Incorporating river: {bathy_dir}")

                # Load in bathymetry
                self.estimated_bathymetry_points = geometry.EstimatedBathymetryPoints(
                    points_files=[bathy_dir],
                    polygon_files=[poly_dir],
                    catchment_geometry=self.catchment_geometry,
                    z_labels=self.get_instruction_general(
                        key="z_labels", subkey="rivers"
                    ),
                )

                # Call interpolate river on the DEM - the class checks to see if any pixels
                # actually fall inside the polygon
                self.hydrologic_dem.interpolate_rivers(
                    estimated_bathymetry=self.estimated_bathymetry_points,
                    method=self.get_instruction_general(
                        key="interpolation", subkey="rivers"
                    ),
                )

    def run(self):
        """This method executes the geofabrics generation pipeline to produce geofabric
        derivatives."""

        # Ensure the results folder has been created
        self.create_results_folder()

        # Only include data in addition to LiDAR if the area_threshold is not covered
        area_threshold = 10.0 / 100  # Used to decide if bathymetry should be included

        # create the catchment geometry object
        self.catchment_geometry = self.create_catchment()

        # setup the hydrologically conditioned DEM generator
        self.hydrologic_dem = dem.HydrologicallyConditionedDem(
            catchment_geometry=self.catchment_geometry,
            raw_dem_path=self.get_instruction_path("raw_dem"),
            interpolation_method=self.get_instruction_general(
                key="interpolation", subkey="no_data"
            ),
        )

        # Check for and add any bathymetry information
        self.add_bathymetry(
            area_threshold=area_threshold,
            catchment_dirs=self.get_instruction_path("extents"),
        )

        # fill combined dem - save results
        self.hydrologic_dem.dem.to_netcdf(
            self.get_instruction_path("result_dem"),
            format="NETCDF4",
            engine="netcdf4",
        )
        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "dem_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)


class RoughnessLengthGenerator(BaseProcessor):
    """RoughnessLengthGenerator executes a pipeline for loading in a hydrologically
    conditioned DEM and LiDAR tiles to produce a roughness length layer that is added to
    the Hydrologically conditioned DEM. The data and pipeline logic is defined in
    the json_instructions file.

    The `RoughnessLengthGenerator` class contains several important class members:
     * catchment_geometry - Defines all relevant regions in a catchment required in the
       generation of a DEM as polygons.
     * roughness_dem - Adds a roughness layer to the hydrologically conditioned DEM.

    """

    def __init__(self, json_instructions: json, debug: bool = True):
        super(RoughnessLengthGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.roughness_dem = None
        self.debug = debug

    def run(self):
        """This method executes the geofabrics generation pipeline to produce geofabric
        derivatives."""

        logging.info("Adding a roughness layer to the geofabric.")

        # Ensure the results folder has been created
        self.create_results_folder()

        # create the catchment geometry object
        self.catchment_geometry = self.create_catchment()

        # Get LiDAR data file-list - this may involve downloading lidar files
        lidar_datasets_info = self.get_lidar_datasets_info()

        # setup the roughness DEM generator
        self.roughness_dem = dem.RoughnessDem(
            catchment_geometry=self.catchment_geometry,
            hydrological_dem_path=self.get_instruction_path("result_dem"),
            elevation_range=self.get_instruction_general("elevation_range"),
            interpolation_method=self.get_instruction_general(
                key="interpolation", subkey="no_data"
            ),
        )

        # Load in LiDAR tiles
        self.roughness_dem.add_lidar(
            lidar_datasets_info=lidar_datasets_info,
            lidar_classifications_to_keep=self.get_instruction_general(
                "lidar_classifications_to_keep"
            ),
            chunk_size=self.get_processing_instructions("chunk_size"),
            metadata=self.create_metadata(),
        )  # Note must be called after all others if it is to be complete

        # Setup Dask cluster and client
        cluster_kwargs = {
            "n_workers": self.get_processing_instructions("number_of_cores"),
            "threads_per_worker": 1,
            "processes": True,
            "memory_limit": self.get_processing_instructions("memory_limit"),
        }
        cluster = distributed.LocalCluster(**cluster_kwargs)
        with cluster, distributed.Client(cluster) as client:
            print("Dask client:", client)
            print("Dask dashboard:", client.dashboard_link)
            # save results
            try:
                self.roughness_dem.dem.to_netcdf(
                    self.get_instruction_path("result_geofabric"),
                    format="NETCDF4",
                    engine="netcdf4",
                )
            except (Exception, KeyboardInterrupt) as caught_exception:
                pathlib.Path(self.get_instruction_path("result_geofabric")).unlink()
                logging.info(
                    f"Caught error {caught_exception} and deleting"
                    "partially created netCDF output "
                    f"{self.get_instruction_path('result_geofabric')}"
                    " before re-raising error."
                )
                raise caught_exception
        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "roughness_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)


class MeasuredRiverGenerator(BaseProcessor):
    """MeasuredRiverGenerator executes a pipeline to interpolate between
    measured river cross section elevations. A json_instructions file defines
    the pipeline logic and data.

    """

    def __init__(self, json_instructions: json, debug: bool = True):
        super(MeasuredRiverGenerator, self).__init__(
            json_instructions=json_instructions
        )

        self.debug = debug

    def get_measured_instruction(self, key: str):
        """Return true if the DEMs are required for later processing


        Parameters:
            instructions  The json instructions defining the behaviour
        """
        defaults = {
            "thalweg_centre": True,
            "estimate_fan": False,
        }

        assert key in defaults or key in self.instructions["measured"], (
            f"The key: {key} is missing from the measured instructions, and"
            " does not have a default value"
        )
        if "measured" in self.instructions and key in self.instructions["measured"]:
            return self.instructions["measured"][key]
        else:
            self.instructions["measured"][key] = defaults[key]
            return defaults[key]

    def estimate_river_mouth_fan(self, defaults: dict):
        """Calculate and save depth estimates along the river mouth fan."""

        # Get the required inputs that already exist
        crs = self.get_crs()["horizontal"]
        cross_section_spacing = self.get_measured_instruction("cross_section_spacing")
        river_elevation_file = self.get_instruction_path(
            "result_elevation", defaults=defaults
        )
        river_polygon_file = self.get_instruction_path(
            "result_polygon", defaults=defaults
        )
        ocean_contour_file = self.get_vector_or_raster_paths(
            key="ocean_contours", data_type="vector"
        )[0]
        ocean_contour_depth_label = self.get_instruction_general(
            key="z_labels", subkey="ocean"
        )
        # Create the required river centreline and bathymtries that don't exist
        # Only create for the two closest points to the river mouth
        # And ensure is perpindicular to the river mouth
        riverlines = geopandas.read_file(self.get_instruction_path("riverbanks"))
        elevations = geopandas.read_file(river_elevation_file)
        # Resample river edges at same spacing as used to interpolate
        n = len(elevations.groupby("level_0"))
        normalised_locations = numpy.arange(n) * 1 / n
        points_0 = riverlines.geometry.iloc[0].interpolate(
            normalised_locations, normalized=True
        )
        points_1 = riverlines.geometry.iloc[1].interpolate(
            normalised_locations, normalized=True
        )
        # Calculate the mouth centre, normal and tangent
        segment_dx = points_0[0].x - points_1[0].x
        segment_dy = points_0[0].y - points_1[0].y
        segment_length = numpy.sqrt(segment_dx**2 + segment_dy**2)
        mouth_tangent = shapely.geometry.Point(
            [segment_dx / segment_length, segment_dy / segment_length]
        )
        mouth_normal = shapely.geometry.Point([-mouth_tangent.y, mouth_tangent.x])
        mouth_centre = shapely.geometry.MultiPoint([points_0[0], points_1[0]]).centroid
        spacing = max(
            points_0[0].distance(points_0[1]),
            points_1[0].distance(points_1[1]),
        )
        # Generate and save out a river centreline file normal to the river mouth
        river_centreline = shapely.geometry.LineString(
            [
                mouth_centre,
                shapely.geometry.Point(
                    [
                        mouth_centre.x + mouth_normal.x * spacing,
                        mouth_centre.y + mouth_normal.y * spacing,
                    ]
                ),
            ]
        )
        defaults["river_centreline"] = "river_centreline_for_fan.geojson"
        river_centreline_file = self.get_instruction_path(
            "river_centreline", defaults=defaults
        )
        river_centreline = geopandas.GeoDataFrame(geometry=[river_centreline], crs=crs)
        river_centreline.to_file(river_centreline_file)
        # Create the river bathmetries with needed widths and geometry
        defaults["river_bathymetry"] = "river_bathymetry_for_fan.geojson"
        river_bathymetry_file = self.get_instruction_path(
            "river_bathymetry", defaults=defaults
        )
        elevations_clean = (
            elevations[["level_0", "z"]].groupby("level_0").min().reset_index(drop=True)
        )
        elevations_clean["geometry"] = elevations[
            elevations["level_1"] == int(elevations["level_1"].median())
        ]["geometry"].reset_index(drop=True)
        elevations_clean = elevations_clean.iloc[[0, 1]]
        elevations_clean = elevations_clean.set_geometry("geometry").set_crs(crs)
        elevations_clean["width"] = [
            point_0.distance(point_1)
            for point_0, point_1 in zip(points_0[0:2], points_1[0:2])
        ]
        # Add signifiers of being zero offset at bank edge and source of data
        elevations_clean["source"] = "measured"
        elevations_clean.to_file(river_bathymetry_file)

        # Create fan object
        fan = geometry.RiverMouthFan(
            aligned_channel_file=river_centreline_file,
            river_bathymetry_file=river_bathymetry_file,
            river_polygon_file=river_polygon_file,
            ocean_contour_file=ocean_contour_file,
            crs=crs,
            cross_section_spacing=cross_section_spacing,
            elevation_labels=["z"],
            ocean_contour_depth_label=ocean_contour_depth_label,
        )

        # Estimate the fan extents and bathymetry
        fan_polygon, fan_bathymetry = fan.polygon_and_bathymetry()
        river_polygon = geopandas.read_file(river_polygon_file)

        # Combine and save the river and fan geometries
        fan_bathymetry["source"] = "fan"
        combined_bathymetry = geopandas.GeoDataFrame(
            pandas.concat([elevations_clean, fan_bathymetry], ignore_index=True),
            crs=elevations_clean.crs,
        )
        combined_bathymetry.to_file(river_bathymetry_file)
        combined_polygon = river_polygon.overlay(fan_polygon, how="union")
        combined_polygon.to_file(river_polygon_file)

    def run(self):
        """This method extracts a main channel then executes the DemGeneration
        pipeline to produce a DEM before sampling this to extimate width, slope
        and eventually depth."""

        logging.info(
            "Interpolating the measured river elevations if not" "already done."
        )

        # Ensure the results folder has been created
        self.create_results_folder()

        # create the measured river interpolator object
        defaults = {
            "result_polygon": "river_polygon.geojson",
            "result_elevation": "river_elevations.geojson",
        }
        result_polygon_file = self.get_instruction_path(
            "result_polygon", defaults=defaults
        )
        result_elevations_file = self.get_instruction_path(
            "result_elevation", defaults=defaults
        )
        # Only rerun if files don't exist
        if not (result_polygon_file.exists() and result_elevations_file.exists()):
            logging.info("Interpolating measured sections.")
            print("Interpolating measured sections.")
            measured_rivers = bathymetry_estimation.InterpolateMeasuredElevations(
                riverbank_file=self.get_instruction_path("riverbanks"),
                measured_sections_file=self.get_instruction_path("measured_sections"),
                cross_section_spacing=self.get_measured_instruction(
                    "cross_section_spacing"
                ),
                crs=self.get_crs()["horizontal"],
            )
            river_polygon, river_elevations = measured_rivers.interpolate(
                samples_per_section=self.get_measured_instruction(
                    "samples_per_section"
                ),
                thalweg_centre=self.get_measured_instruction("thalweg_centre"),
            )

            # Save the generated polygon and measured elevations
            river_polygon.to_file(result_polygon_file)
            river_elevations.to_file(result_elevations_file)

        # Estimate a river fan if `estimate_fan` is True
        if self.get_measured_instruction("estimate_fan"):
            # Check if already done
            result_elevations = geopandas.read_file(result_elevations_file)
            if not ("fan" == result_elevations["source"]).any():
                logging.info("Estimating the fan bathymetry.")
                print("Estimating the fan bathymetry.")
                self.estimate_river_mouth_fan(defaults)

        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "measured_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)


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

        files_exist = river_bathymetry_file.is_file() and river_polygon_file.is_file()

        if not self.get_bathymetry_instruction("estimate_fan"):
            # If no fan just check if the files exist
            return files_exist
        else:
            # Otherwise check if the fan has been added to the outputs
            if not files_exist:
                # False as files don't exist
                return files_exist
            else:
                # True if fan source in bathymetry
                bathymetry = geopandas.read_file(river_bathymetry_file)
                return (bathymetry["source"] == "fan").any()

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
            area_threshold = self.get_bathymetry_instruction("area_threshold")

            # key to output name mapping
            name_dictionary = {
                "aligned": f"aligned_river_centreline_{area_threshold}.geojson",
                "river_characteristics": "river_characteristics.geojson",
                "river_polygon": "river_polygon.geojson",
                "river_bathymetry": "river_bathymetry.geojson",
                "gnd_dem": "raw_gnd_dem.nc",
                "gnd_dem_extents": "raw_gnd_extents.geojson",
                "veg_dem": "raw_veg_dem.nc",
                "veg_dem_extents": "raw_veg_extents.geojson",
                "catchment": f"river_catchment_{area_threshold}.geojson",
                "network": f"network_river_centreline_{area_threshold}.geojson",
                "network_smoothed": f"network_river_centreline_{area_threshold}_"
                "smoothed.geojson",
            }
            return name_dictionary[key]
        else:
            return name

    def get_result_file_path(self, key: str = None, name: str = None) -> pathlib.Path:
        """Return the file name of the file to save with the local cache path.


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        subfolder = self.get_instruction_path("subfolder")

        name = self.get_result_file_name(key=key, name=name)

        return subfolder / name

    def get_bathymetry_instruction(self, key: str):
        """Return true if the DEMs are required for later processing


        Parameters:
            instructions  The json instructions defining the behaviour
        """
        defaults = {
            "sampling_direction": -1,
            "minimum_slope": 0.0001,  # 0.1m per 1km
            "keep_downstream_osm": False,
            "estimate_fan": False,
            "upstream_smoothing_factor": 25,  # i.e. 25 x cross_section_spacing
        }

        assert key in defaults or key in self.instructions["rivers"], (
            f"The key: {key} is missing from the river instructions, and"
            " does not have a default value"
        )
        if "rivers" in self.instructions and key in self.instructions["rivers"]:
            return self.instructions["rivers"][key]
        else:
            self.instructions["rivers"][key] = defaults[key]
            return defaults[key]

    def get_network_channel(self) -> bathymetry_estimation.Channel:
        """Read in or create a channel from a river network."""

        # Get instructions
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")

        # Check if file exists
        network_name = self.get_result_file_path(key="network")
        if network_name.is_file():
            channel = bathymetry_estimation.Channel(
                channel=geopandas.read_file(network_name),
                resolution=cross_section_spacing,
                sampling_direction=self.get_bathymetry_instruction(
                    "sampling_direction"
                ),
            )
        else:
            # Else, create if it doesn't exist
            channel = bathymetry_estimation.Channel.from_network(
                network_file=self.get_bathymetry_instruction("network_file"),
                crs=self.get_crs()["horizontal"],
                starting_id=self.get_bathymetry_instruction("network_id"),
                resolution=cross_section_spacing,
                area_threshold=self.get_bathymetry_instruction("area_threshold"),
                name_dict=self.get_bathymetry_instruction("network_columns"),
                sampling_direction=self.get_bathymetry_instruction(
                    "sampling_direction"
                ),
            )

            if self.debug:
                # Save the channel and smoothed channel derived from a river network
                network_name = self.get_result_file_path(key="network")
                if not network_name.is_file():
                    channel.channel.to_file(network_name)
                smoothed_rec_name = self.get_result_file_path(key="network_smoothed")
                if not smoothed_rec_name.is_file():
                    channel.get_parametric_spline_fit().to_file(smoothed_rec_name)
        return channel

    def get_dems(self, channel: geometry.CatchmentGeometry) -> tuple:
        """Allow selection of the ground or vegetation DEM, and either create
        or load it.


        Parameters:
            instructions  The json instructions defining the behaviour
        """

        instruction_paths = self.instructions["data_paths"]

        # Extract instructions from JSON
        river_corridor_width = self.get_bathymetry_instruction("river_corridor_width")

        # Define ground and veg files
        gnd_file = self.get_result_file_path(key="gnd_dem")
        veg_file = self.get_result_file_path(key="veg_dem")

        # Ensure channel catchment exists and is up to date if needed
        if not gnd_file.is_file() or not veg_file.is_file():
            catchment_file = self.get_result_file_path(key="catchment")
            instruction_paths["extents"] = str(
                self.get_result_file_name(key="catchment")
            )
            channel_catchment = channel.get_channel_catchment(
                corridor_radius=river_corridor_width / 2
            )
            channel_catchment.to_file(catchment_file)
        # Remove bathymetry contour information if it exists while creating DEMs
        bathy_data_paths = None
        bathy_apis = None
        if "ocean_contours" in instruction_paths:
            bathy_data_paths = instruction_paths.pop("ocean_contours")
        if (
            "vector" in self.instructions["datasets"]
            and "ocean_contours" in self.instructions["datasets"]["vector"]["linz"]
        ):
            bathy_apis = self.instructions["datasets"]["vector"]["linz"].pop(
                "ocean_contours"
            )
        # Get the ground DEM
        if not gnd_file.is_file():
            # Create the ground DEM file if this has not be created yet!
            print("Generating ground DEM.")
            instruction_paths["raw_dem"] = str(self.get_result_file_name(key="gnd_dem"))
            runner = RawLidarDemGenerator(self.instructions)
            runner.run()
            instruction_paths.pop("raw_dem")
        # Load the Ground DEM
        print("Loading ground DEM.")  # drop band added by rasterio.open()
        gnd_dem = rioxarray.rioxarray.open_rasterio(gnd_file, masked=True).squeeze(
            "band", drop=True
        )
        # Get the vegetation DEM
        if not veg_file.is_file():
            # Create the catchment file if this has not be created yet!
            print("Generating vegetation DEM.")
            self.instructions["general"][
                "lidar_classifications_to_keep"
            ] = self.get_bathymetry_instruction("veg_lidar_classifications_to_keep")
            instruction_paths["raw_dem"] = str(self.get_result_file_name(key="veg_dem"))
            runner = RawLidarDemGenerator(self.instructions)
            runner.run()
            instruction_paths.pop("raw_dem")
        # Load the Veg DEM - drop band added by rasterio.open()
        print("Loading the vegetation DEM.")
        veg_dem = dem.rioxarray.rioxarray.open_rasterio(veg_file, masked=True).squeeze(
            "band", drop=True
        )
        # Replace bathymetry contour information if it exists
        if bathy_data_paths is not None:
            instruction_paths["ocean_contours"] = bathy_data_paths
        if bathy_apis is not None:
            self.instructions["datasets"]["vector"]["linz"][
                "ocean_contours"
            ] = bathy_apis
        return gnd_dem, veg_dem

    def align_channel(
        self,
        channel_width: bathymetry_estimation.ChannelCharacteristics,
        channel: bathymetry_estimation.Channel,
    ) -> geopandas.GeoDataFrame:
        """Align the river network defined channel based on LiDAR and save the aligned
        channel.


        Parameters:
            channel_width  The class for characterising channel width and other
                properties
            channel  The river network defined channel alignment
        """

        # Get instruciton parameters
        min_channel_width = self.get_bathymetry_instruction("min_channel_width")
        network_alignment_tolerance = self.get_bathymetry_instruction(
            "network_alignment_tolerance"
        )
        river_corridor_width = self.get_bathymetry_instruction("river_corridor_width")

        bank_threshold = self.get_bathymetry_instruction("min_bank_height")
        width_centre_smoothing_multiplier = self.get_bathymetry_instruction(
            "width_centre_smoothing"
        )

        # The width of cross sections to sample
        corridor_radius = river_corridor_width / 2 + network_alignment_tolerance

        aligned_channel, sampled_cross_sections = channel_width.align_channel(
            threshold=bank_threshold,
            min_channel_width=min_channel_width,
            initial_channel=channel,
            search_radius=network_alignment_tolerance,
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
    ) -> tuple:
        """Align the river network defined channel based on LiDAR and save the aligned
        channel.


        Parameters:
            channel_width  The class for characterising channel width and other
                properties
            channel  The river network defined channel alignment
        """

        # Get instruciton parameters
        max_channel_width = self.get_bathymetry_instruction("max_channel_width")
        min_channel_width = self.get_bathymetry_instruction("min_channel_width")
        bank_threshold = self.get_bathymetry_instruction("min_bank_height")
        max_bank_height = self.get_bathymetry_instruction("max_bank_height")
        river_corridor_width = self.get_bathymetry_instruction("river_corridor_width")

        (
            sampled_cross_sections,
            river_polygon,
        ) = channel_width.estimate_width_and_slope(
            aligned_channel=aligned_channel,
            threshold=bank_threshold,
            cross_section_radius=river_corridor_width / 2,
            search_radius=max_channel_width / 2,
            min_channel_width=min_channel_width,
            max_threshold=max_bank_height,
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
                or "bank_i" in column_name
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

    def characterise_channel(self):
        """Calculate the channel width, slope and other characteristics. This requires a
        ground and vegetation DEM. This also may require alignment of the channel
        centreline.

        """

        logging.info("The channel hasn't been characerised. Charactreising now.")

        # Decide if aligning from river network alone, or OSM and river network
        if "osm_id" in self.instructions["rivers"]:
            channel_width, aligned_channel = self.align_channel_from_osm()
        else:
            channel_width, aligned_channel = self.align_channel_from_rec()
        # calculate the channel width and save results
        print("Characterising the aligned channel.")
        self.calculate_channel_characteristics(
            channel_width=channel_width, aligned_channel=aligned_channel
        )

    def align_channel_from_rec(self) -> tuple:
        """Calculate the channel width, slope and other characteristics. This requires a
        ground and vegetation DEM. This also may require alignment of the channel
        centreline.

        """

        logging.info("Align from river network.")

        # Extract instructions
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        resolution = self.get_resolution()

        # Create river network defined channel
        channel = self.get_network_channel()

        # Get DEMs - create and save if don't exist
        gnd_dem, veg_dem = self.get_dems(channel=channel)

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

            # Align and save the river network defined channel
            aligned_channel = self.align_channel(
                channel_width=channel_width, channel=channel
            )
        else:
            aligned_channel_file = self.get_result_file_path(key="aligned")
            aligned_channel = geopandas.read_file(aligned_channel_file)
        return channel_width, aligned_channel

    def align_channel_from_osm(
        self,
    ) -> bathymetry_estimation.ChannelCharacteristics:
        """Calculate the channel width, slope and other characteristics. This requires a
        ground and vegetation DEM. This also may require alignment of the channel
        centreline.

        """

        logging.info("Align from OSM.")

        # Extract instructions
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        resolution = self.get_resolution()

        # Create river network defined channel
        channel = self.get_network_channel()
        crs = self.get_crs()["horizontal"]

        # Create OSM defined channel
        osm_id = self.get_bathymetry_instruction("osm_id")
        query = f"(way[waterway]({osm_id});); out body geom;"
        overpass = OSMPythonTools.overpass.Overpass()
        if "osm_date" in self.instructions["rivers"]:
            osm_channel = overpass.query(
                query,
                date=self.get_bathymetry_instruction("osm_date"),
                timeout=60,
            )
        else:
            osm_channel = overpass.query(query, timeout=60)
        osm_channel = osm_channel.elements()[0]
        osm_channel = geopandas.GeoDataFrame(
            {
                "geometry": [osm_channel.geometry()],
                "OSM_id": [osm_channel.id()],
                "waterway": [osm_channel.tags()["waterway"]],
            },
            crs=self.OSM_CRS,
        ).to_crs(crs)
        if self.debug:
            osm_channel.to_file(
                self.get_result_file_path(name="osm_channel_full.geojson")
            )
        # Cut the OSM to size - give warning if OSM line shorter than network
        # Get the start and end point of the smoothed network line
        channel = channel.get_parametric_spline_fit()
        network_extents = channel.boundary.explode(index_parts=False)
        network_start, network_end = (
            network_extents.iloc[0],
            network_extents.iloc[1],
        )
        # Get the distance along the OSM that the start/end points are.
        # Note projection function is limited between [0, osm_channel.length]
        end_split_length = float(osm_channel.project(network_end))
        start_split_length = float(osm_channel.project(network_start))
        # Ensure the OSM line is defined upstream
        if start_split_length > end_split_length:
            # Reverse direction of the geometry
            osm_channel.loc[0, "geometry"] = shapely.geometry.LineString(
                list(osm_channel.iloc[0].geometry.coords)[::-1]
            )

        # Cut the OSM to the length of the network. Give warning if shorter.
        start_split_length = float(osm_channel.project(network_start))
        if start_split_length > 0 and not self.get_bathymetry_instruction(
            "keep_downstream_osm"
        ):
            split_point = osm_channel.interpolate(start_split_length)
            osm_channel = shapely.ops.snap(
                osm_channel.loc[0].geometry, split_point.loc[0], tolerance=0.1
            )
            osm_channel = geopandas.GeoDataFrame(
                {
                    "geometry": [
                        list(shapely.ops.split(osm_channel, split_point.loc[0]).geoms)[
                            1
                        ]
                    ]
                },
                crs=crs,
            )
        else:
            logging.warning(
                "The OSM reference line starts upstream of the"
                "network line. The bottom of the network will be"
                "ignored over a stright line distance of "
                f"{osm_channel.distance(network_start)}"
            )
        # Clip end if needed - recacluate clip position incase front clipped.
        end_split_length = float(osm_channel.project(network_end))
        if end_split_length < float(osm_channel.length):
            split_point = osm_channel.interpolate(end_split_length)
            osm_channel = shapely.ops.snap(
                osm_channel.loc[0].geometry, split_point.loc[0], tolerance=0.1
            )
            osm_channel = geopandas.GeoDataFrame(
                {
                    "geometry": [
                        list(shapely.ops.split(osm_channel, split_point.loc[0]).geoms)[
                            0
                        ]
                    ]
                },
                crs=crs,
            )
        else:
            logging.warning(
                "The OSM reference line ends downstream of the"
                "network line. The top of the network will be"
                "ignored over a stright line distance of "
                f"{osm_channel.distance(network_end)}"
            )

        if self.debug:
            osm_channel.to_file(
                self.get_result_file_path(name="osm_channel_cut.geojson")
            )
        # smooth
        osm_channel = bathymetry_estimation.Channel(
            channel=osm_channel,
            resolution=cross_section_spacing,
            sampling_direction=1,
        )
        smoothed_osm_channel = osm_channel.get_parametric_spline_fit()
        smoothed_osm_channel.to_file(self.get_result_file_path(key="aligned"))
        # Get DEMs - create and save if don't exist
        gnd_dem, veg_dem = self.get_dems(channel=osm_channel)

        # Create the channel width object
        channel_width = bathymetry_estimation.ChannelCharacteristics(
            gnd_dem=gnd_dem,
            veg_dem=veg_dem,
            cross_section_spacing=cross_section_spacing,
            resolution=resolution,
            debug=self.debug,
        )

        return channel_width, smoothed_osm_channel

    def _rolling_mean_with_padding(
        self, data: geopandas.GeoSeries, number_of_samples: int
    ) -> numpy.ndarray:
        """Calculate the rolling mean of an array after padding the array with
        the edge value to ensure the derivative is smooth.

        Parameters
        ----------

        data
            The array to pad then smooth.
        number_of_samples
            The width in samples of the averaging filter
        """
        assert (
            number_of_samples > 0 and type(number_of_samples) == int
        ), "Must be more than 0 and an int"
        rolling_mean = (
            numpy.convolve(
                numpy.pad(data, int(number_of_samples / 2), "symmetric"),
                numpy.ones(number_of_samples),
                "valid",
            )
            / number_of_samples
        )
        return rolling_mean

    def _apply_upstream_smoothing(self, width_values: geopandas.GeoDataFrame) -> str:
        """Apply upstream smoothing to the width, flat_widths, thresholds, and
        slope. Store in the widths values Then result the smoothing label.
        """

        # Get the level of upstream smoothing to apply
        upstream_smoothing_factor = self.get_bathymetry_instruction(
            "upstream_smoothing_factor"
        )
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        # Cycle through and caluclate the rolling mean
        label = f"{cross_section_spacing*upstream_smoothing_factor/1000}km"

        # Apply smoothing via '_rolling_mean_with_padding' to each measurement
        # Slope
        width_values[f"slope_mean_{label}"] = self._rolling_mean_with_padding(
            width_values["slope"], upstream_smoothing_factor
        )
        # Width
        widths_no_nan = width_values["valid_widths"].interpolate(
            "index", limit_direction="both"
        )
        width_values[f"widths_mean_{label}"] = self._rolling_mean_with_padding(
            widths_no_nan, upstream_smoothing_factor
        )
        # Flat width
        flat_widths_no_nan = width_values["valid_flat_widths"].interpolate(
            "index", limit_direction="both"
        )
        width_values[f"flat_widths_mean_{label}"] = self._rolling_mean_with_padding(
            flat_widths_no_nan, upstream_smoothing_factor
        )
        # Threshold
        thresholds_no_nan = width_values["valid_threhold"].interpolate(
            "index", limit_direction="both"
        )
        width_values[f"thresholds_mean_{label}"] = self._rolling_mean_with_padding(
            thresholds_no_nan, upstream_smoothing_factor
        )

        return label

    def calculate_river_bed_elevations(self):
        """Calculate and save depth estimates along the channel using various
        approaches.

        """

        # Read in the flow file and calcaulate the depths - write out the results
        width_values = geopandas.read_file(
            self.get_result_file_path(key="river_characteristics")
        )
        width_values["source"] = "river"  # Specify as coming form river estimation
        channel = self.get_network_channel()

        # Match each channel midpoint to a reach ID - based on what reach is closest
        width_values["id"] = (
            numpy.ones(len(width_values["widths"]), dtype=float) * numpy.nan
        )
        # Add the friction and flow values to the widths and slopes
        width_values["mannings_n"] = numpy.zeros(len(width_values["id"]), dtype=int)
        width_values["flow"] = numpy.zeros(len(width_values["id"]), dtype=int)
        for i, row in width_values.iterrows():
            if row.geometry is not None and not row.geometry.is_empty:
                distances = channel.channel.distance(width_values.loc[i].geometry)
                width_values.loc[i, ("id", "flow", "mannings_n")] = channel.channel[
                    distances == distances.min()
                ][["id", "flow", "mannings_n"]].min()
        # Fill in any missing values
        width_values["id"] = (
            width_values["id"].fillna(method="ffill").fillna(method="bfill")
        )
        width_values["id"] = width_values["id"].astype("int")
        width_values["flow"] = (
            width_values["flow"].fillna(method="ffill").fillna(method="bfill")
        )
        width_values["mannings_n"] = (
            width_values["mannings_n"].fillna(method="ffill").fillna(method="bfill")
        )

        # Get the level of upstream smoothing to apply
        label = self._apply_upstream_smoothing(width_values)

        # Names of values to use
        slope_name = f"slope_mean_{label}"
        min_z_name = "min_z_centre_unimodal"
        width_name = f"widths_mean_{label}"
        flat_width_name = f"flat_widths_mean_{label}"
        threshold_name = f"thresholds_mean_{label}"

        # Enfore a minimum slope - as specified in the instructions
        minimum_slope = self.get_bathymetry_instruction("minimum_slope")
        logging.info(f"Enforcing a minimum slope of {minimum_slope}")
        width_values.loc[
            width_values[slope_name] < minimum_slope, slope_name
        ] = minimum_slope

        # Calculate depths and bed elevation using the Neal et al approach (Uniform flow
        # theory)
        full_bank_depth = self._calculate_neal_et_al_depth(
            width_values=width_values,
            width_name=width_name,
            slope_name=slope_name,
            threshold_name=threshold_name,
        )
        if not (full_bank_depth >= 0).all():
            logging.warning(
                "Unexpected negative depths. Setting to zero. "
                "Check `river_characteristics.geojson` file."
            )
            full_bank_depth[full_bank_depth < 0] = 0
        active_channel_bank_depth = self._convert_full_bank_to_channel_depth(
            full_bank_depth=full_bank_depth,
            threshold_name=threshold_name,
            flat_width_name=flat_width_name,
            full_bank_width_name=width_name,
            width_values=width_values,
        )
        # Ensure valid depths before converting to bed elevations
        if not ((full_bank_depth - width_values[threshold_name]) >= 0).all():
            mask = (full_bank_depth - width_values[threshold_name]) < 0
            logging.warning(
                "Depths less than the thresholds. Try reduce the "
                "`max_bank_height`. Setting to thresholds. "
                f"{mask.sum()} affected."
            )
            full_bank_depth[mask] = width_values[threshold_name][mask]
        width_values["bed_elevation_Neal_et_al"] = (
            width_values[min_z_name] - active_channel_bank_depth
        )
        if self.debug:
            # Optionally write out additional depth information
            width_values["area_adjusted_depth_Neal_et_al"] = active_channel_bank_depth
            width_values["depth_Neal_et_al"] = full_bank_depth
        # Calculate depths and bed elevation using the Rupp & Smart approach (Hydrologic
        # geometry)
        full_bank_depth = self._calculate_rupp_and_smart_depth(
            width_values=width_values,
            width_name=width_name,
            slope_name=slope_name,
            threshold_name=threshold_name,
        )
        if not (full_bank_depth >= 0).all():
            logging.warning(
                "Unexpected negative depths. Setting to zero. "
                "Check `river_characteristics.geojson` file."
            )
            full_bank_depth[full_bank_depth < 0] = 0
        active_channel_bank_depth = self._convert_full_bank_to_channel_depth(
            full_bank_depth=full_bank_depth,
            threshold_name=threshold_name,
            flat_width_name=flat_width_name,
            full_bank_width_name=width_name,
            width_values=width_values,
        )
        # Ensure valid depths before converting to bed elevations
        if not ((full_bank_depth - width_values[threshold_name]) >= 0).all():
            mask = (full_bank_depth - width_values[threshold_name]) < 0
            logging.warning(
                "Depths less than the thresholds. Try reduce the "
                "`max_bank_height`. Setting to thresholds. "
                f"{mask.sum()} affected."
            )
            full_bank_depth[mask] = width_values[threshold_name][mask]
        width_values["bed_elevation_Rupp_and_Smart"] = (
            width_values[min_z_name] - active_channel_bank_depth
        )
        if self.debug:
            # Optionally write out additional depth information
            width_values[
                "area_adjusted_depth_Rupp_and_Smart"
            ] = active_channel_bank_depth
            width_values["depth_Rupp_and_Smart"] = full_bank_depth
        # Save the bed elevations
        values_to_save = [
            "geometry",
            "bed_elevation_Neal_et_al",
            "bed_elevation_Rupp_and_Smart",
            min_z_name,
            width_name,
            flat_width_name,
            "source",
        ]
        if self.debug:
            # Optionally write out additional depth information
            values_to_save.extend(
                [
                    "depth_Neal_et_al",
                    "depth_Rupp_and_Smart",
                    "area_adjusted_depth_Neal_et_al",
                    "area_adjusted_depth_Rupp_and_Smart",
                ]
            )
        # Save the widths and depths
        width_values[values_to_save].rename(
            columns={min_z_name: "bank_height", width_name: "width"}
        ).to_file(self.get_result_file_path(key="river_bathymetry"))

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

        # Calculate the estimated full bank flow area
        full_flood_area = full_bank_depth * width_values[full_bank_width_name]

        # Threshold is the depth of flood water above the water surface
        threshold = width_values[threshold_name]

        # Calculate the area of flood waters (i.e. above the water surface)
        above_water_area = threshold * width_values[full_bank_width_name]
        # Calculate the above water area that is actually banks (assume linear)
        exposed_bank_area = (
            threshold
            * (width_values[full_bank_width_name] - width_values[flat_width_name])
            / 2
        )
        if not (exposed_bank_area >= 0).all():
            logging.warning(
                "The exposed bank area is not always postive. " "Setting to zero."
            )
            exposed_bank_area[exposed_bank_area < 0] = 0
        extra_flood_area = above_water_area - exposed_bank_area

        # The area to convert to the active 'flat' channel depth
        flat_flow_area = full_flood_area - extra_flood_area

        # Calculate the depth from the area
        flat_flow_depth = flat_flow_area / width_values[flat_width_name]

        # Ensure not negative
        if not (flat_flow_depth >= 0).all():
            logging.warning(
                "Negative area-adjusted depths. Try reduce the "
                "`max_bank_height`. Setting to zero. # affected: "
                f"{len(flat_flow_depth[flat_flow_depth < 0])}."
            )
            flat_flow_depth[flat_flow_depth < 0] = 0

        return flat_flow_depth

    def estimate_river_mouth_fan(self):
        """Calculate and save depth estimates along the river mouth fan."""

        # Required inputs
        crs = self.get_crs()["horizontal"]
        cross_section_spacing = self.get_bathymetry_instruction("cross_section_spacing")
        river_bathymetry_file = self.get_result_file_path(key="river_bathymetry")
        river_polygon_file = self.get_result_file_path(key="river_polygon")
        ocean_contour_file = self.get_vector_or_raster_paths(
            key="ocean_contours", data_type="vector"
        )[0]
        aligned_channel_file = self.get_result_file_path(key="aligned")
        ocean_contour_depth_label = self.get_instruction_general(
            key="z_labels", subkey="ocean"
        )

        # Create fan object
        fan = geometry.RiverMouthFan(
            aligned_channel_file=aligned_channel_file,
            river_bathymetry_file=river_bathymetry_file,
            river_polygon_file=river_polygon_file,
            ocean_contour_file=ocean_contour_file,
            crs=crs,
            cross_section_spacing=cross_section_spacing,
            elevation_labels=[
                "bed_elevation_Neal_et_al",
                "bed_elevation_Rupp_and_Smart",
            ],
            ocean_contour_depth_label=ocean_contour_depth_label,
        )

        # Estimate the fan extents and bathymetry
        fan_polygon, fan_bathymetry = fan.polygon_and_bathymetry()
        river_bathymetry = geopandas.read_file(river_bathymetry_file)
        river_polygon = geopandas.read_file(river_polygon_file)

        # Combine and save the river and fan geometries
        fan_bathymetry["source"] = "fan"
        fan_bathymetry["bank_height"] = numpy.nan  # No bank height info in ocean
        combined_bathymetry = geopandas.GeoDataFrame(
            pandas.concat(
                [fan_bathymetry.loc[::-1], river_bathymetry], ignore_index=True
            ),
            crs=river_bathymetry.crs,
        )
        combined_bathymetry.to_file(river_bathymetry_file)
        combined_polygon = river_polygon.overlay(fan_polygon, how="union")
        combined_polygon = (
            geopandas.GeoDataFrame(
                geometry=combined_polygon.buffer(self.get_resolution())
            )
            .dissolve()
            .buffer(-self.get_resolution())
        )
        combined_polygon.to_file(river_polygon_file)

    def run(self):
        """This method extracts a main channel then executes the DemGeneration
        pipeline to produce a DEM before sampling this to extimate width, slope
        and eventually depth."""

        logging.info("Adding river and fan bathymetry if it doesn't already exist.")

        # Ensure the results folder has been created
        self.create_results_folder()

        # Characterise river channel if not already done - may generate DEMs
        if not self.channel_characteristics_exist():
            self.characterise_channel()
        # Estimate channel and fan depths if not already done
        if not self.channel_bathymetry_exist():
            logging.info("Estimating the channel bathymetry.")
            print("Estimating the channel bathymetry.")

            # Calculate and save river bathymetry depths
            self.calculate_river_bed_elevations()
            # check if the river mouth is to be estimated
            if self.get_bathymetry_instruction("estimate_fan"):
                logging.info("Estimating the fan bathymetry.")
                print("Estimating the fan bathymetry.")
                self.estimate_river_mouth_fan()
        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "rivers_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)


class WaterwayBedElevationEstimator(BaseProcessor):
    """WaterwayBedElevationGenerator executes a pipeline to pull in OpenStreetMap waterway
    and tunnel information. A DEM is generated of the surrounding area and this used to
    unblock waterways and tunnels - by taking the lowest value in the area around a tunnel
    to be the tunnel elevation, and ensuring opne waterways flow downhill.

    """

    def __init__(self, json_instructions: json, debug: bool = True):
        super(WaterwayBedElevationEstimator, self).__init__(
            json_instructions=json_instructions
        )

        self.debug = debug

    def get_result_file_name(self, key: str) -> str:
        """Return the name of the file to save."""

        # key to output name mapping
        name_dictionary = {
            "raw_dem": "waterways_raw_dem.nc",
            "open_polygon": "open_waterways_polygon.geojson",
            "open_elevation": "open_waterways_elevation.geojson",
            "closed_polygon": "closed_waterways_polygon.geojson",
            "closed_elevation": "closed_waterways_elevation.geojson",
            "waterways_polygon": "waterways_polygon.geojson",
            "waterways": "waterways.geojson",
        }
        return name_dictionary[key]

    def get_result_file_path(self, key: str) -> pathlib.Path:
        """Return the file name of the file to save with the local cache path.

        Parameters:
            instructions  The json instructions defining the behaviour
        """

        subfolder = self.get_instruction_path("subfolder")

        name = self.get_result_file_name(key=key)

        return subfolder / name

    def waterway_elevations_exists(self):
        """Check to see if the waterway and culvert bathymeties have already been
        estimated."""

        closed_polygon_file = self.get_result_file_path(key="closed_polygon")
        closed_elevation_file = self.get_result_file_path(key="closed_elevation")
        open_polygon_file = self.get_result_file_path(key="open_polygon")
        open_elevation_file = self.get_result_file_path(key="open_elevation")
        if (
            closed_polygon_file.is_file()
            and closed_elevation_file.is_file()
            and open_polygon_file.is_file()
            and open_elevation_file.is_file()
        ):
            return True
        else:
            return False

    def minimum_elevation_in_polygon(
        self, geometry: shapely.geometry.Polygon, dem: xarray.Dataset
    ):
        """Determine the minimum value in each polygon. Select only coordinates
        within the polygon bounding box before clipping to the bounding box and
        then returning the minimum elevation."""

        # Index in polygon bbox
        bbox = geometry.bounds

        # Select DEM within the bounding box - allow ascending or decending dimensions
        y_slice = (
            slice(bbox[1], bbox[3])
            if dem.y[-1] - dem.y[0] > 0
            else slice(bbox[3], bbox[1])
        )
        x_slice = (
            slice(bbox[0], bbox[2])
            if dem.x[-1] - dem.x[0] > 0
            else slice(bbox[2], bbox[0])
        )
        small_z = dem.z.sel(x=x_slice, y=y_slice)

        # clip to polygon and return minimum elevation
        return float(small_z.rio.clip([geometry]).min())

    def estimate_closed_elevations(
        self, waterways: geopandas.GeoDataFrame, dem: xarray.Dataset
    ):
        """Sample the DEM around the tunnels to estimate the bed elevation."""

        # Check if already generated
        polygon_file = self.get_result_file_path(key="closed_polygon")
        elevation_file = self.get_result_file_path(key="closed_elevation")
        if polygon_file.is_file() and elevation_file.is_file():
            print("Closed waterways already recorded. ")
            logging.info(
                "Estimating closed waterway and tunnel bed elevation from OpenStreetMap."
            )
            return
        # If not - estimate elevations along close waterways
        closed_waterways = waterways[waterways["tunnel"]]
        closed_waterways["polygon"] = closed_waterways.buffer(closed_waterways["width"])
        # If no closed waterways write out empty files and return
        if len(closed_waterways) == 0:
            closed_waterways["elevation"] = []
            closed_waterways.set_geometry("polygon", drop=True)[
                ["geometry", "width", "elevation"]
            ].to_file(polygon_file)
            closed_waterways.drop(columns=["polygon"]).to_file(elevation_file)
            return

        # Sample the minimum elevation at each tunnel
        elevations = closed_waterways.apply(
            lambda row: self.minimum_elevation_in_polygon(
                geometry=row["polygon"], dem=dem
            ),
            axis=1,
        )

        # Create sampled points to go with the sampled elevations
        points = closed_waterways["geometry"].apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    # Ensure even spacing across the length of the waterway
                    row.interpolate(
                        i
                        * row.length
                        / int(numpy.ceil(row.length / self.get_resolution()))
                    )
                    for i in range(
                        int(numpy.ceil(row.length / self.get_resolution())) + 1
                    )
                ]
            )
        )
        points = geopandas.GeoDataFrame(
            {
                "elevation": elevations,
                "geometry": points,
                "width": closed_waterways["width"],
            },
            crs=closed_waterways.crs,
        )
        closed_waterways["elevation"] = elevations
        # Remove any NaN areas (where no LiDAR data to estimate elevations)
        nan_filter = (
            points.explode(ignore_index=False, index_parts=True)["elevation"]
            .notnull()
            .groupby(level=0)
            .all()
            .values
        )
        if not nan_filter.all():
            logging.warning(
                "Some open waterways are being ignored as there is not enough data to "
                "estimate their elevations."
            )
        points_exploded = points[nan_filter].explode(
            ignore_index=False, index_parts=True
        )
        closed_waterways = closed_waterways[nan_filter]

        # Save out polygons and elevations
        closed_waterways.set_geometry("polygon", drop=True)[
            ["geometry", "width", "elevation"]
        ].to_file(polygon_file)
        points_exploded.to_file(elevation_file)

    def estimate_open_elevations(
        self, waterways: geopandas.GeoDataFrame, dem: xarray.Dataset
    ):
        """Sample the DEM along the open waterways to enforce a decreasing elevation."""

        # Check if already generated
        polygon_file = self.get_result_file_path(key="open_polygon")
        elevation_file = self.get_result_file_path(key="open_elevation")
        if polygon_file.is_file() and elevation_file.is_file():
            print("Open waterways already recorded. ")
            logging.info(
                "Estimating open waterway and tunnel bed elevation from OpenStreetMap."
            )
            return
        # If not - estimate the elevations along the open waterways
        open_waterways = waterways[numpy.logical_not(waterways["tunnel"])]

        # sample the ends of the waterway - sample over a polygon at each end
        polygons = open_waterways.interpolate(0).buffer(open_waterways["width"])
        open_waterways["start_elevation"] = polygons.apply(
            lambda geometry: self.minimum_elevation_in_polygon(
                geometry=geometry, dem=dem
            )
        )
        polygons = open_waterways.interpolate(open_waterways.length).buffer(
            open_waterways["width"]
        )
        open_waterways["end_elevation"] = polygons.geometry.apply(
            lambda geometry: self.minimum_elevation_in_polygon(
                geometry=geometry, dem=dem
            )
        )

        # Remove any waterways without data to assess elevations
        nan_filter = (
            open_waterways[["start_elevation", "end_elevation"]]
            .notnull()
            .all(axis=1)
            .values
        )
        if not nan_filter.all():
            logging.warning(
                "Some open waterways are being ignored as there is not enough data to "
                "estimate their elevations."
            )
        open_waterways = open_waterways[nan_filter]

        # save out the polygons
        open_waterways.buffer(open_waterways["width"]).to_file(polygon_file)

        # Sample down-slope location along each line
        def sample_location_down_slope(row):
            """Sample evenly space poinst along polylines in the downslope direction"""

            if row["start_elevation"] > row["end_elevation"]:
                sample_range = range(
                    int(numpy.ceil(row.geometry.length / self.get_resolution())) + 1
                )
            else:
                sample_range = range(
                    int(numpy.ceil(row.geometry.length / self.get_resolution())),
                    -1,
                    -1,
                )
            sampled_sampled_multipoints = [
                # Ensure even spacing across the length of the waterway
                row.geometry.interpolate(
                    i
                    * row.geometry.length
                    / int(numpy.ceil(row.geometry.length / self.get_resolution()))
                )
                for i in sample_range
            ]
            sampled_multipoints = shapely.geometry.MultiPoint(
                sampled_sampled_multipoints
            )

            return sampled_multipoints

        open_waterways["points"] = open_waterways.apply(
            lambda row: sample_location_down_slope(row=row),
            axis=1,
        )

        open_waterways = open_waterways.set_geometry("points", drop=True)[
            ["geometry", "width"]
        ]
        # sort index needed to ensure correct behaviour of the explode function
        open_waterways = open_waterways.sort_index(ascending=True).explode(
            ignore_index=False, index_parts=True, column="geometry"
        )
        open_waterways["polygons"] = open_waterways.buffer(open_waterways["width"])
        # Sample the minimum elevations along each  open waterway
        open_waterways["elevation"] = open_waterways["polygons"].apply(
            lambda geometry: self.minimum_elevation_in_polygon(
                geometry=geometry, dem=dem
            )
        )  # TODO - look to optimise as runs slowely
        # Check open waterways take into account culvert bed elevations
        closed_polygons = geopandas.read_file(
            self.get_result_file_path(key="closed_polygon")
        )
        # Check each culvert
        if len(closed_polygons) > 0:
            for closed_polygon, closed_elevation in zip(
                closed_polygons["geometry"], closed_polygons["elevation"]
            ):
                # Take the nearest closed elevation if lower
                elevations_near_culvert = open_waterways.clip(closed_polygon)
                indices_to_replace = elevations_near_culvert.index[
                    elevations_near_culvert["elevation"] > closed_elevation
                ]
                open_waterways.loc[indices_to_replace, "elevation"] = closed_elevation
        # Ensure the sampled elevations monotonically decrease
        for index, waterway_points in open_waterways.groupby(level=0):
            open_waterways.loc[(index,), ("elevation")] = numpy.fmin.accumulate(
                waterway_points["elevation"]
            )
        # Save bathymetry
        open_waterways[["geometry", "width", "elevation"]].to_file(elevation_file)

    def create_dem(self, waterways: geopandas.GeoDataFrame) -> xarray.Dataset:
        """Create and return a DEM at a resolution 1.5x the waterway width."""

        dem_file = self.get_result_file_path(key="raw_dem")

        # Load already created DEM file in
        if not dem_file.is_file():
            # Create DEM over the waterway region
            # Save out the waterway polygons as a file with a single multipolygon
            waterways_polygon_file = self.get_result_file_path(key="waterways_polygon")
            waterways_polygon = waterways.buffer(waterways["width"])
            waterways_polygon = geopandas.GeoDataFrame(
                geometry=[shapely.ops.unary_union(waterways_polygon.geometry.array)],
                crs=waterways_polygon.crs,
            )
            waterways_polygon.to_file(waterways_polygon_file)

            # Create DEM generation instructions
            dem_instructions = self.instructions
            dem_instruction_paths = dem_instructions["data_paths"]
            dem_instruction_paths["extents"] = self.get_result_file_name(
                key="waterways_polygon"
            )
            dem_instruction_paths["raw_dem"] = self.get_result_file_name(key="raw_dem")

            # Create the ground DEM file if this has not be created yet!
            print("Generating waterway DEM.")
            runner = RawLidarDemGenerator(self.instructions)
            runner.run()
        # Load in the DEM
        dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True).squeeze(
            "band", drop=True
        )
        return dem

    def download_osm_values(self) -> bool:
        """Download OpenStreetMap waterways and tunnels within the catchment BBox."""

        waterways_file_path = self.get_result_file_path(key="waterways")

        if waterways_file_path.is_file():  # already created. Load in.
            waterways = geopandas.read_file(waterways_file_path)
        else:  # Download from OSM
            # Create area to query within
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
            if "osm_date" in self.instructions["drains"]:
                waterways = overpass.query(
                    query,
                    date=self.instructions["drains"]["osm_date"],
                    timeout=60,
                )
            else:
                waterways = overpass.query(query, timeout=60)

            # Extract information
            element_dict = {
                "geometry": [],
                "OSM_id": [],
                "waterway": [],
                "tunnel": [],
            }

            for element in waterways.elements():
                element_dict["geometry"].append(element.geometry())
                element_dict["OSM_id"].append(element.id())
                element_dict["waterway"].append(element.tags()["waterway"])
                element_dict["tunnel"].append("tunnel" in element.tags().keys())
            waterways = (
                geopandas.GeoDataFrame(element_dict, crs=self.OSM_CRS)
                .to_crs(self.catchment_geometry.crs["horizontal"])
                .set_index("OSM_id", drop=True)
            )

            # Remove polygons
            waterways = waterways[waterways.geometry.type == "LineString"].sort_index(
                ascending=True
            )

            # Get specified widths
            widths = self.instructions["drains"]["widths"]
            # Check if rivers are specified and remove if not
            if "ditch" not in widths.keys():
                widths["ditch"] = widths["drain"]
            # Identify and remove undefined waterway types
            for waterway_label in waterways["waterway"].unique():
                if waterway_label not in widths.keys():
                    waterways = waterways[waterways["waterway"] != waterway_label]
                    print(
                        f"{waterway_label} is not in the specified widths and"
                        " is being removed"
                    )
                    logging.info(
                        f"{waterway_label} is not in the specified widths and"
                        " is being removed"
                    )
            # Add width label
            waterways["width"] = waterways["waterway"].apply(
                lambda waterway: widths[waterway]
            )
            # Clip to land
            waterways = waterways.clip(self.catchment_geometry.land).sort_index(
                ascending=True
            )

            # Save file
            waterways.to_file(waterways_file_path)
        return waterways

    def run(self):
        """This method runs a pipeline that:
        * downloads all tunnels and waterways within a catchment.
        * creates and samples a DEM around each feature to estimate the bed
          elevation.
        * saves out extents and bed elevations of the waterway and tunnel network
        """

        # Don't reprocess if already estimated
        if self.waterway_elevations_exists():
            logging.info("Waterway and tunnel bed elevations already estimated.")
            return
        logging.info("Estimating waterway and tunnel bed elevation from OpenStreetMap.")

        # Ensure the results folder has been created
        self.create_results_folder()

        # Load in catchment
        self.catchment_geometry = self.create_catchment()

        # Download waterways and tunnels from OSM
        waterways = self.download_osm_values()

        # Create a DEM where the waterways and tunnels are
        dem = self.create_dem(waterways=waterways)

        # Estimate the waterway and tunnel bed elevations from the DEM
        self.estimate_closed_elevations(waterways=waterways, dem=dem)
        self.estimate_open_elevations(waterways=waterways, dem=dem)

        if self.debug:
            # Record the parameter used during execution - append to existing
            with open(
                self.get_instruction_path("subfolder") / "waterway_instructions.json",
                "a",
            ) as file_pointer:
                json.dump(self.instructions, file_pointer)
