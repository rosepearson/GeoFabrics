# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:10:55 2021

@author: pearsonra
"""

import urllib
import pathlib
import requests
import shapely
import geopandas
import typing
import math
from . import geometry
import matplotlib
import matplotlib.pyplot


class LinzTiles:
    """ A class to manage fetching Vector data from LINZ.

    API details at: https://help.koordinates.com/query-api-and-web-services/vector-query/
    """

    SCHEME = "https"
    NETLOC_API = "data.linz.govt.nz"
    JSON_PATH_API = "/services/query/v1/vector.json"
    WFS_PATH_API_START = "/services;key="
    WFS_PATH_API_END = "/wfs"
    LINZ_CRS = "EPSG:4326"

    MAX_RESULTS = 100
    MAX_RADIUS = 100000

    def __init__(self, key: str, catchment_geometry: geometry.CatchmentGeometry, verbose: bool = False):
        """ Load in Vector dataset processing chain.

        Zip results
        """

        self.key = key
        self.catchment_geometry = catchment_geometry
        self.verbose = verbose

        self.json_string = None
        self.tile_names = None

    def run(self, layer: int, prefix: str):
        """ Query for tiles within a catchment for a specified layer and return a list of the tile names within
        the catchment """

        tile_names = self.get_tiles_inside_catchment(layer, prefix)

        return tile_names

    def query_vector_wfs(self, bounds, layer: int):
        """ Function to check for tiles in search rectangle using the LINZ WFS vector query API
        https://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/wfs-spatial-filtering

        Note that depending on the LDS layer the geometry name may be 'shape' - most property/titles,
        or GEOMETRY - most other layers including Hydrographic and Topographic data.

        bounds defines the bounding box containing in the catchment boundary CRS specified by SRSName """

        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API,
                                            f"{self.WFS_PATH_API_START}{self.key}{self.WFS_PATH_API_END}",
                                            "", "", ""))

        api_queary = {
            "service": "WFS",
            "version": 2.0,
            "request": "GetFeature",
            "typeNames": f"layer-{layer}",
            "outputFormat": "json",
            "SRSName": f"EPSG:{self.catchment_geometry.crs}",
            "cql_filter": f"bbox(GEOMETRY, {bounds['maxy'].max()}, {bounds['maxx'].max()}, " +
                          f"{bounds['miny'].min()}, {bounds['minx'].min()})"
        }

        response = requests.get(data_url, params=api_queary, stream=True)

        response.raise_for_status()
        return response.json()

    def get_tiles_inside_catchment(self, layer: int, prefix: str):
        """ Get a list of tiles within the catchment boundary """

        # radius in metres
        catchment_bounds = self.catchment_geometry.catchment.geometry.bounds
        feature_collection = self.query_vector_wfs(catchment_bounds, layer)

        if self.verbose:  # Plot catchment
            figure = matplotlib.pyplot.figure(figsize=(10, 10))
            gs = figure.add_gridspec(1, 1)

            ax1 = figure.add_subplot(gs[0, 0])
            self.catchment_geometry.catchment.plot(ax=ax1)

        # Cycle through each tile getting name and coordinates
        tile_names = []
        for json_tile in feature_collection['features']:
            json_geometry = json_tile['geometry']

            assert json_geometry['type'] == 'Polygon', f"Unexpected tile geometry of type {json_geometry['type']} " \
                "instead of Polygon"

            tile_coords = json_geometry['coordinates'][0]
            tile = shapely.geometry.Polygon([(tile_coords[0][0], tile_coords[0][1]),
                                             (tile_coords[1][0], tile_coords[1][1]),
                                             (tile_coords[2][0], tile_coords[2][1]),
                                             (tile_coords[3][0], tile_coords[3][1])])

            # check intersection of tile and catchment in LINZ CRS
            if self.catchment_geometry.catchment.intersects(tile).any():
                tile_names.append(f"{prefix}{json_tile['properties']['tilename']}.laz")

                if self.verbose:  # Plot overlapping catchment in red
                    matplotlib.pyplot.plot(*tile.exterior.xy, color="red")
            elif self.verbose:  # Plot outside catchment in red
                matplotlib.pyplot.plot(*tile.exterior.xy, color="blue")

        return sorted(tile_names)
