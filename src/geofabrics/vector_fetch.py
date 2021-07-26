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
    PATH_API = "/services/query/v1/vector.json"
    LINZ_CRS = "EPSG:4326"

    MAX_RESULTS = 100
    MAX_RADIUS = 100000

    def __init__(self, key: str, layer: int, catchment_geometry: geometry.CatchmentGeometry,
                 cache_path: typing.Union[str, pathlib.Path], verbose: bool = False):
        """ Load in Vector dataset processing chain.

        Zip results
        """

        self.key = key
        self.layer = layer
        self.catchment_geometry = catchment_geometry
        self.cache_path = pathlib.Path(cache_path)
        self.verbose = verbose

        self.json_string = None
        self.tile_names = None

    def run(self):
        """ Query for tiles within a catchment construct a list of tiles names within the catchment """
        self.tile_names = self.get_tiles_inside_catchment()

    def query_vector_inside_radius(self, x: float, y: float, radius: float):
        """ Function to check for tiles in search region using the Koordinates vector query API
        https://help.koordinates.com/query-api-and-web-services/vector-query/

        x and y define the centre of the search radius in decimal degrees (WGS84/EPSG:4326), and the radius is the
        search radius in metres """

        radius = min(math.ceil(radius), self.MAX_RADIUS)

        api_queary = {
            "key": self.key,
            "layer": self.layer,
            "x": x,
            "y": y,
            "max_results": self.MAX_RESULTS,
            "radius": radius,
            "geometry": "true",
            "with_field_names": "true",
            "Accept-Encoding": "gzip"
        }

        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API, self.PATH_API, "", "", ""))

        response = requests.get(data_url, params=api_queary, stream=True)
        response.raise_for_status()
        return response.json()

    def get_tiles_inside_catchment(self):
        """ Get a list of tiles within the catchment boundary """

        # radius in metres
        catchment_bounds = self.catchment_geometry.catchment.geometry.bounds
        width = (catchment_bounds['maxx'].max() - catchment_bounds['minx'].min()) / 2
        height = (catchment_bounds['maxy'].max() - catchment_bounds['miny'].min()) / 2
        catchment_radius = max(width, height)

        assert catchment_radius < self.MAX_RADIUS, "The catchment region is larger than that supported by the " \
            "Koordinates vector API query and support has not yet been added for pulling tiles from larger areas"

        # x and y in decimal degrees (WGS84/EPSG:4326)
        catchment_linz_crs = self.catchment_geometry.catchment.geometry.to_crs(self.LINZ_CRS)
        catchment_bounds = catchment_linz_crs.bounds
        x = (catchment_bounds['minx'].min() + catchment_bounds['maxx'].max()) / 2
        y = (catchment_bounds['miny'].min() + catchment_bounds['maxy'].max()) / 2
        json_response = self.query_vector_inside_radius(x, y, catchment_radius)

        feature_collection = json_response['vectorQuery']['layers']['105448']

        assert feature_collection['crs']['properties']['name'] == f"{self.LINZ_CRS}", "Feature collection has an " \
            f"unexpected CRS of {feature_collection['crs']['properties']['name']}, when {self.LINZ_CRS} was expected."

        f = matplotlib.pyplot.figure(figsize=(10, 10))
        gs = f.add_gridspec(1, 1)

        ax1 = f.add_subplot(gs[0, 0])
        catchment_linz_crs.plot(ax=ax1)
        
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
            if catchment_linz_crs.intersects(tile).any():
                tile_names.append(json_tile['properties']['tilename'])
                matplotlib.pyplot.plot(*tile.exterior.xy, color="red")
            else:
                matplotlib.pyplot.plot(*tile.exterior.xy, color="blue")

        return tile_names
