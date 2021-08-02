# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:10:55 2021

@author: pearsonra
"""

import urllib
import requests
import shapely
import shapely.geometry
import geopandas
from . import geometry


class Linz:
    """ A class to manage fetching Vector data from LINZ.

    API details at: https://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/wfs-spatial-filtering

    The specified vector layer is queried each time run is called and any vectors passing though the catchment defined
    in the catchment_geometry are returned. """

    SCHEME = "https"
    NETLOC_API = "data.linz.govt.nz"
    WFS_PATH_API_START = "/services;key="
    WFS_PATH_API_END = "/wfs"

    def __init__(self, key: str, catchment_geometry: geometry.CatchmentGeometry, verbose: bool = False):
        """ Load in vector information from LINZ. Specify the layer to import during run.
        """

        self.key = key
        self.catchment_geometry = catchment_geometry
        self.verbose = verbose

    def run(self, layer: int, geometry_type: str):
        """ Query for tiles within a catchment for a specified layer and return a list of the vector features names
        within the catchment """

        features = self.get_features_inside_catchment(layer, geometry_type)

        return features

    def query_vector_wfs(self, bounds, layer: int, geometry_type: str):
        """ Function to check for tiles in search rectangle using the LINZ WFS vector query API
        https://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/wfs-spatial-filtering

        Note that depending on the LDS layer the geometry name may be 'shape' - most property/titles,
        or GEOMETRY - most other layers including Hydrographic and Topographic data.

        bounds defines the bounding box containing in the catchment boundary """

        data_url = urllib.parse.urlunparse((self.SCHEME, self.NETLOC_API,
                                            f"{self.WFS_PATH_API_START}{self.key}{self.WFS_PATH_API_END}",
                                            "", "", ""))

        api_query = {
            "service": "WFS",
            "version": 2.0,
            "request": "GetFeature",
            "typeNames": f"layer-{layer}",
            "outputFormat": "json",
            "SRSName": f"EPSG:{self.catchment_geometry.crs}",
            "cql_filter": f"bbox({geometry_type}, {bounds['maxy'].max()}, {bounds['maxx'].max()}, " +
                          f"{bounds['miny'].min()}, {bounds['minx'].min()}, " +
                          f"'urn:ogc:def:crs:EPSG:{self.catchment_geometry.crs}')"
        }

        response = requests.get(data_url, params=api_query, stream=True)

        response.raise_for_status()
        return response.json()

    def get_features_inside_catchment(self, layer: int, geometry_type: str):
        """ Get a list of features within the catchment boundary """

        # radius in metres
        catchment_bounds = self.catchment_geometry.catchment.geometry.bounds
        feature_collection = self.query_vector_wfs(catchment_bounds, layer, geometry_type)

        # Cycle through each feature getting name and coordinates
        features = []
        for feature in feature_collection['features']:

            shapely_geometry = shapely.geometry.shape(feature['geometry'])

            # check intersection of tile and catchment in LINZ CRS
            if self.catchment_geometry.catchment.intersects(shapely_geometry).any():

                # convert any one Polygon MultiPolygons to a straight Polygon
                if (shapely_geometry.geometryType() == 'MultiPolygon' and len(shapely_geometry) == 1):
                    shapely_geometry = shapely_geometry[0]

                features.append(shapely_geometry)

        # Convert to a geopandas dataframe
        if len(features) > 0:
            features = geopandas.GeoDataFrame(index=list(range(len(features))), geometry=features,
                                              crs=self.catchment_geometry.crs)
        else:
            features = None

        return features
