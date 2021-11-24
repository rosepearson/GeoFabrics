# -*- coding: utf-8 -*-
"""
This module contains classes associated with estiming channel bathymetry information.
"""

import geopandas


def get_up_stream_reaches(rec_network: geopandas.GeoDataFrame,
                          reach_id: int,
                          reaches: geopandas.GeoDataFrame = None,
                          max_iterations: int = 10000,
                          iteration: int = 0):
    """ A recurive function to trace all up reaches from the reach_id.
    The default values for reaches and iteration are set for the
    initial call to the recursive function.

    Parameters
    ----------

    rec_network
        Contains association information between upstream and
        downstream reaches.
    reach_id
        The `nzsegment` id of the reach to trace upstream from.
    reaches
        The already traced downstream reaches to append to.
    max_iterations
        The maximum number of iterations along a single strand to trace
        upstream.
    iteration
        The number of iterations traveled upstream.
    """
    if reaches is None:
        reaches = rec_network[rec_network['nzsegment'] == reach_id]
    if iteration > max_iterations:
        print(f"Reached recursion limit at: {iteration}")
        return reaches, iteration
    iteration += 1
    up_stream_reaches = rec_network[rec_network['NextDownID']
                                    == reach_id]
    reaches = reaches.append(up_stream_reaches)
    for index, up_stream_reach in up_stream_reaches.iterrows():
        if not up_stream_reach['Headwater']:
            reaches, iteration = get_up_stream_reaches(
                rec_network=rec_network,
                reach_id=up_stream_reach['nzsegment'],
                reaches=reaches,
                iteration=iteration)

    return reaches, iteration


def threshold_channel(reaches: geopandas.GeoDataFrame,
                      area_threshold: float,
                      channel_corridor_radius: float):
    """ Drop all channel reaches less than the specified area_threshold.

    Parameters
    ----------

    reaches
        The channel reaches
    area_threshold
        The area threshold in metres squared below which to ignore a reach.
    channel_corridor_radius
        The radius of hte channel corridor. This will determine the width of
        the channel catchment.
    """
    main_channel = reaches[reaches['CUM_AREA'] > area_threshold]
    main_channel_polygon = geopandas.GeoDataFrame(geometry=main_channel.buffer(channel_corridor_radius))
    main_channel_polygon['label'] = 1
    main_channel_polygon = main_channel_polygon.dissolve(by='label')
    return main_channel, main_channel_polygon
