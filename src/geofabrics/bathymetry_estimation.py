# -*- coding: utf-8 -*-
"""
This module contains classes associated with estiming channel bathymetry information.
"""

import geopandas
import shapely
import numpy
import xarray
import scipy


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


def subsample_channels(channel_polylines: geopandas.GeoDataFrame,
                       sampling_resolution: float):
    """ Subsample along all polylines at the sampling resolution.

    Parameters
    ----------

    channel_polylines
        The channel reaches with reache geometry defined as polylines.
    sampling_resolution
        The resolution to subsample at.
    """

    sampled_polylines = []
    for index, row in channel_polylines.iterrows():
        sampled_polylines.append(shapely.geometry.LineString(
            [row.geometry.interpolate(i * sampling_resolution) for i in
             range(int(numpy.ceil(row.geometry.length / sampling_resolution)))]))

    sampled_channel_polylines = channel_polylines.set_geometry(sampled_polylines)
    return sampled_channel_polylines


def transects_along_reaches_at_node(channel_polylines: geopandas.GeoDataFrame,
                                    transect_radius: float):
    """ Calculate transects along a channel at the midpoint of each segment.

    Parameters
    ----------

    channel_polylines
        The channel reaches with reach geometry defined as polylines.
    transect_length
        The radius of the transect (or half length).
    """

    transects_dict = {'geometry': [],
                      'nzsegment': [],
                      'nx': [],
                      'ny': [],
                      'midpoint': [],
                      'length': []}

    for index, row in channel_polylines.iterrows():

        (x_array, y_array) = row.geometry.xy
        nzsegment = row['nzsegment']
        for i in range(len(x_array)):

            # Recorde the NZ segment
            transects_dict['nzsegment'] = nzsegment

            # calculate midpoint
            midpoint = shapely.geometry.Point([x_array[i], y_array[i]])

            # caclulate slope along segment
            def segment_slope(x_array, y_array, index):
                length = numpy.sqrt(
                    (x_array[index + 1] - x_array[index]) ** 2
                    + (y_array[index + 1] - y_array[index]) ** 2)
                dx = (x_array[index + 1] - x_array[index]) \
                    / length
                dy = (y_array[index + 1] - y_array[index]) \
                    / length
                return dx, dy, length

            if i == 0:
                # first segment - slope of next segment
                dx, dy, length = segment_slope(x_array, y_array, i)
            elif i == len(x_array) - 1:
                # last segment - slope of previous segment
                dx, dy, length = segment_slope(x_array, y_array, i - 1)
            else:
                # slope of the length weighted mean of both sgments
                dx_prev, dy_prev, l_prev = segment_slope(x_array,
                                                         y_array,
                                                         i - 1)
                dx_next, dy_next, l_next = segment_slope(x_array,
                                                         y_array,
                                                         i - 1)
                dx = (dx_prev * l_prev + dx_next * l_next) / (l_prev + l_next)
                dy = (dy_prev * l_prev + dy_next * l_next) / (l_prev + l_next)
                length = (l_prev + l_next) / 2

            normal_x = -dy
            normal_y = dx

            # record normal to a segment nx and ny
            transects_dict['nx'].append(normal_x)
            transects_dict['ny'].append(normal_y)

            # calculate transect - using effectively nx and ny
            transects_dict['geometry'].append(shapely.geometry.LineString([
                [midpoint.x - transect_radius * normal_x,
                 midpoint.y - transect_radius * normal_y],
                midpoint,
                [midpoint.x + transect_radius * normal_x,
                 midpoint.y + transect_radius * normal_y]]))
            transects_dict['midpoint'].append(midpoint)

            # record the length of the line segment
            transects_dict['length'].append(length)

    transects = geopandas.GeoDataFrame(transects_dict,
                                       crs=channel_polylines.crs)
    return transects


def transects_along_reaches_at_midpoint(channel_polylines: geopandas.GeoDataFrame,
                            transect_radius: float):
    """ Calculate transects along a channel at the midpoint of each segment.

    Parameters
    ----------

    channel_polylines
        The channel reaches with reach geometry defined as polylines.
    transect_length
        The radius of the transect (or half length).
    """

    transects_dict = {'geometry': [],
                      'nzsegment': [],
                      'nx': [],
                      'ny': [],
                      'midpoint': [],
                      'length': []}

    for index, row in channel_polylines.iterrows():

        (x_array, y_array) = row.geometry.xy
        nzsegment = row['nzsegment']
        for i in range(len(x_array) - 1):

            # Recorde the NZ segment
            transects_dict['nzsegment'] = nzsegment

            # calculate midpoint
            midpoint = [(x_array[i] + x_array[i+1])/2,
                        (y_array[i] + y_array[i+1])/2]

            # caclulate slope along segment
            dx = (x_array[i+1] - x_array[i]) \
                / numpy.sqrt((x_array[i+1] - x_array[i]) ** 2
                             + (y_array[i+1] - y_array[i]) ** 2)
            dy = (y_array[i+1] - y_array[i]) \
                / numpy.sqrt((x_array[i+1] - x_array[i]) ** 2
                             + (y_array[i+1] - y_array[i]) ** 2)
            normal_x = -dy
            normal_y = dx

            # record normal to a segment nx and ny
            transects_dict['nx'].append(normal_x)
            transects_dict['ny'].append(normal_y)

            # calculate transect - using effectively nx and ny
            transects_dict['geometry'].append(shapely.geometry.LineString([
                [midpoint[0] - transect_radius * normal_x,
                 midpoint[1] - transect_radius * normal_y],
                midpoint,
                [midpoint[0] + transect_radius * normal_x,
                 midpoint[1] + transect_radius * normal_y]]))
            transects_dict['midpoint'].append(shapely.geometry.Point(midpoint))

            # record the length of the line segment
            transects_dict['length'].append(
                numpy.sqrt((x_array[i+1]-x_array[i]) ** 2
                           + (y_array[i+1] - y_array[i]) ** 2))

    transects = geopandas.GeoDataFrame(transects_dict,
                                       crs=channel_polylines.crs)
    return transects


def sample_from_transects(transects: geopandas.GeoDataFrame,
                          dem: xarray.core.dataarray.DataArray,
                          resolution: float):
    """ Sample at the sampling resolution along transects

    Parameters
    ----------

    transects
        The transects with geometry defined as polylines.
    dem
        The DEM raster to sample from.
    resolution
        The resolution to sample at
    """

    # The number of transect samples - ensure odd - defined from the first
    number_of_samples = int(numpy.floor(transects.iloc[0].geometry.length
                                        / resolution) * 2 - 1)
    sample_index_array = numpy.arange(-numpy.floor(number_of_samples / 2),
                                      numpy.floor(number_of_samples / 2) + 1,
                                      1)

    transect_samples = {'elevations': [], 'xx': [], 'yy': [], 'min_z': []}

    # create tree to sample from
    grid_x, grid_y = numpy.meshgrid(dem.x, dem.y)
    xy_in = numpy.concatenate([[grid_x.flatten()],
                               [grid_y.flatten()]], axis=0).transpose()
    tree = scipy.spatial.KDTree(xy_in)

    # cycle through each transect - calculate sample points then look up
    for index, row in transects.iterrows():

        # Calculate xx, and yy points to sample at
        if row['nx'] == 0:
            xx = row.midpoint.x + numpy.zeros(number_of_samples)
        else:
            xx = row.midpoint.x + sample_index_array * resolution * row['nx']
        if row['ny'] == 0:
            yy = row.midpoint.y + numpy.zeros(number_of_samples)
        else:
            yy = row.midpoint.y + sample_index_array * resolution * row['ny']

        # Sample the elevations at along the transect
        xy_points = numpy.concatenate([[xx], [yy]], axis=0).transpose()
        distances, indices = tree.query(xy_points)
        elevations = dem.data.flatten()[indices]
        transect_samples['elevations'].append(elevations)
        transect_samples['min_z'].append(numpy.nanmin(elevations))

    return transect_samples


def align_channel(channel: geopandas.GeoDataFrame,
                  transects: geopandas.GeoDataFrame,
                  transect_samples: dict,
                  threshold: float,
                  resolution: float):
    """ Estimate the channel centre from transect samples

    Parameters
    ----------

    transects
        The transects with geometry defined as polylines.
    transect_samples
        The sampled values along the transects.
    threshold
        The height above the water level to detect as a bank.
    resolution
        The resolution to sample at.
    """

    # start at centre of 
    widths = {'widths': [], 'first_widths': [], 'last_widths': []}

    for j in range(len(transect_samples['elevations'])):

        assert numpy.floor(len(transect_samples['elevations'][j]) / 2) \
            != len(transect_samples['elevations'][j])/2, "Expect an odd length"
        start_i = numpy.nan
        stop_i = numpy.nan
        centre_index = int(numpy.floor(len(transect_samples['elevations'][j])/2))

        for i in numpy.arange(0, centre_index, 1):

            # work forward checking height
            if transect_samples['elevations'][j][centre_index + i] \
                - transects.loc[j]['upstream_min_z'] > threshold \
                    and numpy.isnan(stop_i):
                stop_i = centre_index + i
            # work backward checking height
            if transect_samples['elevations'][j][centre_index - i] \
                - transects.loc[j]['upstream_min_z'] > threshold \
                    and numpy.isnan(start_i):
                start_i = centre_index - i

        widths['first_widths'].append((centre_index-start_i)*resolution)
        widths['last_widths'].append((stop_i-centre_index)*resolution)
        widths['widths'].append((stop_i - start_i)*resolution)

    return widths


def transect_widths_by_threshold(transects: geopandas.GeoDataFrame,
                                 transect_samples: dict,
                                 threshold: float,
                                 resolution: float):
    """ Estimate width based on a thresbold of bank height above water level

    Parameters
    ----------

    transects
        The transects with geometry defined as polylines.
    transect_samples
        The sampled values along the transects.
    threshold
        The height above the water level to detect as a bank.
    resolution
        The resolution to sample at.
    """

    widths = {'widths': [], 'first_widths': [], 'last_widths': []}

    for j in range(len(transect_samples['elevations'])):

        assert numpy.floor(len(transect_samples['elevations'][j]) / 2) \
            != len(transect_samples['elevations'][j])/2, "Expect an odd length"
        start_i = numpy.nan
        stop_i = numpy.nan
        centre_index = int(numpy.floor(len(transect_samples['elevations'][j])/2))

        for i in numpy.arange(0, centre_index, 1):

            # work forward checking height
            if transect_samples['elevations'][j][centre_index + i] \
                - transects.loc[j]['min_z'] > threshold \
                    and numpy.isnan(stop_i):
                stop_i = centre_index + i
            # work backward checking height
            if transect_samples['elevations'][j][centre_index - i] \
                - transects.loc[j]['min_z'] > threshold \
                    and numpy.isnan(start_i):
                start_i = centre_index - i

        widths['first_widths'].append((centre_index-start_i)*resolution)
        widths['last_widths'].append((stop_i-centre_index)*resolution)
        widths['widths'].append((stop_i - start_i)*resolution)

    return widths
