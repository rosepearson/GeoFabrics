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


class ChannelBathymetry:
    """ A class to estimate the width, slope and depth of a channel from
    a detailed DEM and a river network. """

    def __init__(self,
                 channel: geopandas.GeoDataFrame,
                 dem: xarray.core.dataarray.DataArray,
                 transect_spacing: float,
                 resolution: float,
                 transect_radius: float):
        """ Load in the reference DEM, clip and extract points transects

        channel
            The channel to estimate bathymetry along defined as a polyline.
        dem
            The DEM along the channel
        transect_samples
            The sampled values along the transects.
        threshold
            The height above the water level to detect as a bank.
        resolution
            The resolution to sample at.
        """

        self.channel = channel
        self.dem = dem
        self.transect_spacing = transect_spacing
        self.resolution = resolution
        self.transect_radius = transect_radius

        self._id = 'nzsegment'

        self.aligned_channel = None

    def subsample_channels(self, channel_polylines: geopandas.GeoDataFrame, sampling_resolution: float, upstream: bool):
        """ Subsample along all polylines at the sampling resolution. Note the
        subsampling is done in the upstream direction.

        Parameters
        ----------

        channel_polylines
            The channel reaches with reache geometry defined as polylines.
        sampling_resolution
            The resolution to subsample at.
        upstream
            True if the channel polyline is defined upstream, False if it is defined downstream
        """

        sampled_polylines = []
        for index, row in channel_polylines.iterrows():
            number_segment_samples = int(numpy.ceil(row.geometry.length / sampling_resolution))
            segment_resolution = row.geometry.length / number_segment_samples
            if upstream:
                indices = numpy.arange(0, number_segment_samples + 1, 1)
            else:
                indices = numpy.arange(number_segment_samples, -1, -1)
            sampled_polylines.append(shapely.geometry.LineString(
                [row.geometry.interpolate(i * segment_resolution) for i in indices]))

        sampled_channel_polylines = channel_polylines.set_geometry(sampled_polylines)
        return sampled_channel_polylines

    def _segment_slope(self, x_array, y_array, index):
        """ Return the slope and length characteristics of a line segment.

        Parameters
        ----------

        x_array
            The x values of all polyline nodes.
        y_array
            The y values of all polyline nodes.
        index
            The segment index (the index of the starting node in the segment)
        """
        length = numpy.sqrt(
            (x_array[index + 1] - x_array[index]) ** 2
            + (y_array[index + 1] - y_array[index]) ** 2)
        dx = (x_array[index + 1] - x_array[index]) \
            / length
        dy = (y_array[index + 1] - y_array[index]) \
            / length
        return dx, dy, length

    def transects_along_reaches_at_node(self, channel_polylines: geopandas.GeoDataFrame,
                                        transect_radius: float):
        """ Calculate transects along a channel at the midpoint of each segment.
        Segments in Rec2 are defined upstream down 

        Parameters
        ----------

        channel_polylines
            The channel reaches with reach geometry defined as polylines.
        transect_length
            The radius of the transect (or half length).
        """

        transects_dict = {'geometry': [],
                          self._id: [],
                          'nx': [],
                          'ny': [],
                          'midpoint': [],
                          'length': []}

        for index, row in channel_polylines.iterrows():

            (x_array, y_array) = row.geometry.xy
            nzsegment = row[self._id]
            for i in range(len(x_array)):

                # Recorde the NZ segment
                transects_dict[self._id].append(nzsegment)

                # define transect midpoint - point on channel reach
                midpoint = shapely.geometry.Point([x_array[i], y_array[i]])

                # caclulate slope along segment
                if i == 0:
                    # first segment - slope of next segment
                    dx, dy, length = self._segment_slope(x_array, y_array, i)
                elif i == len(x_array) - 1:
                    # last segment - slope of previous segment
                    dx, dy, length = self._segment_slope(x_array, y_array, i - 1)
                else:
                    # slope of the length weighted mean of both sgments
                    dx_prev, dy_prev, l_prev = self._segment_slope(x_array,
                                                                   y_array,
                                                                   i)
                    dx_next, dy_next, l_next = self._segment_slope(x_array,
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

    def transects_along_reaches_at_midpoint(self, channel_polylines: geopandas.GeoDataFrame,
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

                # caclulate slope and normal for the segment
                dx, dy, length = self._segment_slope(x_array, y_array, i)
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
                transects_dict['length'].append(length)

        transects = geopandas.GeoDataFrame(transects_dict,
                                           crs=channel_polylines.crs)
        return transects

    def sample_from_transects(self, transects: geopandas.GeoDataFrame):
        """ Sample at the sampling resolution along transects

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines.

        """

        # The number of transect samples - ensure odd - defined from the first
        number_of_samples = int(numpy.floor(transects.iloc[0].geometry.length
                                            / self.resolution) - 1)
        sample_index_array = numpy.arange(-numpy.floor(number_of_samples / 2),
                                          numpy.floor(number_of_samples / 2) + 1,
                                          1)

        transect_samples = {'elevations': [], 'xx': [], 'yy': [], 'min_z': [],
                            'min_i': []}

        # create tree to sample from
        grid_x, grid_y = numpy.meshgrid(self.dem.x, self.dem.y)
        xy_in = numpy.concatenate([[grid_x.flatten()],
                                   [grid_y.flatten()]], axis=0).transpose()
        tree = scipy.spatial.KDTree(xy_in)

        # cycle through each transect - calculate sample points then look up
        for index, row in transects.iterrows():

            # Calculate xx, and yy points to sample at
            xx = row.midpoint.x + sample_index_array * self.resolution * row['nx']
            yy = row.midpoint.y + sample_index_array * self.resolution * row['ny']

            # Sample the elevations at along the transect
            xy_points = numpy.concatenate([[xx], [yy]], axis=0).transpose()
            distances, indices = tree.query(xy_points)
            elevations = self.dem.data.flatten()[indices]
            transect_samples['elevations'].append(elevations)
            if len(elevations) - numpy.sum(numpy.isnan(elevations)) > 0:
                min_index = numpy.nanargmin(elevations)
                transect_samples['min_z'].append(elevations[min_index])
                transect_samples['min_i'].append(min_index)
            else:
                transect_samples['min_z'].append(numpy.nan)
                transect_samples['min_i'].append(numpy.nan)

        return transect_samples

    def transect_widths_by_threshold_outwards(self, transects: geopandas.GeoDataFrame,
                                              transect_samples: dict,
                                              threshold: float,
                                              resolution: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected.

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

        widths = {'widths': [], 'first_bank': [], 'last_bank': []}

        for j in range(len(transect_samples['elevations'])):

            number_of_samples = len(transect_samples['elevations'][j])
            assert numpy.floor(number_of_samples / 2) \
                != number_of_samples / 2, "Expect an odd length"

            start_i = numpy.nan
            stop_i = numpy.nan
            start_index = transect_samples['min_i'][j]
            centre_index = int(numpy.floor(number_of_samples / 2))

            for i in numpy.arange(start_index, number_of_samples, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transect_samples['min_z'][j]
                if numpy.isnan(stop_i) and elevation_over_minimum > threshold:
                    stop_i = i

            for i in numpy.arange(start_index, -1, -1):

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transect_samples['min_z'][j]
                if numpy.isnan(start_i) and elevation_over_minimum > threshold:
                    start_i = i

            widths['first_bank'].append((centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - centre_index) * resolution)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def aligned_transect_widths_by_threshold_outwards(self, transects: geopandas.GeoDataFrame,
                                                      transect_samples: dict,
                                                      threshold: float,
                                                      resolution: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected.

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

        widths = {'widths': [], 'first_bank': [], 'last_bank': []}

        for j in range(len(transect_samples['elevations'])):

            number_of_samples = len(transect_samples['elevations'][j])
            assert numpy.floor(number_of_samples / 2) \
                != number_of_samples / 2, "Expect an odd length"

            sub_threshold_detected = False  # True when detected in either direction
            start_i = numpy.nan
            stop_i = numpy.nan
            centre_index = int(numpy.floor(number_of_samples / 2))

            for i in numpy.arange(0, centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][centre_index + i] \
                    - transect_samples['min_z'][j]
                if sub_threshold_detected and numpy.isnan(stop_i) \
                        and elevation_over_minimum > threshold:
                    stop_i = centre_index + i
                elif elevation_over_minimum < threshold:
                    sub_threshold_detected = True

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][centre_index - i] \
                    - transect_samples['min_z'][j]
                if sub_threshold_detected and numpy.isnan(start_i) \
                        and elevation_over_minimum > threshold:
                    start_i = centre_index - i
                elif elevation_over_minimum < threshold:
                    sub_threshold_detected = True

            widths['first_bank'].append((centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - centre_index) * resolution)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def transect_widths_by_threshold_inwards(self, transects: geopandas.GeoDataFrame,
                                             transect_samples: dict,
                                             threshold: float,
                                             resolution: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start at the end of the transect and work in.

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

        widths = {'widths': [], 'first_bank': [], 'last_bank': []}

        for j in range(len(transect_samples['elevations'])):

            number_of_samples = len(transect_samples['elevations'][j])
            assert numpy.floor(number_of_samples / 2) \
                != number_of_samples / 2, "Expect an odd length"
            start_i = numpy.nan
            stop_i = numpy.nan
            centre_index = int(numpy.floor(number_of_samples / 2))

            for i in numpy.arange(0, centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transect_samples['min_z'][j]
                if elevation_over_minimum > threshold:
                    start_i = i
                elif not numpy.isnan(start_i) and not numpy.isnan(elevation_over_minimum):
                    break

            for i in numpy.arange(number_of_samples - 1, centre_index - 1, -1):

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transect_samples['min_z'][j]
                if elevation_over_minimum > threshold:
                    stop_i = i
                elif not numpy.isnan(stop_i) and not numpy.isnan(elevation_over_minimum):
                    break

            widths['first_bank'].append((centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - centre_index) * resolution)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def _plot_results(self, transects: geopandas.GeoDataFrame,
                      transect_samples: dict,
                      threshold: float,
                      channel_polygon=None, include_transects: bool = True):
        """ Function used for debugging or interactively to visualised the
        samples and widths

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        transect_samples
            The sampled transect values.
        threshold
            The bank detection threshold.
        channel_polygon
            The channel polygon as estimated from the widths. Optional.
        """

        import matplotlib

        # Plot all sampled transect values
        f, ax = matplotlib.pyplot.subplots(figsize=(11, 4))
        for elevations, min_z in zip(transect_samples['elevations'], transect_samples['min_z']):
            matplotlib.pyplot.plot(elevations - min_z)
        ax.set(title=f"Sampled transects. Thresh {threshold}")

        # Plot a specific transect alongside various threshold values
        i = 10
        f, ax = matplotlib.pyplot.subplots(figsize=(11, 4))
        matplotlib.pyplot.plot(transect_samples['elevations'][i] - transect_samples['min_z'][i], label="Transects")
        matplotlib.pyplot.plot([0, 300], [0.25, 0.25], label="0.25 Thresh")
        matplotlib.pyplot.plot([0, 300], [0.5, 0.5], label="0.75 Thresh")
        matplotlib.pyplot.plot([0, 300], [0.75, 0.75], label="0.5 Thresh")
        matplotlib.pyplot.plot([0, 300], [1, 1], label="1.0 Thresh")
        matplotlib.pyplot.legend()
        ax.set(title=f"Sampled transects. Thresh {threshold}, segment {i + 1}")

        # Create width lines for plotting
        def apply_bank_width(midpoint, nx, ny, first_bank, last_bank, resolution):
            import shapely
            return shapely.geometry.LineString([
                [midpoint.x - first_bank * resolution * nx,
                 midpoint.y - first_bank * resolution * ny],
                [midpoint.x + last_bank * resolution * nx,
                 midpoint.y + last_bank * resolution * ny]])
        transects['width_line'] = transects.apply(lambda x:
                                                  apply_bank_width(x['midpoint'],
                                                                   x['nx'],
                                                                   x['ny'],
                                                                   x['first_bank'],
                                                                   x['last_bank'],
                                                                   self.resolution), axis=1)
        transect_width_df = transects.set_geometry('width_line')

        # Plot transects, widths, and centrelines on the DEM
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
        self.dem.plot(ax=ax, label='DEM')
        if include_transects:
            transects.plot(ax=ax, color='blue', linewidth=1, label='transects')
        transect_width_df.plot(ax=ax, color='red', linewidth=1.5, label='widths')
        if channel_polygon is not None and type(channel_polygon) is shapely.geometry.MultiPolygon:
            for i, channel_polygon_i in enumerate(channel_polygon):
                matplotlib.pyplot.plot(*channel_polygon_i.exterior.xy, label=f'channel polygon {i}')
        elif channel_polygon is not None and type(channel_polygon) is shapely.geometry.Polygon:
            matplotlib.pyplot.plot(*channel_polygon.exterior.xy, label='channel polygon')
        self.channel.plot(ax=ax, color='black', linewidth=1.5, linestyle='--', label='original channel')
        self.aligned_channel.plot(ax=ax, linewidth=2, color='green', zorder=4, label='aligned channel')
        ax.set(title=f"Raster Layer with Vector Overlay. Thresh {threshold}")
        ax.axis('off')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()

        # Plot the various min_z values if they have been added to the transects
        f, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
        min_z_columns = [column_name for column_name in transects.columns if 'min_z' in column_name]
        if len(min_z_columns) > 0:
            transects[min_z_columns].plot(ax=ax)

        # Plot the widths
        f, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
        width_columns = [column_name for column_name in transects.columns if 'widths' in column_name]
        if len(width_columns) > 0:
            transects[width_columns].plot(ax=ax)

        # Plot the slopes
        f, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
        slope_columns = [column_name for column_name in transects.columns if 'slope' in column_name]
        if len(slope_columns) > 0:
            transects[slope_columns].plot(ax=ax)
        matplotlib.pyplot.ylim((0, None))

    def _estimate_centreline_using_polygon(self, transects: geopandas.GeoDataFrame,
                                           erosion_factor: float = -2,
                                           dilation_factor: float = 3,
                                           simplification_factor: float = 5):
        """ Create a polygon representing the channel from transect width
        measurements. Use erosion and dilation to reduce the impact of poor
        width estimates. Estimate a centreline using the transect intersections
        with the polygon.

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        erosion_factor
            The number of times the transect spacing to erode the polygon by.
        dilation_factor
            The number of times the transect spacing to dilate the polygon by.
        simplification_factor
            The number of times the transect spacing to simplify the centreline
            by.
        """

        # Create channel polygon
        channel_polygon = []
        for index, row in transects.iterrows():
            if not numpy.isnan(row['first_bank']) and not numpy.isnan(row['last_bank']):
                channel_polygon.append([row['midpoint'].x - row['first_bank'] * self.resolution * row['nx'],
                                        row['midpoint'].y - row['first_bank'] * self.resolution * row['ny']])
                channel_polygon.insert(0, [row['midpoint'].x + row['last_bank'] * self.resolution * row['nx'],
                                           row['midpoint'].y + row['last_bank'] * self.resolution * row['ny']])
        channel_polygon = shapely.geometry.Polygon(channel_polygon)
        channel_polygon = channel_polygon.buffer(
            self.transect_spacing * dilation_factor).buffer(self.transect_spacing * erosion_factor)

        # Estimate channel midpoints from the intersections of channel_polygon and transects
        aligned_channel = {'geometry': [], self._id: []}
        reach_id = transects.iloc[0][self._id]
        centre_points = []
        for index, row in transects.iterrows():
            centre_point = channel_polygon.intersection(row.geometry).centroid
            if not centre_point.is_empty:
                centre_points.append(centre_point)

            # check if moved to a new reach
            if reach_id != row[self._id] and len(centre_points) > 1:  # New reach
                # Add to dictionary
                aligned_channel['geometry'].append(
                    shapely.geometry.LineString(centre_points).simplify(self.transect_spacing * simplification_factor))
                aligned_channel[self._id].append(reach_id)
                # Reset for the next reach
                reach_id = row[self._id]
                centre_points = [centre_point]

        if len(centre_points) > 0:  # Store the final reach
            aligned_channel['geometry'].append(
                shapely.geometry.LineString(centre_points).simplify(self.transect_spacing * simplification_factor))
            aligned_channel[self._id].append(reach_id)

        # Create a aligned channel dataframe
        aligned_channel = geopandas.GeoDataFrame(aligned_channel, crs=transects.crs)
        return aligned_channel, channel_polygon

    def _unimodal_smoothing(self, y: numpy.ndarray):
        """ Fit a monotonically increasing cublic spline to the data.

        Monotonically increasing cublic splines
        - https://stats.stackexchange.com/questions/467126/monotonic-splines-in-python
        -- https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/epdf/10.1002/cem.935

        Parameters
        ----------

        y
            A 1D array of data to fit a monotonically increasing polynomial fit to.
        """

        x = numpy.arange(len(y))

        # Prepare bases (Imat) and penalty
        dd = 3
        la = 100
        kp = 10000000
        E  = numpy.eye(len(x))
        D3 = numpy.diff(E, n = dd, axis=0)
        D1 = numpy.diff(E, n = 1, axis=0)

        # Monotone smoothing
        ws = numpy.zeros(len(x) - 1)
        for it in range(30):
            Ws = numpy.diag(ws * kp)
            mon_cof = numpy.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, y)  # Polynomial fit, not monotonically constrained
            ws_new = (D1 @ mon_cof < 0.0) * 1
            dw = numpy.sum(ws != ws_new)
            ws = ws_new
            if(dw == 0):
                break
            #print(dw)

        # Monotonic and non monotonic fits
        unconstrained_polynomial_fit = numpy.linalg.solve(E + la * D3.T @ D3, y)

        return mon_cof

    def align_channel(self, threshold: float):
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

        # Sample channel
        sampled_channel = self.subsample_channels(self.channel, self.transect_spacing, upstream=False)

        # Create transects
        transects = self.transects_along_reaches_at_node(
                    channel_polylines=sampled_channel,
                    transect_radius=self.transect_radius)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # Bank estimates - outside in
        self.transect_widths_by_threshold_outwards(transects=transects,
                                                   transect_samples=transect_samples,
                                                   threshold=threshold,
                                                   resolution=self.resolution)

        # Create channel polygon with erosion and dilation to reduce sensitivity to poor width measurements
        self.aligned_channel, channel_polygon = self._estimate_centreline_using_polygon(transects)

        # Plot results
        self._plot_results(transects, transect_samples, threshold, channel_polygon)

    def estimate_width_and_slope(self, manual_aligned_channel: geopandas.GeoDataFrame, threshold: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        threshold
            The height above the water level to detect as a bank.
        """

        # Subsample transects
        sampled_aligned_channel = self.subsample_channels(manual_aligned_channel, self.transect_spacing, upstream=True)

        # Define transects
        transects = self.transects_along_reaches_at_node(
                    channel_polylines=sampled_aligned_channel,
                    transect_radius=self.transect_radius)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # Estimate widths
        self.aligned_transect_widths_by_threshold_outwards(transects=transects,
                                                           transect_samples=transect_samples,
                                                           threshold=threshold,
                                                           resolution=self.resolution)

        # Smooth slope and width estimates
        # Create channel polygon with erosion and dilation to reduce sensitivity to poor width measurements
        aligned_channel_2, channel_polygon = self._estimate_centreline_using_polygon(transects)

        # Estimate width: Repeat process a second time
        # 1. transects, 2. transect samples, 3. aligned widths outwards

        # Width smoothing - either from polygon if good enough, or function fit to aligned_widths_outward
        transects['widths_mean'] = transects['widths'].rolling(5, min_periods=1, center=True).mean()
        transects['widths_median'] = transects['widths'].rolling(5, min_periods=1, center=True).median()

        # Estimate slopes - smoothing!
        transects['min_z'] = transect_samples['min_z']
        transects['mean_min_z'] = transects['min_z'].rolling(5, min_periods=1, center=True).mean()
        min_z = transects['mean_min_z']
        upstream_min_z = numpy.zeros(len(min_z))
        upstream_min_z[-1] = min_z[len(min_z) - 1]
        for i in range(len(min_z) - 2, -1, -1):  # range [len-1, len-2, len-3, ..., 2, 1, 0]
            upstream_min_z[i] = min_z[i] if min_z[i] < upstream_min_z[i + 1] else upstream_min_z[i + 1]
        transects['upstream_min_z'] = upstream_min_z

        # Monotonically increasing splines fit
        transects['min_z_poly_constrained'] = self._monotonically_increasing_cubic_spline(transects['min_z'])

        # Plot results
        self._plot_results(transects, transect_samples, threshold, channel_polygon, include_transects=False)

        # Return results for now
        return transects
