# -*- coding: utf-8 -*-
"""
This module contains classes associated with estiming channel bathymetry information.
"""

import geopandas
import shapely
import numpy
import xarray
import scipy
import scipy.signal
import scipy.interpolate


class Channel:
    """ A class to define a channel centre line. """

    def __init__(self,
                 channel: geopandas.GeoDataFrame,
                 resolution: float,
                 sampling_direction: int = 1):
        """ A channel centreline and functions to support sampling or smoothing
        the centreline.

        Parameters
        ----------

        channel
            The channel to estimate bathymetry along defined as a polyline.
        resolution
            The resolution to sample at.
        sampling_direction
            The ordering of the points in the polylines.
        """

        self.channel = channel
        self.resolution = resolution
        self.sampling_direction = sampling_direction

    @classmethod
    def from_rec(cls,
                 rec_network: geopandas.GeoDataFrame,
                 reach_id: int,
                 resolution: float,
                 area_threshold: float,
                 max_iterations: int = 10000):
        """ Create a channel object from a REC file.

        Parameters
        ----------

        rec_network
            Contains association information between upstream and
            downstream reaches.
        reach_id
            The name of the reach ID in the REC channel.
        area_threshold
            The area threshold in metres squared below which to ignore a reach.
        max_iterations
            The maximum number of iterations along a single strand to trace
            upstream.
        iteration
            The number of iterations traveled upstream.
        """
        reaches, iteration = cls._get_up_stream_reaches(rec_network=rec_network,
                                                        reach_id=reach_id,
                                                        reaches=None,
                                                        max_iterations=max_iterations,
                                                        iteration=0)
        reaches = reaches[reaches['CUM_AREA'] > area_threshold]
        sampling_direction = -1
        channel = cls(channel=reaches,
                      resolution=resolution,
                      sampling_direction=sampling_direction)
        return channel

    @classmethod
    def _get_up_stream_reaches(cls,
                               rec_network: geopandas.GeoDataFrame,
                               reach_id: int,
                               reaches: geopandas.GeoDataFrame,
                               max_iterations: int,
                               iteration: int):
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
                reaches, iteration = cls._get_up_stream_reaches(
                    rec_network=rec_network,
                    reach_id=up_stream_reach['nzsegment'],
                    reaches=reaches,
                    max_iterations=max_iterations,
                    iteration=iteration)

        return reaches, iteration

    def get_sampled_spline_fit(self):
        """ Return the smoothed channel sampled at the resolution after it has
        been fit with a spline between corner points.

        Parameters
        ----------

        catchment_corridor_radius
            The radius of the channel corridor. This will determine the width of
            the channel catchment.
        """

        # Use spaced points as the insures consistent distance between sampled points along the spline
        xy = self.get_spaced_points(channel=self.channel,
                                    sampling_direction=self.sampling_direction,
                                    spacing=self.resolution * 10)
        '''xy = self._get_corner_points(channel=self.channel,
                                     sampling_direction=self.sampling_direction)'''
        xy = self._fit_spline_between_xy(xy)
        smooth_channel = shapely.geometry.LineString(xy.T)
        smooth_channel = geopandas.GeoDataFrame(geometry=[smooth_channel],
                                                crs=self.channel.crs)
        return smooth_channel

    def get_channel_catchment(self,
                              corridor_radius: float):
        """ Create a catchment from the smooth channel and the specified
        radius.

        Parameters
        ----------

        corridor_radius
            The radius of the channel corridor. This will determine the width of
            the channel catchment.
        """

        smooth_channel = self.get_sampled_spline_fit()
        channel_catchment = geopandas.GeoDataFrame(geometry=smooth_channel.buffer(corridor_radius))
        return channel_catchment

    def _remove_duplicate_points(cls, xy):
        """ Remove duplicate xy pairs in a list of xy points.

        Parameters
        ----------

        xy
            A paired nx2 array of x, y points.
        """
        xy_unique, indices = numpy.unique(xy, axis=1, return_index=True)
        indices.sort()
        xy = xy[:, indices]
        return xy

    def _get_corner_points(cls, channel, sampling_direction: int) -> numpy.ndarray:
        """ Extract the corner points from the provided polyline.

        Parameters
        ----------

        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        x = []
        y = []
        for line_string in channel.geometry:
            xy = line_string.xy
            x.extend(xy[0][::sampling_direction])
            y.extend(xy[1][::sampling_direction])

        xy = numpy.array([x, y])
        xy = cls._remove_duplicate_points(xy)
        return xy

    def get_spaced_points(self, channel,
                          spacing,
                          sampling_direction: int) -> numpy.ndarray:
        """  Sample at the specified spacing along the entire line.

        Parameters
        ----------

        spacing
            The spacing between sampled points along straight segments
        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        # Combine the channel centreline into a single geometry
        xy_corner_points = self._get_corner_points(channel, sampling_direction)
        line_string = shapely.geometry.LineString(xy_corner_points.T)

        # Calculate the number of segments to break the line into.
        number_segment_samples = max(numpy.round(line_string.length / spacing), 2)
        segment_resolution = line_string.length / (number_segment_samples - 1)

        # Equally space points along the entire line string
        xy_spaced = [line_string.interpolate(i * segment_resolution) for i in
                     numpy.arange(0, number_segment_samples, 1)]

        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def get_spaced_points_with_corners(self, channel,
                                       spacing,
                                       sampling_direction: int) -> numpy.ndarray:
        """ Sample at the specified spacing along each straight segment.

        Parameters
        ----------

        spacing
            The spacing between sampled points along straight segments
        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        # Combine the channel centreline into a single geometry
        xy_corner_points = self._get_corner_points(channel, sampling_direction)
        line_string = shapely.geometry.LineString(xy_corner_points.T)

        xy_segment = line_string.xy
        x = xy_segment[0]
        y = xy_segment[1]
        xy_spaced = []

        # Cycle through each segment sampling along it
        for i in numpy.arange(len(x) - 1, 0, -1):
            line_segment = shapely.geometry.LineString([[x[i], y[i]],
                                                        [x[i - 1], y[i - 1]]])

            number_segment_samples = max(numpy.round(line_segment.length / spacing), 2)
            segment_resolution = line_segment.length / (number_segment_samples - 1)

            xy_spaced.extend([line_segment.interpolate(i * segment_resolution)
                              for i in numpy.arange(0, number_segment_samples)])

        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def _fit_spline_through_xy(self, xy, smoothing_multiplier: int = 50) -> numpy.ndarray:
        """ Fit a spline to the aligned centreline points and sampled at the resolution.

        Parameters
        ----------

        smoothing_multiplier
            This is multiplied by the number of aligned_centreline points and
            passed into the scipy.interpolate.splprep.
        """

        smoothing_factor = smoothing_multiplier * len(xy[0])

        tck_tuple, u_input = scipy.interpolate.splprep(xy, s=smoothing_factor)

        # Sample every roughly res along the spine
        line_length = shapely.geometry.LineString(xy.T).length
        sample_step_u = 1 / round(line_length / self.resolution)
        u_sampled = numpy.arange(0, 1 + sample_step_u, sample_step_u)
        xy_sampled = scipy.interpolate.splev(u_sampled, tck_tuple)
        xy_sampled = numpy.array(xy_sampled)

        return xy_sampled

    def _fit_spline_between_xy(self, xy, k=3) -> numpy.ndarray:
        """ Fit a spline to the aligned centreline points and sampled at the resolution.

        Parameters
        ----------

        k
            The polynomial degree. Should be off. 1<= k <= 5.
        """

        knotspace = range(len(xy[0]))
        knots = scipy.interpolate.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
        knots_full = numpy.concatenate(([knots[0]] * k, knots, [knots[-1]] * k))

        tckX = knots_full, xy[0], k
        tckY = knots_full, xy[1], k

        splineX = scipy.interpolate.UnivariateSpline._from_tck(tckX)
        splineY = scipy.interpolate.UnivariateSpline._from_tck(tckY)

        # get number of points to sample spline at
        line_length = shapely.geometry.LineString(xy.T).length
        number_of_samples = round(line_length / self.resolution)

        u_sampled = numpy.linspace(0, len(xy[0]) - 1, number_of_samples)
        x_sampled = splineX(u_sampled)
        y_sampled = splineY(u_sampled)

        return numpy.array([x_sampled, y_sampled])

    def sampled_smoothed_centreline(self) -> numpy.ndarray:
        """ Return the spline smoothed aligned_centreline sampled at the
        resolution.
        """

        xy = self._get_corner_points(self.get_sampled_spline_fit(), sampling_direction=1)
        xy = self._fit_spline_through_xy(xy, 5 * self.resolution)

        return xy

class ChannelOld:
    """ A class to trace the channel centre line. """

    def __init__(self,
                 channel: geopandas.GeoDataFrame,
                 spacing: float,
                 reach_id: str = 'nzsegment',
                 aligned_centreline: numpy.ndarray = None,
                 sampling_direction: int = -1):
        """ Track the reference of various aligned channel centre lines to the
        REC metadata.

        Parameters
        ----------

        channel
            The channel to estimate bathymetry along defined as a polyline.
        resolution
            The resolution to sample at.
        reach_id
            The name of the reach ID in the REC channel.
        aligned_centreline
            An optionally provided channel centreline as an [x, y] array.
        """

        self._channel = channel
        self._spacing = spacing
        self._reach_id = reach_id
        if aligned_centreline is not None:
            self._aligned_centreline = aligned_centreline
        else:
            self._aligned_centreline = self.get_spaced_points(
                channel=channel,
                spacing=5 * spacing,
                sampling_direction=sampling_direction)

    @property
    def original_channel(self) -> geopandas.GeoDataFrame:
        """ Return the REC channel along with all properties. """

        return self._channel

    @property
    def aligned_centreline(self) -> numpy.ndarray:
        """ Return the latest aligned centreline. """

        return self._aligned_centreline

    @aligned_centreline.setter
    def aligned_centreline(self, aligned_centreline: numpy.ndarray):
        """ Update the latest aligned centreline.

        Parameters
        ----------

        aligned_centreline
            A paired nx2 array of x, y points.
        """

        self._aligned_centreline = aligned_centreline

    def _remove_duplicate_points(self, xy):
        """ Remove duplicate xy pairs in a list of xy points.

        Parameters
        ----------

        xy
            A paired nx2 array of x, y points.
        """
        xy_unique, indices = numpy.unique(xy, axis=1, return_index=True)
        indices.sort()
        xy = xy[:, indices]
        return xy

    def _get_corner_points(self, channel, sampling_direction: int) -> numpy.ndarray:
        """ Extract the corner points from the provided polyline.

        Parameters
        ----------

        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        x = []
        y = []
        for line_string in channel.geometry:
            xy = line_string.xy
            x.extend(xy[0][::sampling_direction])
            y.extend(xy[1][::sampling_direction])

        xy = numpy.array([x, y])
        xy = self._remove_duplicate_points(xy)
        return xy

    def get_spaced_points(self, channel,
                          spacing,
                          sampling_direction: int) -> numpy.ndarray:
        """  Sample at the specified spacing along the entire line.

        Parameters
        ----------

        spacing
            The spacing between sampled points along straight segments
        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        # Combine the channel centreline into a single geometry
        xy_corner_points = self._get_corner_points(channel, sampling_direction)
        line_string = shapely.geometry.LineString(xy_corner_points.T)

        # Calculate the number of segments to break the line into.
        number_segment_samples = max(numpy.round(line_string.length / spacing), 2)
        segment_resolution = line_string.length / (number_segment_samples - 1)

        # Equally space points along the entire line string
        xy_spaced = [line_string.interpolate(i * segment_resolution) for i in
                     numpy.arange(0, number_segment_samples, 1)]

        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def get_spaced_points_with_corners(self, channel,
                                       spacing,
                                       sampling_direction: int) -> numpy.ndarray:
        """ Sample at the specified spacing along each straight segment.

        Parameters
        ----------

        spacing
            The spacing between sampled points along straight segments
        sample_direction
            Are the reaches sampled in the same direction they are ordered.
            1 if in the same direction, -1 if in the opposite direction.
        """

        # Combine the channel centreline into a single geometry
        xy_corner_points = self._get_corner_points(channel, sampling_direction)
        line_string = shapely.geometry.LineString(xy_corner_points.T)

        xy_segment = line_string.xy
        x = xy_segment[0]
        y = xy_segment[1]
        xy_spaced = []

        # Cycle through each segment sampling along it
        for i in numpy.arange(len(x) - 1, 0, -1):
            line_segment = shapely.geometry.LineString([[x[i], y[i]],
                                                        [x[i - 1], y[i - 1]]])

            number_segment_samples = max(numpy.round(line_segment.length / spacing), 2)
            segment_resolution = line_segment.length / (number_segment_samples - 1)

            xy_spaced.extend([line_segment.interpolate(i * segment_resolution)
                              for i in numpy.arange(0, number_segment_samples)])

        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def fit_spline_to_aligned_centreline(self, xy, smoothing_multiplier: int = 50) -> numpy.ndarray:
        """ Fit a spline to the aligned centreline points and sampled at the resolution.

        Parameters
        ----------

        smoothing_multiplier
            This is multiplied by the number of aligned_centreline points and
            passed into the scipy.interpolate.splprep.
        """

        smoothing_factor = smoothing_multiplier * len(xy[0])

        tck_tuple, u_input = scipy.interpolate.splprep(xy, s=smoothing_factor)

        # Sample every roughly res along the spine
        line_length = shapely.geometry.LineString(xy.T).length
        sample_step_u = 1 / round(line_length / self._spacing)
        u_sampled = numpy.arange(0, 1 + sample_step_u, sample_step_u)
        xy_sampled = scipy.interpolate.splev(u_sampled, tck_tuple)
        xy_sampled = numpy.array(xy_sampled)

        return xy_sampled

    def fit_spline_to_points_from_knots(self, xy, k=3) -> numpy.ndarray:
        """ Fit a spline to the aligned centreline points and sampled at the resolution.

        Parameters
        ----------

        k
            The polynomial degree. Should be off. 1<= k <= 5.
        """

        knotspace = range(len(xy[0]))
        knots = scipy.interpolate.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
        knots_full = numpy.concatenate(([knots[0]] * k, knots, [knots[-1]] * k))

        tckX = knots_full, xy[0], k
        tckY = knots_full, xy[1], k

        splineX = scipy.interpolate.UnivariateSpline._from_tck(tckX)
        splineY = scipy.interpolate.UnivariateSpline._from_tck(tckY)

        # get number of points to sample spline at
        line_length = shapely.geometry.LineString(xy.T).length
        number_of_samples = round(line_length / self._spacing)

        u_sampled = numpy.linspace(0, len(xy[0]) - 1, number_of_samples)
        x_sampled = splineX(u_sampled)
        y_sampled = splineY(u_sampled)

        return numpy.array([x_sampled, y_sampled])

    def sampled_smoothed_centreline(self) -> numpy.ndarray:
        """ Return the spline smoothed aligned_centreline sampled at the
        resolution.
        """

        xy = self.fit_spline_to_aligned_centreline(self._aligned_centreline, 5 * self._spacing)

        return xy


class ChannelBathymetry:
    """ A class to estimate the width, slope and depth of a channel from
    a detailed DEM and a river network. """

    def __init__(self,
                 channel: Channel,
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
        self.aligned_channel = None
        self.dem = dem
        self.transect_spacing = transect_spacing
        self.resolution = resolution
        self.transect_radius = transect_radius

    @property
    def number_of_samples(self) -> int:
        """ Return the number of samples to take along transects. This should
        be an odd number. Subtract 1 instead of adding to ensure within the
        generated DEM. """

        return int(self.transect_radius / self.resolution) * 2 - 1

    @property
    def centre_index(self) -> int:
        """ Return the centre index for samples taken along a transect. """
        return int(numpy.floor(self.number_of_samples / 2))

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

    def transects_along_reaches_at_node(self, sampled_channel: geopandas.GeoDataFrame):
        """ Calculate transects along a channel at the midpoint of each segment.

        Parameters
        ----------

        sampled_channel
            The sampled channel defined as a single polyline. Any branches described
            separately.
        transect_length
            The radius of the transect (or half length).
        """

        transects_dict = {'geometry': [],
                          'nx': [],
                          'ny': [],
                          'midpoint': [],
                          'length': []}

        assert len(sampled_channel) == 1, "Expect only one polyline " \
            "geometry per channel. Instead got {len(channel_polyline)}"

        (x_array, y_array) = sampled_channel.iloc[0].geometry.xy
        for i in range(len(x_array)):

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
                [midpoint.x - self.transect_radius * normal_x,
                 midpoint.y - self.transect_radius * normal_y],
                midpoint,
                [midpoint.x + self.transect_radius * normal_x,
                 midpoint.y + self.transect_radius * normal_y]]))
            transects_dict['midpoint'].append(midpoint)

            # record the length of the line segment
            transects_dict['length'].append(length)

        transects = geopandas.GeoDataFrame(transects_dict,
                                           crs=sampled_channel.crs)
        return transects

    def transects_along_reaches_at_midpoint(self, sampled_channel: geopandas.GeoDataFrame):
        """ Calculate transects along a channel at the midpoint of each segment.

        Parameters
        ----------

        sampled_channel
            The sampled channel defined as a single polyline.
        transect_length
            The radius of the transect (or half length).
        """

        transects_dict = {'geometry': [],
                          'nx': [],
                          'ny': [],
                          'midpoint': [],
                          'length': []}

        assert len(sampled_channel) == 1, "Expect only one polyline " \
            "geometry per channel. Instead got {len(channel_polyline)}"

        (x_array, y_array) = sampled_channel.iloc[0].geometry.xy
        for i in range(len(x_array) - 1):

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
                [midpoint[0] - self.transect_radius * normal_x,
                 midpoint[1] - self.transect_radius * normal_y],
                midpoint,
                [midpoint[0] + self.transect_radius * normal_x,
                 midpoint[1] + self.transect_radius * normal_y]]))
            transects_dict['midpoint'].append(shapely.geometry.Point(midpoint))

            # record the length of the line segment
            transects_dict['length'].append(length)

        transects = geopandas.GeoDataFrame(transects_dict,
                                           crs=sampled_channel.crs)
        return transects

    def _estimate_water_level_and_slope(self, transects: geopandas.GeoDataFrame,
                                        transect_samples: dict,
                                        smoothing_distance: float = 1000):
        """ Estimate the water level and slope from the minimumz heights along
        the sampled transects after applying appropiate smoothing and
        constraints to ensure it is monotonically increasing. Values are stored
        in the transects.

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines.
        transect_samples
            The sampled values along each transect
        """

        smoothing_samples = int(numpy.ceil(smoothing_distance/self.transect_spacing))

        # water surface - including monotonically increasing splines fit
        transects['min_z'] = transect_samples['min_z']
        transects['min_z_unimodal'] = self._unimodal_smoothing(transects['min_z'])
        transects[f'min_z_unimodal_{smoothing_distance/1000}km_rolling_mean'] = \
            transects['min_z_unimodal'].rolling(
            smoothing_samples, min_periods=1, center=True).mean()
        transects['min_z_savgol'] = scipy.signal.savgol_filter(
            transects['min_z'].interpolate('index', limit_direction='both'),
            int(smoothing_samples / 2) * 2 + 1,  # Must be odd - number of samples to include
            3)

        transects['min_z_centre'] = transect_samples['min_z_centre']
        transects['min_z_centre_unimodal'] = self._unimodal_smoothing(transects['min_z_centre'])
        transects['min_z_centre_savgol'] = scipy.signal.savgol_filter(
            transects['min_z_centre'].interpolate('index', limit_direction='both'),
            int(smoothing_samples / 2) * 2 + 1,  # Must be odd - number of samples to include
            3)

        # Set the water z value to use for width thresholding
        transects['min_z_water'] = transects[f'min_z_unimodal_{smoothing_distance/1000}km_rolling_mean']

        # Slope - from the water z
        transects['slope'] = transects['min_z_water'].diff() / self.transect_spacing

    def sample_from_transects(self, transects: geopandas.GeoDataFrame):
        """ Sample at the sampling resolution along transects

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines.

        """

        # The number of transect samples - ensure odd - defined from the first
        sample_index_array = numpy.arange(-numpy.floor(self.number_of_samples / 2),
                                          numpy.floor(self.number_of_samples / 2) + 1,
                                          1)

        transect_samples = {'elevations': [], 'xx': [], 'yy': [], 'min_z': [],
                            'min_i': [], 'min_xy': [], 'min_z_centre': [],
                            'min_i_centre': [], 'min_xy_centre': []}

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
                transect_samples['min_xy'].append(shapely.geometry.Point(xy_points[min_index]))
            else:
                transect_samples['min_z'].append(numpy.nan)
                transect_samples['min_i'].append(numpy.nan)
                transect_samples['min_xy'].append(shapely.geometry.Point([numpy.nan,
                                                                          numpy.nan]))

            # Find the min of just the centre 1/3 of samples
            start_i = 99
            stop_i = 199
            if len(elevations[start_i:stop_i]) - numpy.sum(numpy.isnan(elevations[start_i:stop_i])) > 0:
                min_index = numpy.nanargmin(elevations[start_i:stop_i])
                transect_samples['min_z_centre'].append(elevations[start_i + min_index])
                transect_samples['min_i_centre'].append(start_i + min_index)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point(xy_points[start_i + min_index]))
            else:
                transect_samples['min_z_centre'].append(numpy.nan)
                transect_samples['min_i_centre'].append(numpy.nan)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point([numpy.nan, numpy.nan]))

        return transect_samples

    def thresholded_widths_outwards_from_min(self, transects: geopandas.GeoDataFrame,
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

        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': []}

        for j in range(len(transect_samples['elevations'])):

            assert len(transect_samples['elevations'][j]) == self.number_of_samples, "Expect fixed length"

            start_i = numpy.nan
            stop_i = numpy.nan
            start_index = transect_samples['min_i'][j]

            for i in numpy.arange(start_index, self.number_of_samples, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transects.iloc[j]['min_z_water']
                if numpy.isnan(stop_i) and elevation_over_minimum > threshold:
                    stop_i = i

            for i in numpy.arange(start_index, -1, -1):

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transects.iloc[j]['min_z_water']
                if numpy.isnan(start_i) and elevation_over_minimum > threshold:
                    start_i = i

            widths['first_bank'].append((self.centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - self.centre_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def thresholded_widths_outwards_from_centre(self, transects: geopandas.GeoDataFrame,
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

        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': []}

        for j in range(len(transect_samples['elevations'])):

            assert len(transect_samples['elevations'][j]) == self.number_of_samples, "Expect fixed length"

            sub_threshold_detected = False  # True when detected in either direction
            start_i = numpy.nan
            stop_i = numpy.nan

            for i in numpy.arange(0, self.centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][self.centre_index + i] \
                    - transects.iloc[j]['min_z_water']
                if sub_threshold_detected and numpy.isnan(stop_i) \
                        and elevation_over_minimum > threshold:
                    stop_i = self.centre_index + i
                elif elevation_over_minimum < threshold:
                    sub_threshold_detected = True

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][self.centre_index - i] \
                    - transects.iloc[j]['min_z_water']
                if sub_threshold_detected and numpy.isnan(start_i) \
                        and elevation_over_minimum > threshold:
                    start_i = self.centre_index - i
                elif elevation_over_minimum < threshold:
                    sub_threshold_detected = True

            widths['first_bank'].append((self.centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - self.centre_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def thresholded_widths_outwards_directional(self, transects: geopandas.GeoDataFrame,
                                                transect_samples: dict,
                                                threshold: float,
                                                resolution: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected. Takes nearest channel to
        centre, but channel doesn't need to include the centre.'

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

        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': []}

        for j in range(len(transect_samples['elevations'])):

            assert len(transect_samples['elevations'][j]) == self.number_of_samples, "Expect fixed length"

            start_i = numpy.nan
            stop_i = numpy.nan
            centre_sub_threshold = transect_samples['elevations'][j][self.centre_index] \
                - transects.iloc[j]['min_z_water'] < threshold
            forward_sub_threshold = False
            backward_sub_threshold = False

            for i in numpy.arange(0, self.centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][self.centre_index + i] \
                    - transects.iloc[j]['min_z_water']
                if (centre_sub_threshold or forward_sub_threshold) \
                        and numpy.isnan(stop_i) and elevation_over_minimum > threshold:
                    # Leaving the channel
                    stop_i = self.centre_index + i
                elif elevation_over_minimum < threshold and not forward_sub_threshold \
                        and not backward_sub_threshold and not centre_sub_threshold:
                    # only just made it forward to the start of the channel
                    forward_sub_threshold = True
                    start_i = self.centre_index + i - 1

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][self.centre_index - i] \
                    - transects.iloc[j]['min_z_water']
                if (centre_sub_threshold or backward_sub_threshold) \
                        and numpy.isnan(start_i) and elevation_over_minimum > threshold:
                    start_i = self.centre_index - i
                elif elevation_over_minimum < threshold and not forward_sub_threshold \
                        and not backward_sub_threshold and not centre_sub_threshold:
                    # only just made it backward to the end of the channel
                    backward_sub_threshold = True
                    stop_i = self.centre_index - i + 1

            widths['first_bank'].append((self.centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - self.centre_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
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

        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': []}

        for j in range(len(transect_samples['elevations'])):

            assert len(transect_samples['elevations'][j]) == self.number_of_samples, "Expect fixed length"
            start_i = numpy.nan
            stop_i = numpy.nan

            for i in numpy.arange(0, self.centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transects.iloc[j]['min_z_water']
                if elevation_over_minimum > threshold:
                    start_i = i
                elif not numpy.isnan(start_i) and not numpy.isnan(elevation_over_minimum):
                    break

            for i in numpy.arange(self.number_of_samples - 1, self.centre_index - 1, -1):

                # work backward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transects.iloc[j]['min_z_water']
                if elevation_over_minimum > threshold:
                    stop_i = i
                elif not numpy.isnan(stop_i) and not numpy.isnan(elevation_over_minimum):
                    break

            widths['first_bank'].append((self.centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - self.centre_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def _plot_results(self, transects: geopandas.GeoDataFrame,
                      transect_samples: dict,
                      threshold: float,
                      aligned_channel: geopandas.GeoDataFrame = None,
                      include_transects: bool = True):
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
        aligned_channel
            The aligned channel generated from the transects
        """

        import matplotlib

        '''# Plot all sampled transect values
        f, ax = matplotlib.pyplot.subplots(figsize=(11, 4))
        for elevations, min_z in zip(transect_samples['elevations'], transect_samples['min_z']):
            matplotlib.pyplot.plot(elevations - min_z)
        ax.set(title=f"Sampled transects. Thresh {threshold}")'''

        # Plot a specific transect alongside various threshold values
        '''i = 10
        f, ax = matplotlib.pyplot.subplots(figsize=(11, 4))
        matplotlib.pyplot.plot(transect_samples['elevations'][i] - transect_samples['min_z'][i], label="Transects")
        matplotlib.pyplot.plot([0, 300], [0.25, 0.25], label="0.25 Thresh")
        matplotlib.pyplot.plot([0, 300], [0.5, 0.5], label="0.75 Thresh")
        matplotlib.pyplot.plot([0, 300], [0.75, 0.75], label="0.5 Thresh")
        matplotlib.pyplot.plot([0, 300], [1, 1], label="1.0 Thresh")
        matplotlib.pyplot.legend()
        ax.set(title=f"Sampled transects. Thresh {threshold}, segment {i + 1}")'''

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

        # Plot transects, widths, and centrelines on the DEM
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
        self.dem.plot(ax=ax, label='DEM')
        if include_transects:
            transects.plot(ax=ax, color='blue', linewidth=1, label='transects')
        transects.set_geometry('width_line').plot(ax=ax, color='red', linewidth=1.5, label='widths')
        self.channel.get_sampled_spline_fit().plot(ax=ax, color='black', linewidth=1.5, linestyle='--',
                                                   label='sampled channel')
        if aligned_channel is not None:
            aligned_channel.plot(ax=ax, linewidth=2, color='green', zorder=4, label='Aligned channel')
        if 'perturbed_midpoints' in transects.columns:
            transects.set_geometry('perturbed_midpoints').plot(ax=ax, color='aqua', zorder=5,
                                                               markersize=5, label='Perturbed midpoints')
        ax.set(title=f"Raster Layer with Vector Overlay. Thresh {threshold}")
        ax.axis('off')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()

        # Plot the various min_z values if they have been added to the transects
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
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
        aligned_channel = []
        for index, row in transects.iterrows():
            centre_point = channel_polygon.intersection(row.geometry).centroid
            aligned_channel.append(centre_point)

        if len(aligned_channel) > 0:  # Store the final reach
            aligned_channel = shapely.geometry.LineString(aligned_channel)
            aligned_channel = geopandas.GeoDataFrame(geometry=[aligned_channel],
                                                     crs=transects.crs)

        return aligned_channel, channel_polygon

    def _perturb_centreline_from_width(self, transects: geopandas.GeoDataFrame,
                                       smoothing_distance):
        """ Offset the transect centre points along the transect based on the
        centre of the estimated width. Note that the width centres are smoothed
        based on the smoothing distance before offsetting. .

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        smoothing_distance
            The metres along the channel to smooth the widths by.
        """

        # Calculate the offset distance between the transect and width centres
        offset_distance = ((transects['last_bank_i'] + transects['first_bank_i']) / 2
                           - self.centre_index) * self.resolution
        offset_distance = self._despike(offset_distance,
                                        smoothing_distance=100,
                                        threshold = 50)

        # Smooth the offset distances
        smoothed_offset_distance = scipy.signal.savgol_filter(
            offset_distance.interpolate('index', limit_direction='both'),
            int(smoothing_distance / self.transect_spacing / 2) * 2 + 1,  # Must be odd - number of samples to include
            3)  # Polynomial order

        # Perterb by smoothed distance
        perturbed_midpoints = []
        perturbed_midpoints_list = []
        for index, row in transects.iterrows():
            midpoint = row['midpoint']

            # Perturb by smoothed offset distance
            perturbed_midpoint = [midpoint.x + smoothed_offset_distance[index] * row['nx'],
                                  midpoint.y + smoothed_offset_distance[index] * row['ny']]
            perturbed_midpoints.append(shapely.geometry.Point(perturbed_midpoint))
            perturbed_midpoints_list.append(perturbed_midpoint)

        transects['perturbed_midpoints'] = perturbed_midpoints
        perturbed_channel_centreline = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.LineString(perturbed_midpoints_list)],
            crs=transects.crs)
        return perturbed_channel_centreline

    def _despike(self, spiky_values: geopandas.GeoSeries,
                 threshold: float,
                 smoothing_distance: float) -> geopandas.GeoSeries:
        """ A function to remove and linearly interpolate over values that are
        deemed a spike.

        Parameters
        ----------

        spiky_values
            The value to run spike detection over.
        threshold
            The threshold for a blip to be deemed a spike.
        smoothing_distance
            The distance down river to smooth along
        """

        # Must be odd - number of samples to include
        samples_to_filter_with = int(smoothing_distance / self.transect_spacing / 2) * 2 + 1

        smoothed_values = scipy.signal.savgol_filter(
            spiky_values.interpolate('index', limit_direction='both'),
            samples_to_filter_with,
            3)

        # Spikes
        spikes = (spiky_values - smoothed_values).abs()

        # despiking
        despiked_values = spiky_values.copy(deep=True)
        despiked_values[spikes > threshold] = numpy.nan
        #despiked_values = despiked_values.interpolate('index', limit_direction='both')
        return despiked_values

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
            mon_cof = numpy.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, y)  # Polynomial fit, monotonically constrained
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
        sampled_channel = self.channel.get_sampled_spline_fit()

        # Create transects
        transects = self.transects_along_reaches_at_node(
                    sampled_channel=sampled_channel)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # record min_i and min_xy
        transects['min_i'] = transect_samples['min_i']
        transects['min_xy'] = transect_samples['min_xy']
        transects['min_i_centre'] = transect_samples['min_i_centre']
        transects['min_xy_centre'] = transect_samples['min_xy_centre']

        # Estimate water surface level and slope - Smooth slope upstream over 1km
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=1000)
        transects['min_z_water'] = transects['min_z_centre_unimodal']

        # Bank estimates - outside in
        '''self.thresholded_widths_outwards_from_centre(transects=transects,
                                                     transect_samples=transect_samples,
                                                     threshold=threshold,
                                                     resolution=self.resolution)'''
        self.thresholded_widths_outwards_directional(transects=transects,
                                                        transect_samples=transect_samples,
                                                        threshold=threshold,
                                                        resolution=self.resolution)
        '''self.thresholded_widths_outwards_from_min(transects=transects,
                                                     transect_samples=transect_samples,
                                                     threshold=threshold,
                                                     resolution=self.resolution)'''

        # Create channel polygon with erosion and dilation to reduce sensitivity to poor width measurements
        aligned_channel = self._perturb_centreline_from_width(transects, smoothing_distance=100)

        # Plot results
        self._plot_results(transects=transects,
                           transect_samples=transect_samples,
                           threshold=threshold,
                           include_transects=False,
                           aligned_channel=aligned_channel)

        # Second alignment step
        '''self.channel.aligned_centreline = numpy.array(aligned_channel.iloc[0].geometry.xy)
        xy = self.channel.sampled_smoothed_centreline()
        sampled_channel = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.LineString(xy.T)],
            crs=self.channel.original_channel.crs)

        # Create transects
        transects = self.transects_along_reaches_at_node(
                    channel_polylines=sampled_channel)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # Estimate water surface level and slope
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=slope_smoothing_distance)
        transects['min_z_water'] = transects['min_z_unimodal']

        # Bank estimates - outside in
        self.transect_widths_by_threshold_outwards(transects=transects,
                                                   transect_samples=transect_samples,
                                                   threshold=threshold,
                                                   resolution=self.resolution)

        # Create channel polygon with erosion and dilation to reduce sensitivity to poor width measurements
        aligned_channel = self._perturb_centreline_from_width(transects, smoothing_distance=500)

        # Plot results
        self._plot_results(transects=transects,
                           transect_samples=transect_samples,
                           threshold=threshold,
                           include_transects=False,
                           channel=aligned_channel)'''
        return aligned_channel, transects

    def estimate_width_and_slope(self, manual_aligned_channel: geopandas.GeoDataFrame, threshold: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        threshold
            The height above the water level to detect as a bank.
        """
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
                True if the channel polyline to sample from is already defined
                upstream, False if it is defined downstream
            """

            sampled_polylines = []
            for index, row in channel_polylines.iterrows():
                number_segment_samples = round(row.geometry.length / sampling_resolution)
                segment_resolution = row.geometry.length / number_segment_samples
                if upstream:
                    indices = numpy.arange(0, number_segment_samples + 1, 1)
                else:
                    indices = numpy.arange(number_segment_samples, -1, -1)
                sampled_polylines.append(shapely.geometry.LineString(
                    [row.geometry.interpolate(i * segment_resolution) for i in indices]))

            sampled_channel_polylines = channel_polylines.set_geometry(sampled_polylines)
            return sampled_channel_polylines

        z_smoothing_distance = 500  # Smooth water surface upstream over 1km
        width_smoothing_distance = 100  # Smooth slope upstream over 1km
        # Subsample transects
        sampled_aligned_channel = self.subsample_channels(manual_aligned_channel, self.transect_spacing, upstream=True)

        # Define transects
        transects = self.transects_along_reaches_at_node(
                    channel_polylines=sampled_aligned_channel)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # Estimate water surface level and slope
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=z_smoothing_distance)
        transects['min_z_water'] = transects['min_z_unimodal']

        # Estimate widths
        self.transect_widths_by_threshold_outwards_from_centre(transects=transects,
                                                               transect_samples=transect_samples,
                                                               threshold=threshold,
                                                               resolution=self.resolution)

        # Update centreline estimation from widths
        sampled_aligned_channel = self._perturb_centreline_from_width(transects)

        # Repeat width estimate with the realigned centreline?
        transects = self.transects_along_reaches_at_node(channel_polylines=sampled_aligned_channel)
        transect_samples = self.sample_from_transects(transects=transects)
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=z_smoothing_distance)
        self.transect_widths_by_threshold_outwards_from_centre(transects=transects,
                                                               transect_samples=transect_samples,
                                                               threshold=threshold,
                                                               resolution=self.resolution)

        # Width smoothing - either from polygon if good enough, or function fit to aligned_widths_outward
        transects['widths_mean'] = transects['widths'].rolling(5,
                                                               min_periods=1, center=True).mean()
        transects['widths_median'] = transects['widths'].rolling(5,
                                                                 min_periods=1, center=True).median()
        transects['widths_Savgol'] = scipy.signal.savgol_filter(
            transects['widths'].interpolate('index', limit_direction='both'),
            int(width_smoothing_distance / self.transect_spacing / 2) * 2 + 1,  # Must be odd - number of samples to include
            3)  # Polynomial order

        # Plot results
        self._plot_results(transects=transects,
                           transect_samples=transect_samples,
                           threshold=threshold,
                           channel=manual_aligned_channel,
                           include_transects=False)

        # Return results for now
        return transects, sampled_aligned_channel
