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
        spline_channel = shapely.geometry.LineString(xy.T)
        spline_channel = geopandas.GeoDataFrame(geometry=[spline_channel],
                                                crs=self.channel.crs)
        return spline_channel

    def get_smoothed_spline_fit(self, smoothing_multiplier) -> numpy.ndarray:
        """ Return the spline smoothed aligned_centreline sampled at the
        resolution.
        """

        xy = self._get_corner_points(channel=self.channel,
                                     sampling_direction=self.sampling_direction)
        xy = self._fit_spline_through_xy(xy, smoothing_multiplier)
        spline_channel = shapely.geometry.LineString(xy.T)
        spline_channel = geopandas.GeoDataFrame(geometry=[spline_channel],
                                                crs=self.channel.crs)

        return spline_channel

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

        # Sample every roughly res along the spine with rough line length estimate
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


class ChannelBathymetry:
    """ A class to estimate the width, slope and depth of a channel from
    a detailed DEM and a river network. """

    def __init__(self,
                 channel: Channel,
                 dem: xarray.core.dataarray.DataArray,
                 transect_spacing: float,
                 resolution: float,
                 transect_radius: float,
                 min_z_radius: float):
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
        self.min_z_radius = min_z_radius

        assert transect_radius >= min_z_radius, "The transect radius must be >= the min_z_radius"

    @property
    def number_of_samples(self) -> int:
        """ Return the number of samples to take along transects. This should
        be an odd number. Subtract 1 instead of adding to ensure within the
        generated DEM. """

        return int(self.transect_radius / self.resolution) * 2 - 1

    @property
    def min_z_start_i(self) -> int:
        """ Return the starting index of samples along each transect to begin
        looking for the minimu z. """

        number_min_z_samples = int(self.min_z_radius / self.resolution) * 2 - 1

        return int((self.number_of_samples - number_min_z_samples) / 2)

    @property
    def min_z_stop_i(self) -> int:
        """ Return the stopping index of samples along each transect to begin
        looking for the minimu z. """

        return int(self.number_of_samples - self.min_z_start_i)

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
                          'length': [],
                          'mid_x': [],
                          'mid_y': []}

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
            transects_dict['mid_x'].append(midpoint.x)
            transects_dict['mid_y'].append(midpoint.y)

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
                          'length': [],
                          'mid_x': [],
                          'mid_y': []}

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
            transects_dict['mid_x'].append(midpoint.x)
            transects_dict['mid_y'].append(midpoint.y)

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

        # water surface - including monotonically increasing splines fit
        '''transects['min_z'] = transect_samples['min_z']
        transects['min_z_unimodal'] = self._unimodal_smoothing(transects['min_z'])
        transects['min_z_savgol'] = scipy.signal.savgol_filter(
            transects['min_z'].interpolate('index', limit_direction='both'),
            int(smoothing_samples / 2) * 2 + 1,  # Must be odd - number of samples to include
            3)'''

        # Min z values as the water surface. Ensure no NaN
        transects['min_z_centre'] = transect_samples['min_z_centre']
        transects['min_z_centre'] = transects['min_z_centre'].interpolate('index', limit_direction='both')
        # Unimodal enforaces monotonically increasing
        transects['min_z_centre_unimodal'] = self._unimodal_smoothing(transects['min_z_centre'])

        # Set the water z value to use for width thresholding
        transects['min_z_water'] = transects['min_z_centre_unimodal']

        # Slope from the water surface - interpolate to fill any Nan
        transects['slope'] = transects['min_z_water'].diff() / self.transect_spacing
        transects['slope'] = transects['slope'].interpolate('index', limit_direction='both')

        # Slopes for a range of smoothings
        for smoothing_distance in [500, 1000, 2000, 3000]:
            # ensure odd number of samples so array length preserved
            smoothing_samples = int(numpy.ceil(smoothing_distance / self.transect_spacing))
            smoothing_samples = int(smoothing_samples / 2) * 2 + 1
            label = f'{smoothing_distance/1000}km'

            # Smoothed min_z_centre_unimodal
            transects[f'min_z_centre_unimodal_mean_{label}'] \
                = self._rolling_mean_with_padding(transects['min_z_centre_unimodal'],
                                                  smoothing_samples)

            # Smoothed slope
            transects[f'slope_mean_{label}'] = self._rolling_mean_with_padding(transects['slope'],
                                                                               smoothing_samples)

    def _rolling_mean_with_padding(self, data: geopandas.GeoSeries, number_of_samples: int) -> numpy.ndarray:
        """ Calculate the rolling mean of an array after padding the array with
        the edge value to ensure the derivative is smooth.

        Parameters
        ----------

        data
            The array to pad then smooth.
        number_of_samples
            The width in samples of the averaging filter
        """
        assert number_of_samples > 0 and type(number_of_samples) == int, "Must be more than 0 and an int"
        rolling_mean = numpy.convolve(
            numpy.pad(data, int(number_of_samples/2), 'symmetric'),
            numpy.ones(number_of_samples), 'valid') / number_of_samples
        return rolling_mean

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
                            'min_i_centre': [], 'min_xy_centre': [],
                            'min_x_centre': [], 'min_y_centre': []}

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
            if len(elevations[self.min_z_start_i:self.min_z_stop_i]) \
                    - numpy.sum(numpy.isnan(elevations[self.min_z_start_i:self.min_z_stop_i])) > 0:
                min_index = numpy.nanargmin(elevations[self.min_z_start_i:self.min_z_stop_i])
                transect_samples['min_z_centre'].append(elevations[self.min_z_start_i + min_index])
                transect_samples['min_i_centre'].append(self.min_z_start_i + min_index)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point(xy_points[self.min_z_start_i
                                                                                          + min_index]))
                transect_samples['min_x_centre'].append(xy_points[self.min_z_start_i + min_index, 0])
                transect_samples['min_y_centre'].append(xy_points[self.min_z_start_i + min_index, 1])
            else:
                transect_samples['min_z_centre'].append(numpy.nan)
                transect_samples['min_i_centre'].append(numpy.nan)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point([numpy.nan, numpy.nan]))
                transect_samples['min_x_centre'].append(numpy.nan)
                transect_samples['min_y_centre'].append(numpy.nan)

        return transect_samples

    def thresholded_widths_outwards_from_min(self, transects: geopandas.GeoDataFrame,
                                             transect_samples: dict,
                                             threshold: float,
                                             resolution: float,
                                             min_name: str = 'min_i'):
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
            start_index = transect_samples[min_name][j]
            if numpy.isnan(start_index):
                start_index = transect_samples['min_i'][j]

            for i in numpy.arange(int(start_index), self.number_of_samples, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][i] - transects.iloc[j]['min_z_water']
                if numpy.isnan(stop_i) and elevation_over_minimum > threshold:
                    stop_i = i

            for i in numpy.arange(int(start_index), -1, -1):

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

    def thresholded_widths_outwards_directional_from_centre(self, transects: geopandas.GeoDataFrame,
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
            all_nan_so_far = numpy.isnan(transect_samples['elevations'][j][self.centre_index])
            forward_sub_threshold = False
            backward_sub_threshold = False

            for i in numpy.arange(0, self.centre_index + 1, 1):

                # work forward checking height
                elevation_over_minimum = transect_samples['elevations'][j][self.centre_index + i] \
                    - transects.iloc[j]['min_z_water']
                # Update the centre_sub_threshold if first non-nan values
                if all_nan_so_far and not numpy.isnan(elevation_over_minimum):
                    centre_sub_threshold = elevation_over_minimum < threshold
                    all_nan_so_far = False
                # Detect banks
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
                # Update the centre_sub_threshold if first non-nan values
                if all_nan_so_far and not numpy.isnan(elevation_over_minimum):
                    centre_sub_threshold = elevation_over_minimum < threshold
                    all_nan_so_far = False
                # Detect bank
                if (centre_sub_threshold or backward_sub_threshold) \
                        and numpy.isnan(start_i) and elevation_over_minimum > threshold:
                    # Leaving the channel
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

    def thresholded_widths_outwards_directional_from_min(self, transects: geopandas.GeoDataFrame,
                                                         transect_samples: dict,
                                                         threshold: float,
                                                         resolution: float,
                                                         min_name: str = 'min_i'):
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

            start_index = transect_samples[min_name][j]
            if numpy.isnan(start_index):
                start_index = transect_samples['min_i'][j]

            centre_sub_threshold = transect_samples['elevations'][j][int(start_index)] \
                - transects.iloc[j]['min_z_water'] < threshold
            all_nan_so_far = numpy.isnan(transect_samples['elevations'][j][int(start_index)])
            forward_sub_threshold = False
            backward_sub_threshold = False

            for i in numpy.arange(0, self.number_of_samples, 1):
                forward_index = int(start_index + i)
                backward_index = int(start_index - i)

                if forward_index >= self.number_of_samples and backward_index < 0:
                    break

                # working forward checking height
                if forward_index < self.number_of_samples:
                    elevation_over_minimum = transect_samples['elevations'][j][forward_index] \
                        - transects.iloc[j]['min_z_water']

                    # Update the centre_sub_threshold if first non-nan values
                    if all_nan_so_far and not numpy.isnan(elevation_over_minimum):
                        centre_sub_threshold = elevation_over_minimum < threshold
                        all_nan_so_far = False

                    if (centre_sub_threshold or forward_sub_threshold) \
                            and numpy.isnan(stop_i) and elevation_over_minimum > threshold:
                        # Leaving the channel
                        stop_i = forward_index
                    elif elevation_over_minimum < threshold and not forward_sub_threshold \
                            and not backward_sub_threshold and not centre_sub_threshold:
                        # only just made it forward to the start of the channel
                        forward_sub_threshold = True
                        start_i = forward_index - 1
                # working backward checking height
                if backward_index >= 0:
                    elevation_over_minimum = transect_samples['elevations'][j][backward_index] \
                        - transects.iloc[j]['min_z_water']

                    # Update the centre_sub_threshold if first non-nan values
                    if all_nan_so_far and not numpy.isnan(elevation_over_minimum):
                        centre_sub_threshold = elevation_over_minimum < threshold
                        all_nan_so_far = False

                    if (centre_sub_threshold or backward_sub_threshold) \
                            and numpy.isnan(start_i) and elevation_over_minimum > threshold:
                        start_i = backward_index
                    elif elevation_over_minimum < threshold and not forward_sub_threshold \
                            and not backward_sub_threshold and not centre_sub_threshold:
                        # only just made it backward to the end of the channel
                        backward_sub_threshold = True
                        stop_i = backward_index + 1

            widths['first_bank'].append((self.centre_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - self.centre_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def thresholded_widths_inwards(self, transects: geopandas.GeoDataFrame,
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
                      initial_spline: geopandas.GeoDataFrame = None,
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

        # Plot transects, widths, and centrelines on the DEM
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
        self.dem.plot(ax=ax, label='DEM')
        if include_transects:
            transects.plot(ax=ax, color='aqua', linewidth=1, label='transects')
        transects.set_geometry('width_line').plot(ax=ax, color='red', linewidth=1.5, label='widths')
        self.channel.get_sampled_spline_fit().plot(ax=ax, color='black', linewidth=1.5, linestyle='--',
                                                   label='sampled channel')
        if aligned_channel is not None:
            aligned_channel.plot(ax=ax, linewidth=2, color='green', zorder=4, label='Aligned channel')
        if initial_spline is not None:
            initial_spline.plot(ax=ax, linewidth=2, color='blue', zorder=3, label='Min Z splne')
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

    def _centreline_from_width_spline(self, transects: geopandas.GeoDataFrame,
                                      smoothing_multiplier):
        """ Fit a spline through the width centres with a healthy dose of
        smoothing.

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        smoothing_multiplier
            The smoothing multiplier to apply to the spline fit.
        """

        # Calculate the offset distance between the transect and width centres
        widths_centre_offset = self.resolution * ((transects['first_bank_i']
                                                   + transects['last_bank_i']) / 2 - self.centre_index)
        widths_centre_xy = numpy.vstack([(transects['mid_x'] + widths_centre_offset * transects['nx']).array,
                                         (transects['mid_y'] + widths_centre_offset * transects['ny']).array]).T
        widths_centre_xy = widths_centre_xy[numpy.isnan(widths_centre_xy).any(axis=1) == False]
        widths_centre_line = geopandas.GeoDataFrame(geometry=[shapely.geometry.LineString(widths_centre_xy)],
                                                    crs=transects.crs)
        widths_centre_line = Channel(widths_centre_line, resolution=self.transect_spacing)

        aligned_spline = widths_centre_line.get_smoothed_spline_fit(smoothing_multiplier)
        return aligned_spline

    def _centreline_from_min_z(self, transects: geopandas.GeoDataFrame,
                               smoothing_multiplier: float):
        """ Fit a spline through the near min z along the transect with a healthy dose of
        smoothing.

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        smoothing_multiplier
            The smoothing multiplier to apply to the spline fit.
        """

        # Calculate the offset distance between the transect and width centres
        min_centre_xy = numpy.vstack([transects['min_x_centre'].array, transects['min_y_centre'].array]).T
        min_centre_xy = min_centre_xy[numpy.isnan(min_centre_xy).any(axis=1) == False]
        min_centre_line = geopandas.GeoDataFrame(geometry=[shapely.geometry.LineString(min_centre_xy)],
                                                 crs=transects.crs)
        min_centre_line = Channel(min_centre_line, resolution=self.transect_spacing)

        min_centre_spline = min_centre_line.get_smoothed_spline_fit(smoothing_multiplier)
        return min_centre_spline

    def _transect_and_spline_intersection(self, transects: geopandas.GeoDataFrame,
                                          spline: geopandas.GeoDataFrame,
                                          entry_name: str):
        """ Find the nearest index of intersection between each transect
        profile and the spline. Save the index values in transect.

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines with width
            estimates.
        spline
            The spline line.
        entry_name
            The name to save the intersection index under
        """
        assert len(spline) == 1, "Expect only one spline entry"
        spline_line = spline.geometry.iloc[0]

        def spline_transect_intersection_index(spline_line,
                                               transect,
                                               midpoint,
                                               nx,
                                               ny,
                                               centre_index,
                                               resolution):
            """ Calculate the intersection index of a spline and the transect.
            """
            if spline_line.intersects(transect):
                # find where the spline intersects the transect line
                intersection_point = spline_line.intersection(transect)
                if type(intersection_point) == shapely.geometry.MultiPoint:
                    # Take the point nearest to the centre if their are multiple intesections
                    offset_distance = numpy.inf
                    offset = []
                    for point in intersection_point:
                        point_offset = [point.x - midpoint.x,
                                        point.y - midpoint.y]
                        point_distance = numpy.sqrt(point_offset[0] ** 2 + point_offset[1] ** 2)
                        if point_distance < offset_distance:
                            offset = point_offset
                            offset_distance = point_distance
                else:
                    # Otherswise do the same calculations
                    offset = [intersection_point.x - midpoint.x,
                              intersection_point.y - midpoint.y]
                    offset_distance = numpy.sqrt(offset[0] ** 2 + offset[1] ** 2)

                # decide if counting up or down
                if (numpy.sign(nx) == numpy.sign(offset[0]) and nx != 0) \
                        or (numpy.sign(ny) == numpy.sign(offset[1]) and ny != 0):
                    direction = +1
                else:
                    direction = -1
                index = centre_index + direction * round(offset_distance / resolution)
            else:
                index = numpy.nan
            return index
        transects[entry_name] = transects.apply(lambda row:
                                                spline_transect_intersection_index(spline_line,
                                                                                   row['geometry'],
                                                                                   row['midpoint'],
                                                                                   row['nx'],
                                                                                   row['ny'],
                                                                                   self.centre_index,
                                                                                   self.resolution), axis=1)

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

    def _apply_bank_width(self, midpoint, nx, ny, first_bank, last_bank):
        """ Generate a line for each width for visualisation.

        Parameters
        ----------

        midpoint
            The centre of the transect.
        nx
            Transect normal x-component.
        ny
            Transect normal y-component.
        first_bank
            The signed distance between the first bank and the transect centre.
        last_bank
            The signed distance between the last bank and the transect centre.
        """
        return shapely.geometry.LineString([
            [midpoint.x - first_bank * nx,
             midpoint.y - first_bank * ny],
            [midpoint.x + last_bank * nx,
             midpoint.y + last_bank * ny]])

    def align_channel(self,
                      threshold: float,
                      min_z_smoothing_multiplier: float,
                      width_centre_smoothing_multiplier: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        threshold
            The height above the water level to detect as a bank.
        smoothing_multiplier
            The number of transects to include in the downstream spline smoothing.
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

        # Create a new centreline estimate from a spline through the near min z
        transects['min_x_centre'] = transect_samples['min_x_centre']
        transects['min_y_centre'] = transect_samples['min_y_centre']
        min_centre_spline = self._centreline_from_min_z(transects=transects, smoothing_multiplier=min_z_smoothing_multiplier)

        # Get spline and transect intersection
        self._transect_and_spline_intersection(transects=transects, spline=min_centre_spline, entry_name='min_spline_i')
        transect_samples['min_spline_i'] = transects['min_spline_i']

        # Estimate water surface level and slope - Smooth slope upstream over 1km
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=1000)

        # Bank estimates - outside in
        '''self.thresholded_widths_outwards_from_centre(transects=transects,
                                                     transect_samples=transect_samples,
                                                     threshold=threshold,
                                                     resolution=self.resolution)'''
        '''self.thresholded_widths_outwards_directional_from_centre(transects=transects,
                                                     transect_samples=transect_samples,
                                                     threshold=threshold,
                                                     resolution=self.resolution)'''
        '''self.thresholded_widths_outwards_from_min(transects=transects,
                                                  transect_samples=transect_samples,
                                                  threshold=threshold,
                                                  resolution=self.resolution,
                                                  min_name='min_spline_i')'''
        self.thresholded_widths_outwards_directional_from_min(transects=transects,
                                                              transect_samples=transect_samples,
                                                              threshold=threshold,
                                                              resolution=self.resolution,
                                                              min_name='min_spline_i')

        # Add width linestring to the transects
        transects['width_line'] = transects.apply(
            lambda x: self._apply_bank_width(x['midpoint'],
                                             x['nx'],
                                             x['ny'],
                                             x['first_bank'],
                                             x['last_bank']), axis=1)

        # Create channel polygon with erosion and dilation to reduce sensitivity to poor width measurements
        aligned_channel = self._centreline_from_width_spline(transects,
                                                             smoothing_multiplier=width_centre_smoothing_multiplier)

        # Plot results
        self._plot_results(transects=transects,
                           transect_samples=transect_samples,
                           threshold=threshold,
                           include_transects=False,
                           aligned_channel=aligned_channel,
                           initial_spline=min_centre_spline)

        return aligned_channel, transects, min_centre_spline

    def estimate_width_and_slope(self,
                                 aligned_channel: geopandas.GeoDataFrame, 
                                 threshold: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        aligned_channel
            The channel centre line. Should be in the channel bed.
        threshold
            The height above the water level to detect as a bank.
        """

        slope_smoothing_distance = 500  # Smooth slope upstream over this many metres
        width_smoothing_distance = 100  # Smooth width upstream over this many metres

        # Create transects
        transects = self.transects_along_reaches_at_node(sampled_channel=aligned_channel)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects)

        # Estimate water surface level and slope
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=slope_smoothing_distance)

        # Estimate widths
        self.thresholded_widths_outwards_directional_from_centre(
            transects=transects,
            transect_samples=transect_samples,
            threshold=threshold,
            resolution=self.resolution)

        # Width smoothing - either from polygon if good enough, or function fit to aligned_widths_outward
        widths_no_nan = transects['widths'].interpolate('index', limit_direction='both')
        for smoothing_distance in [150, 200, 250, 2000, 3000]:
            # ensure odd number of samples so array length preserved
            smoothing_samples = int(numpy.ceil(smoothing_distance / self.transect_spacing))
            smoothing_samples = int(smoothing_samples / 2) * 2 + 1
            label = f"{smoothing_distance/1000}km"
            # try a variety of smoothing approaches
            transects[f'widths_mean_{label}'] = self._rolling_mean_with_padding(widths_no_nan, smoothing_samples)
            '''transects[f'widths_median_{label}'] = transects['widths'].rolling(smoothing_samples,
                                                                              min_periods=1, center=True).median()
            transects[f'widths_Savgol_{label}'] = scipy.signal.savgol_filter(
                transects['widths'].interpolate('index', limit_direction='both'),
                smoothing_samples,  # Ensure odd. number of samples included
                3)  # Polynomial order'''

        transects['width_line'] = transects.apply(
            lambda x: self._apply_bank_width(x['midpoint'],
                                             x['nx'],
                                             x['ny'],
                                             x['first_bank'],
                                             x['last_bank']), axis=1)

        # Plot results
        self._plot_results(transects=transects,
                           transect_samples=transect_samples,
                           threshold=threshold,
                           aligned_channel=aligned_channel,
                           include_transects=False)

        # Return results for now
        return transects
