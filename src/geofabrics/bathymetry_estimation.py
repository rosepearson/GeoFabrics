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
                 veg_dem: xarray.core.dataarray.DataArray,
                 transect_spacing: float,
                 resolution: float):
        """ Load in the reference DEM, clip and extract points transects

        channel
            The channel to estimate bathymetry along defined as a polyline.
        dem
            The ground DEM along the channel
        veg_dem
            The vegetation DEM along the channel
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
        self.veg_dem = veg_dem
        self.transect_spacing = transect_spacing
        self.resolution = resolution
        self.transect_radius = None

    @property
    def number_of_samples(self) -> int:
        """ Return the number of samples to take along transects. This should
        be an odd number. Subtract 1 instead of adding to ensure within the
        generated DEM. """

        assert self.transect_radius is not None, "Transect radius must be set before this is called"
        return int(self.transect_radius / self.resolution) * 2 - 1

    def calculate_min_z_start_i(self, min_z_search_radius) -> int:
        """ Return the starting index of samples along each transect to begin
        looking for the minimu z. """

        number_min_z_samples = int(min_z_search_radius / self.resolution) * 2 - 1

        return int((self.number_of_samples - number_min_z_samples) / 2)

    def calculate_min_z_stop_i(self, min_z_search_radius) -> int:
        """ Return the stopping index of samples along each transect to begin
        looking for the minimu z. """

        return int(self.number_of_samples - self.calculate_min_z_start_i(min_z_search_radius))

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
        transects['min_z'] = transect_samples['min_z']
        '''transects['min_z_unimodal'] = self._unimodal_smoothing(transects['min_z'])
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

    def sample_from_transects(self,
                              transects: geopandas.GeoDataFrame,
                              min_z_search_radius: float):
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

        min_z_start_i = self.calculate_min_z_start_i(min_z_search_radius)
        min_z_stop_i = self.calculate_min_z_stop_i(min_z_search_radius)

        transect_samples = {'gnd_elevations': [], 'veg_elevations': [],
                            'xx': [], 'yy': [], 'min_z': [],
                            'min_i': [], 'min_xy': [], 'min_z_centre': [],
                            'min_i_centre': [], 'min_xy_centre': [],
                            'min_x_centre': [], 'min_y_centre': []}

        # create tree of ground values to sample from
        grid_x, grid_y = numpy.meshgrid(self.dem.x, self.dem.y)
        xy_in = numpy.concatenate([[grid_x.flatten()],
                                   [grid_y.flatten()]], axis=0).transpose()
        gnd_tree = scipy.spatial.KDTree(xy_in)

        # create tree of vegetation values to sample from
        grid_x, grid_y = numpy.meshgrid(self.veg_dem.x, self.veg_dem.y)
        xy_in = numpy.concatenate([[grid_x.flatten()],
                                   [grid_y.flatten()]], axis=0).transpose()
        veg_tree = scipy.spatial.KDTree(xy_in)

        # cycle through each transect - calculate sample points then look up
        for index, row in transects.iterrows():

            # Calculate xx, and yy points to sample at
            xx = row.midpoint.x + sample_index_array * self.resolution * row['nx']
            yy = row.midpoint.y + sample_index_array * self.resolution * row['ny']
            xy_points = numpy.concatenate([[xx], [yy]], axis=0).transpose()

            # Sample the vegetation elevations at along the transect
            distances, indices = veg_tree.query(xy_points)
            elevations = self.veg_dem.data.flatten()[indices]
            transect_samples['veg_elevations'].append(elevations)

            # Sample the ground elevations at along the transect
            distances, indices = gnd_tree.query(xy_points)
            elevations = self.dem.data.flatten()[indices]
            transect_samples['gnd_elevations'].append(elevations)

            # Find the min elevation and index of it along each cross section
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
            if len(elevations[min_z_start_i:min_z_stop_i]) \
                    - numpy.sum(numpy.isnan(elevations[min_z_start_i:min_z_stop_i])) > 0:
                min_index = numpy.nanargmin(elevations[min_z_start_i:min_z_stop_i])
                transect_samples['min_z_centre'].append(elevations[min_z_start_i + min_index])
                transect_samples['min_i_centre'].append(min_z_start_i + min_index)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point(xy_points[min_z_start_i
                                                                                          + min_index]))
                transect_samples['min_x_centre'].append(xy_points[min_z_start_i + min_index, 0])
                transect_samples['min_y_centre'].append(xy_points[min_z_start_i + min_index, 1])
            else:
                transect_samples['min_z_centre'].append(numpy.nan)
                transect_samples['min_i_centre'].append(numpy.nan)
                transect_samples['min_xy_centre'].append(shapely.geometry.Point([numpy.nan, numpy.nan]))
                transect_samples['min_x_centre'].append(numpy.nan)
                transect_samples['min_y_centre'].append(numpy.nan)

        return transect_samples

    def fixed_thresholded_widths_from_centre_within_radius(self, transects: geopandas.GeoDataFrame,
                                                           sampled_elevations: dict,
                                                           threshold: float,
                                                           resolution: float,
                                                           search_radius: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected. Takes the widest channel within
        the radius.'

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

        search_radius_index = int(search_radius / self.resolution)
        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': []}

        for j in range(len(sampled_elevations['gnd_elevations'])):

            assert len(sampled_elevations['gnd_elevations'][j]) == self.number_of_samples, "Expect fixed length"

            gnd_samples = sampled_elevations['gnd_elevations'][j]
            veg_samples = sampled_elevations['veg_elevations'][j]
            start_index = self.centre_index
            z_water = transects.iloc[j]['min_z_water']

            start_i, stop_i = self.fixed_threshold_width(gnd_samples=gnd_samples,
                                                         veg_samples=veg_samples,
                                                         start_index=start_index,
                                                         z_water=z_water,
                                                         threshold=threshold,
                                                         search_radius_index=search_radius_index)

            # assign the longest width
            widths['first_bank'].append((start_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - start_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)

        for key in widths.keys():
            transects[key] = widths[key]

    def variable_thresholded_widths_from_centre_within_radius(self, transects: geopandas.GeoDataFrame,
                                                              sampled_elevations: dict,
                                                              threshold: float,
                                                              resolution: float,
                                                              search_radius: float,
                                                              maximum_threshold: float):
        """ Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected. Takes the widest channel within
        the radius.'

        Parameters
        ----------

        transects
            The transects with geometry defined as polylines.
        sampled_elevations
            The sampled values along the transects.
        threshold
            The height above the water level to detect as a bank.
        resolution
            The resolution to sample at.
        """

        search_radius_index = int(search_radius / self.resolution)
        print(f'search radius index = {search_radius_index}')
        widths = {'widths': [], 'first_bank': [], 'last_bank': [],
                  'first_bank_i': [], 'last_bank_i': [], 'threshold': []}

        for j in range(len(sampled_elevations['gnd_elevations'])):

            assert len(sampled_elevations['gnd_elevations'][j]) == self.number_of_samples, "Expect fixed length"

            gnd_samples = sampled_elevations['gnd_elevations'][j]
            veg_samples = sampled_elevations['veg_elevations'][j]
            start_index = self.centre_index
            z_water = transects.iloc[j]['min_z_water']

            # Get width based on fixed threshold
            start_i, stop_i = self.fixed_threshold_width(gnd_samples=gnd_samples,
                                                         veg_samples=veg_samples,
                                                         start_index=start_index,
                                                         z_water=z_water,
                                                         threshold=threshold,
                                                         search_radius_index=search_radius_index)

            # Iterate out from the fixed threshold width until the banks go down, or the max threshold is reached
            maximum_z = z_water + maximum_threshold
            if numpy.isnan(start_i) or numpy.isnan(stop_i):
                dz_bankfull = threshold
            else:
                z_bankfull = numpy.nanmin([gnd_samples[start_i], gnd_samples[stop_i]]) \
                    if not numpy.isnan([gnd_samples[start_i], gnd_samples[stop_i]]).all() \
                    else threshold
                start_i_bf = start_i
                stop_i_bf = stop_i

                while start_i_bf > 0 and stop_i_bf < self.number_of_samples - 1 \
                        and z_bankfull < maximum_z:

                    # break if going down
                    if gnd_samples[start_i_bf - 1] < z_bankfull or veg_samples[start_i_bf - 1] < z_bankfull:
                        break
                    if gnd_samples[stop_i_bf + 1] < z_bankfull or veg_samples[start_i_bf + 1] < z_bankfull:
                        break

                    # if not, extend whichever bank is lower
                    if gnd_samples[start_i_bf - 1] > gnd_samples[stop_i_bf + 1]:
                        stop_i_bf += 1
                    elif gnd_samples[start_i_bf - 1] < gnd_samples[stop_i_bf + 1]:
                        start_i_bf -= 1
                    elif gnd_samples[start_i_bf - 1] == gnd_samples[stop_i_bf + 1]:
                        start_i_bf -= 1
                        stop_i_bf += 1
                    else:
                        # extend if value is nan and not vegetated, or if vegetation is under limit
                        if numpy.isnan(gnd_samples[start_i_bf - 1]) and numpy.isnan(veg_samples[start_i_bf - 1]):
                            start_i_bf -= 1
                        elif numpy.isnan(gnd_samples[start_i_bf - 1]) \
                                and veg_samples[start_i_bf - 1] < z_water + maximum_threshold:
                            start_i_bf -= 1
                        if numpy.isnan(gnd_samples[stop_i_bf + 1]) and numpy.isnan(veg_samples[start_i_bf + 1]):
                            stop_i_bf += 1
                        elif numpy.isnan(gnd_samples[start_i_bf + 1]) \
                                and veg_samples[start_i_bf + 1] < z_water + maximum_threshold:
                            stop_i_bf += 1

                    # Break if the threshold has been meet before updating maz_z
                    if gnd_samples[start_i_bf] >= z_water + maximum_threshold \
                            or gnd_samples[stop_i_bf] >= z_water + maximum_threshold:
                        break
                    # Break if ground is nan and the vegetation is over the limit
                    if numpy.isnan(gnd_samples[start_i_bf]) and veg_samples[start_i_bf] >= z_water + maximum_threshold:
                        break
                    if numpy.isnan(gnd_samples[stop_i_bf]) and veg_samples[stop_i_bf] >= z_water + maximum_threshold:
                        break

                    # update maximum value so far
                    if not numpy.isnan([gnd_samples[start_i_bf], gnd_samples[stop_i_bf], veg_samples[start_i_bf], veg_samples[stop_i_bf]]).all() \
                            and numpy.nanmin([gnd_samples[start_i_bf], gnd_samples[stop_i_bf],
                                              veg_samples[start_i_bf], veg_samples[stop_i_bf]]) > z_bankfull:
                        z_bankfull = numpy.nanmin([gnd_samples[start_i_bf], gnd_samples[stop_i_bf],
                                                   veg_samples[start_i_bf], veg_samples[stop_i_bf]])

                # set to nan if either end of the cross section has been reached
                if start_i_bf <= 0 or stop_i >= self.number_of_samples - 1:
                    dz_bankfull = threshold
                    start_i = numpy.nan
                    stop_i = numpy.nan
                else:
                    dz_bankfull = z_bankfull - z_water
                    start_i = start_i_bf
                    stop_i = stop_i_bf

            # assign the longest width
            widths['first_bank'].append((start_index - start_i) * resolution)
            widths['last_bank'].append((stop_i - start_index) * resolution)
            widths['first_bank_i'].append(start_i)
            widths['last_bank_i'].append(stop_i)
            widths['widths'].append((stop_i - start_i) * resolution)
            widths['threshold'].append(dz_bankfull)

        for key in widths.keys():
            transects[key] = widths[key]

    def fixed_threshold_width(self,
                              gnd_samples: numpy.ndarray,
                              veg_samples: numpy.ndarray,
                              start_index: int,
                              z_water: float,
                              threshold: float,
                              search_radius_index: int):
        """ Calculate the maximum width for a cross section given a fixed
        threshold - checking outwards, forewards and backwards within the
        search radius.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        threshold
            The height above the water level to detect as a bank.
        z_water
            The elevation of the water.
        search_radius_index
            The distance in indices to search for the start of a channel away
            from the start_index
        """

        start_i_list = []
        stop_i_list = []

        forwards_index = start_index
        backwards_index = start_index

        # check outwards
        start_i, stop_i = self.fixed_threshold_width_outwards(gnd_samples=gnd_samples,
                                                              veg_samples=veg_samples,
                                                              start_index=start_index,
                                                              z_water=z_water,
                                                              threshold=threshold)
        if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
            start_i_list.append(start_i)
            stop_i_list.append(stop_i)
            forwards_index = stop_i + 1
            backwards_index = start_i - 1

        # check forewards
        while forwards_index - start_index < search_radius_index:
            start_i, stop_i = self.fixed_threshold_width_forewards(gnd_samples=gnd_samples,
                                                                   veg_samples=veg_samples,
                                                                   start_index=forwards_index,
                                                                   z_water=z_water,
                                                                   threshold=threshold,
                                                                   search_range=search_radius_index)
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                start_i_list.append(start_i)
                stop_i_list.append(stop_i)
                forwards_index = stop_i + 1
            else:
                break

        # check backwards
        while start_index - backwards_index < search_radius_index:
            start_i, stop_i = self.fixed_threshold_width_backwards(gnd_samples=gnd_samples,
                                                                   veg_samples=veg_samples,
                                                                   start_index=backwards_index,
                                                                   z_water=z_water,
                                                                   threshold=threshold,
                                                                   search_range=search_radius_index)
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                start_i_list.append(start_i)
                stop_i_list.append(stop_i)
                backwards_index = start_i - 1
            else:
                break

        # cycle through getting the longest width
        start_i = numpy.nan
        stop_i = numpy.nan
        longest_width = 0
        for i in range(len(start_i_list)):
            if stop_i_list[i] - start_i_list[i] > longest_width:
                longest_width = stop_i_list[i] - start_i_list[i]
                start_i = start_i_list[i]
                stop_i = stop_i_list[i]

        return start_i, stop_i

    def fixed_threshold_width_outwards(self,
                                       gnd_samples: numpy.ndarray,
                                       veg_samples: numpy.ndarray,
                                       start_index: int,
                                       z_water: float,
                                       threshold: float):
        """ If the start_index is nan or less than the threshold, then cycle
        outwards until each side has gone above the threshold.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        threshold
            The height above the water level to detect as a bank.
        z_water
            The elevation of the water.
        """

        start_i = numpy.nan
        stop_i = numpy.nan

        if gnd_samples[start_index] - z_water < threshold or \
                (numpy.isnan(gnd_samples[start_index]) and numpy.isnan(veg_samples[start_index])):

            for i in numpy.arange(0, self.centre_index + 1, 1):

                # work forward checking height
                gnd_elevation_over_minimum = gnd_samples[start_index + i] - z_water
                veg_elevation_over_minimum = veg_samples[start_index + i] - z_water

                # Detect banks - either ground above threshold, or no ground with vegetation over threshold
                if numpy.isnan(stop_i) and (gnd_elevation_over_minimum > threshold
                                            or (numpy.isnan(gnd_elevation_over_minimum)
                                                and veg_elevation_over_minimum > threshold)):
                    # Leaving the channel
                    stop_i = start_index + i

                # work backward checking height
                gnd_elevation_over_minimum = gnd_samples[start_index - i] - z_water
                veg_elevation_over_minimum = veg_samples[start_index - i] - z_water

                # Detect bank
                if numpy.isnan(start_i) and (gnd_elevation_over_minimum > threshold
                                             or (numpy.isnan(gnd_elevation_over_minimum)
                                                 and veg_elevation_over_minimum > threshold)):
                    # Leaving the channel
                    start_i = start_index - i

                # break if both edges detected
                if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                    break

        return start_i, stop_i

    def fixed_threshold_width_forewards(self,
                                        gnd_samples: numpy.ndarray,
                                        veg_samples: numpy.ndarray,
                                        start_index: int,
                                        z_water: float,
                                        threshold: float,
                                        search_range: int):
        """ Check for channels approaching foreward.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        threshold
            The height above the water level to detect as a bank.
        z_water
            The elevation of the water.
        """

        start_i = numpy.nan
        stop_i = numpy.nan

        for i in numpy.arange(start_index, self.number_of_samples, 1):

            # work forward checking height
            gnd_elevation_over_minimum = gnd_samples[i] - z_water
            veg_elevation_over_minimum = veg_samples[i] - z_water

            # Detect banks
            if numpy.isnan(start_i) and \
                (gnd_elevation_over_minimum < threshold or veg_elevation_over_minimum < threshold
                 or (numpy.isnan(gnd_samples[i]) and numpy.isnan(veg_samples[i]))):
                # Entering the channel
                start_i = i - 1
            if numpy.isnan(stop_i) and not numpy.isnan(start_i) and \
                    (gnd_elevation_over_minimum > threshold or (numpy.isnan(gnd_samples[i])
                                                                and veg_elevation_over_minimum > threshold)):
                # Leaving the channel
                stop_i = i

            # break if both edges detected
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                break
            # break if the first edge is not detected in the search range
            if numpy.isnan(start_i) and i > start_index + search_range:
                break

        return start_i, stop_i

    def fixed_threshold_width_backwards(self,
                                        gnd_samples: numpy.ndarray,
                                        veg_samples: numpy.ndarray,
                                        start_index: int,
                                        z_water: float,
                                        threshold: float,
                                        search_range: int):
        """ Check for channels approaching foreward.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        threshold
            The height above the water level to detect as a bank.
        z_water
            The elevation of the water.
        """

        start_i = numpy.nan
        stop_i = numpy.nan

        for i in numpy.arange(start_index, -1, -1):

            # work forward checking height
            gnd_elevation_over_minimum = gnd_samples[i] - z_water
            veg_elevation_over_minimum = veg_samples[i] - z_water

            # Detect banks
            if numpy.isnan(stop_i) and \
                (gnd_elevation_over_minimum < threshold or veg_elevation_over_minimum < threshold
                 or (numpy.isnan(gnd_samples[i]) and numpy.isnan(veg_samples[i]))):
                # Entering the channel backwards
                stop_i = i + 1
            if numpy.isnan(start_i) and not numpy.isnan(stop_i) and \
                    (gnd_elevation_over_minimum > threshold or (numpy.isnan(gnd_samples[i])
                                                                and veg_elevation_over_minimum > threshold)):
                # Leaving the channel backwards
                start_i = i

            # break if both edges detected
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                break
            # break if the first edge is not detected in the search range
            if numpy.isnan(stop_i) and i < start_index - search_range:
                break

        return start_i, stop_i

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
        for elevations, min_z in zip(transect_samples['gnd_elevations'], transect_samples['min_z']):
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
                      min_z_search_radius: float,
                      width_centre_smoothing_multiplier: float,
                      transect_radius: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        threshold
            The height above the water level to detect as a bank.
        smoothing_multiplier
            The number of transects to include in the downstream spline smoothing.
        """

        assert transect_radius >= min_z_search_radius, "The transect radius must be >= the min_z_radius"
        self.transect_radius = transect_radius

        # Sample channel
        sampled_channel = self.channel.get_sampled_spline_fit()

        # Create transects
        transects = self.transects_along_reaches_at_node(
                    sampled_channel=sampled_channel)

        # Sample along transects
        transect_samples = self.sample_from_transects(transects=transects,
                                                      min_z_search_radius=min_z_search_radius)

        # record min_i and min_xy
        transects['min_i'] = transect_samples['min_i']
        transects['min_xy'] = transect_samples['min_xy']
        transects['min_i_centre'] = transect_samples['min_i_centre']
        transects['min_xy_centre'] = transect_samples['min_xy_centre']

        # Create a new centreline estimate from a spline through the near min z
        transects['min_x_centre'] = transect_samples['min_x_centre']
        transects['min_y_centre'] = transect_samples['min_y_centre']
        min_centre_spline = self._centreline_from_min_z(transects=transects,
                                                        smoothing_multiplier=min_z_smoothing_multiplier)

        # Get spline and transect intersection
        self._transect_and_spline_intersection(transects=transects, spline=min_centre_spline, entry_name='min_spline_i')
        transect_samples['min_spline_i'] = transects['min_spline_i']

        # Estimate water surface level and slope - Smooth slope upstream over 1km
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=transect_samples,
                                             smoothing_distance=1000)

        # Bank estimates - outside in
        self.fixed_thresholded_widths_from_centre_within_radius(
            transects=transects,
            transect_samples=transect_samples,
            threshold=threshold,
            search_radius=min_z_search_radius,
            resolution=self.resolution)

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
                                 threshold: float,
                                 transect_radius: float,
                                 min_z_search_radius: float):
        """ Estimate the channel centre from transect samples

        Parameters
        ----------

        aligned_channel
            The channel centre line. Should be in the channel bed.
        threshold
            The height above the water level to detect as a bank.
        """

        assert transect_radius >= min_z_search_radius, "The transect radius must be >= the min_z_radius"
        self.transect_radius = transect_radius

        slope_smoothing_distance = 500  # Smooth slope upstream over this many metres

        # Create transects
        transects = self.transects_along_reaches_at_node(sampled_channel=aligned_channel)

        # Sample along transects
        sampled_elevations = self.sample_from_transects(transects=transects,
                                                        min_z_search_radius=min_z_search_radius)

        # Estimate water surface level and slope
        self._estimate_water_level_and_slope(transects=transects,
                                             transect_samples=sampled_elevations,
                                             smoothing_distance=slope_smoothing_distance)

        # Estimate widths
        #self.fixed_thresholded_widths_from_centre_within_radius(
        self.variable_thresholded_widths_from_centre_within_radius(
            transects=transects,
            sampled_elevations=sampled_elevations,
            threshold=threshold,
            resolution=self.resolution,
            search_radius=min_z_search_radius/10,
            maximum_threshold=7*threshold)

        # Width smoothing - either from polygon if good enough, or function fit to aligned_widths_outward
        widths_no_nan = transects['widths'].interpolate('index', limit_direction='both')
        for smoothing_distance in [150, 200, 250, 2000, 3000]:
            # ensure odd number of samples so array length preserved
            smoothing_samples = int(numpy.ceil(smoothing_distance / self.transect_spacing))
            smoothing_samples = int(smoothing_samples / 2) * 2 + 1
            label = f"{smoothing_distance/1000}km"
            # try a variety of smoothing approaches
            transects[f'widths_mean_{label}'] = self._rolling_mean_with_padding(widths_no_nan, smoothing_samples)
            '''transects[f'widths_Savgol_{label}'] = scipy.signal.savgol_filter(
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
                           transect_samples=sampled_elevations,
                           threshold=threshold,
                           aligned_channel=aligned_channel,
                           include_transects=False)

        # Return results for now
        return transects
