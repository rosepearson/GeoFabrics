# -*- coding: utf-8 -*-
"""
This module contains classes associated with characterising channel geometry
information.
"""

import geopandas
import pandas
import pathlib
import shapely
import numpy
import xarray
import scipy
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot
import logging


class Channel:
    """A class to define a channel centre line from a digital network."""

    def __init__(
        self,
        channel: geopandas.GeoDataFrame,
        resolution: float,
        sampling_direction: int = 1,
    ):
        """A channel centreline and functions to support sampling or smoothing
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
    def from_network(
        cls,
        network_file: pathlib.Path,
        crs: int,
        starting_id: int,
        resolution: float,
        area_threshold: float,
        name_dict: dict,
        sampling_direction: int = -1,
    ):
        """Create a channel object from a REC file.

        Parameters
        ----------

        network
            Contains geometry and relational information between upstream and downstream
            reaches.
        starting_id
            The ID of the reach to trace upstream from.
        name_dict
            The column names of the network.
        area_threshold
            The minimum upstream area for upstream reaches to be included.
        resolution
            The resolution of the upstream polyline trace to create from the network.
        sampling_direction
            The direction to sample each reach polyline. 1 if each reach is defined
            downstream to upstream, -1 otherwise.
        """

        # Load in network and remove unneeded columns
        network = geopandas.read_file(network_file).to_crs(crs)
        # Drop any non-required columns
        network = network[
            [
                name_dict["id"],
                name_dict["to_node"],
                name_dict["from_node"],
                name_dict["flow"],
                name_dict["mannings_n"],
                name_dict["area"],
                "geometry",
            ]
        ]
        network = network.rename(
            columns={value: key for key, value in name_dict.items()}
        )
        assert len(network["id"]) == len(network["id"].unique()), (
            "The reach IDs much be unique in for the network to be valid, but there are"
            "duplicates"
        )
        reaches = cls._get_up_stream_reaches(
            network=network,
            starting_id=starting_id,
            area_threshold=area_threshold,
        )
        channel = cls(
            channel=reaches,
            resolution=resolution,
            sampling_direction=sampling_direction,
        )
        return channel

    @classmethod
    def _get_up_stream_reaches(
        cls,
        network: geopandas.GeoDataFrame,
        starting_id: int,
        area_threshold: float = None,
    ):
        """A function to trace the largest network branch upstream from a starting id.

        There must only be one reference to each ID, and no circular relationships in
        the to_node and from_node's.'

        Parameters
        ----------

        network
            Contains geometry and relational information between upstream and downstream
            reaches.
        starting_id
            The ID of the reach to trace upstream from.
        name_dict
            The column names of the network.
        area_threshold
            The minimum upstream area for upstream reaches to be included.
        """
        reaches_index = network[network["id"] == starting_id].index.tolist()

        # While loop with several break statements and asserts to avoid run-away code
        while True:
            upstream_reaches = network[
                network["to_node"] == network.loc[reaches_index[-1]]["from_node"]
            ][["area", "id"]]
            if len(upstream_reaches) == 0:
                # No more upstream reached
                break
            # Select the reach with the largest upstream area
            max_index = upstream_reaches["area"].idxmax()
            if (
                area_threshold is not None
                and upstream_reaches.loc[max_index]["area"] < area_threshold
            ):
                # The minimum upstream area threshold has been reached
                break
            next_index = upstream_reaches.loc[max_index]["id"]
            assert next_index not in numpy.array(reaches_index), (
                "The same reach ID is referenced twice indicating circular network "
                "connections"
            )
            reaches_index.append(network[network["id"] == next_index].index[0])
        reaches = network.loc[reaches_index]
        return reaches

    def get_parametric_spline_fit_points(self, k: int = 3, spacing: float = None):
        """Return the spline smoothed polyline points created from two splines defining
        a parametric curve fit to points along the channel. The curve is resampled at
        even spacing once the splines are fit.
        If no spacing is provided the channel corner points are used, if a spacing is
        provided the channel corner points with sampling at the spacing are used.

        Parameters
        ----------

        k
            The polynomial degree. Should be off. 1 <= k <= 5.
        spacing
            The spacing between sampled points along straight segments
        """

        # Get points along channel
        if spacing is None:
            xy = self._get_corner_points(
                channel=self.channel, sampling_direction=self.sampling_direction
            )
        else:
            xy = self.get_spaced_points_with_corners(
                channel=self.channel,
                sampling_direction=self.sampling_direction,
                spacing=spacing,
            )
        if len(xy[0]) > 3:  # default k= 3, must be greater to fit with knots
            xy = self._fit_spline_between_xy(xy, k)
        spline_channel = shapely.geometry.LineString(xy.T)
        # resample at even spacing
        xy = []
        for distance in numpy.arange(
            0, spline_channel.length + self.resolution / 2, self.resolution
        ):
            xy.append(spline_channel.interpolate(distance))
        spline_channel = shapely.geometry.LineString(xy)
        return numpy.array(spline_channel.xy)

    def get_parametric_spline_fit(self, k: int = 3, spacing: float = None):
        """Return the spline smoothed polyline created from two splines defining a
        parametric curve fit to points along the channel. The curve is resampled at even
        spacing once the splines are fit.
        If no spacing is provided the channel corner points are used, if a spacing is
        provided the channel corner points with sampling at the spacing are used.

        Parameters
        ----------

        k
            The polynomial degree. Should be off. 1 <= k <= 5.
        spacing
            The spacing between sampled points along straight segments
        """

        # Get spline points
        xy = self.get_parametric_spline_fit_points(k=k, spacing=spacing)
        # create dataframe
        spline_channel = shapely.geometry.LineString(xy.T)
        spline_channel = geopandas.GeoDataFrame(
            geometry=[spline_channel], crs=self.channel.crs
        )
        return spline_channel

    def get_b_spline_fit(
        self, smoothing_multiplier: float = 50, spacing: float = None
    ) -> numpy.ndarray:
        """Return the spline smoothed polyline created using a B-spline fit to corner
        points.
        If no spacing is provided the channel corner points are used, if a spacing is
        provided the channel corner points with sampling at the spacing are used.

        Parameters
        ----------

        smoothing_multiplier
            The polynomial degree. Should be off. 1 <= k <= 5.
        spacing
            The spacing between sampled points along straight segments
        """
        xy = self.get_b_spline_fit_points(
            smoothing_multiplier=smoothing_multiplier, spacing=spacing
        )
        spline_channel = shapely.geometry.LineString(xy)
        spline_channel = geopandas.GeoDataFrame(
            geometry=[spline_channel], crs=self.channel.crs
        )
        return spline_channel

    def get_b_spline_fit_points(
        self, smoothing_multiplier: float = 50, spacing: float = None
    ) -> numpy.ndarray:
        """Return the spline smoothed polyline points created using a B-spline fit to
        corner points.
        If no spacing is provided the channel corner points are used, if a spacing is
        provided the channel corner points with sampling at the spacing are used.

        Parameters
        ----------

        smoothing_multiplier
            The polynomial degree. Should be off. 1 <= k <= 5.
        spacing
            The spacing between sampled points along straight segments
        """
        if spacing is None:
            xy = self._get_corner_points(
                channel=self.channel, sampling_direction=self.sampling_direction
            )
        else:
            xy = self.get_spaced_points_with_corners(
                channel=self.channel,
                sampling_direction=self.sampling_direction,
                spacing=spacing,
            )
        if len(xy[0]) > 3:  # There must be more than three points to fit a spline
            xy = self._fit_spline_through_xy(xy, smoothing_multiplier)
        return xy.T

    def get_channel_catchment(self, corridor_radius: float) -> geopandas.GeoDataFrame:
        """Create a catchment from the smooth channel and the specified
        radius.

        Parameters
        ----------

        corridor_radius
            The radius of the channel corridor. This will determine the width of
            the channel catchment.
        """

        smooth_channel = self.get_parametric_spline_fit()
        channel_catchment = geopandas.GeoDataFrame(
            geometry=smooth_channel.buffer(corridor_radius)
        )
        return channel_catchment

    def _remove_duplicate_points(cls, xy: numpy.ndarray) -> numpy.ndarray:
        """Remove duplicate xy pairs in a list of xy points.

        Parameters
        ----------

        xy
            A paired n x 2 array of x, y points.
        """
        xy_unique, indices = numpy.unique(xy, axis=1, return_index=True)
        indices.sort()
        xy = xy[:, indices]
        return xy

    def _get_corner_points(cls, channel, sampling_direction: int) -> numpy.ndarray:
        """Extract the corner points from the provided polyline.

        Parameters
        ----------

        channel
            A data frame with the geometry defining the channel centreline.
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

    def get_spaced_points(
        self, channel, spacing: float, sampling_direction: int
    ) -> numpy.ndarray:
        """Sample at the specified spacing along the entire line.

        Parameters
        ----------

        channel
            A data frame with the geometry defining the channel centreline.
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
        xy_spaced = [
            line_string.interpolate(i * segment_resolution)
            for i in numpy.arange(0, number_segment_samples, 1)
        ]

        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def get_spaced_points_with_corners(
        self, channel, spacing: float, sampling_direction: int
    ) -> numpy.ndarray:
        """Sample at the specified spacing along each straight segment.

        Parameters
        ----------

        channel
            A data frame with the geometry defining the channel centreline.
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
        for i in numpy.arange(0, len(x) - 1, 1):
            line_segment = shapely.geometry.LineString(
                [[x[i], y[i]], [x[i + 1], y[i + 1]]]
            )

            number_segment_samples = max(numpy.round(line_segment.length / spacing), 2)
            segment_resolution = line_segment.length / (number_segment_samples - 1)

            xy_spaced.extend(
                [
                    line_segment.interpolate(i * segment_resolution)
                    for i in numpy.arange(0, number_segment_samples)
                ]
            )
        # Check for and remove duplicate points
        xy = numpy.array(shapely.geometry.LineString(xy_spaced).xy)
        xy = self._remove_duplicate_points(xy)

        return xy

    def _fit_spline_through_xy(
        self, xy: numpy.ndarray, smoothing_multiplier: float
    ) -> numpy.ndarray:
        """Fits a B-spline representation of the curve represented by xy. This is
        sampled at the channel resolution.

        Parameters
        ----------

        xy
            A paired n x 2 array of x, y points.
        smoothing_multiplier
            This is multiplied by the number of aligned_centreline points and
            passed into the scipy.interpolate.splprep.
        """

        smoothing_factor = smoothing_multiplier * len(xy[0])

        tck_tuple, u_input = scipy.interpolate.splprep(xy, s=smoothing_factor)

        # Sample every roughly res along the spine with rough line length estimate
        line_length = shapely.geometry.LineString(xy.T).length
        sample_step_u = 1 / round(line_length / self.resolution)
        # Increase sample range 1 past in each direction to avoid shrtinage of the line
        u_sampled = numpy.arange(-sample_step_u, 1 + 2 * sample_step_u, sample_step_u)
        xy_sampled = scipy.interpolate.splev(u_sampled, tck_tuple)
        xy_sampled = numpy.array(xy_sampled)

        return xy_sampled

    def _fit_spline_between_xy(self, xy: numpy.ndarray, k: int) -> numpy.ndarray:
        """Fits two splines using knots to define a parametric curve made of x(t) and
        y(t) curves.
        Note that the xy points must be evenly spaced if the output points are to be
        evenly spaced.

        Parameters
        ----------

        xy
            A paired n x 2 array of x, y points.
        k
            The polynomial degree. Should be off. 1 <= k <= 5.
        """

        num_points = len(xy[0])

        assert num_points > k, (
            "scipy.interpolate require num_points > k." + "Select a larger k."
        )

        knotspace = range(num_points)
        knots = scipy.interpolate.InterpolatedUnivariateSpline(
            knotspace, knotspace, k=k
        ).get_knots()
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


class InterpolateMeasuredElevations:
    """Interpolate measured river bed elevations. Requires LineStrings of
    riverbed elevations, and the LineStrings defining the left and right
    riverbank."""

    def __init__(
        self,
        riverbank_file: pathlib.Path,
        measured_sections_file: pathlib.Path,
        cross_section_spacing: int,
    ):
        """Setup for interpolating the measured riverbed elevations along
        equally spaced cross sections."""
        self.riverbanks = geopandas.read_file(riverbank_file)
        self.measured_sections = geopandas.read_file(measured_sections_file)
        self.cross_section_spacing = cross_section_spacing
        self.check_valid_inputs()

    def check_valid_inputs(self):
        """Check expected entries in the input data"""

        # Check expected riverbank information
        if (
            len(self.riverbanks.columns) != 2
            and "Name" not in self.riverbanks.columns
            and "geometry" not in self.riverbanks.columns
        ):
            raise Exception(
                "Unexpected columns for self.riverbanks. Expecting"
                " ['Name', 'geometry'], but got"
                f" {self.riverbanks.columns}"
            )

        if (
            len(self.riverbanks) != 2
            and not (self.riverbanks["Name"] == "Right").any()
            and not (self.riverbanks["Name"] == "Left").any()
        ):
            raise Exception(
                "Unexpected row for self.riverbanks. Expecting two"
                " rows with names ['Right', 'Left'], but got"
                f" {self.riverbanks['Name']}"
            )

        # Check the riverbank lines are both in the same direction
        if self.riverbanks.iloc[0].geometry.interpolate(0, normalized=True).distance(
            self.riverbanks.loc[1].geometry.interpolate(0, normalized=True)
        ) > self.riverbanks.iloc[0].geometry.interpolate(0, normalized=True).distance(
            self.riverbanks.loc[1].geometry.interpolate(1, normalized=True)
        ) or (
            self.riverbanks.iloc[0]
            .geometry.interpolate(1, normalized=True)
            .distance(self.riverbanks.loc[1].geometry.interpolate(1, normalized=True))
            > self.riverbanks.iloc[0]
            .geometry.interpolate(1, normalized=True)
            .distance(self.riverbanks.loc[1].geometry.interpolate(0, normalized=True))
        ):
            logging.Warning(
                "It appears that the `right` and `left` banks are"
                "defined in different directions based on the "
                "relative distance between the endpoints. Please "
                "define both as either upstream or downstream."
            )

        if self.riverbanks.crs is None:
            raise Exception(
                "self.riverbanks must have a valid crs. Instead"
                f"has {self.riverbanks.crs}"
            )

        if (
            self.measured_sections.crs is None
            or self.riverbanks.crs != self.measured_sections.crs
        ):
            raise Exception(
                "self.measured_sections must have a valid crs that matches self.riverbanks. Instead"
                f"has {self.measured_sections.crs}"
            )

    def interpolate(self, samples_per_section: int, thalweg_centre: bool = True):
        """Interpolate with equally spaced points along each cross section

        Parameters
        ----------

        number_samples
            The number of samples along each cross section."""

        # Define river polygon and clip the measured sections
        polygon = geopandas.GeoDataFrame(
            geometry=[
                shapely.geometry.Polygon(
                    numpy.append(
                        self.riverbanks[self.riverbanks["Name"] == "Right"]
                        .iloc[0]
                        .geometry.coords,
                        self.riverbanks[self.riverbanks["Name"] == "Left"]
                        .iloc[0]
                        .geometry.coords[::-1],
                        axis=0,
                    )
                )
            ],
            crs=self.riverbanks.crs,
        )
        # Clip measured sections
        self.measured_sections = self.measured_sections.clip(polygon).sort_index()
        measured_sections_exploded = self.measured_sections.apply(
            lambda row: shapely.geometry.MultiPoint(row["geometry"].coords), axis=1
        ).explode(index_parts=True)

        # work out number of cross sections
        n_cross_sections = round(
            self.riverbanks.length.max() / self.cross_section_spacing
        )

        # Split riverbank polylines
        cross_sections = self.riverbanks.copy(deep=True)
        normalised_sample_locations = numpy.arange(
            0, 1 + 1 / n_cross_sections, 1 / n_cross_sections
        )
        cross_sections["geometry"] = cross_sections["geometry"].apply(
            lambda row: shapely.geometry.MultiPoint(
                row.interpolate(normalised_sample_locations, normalized=True)
            )
        )

        # Reshape to separate right and left column - split lines to points
        cross_sections = cross_sections.explode(index_parts=True)
        cross_sections = pandas.concat(
            [
                cross_sections[cross_sections["Name"] == "Right"]
                .droplevel(0)
                .geometry.rename("Right"),
                cross_sections[cross_sections["Name"] == "Left"]
                .droplevel(0)
                .geometry.rename("Left"),
            ],
            axis=1,
        )

        # Create cross sections
        cross_sections["Sections"] = cross_sections.apply(
            lambda row: shapely.geometry.LineString([row["Right"], row["Left"]]), axis=1
        )

        # Define the node spacings along the cross sections
        if thalweg_centre:
            self.thalweg_centred_spacing(
                samples_per_section=samples_per_section,
                cross_sections=cross_sections,
                measured_sections_exploded=measured_sections_exploded)
        else:
            self.uniform_spacing(
                samples_per_section=samples_per_section,
                cross_sections=cross_sections)

        # Explode the cross section points
        cross_sections_exploded = (
            cross_sections[["Sections"]]
            .set_geometry("Sections", crs=self.riverbanks.crs)
            .explode(index_parts=True)
        )

        # Map measured section values then interpolated down/up river
        nearest_cross_section_point_indices = measured_sections_exploded.apply(
            lambda row: row.distance(
                cross_sections["Sections"].explode(index_parts=True)
            ).idxmin()
        ).unique()
        cross_sections_exploded["z"] = numpy.nan
        cross_sections_exploded.loc[
            nearest_cross_section_point_indices, "z"
        ] = cross_sections_exploded.loc[
            nearest_cross_section_point_indices, "Sections"
        ].apply(
            lambda point: measured_sections_exploded.loc[
                point.distance(measured_sections_exploded).idxmin()
            ].z
        )
        cross_sections_exploded["z"] = (
            cross_sections_exploded["z"]
            .groupby(level=1, group_keys=False)
            .apply(lambda group: group.interpolate(limit_direction="both"))
        )

        # Return river polygon and sampled cross sections
        return polygon, cross_sections_exploded

    def uniform_spacing(self, samples_per_section: int, cross_sections):
        """Interpolate with equally spaced points along each cross section

        Parameters
        ----------

        number_samples
            The number of samples along each cross section."""

        normalised_node_locations = numpy.arange(0,
                                                 1 + 1.0/samples_per_section,
                                                 1.0/samples_per_section)
        cross_sections['Sections'] = (
            cross_sections['Sections'].apply(
                lambda line:
                    shapely.geometry.MultiPoint(
                        line.interpolate(normalised_node_locations,
                                         normalized=True))
                    )
                )

    def thalweg_centred_spacing(
        self,
        samples_per_section: int,
        cross_sections: geopandas.GeoDataFrame,
        measured_sections_exploded: geopandas.GeoDataFrame,
    ):
        """Interpolate with equal number of equally spaced points on each side
        of the Thalweg.

        Parameters
        ----------

        number_samples
            The number of samples along each cross section."""

        # Calculate the normalised thalweg location at each measured section
        def calculate_normalised_thalweg_location(indices: tuple, sections_exploded):
            """Tuple of the section index, and the point index along the section"""
            thalweg_point = sections_exploded[indices]
            left_bank_distance = thalweg_point.distance(
                sections_exploded.loc[indices[0]][0]
            )
            right_bank_distance = thalweg_point.distance(
                sections_exploded.loc[indices[0]][
                    len(sections_exploded.loc[indices[0]]) - 1
                ]
            )
            normalised_thalweg_location = left_bank_distance / (
                left_bank_distance + right_bank_distance
            )

            return normalised_thalweg_location

        thalweg_indices = (
            measured_sections_exploded.apply(lambda row: row.z)
            .groupby(level=0)
            .idxmin()
        )
        thalweg_points = measured_sections_exploded[thalweg_indices]

        # Linearly interpolate the normalised Thalewg location along the cross sections
        cross_sections["Thalweg ratio"] = numpy.nan
        cross_sections.loc[
            thalweg_indices.apply(
                lambda row: cross_sections["Sections"]
                .distance(thalweg_points[row])
                .idxmin()
            ),
            "Thalweg ratio",
        ] = thalweg_indices.apply(
            lambda row: calculate_normalised_thalweg_location(
                row, measured_sections_exploded
            )
        ).array
        cross_sections["Thalweg ratio"].interpolate(
            limit_direction="both", inplace=True
        )

        # Define equal sample locations on each side of the Thalweg
        def split_line_to_nodes(line, thalweg_ratio, samples_per_section):
            """Split line into n_nodes points half equally spaced on each side of the Thalweg"""
            normalised_node_locations = numpy.concatenate(
                [
                    numpy.arange(samples_per_section) * thalweg_ratio/samples_per_section,
                    numpy.arange(samples_per_section + 1) * (1-thalweg_ratio)/samples_per_section + thalweg_ratio
                ]
            )
            return shapely.geometry.MultiPoint(
                line.interpolate(normalised_node_locations, normalized=True)
            )

        # Apply the function for sampling along each cross section
        cross_sections["Sections"] = cross_sections[
            ["Sections", "Thalweg ratio"]
        ].apply(
            lambda row: split_line_to_nodes(
                row["Sections"],
                row["Thalweg ratio"],
                samples_per_section=samples_per_section,
            ),
            axis=1,
        )


class ChannelCharacteristics:
    """A class to estimate the width, slope and other characteristics of a
    channel from a detailed DEM and a river network."""

    def __init__(
        self,
        gnd_dem: xarray.Dataset,
        veg_dem: xarray.Dataset,
        cross_section_spacing: float,
        resolution: float,
        debug: bool = False,
    ):
        """Setup for estimating river characteristics from DEM cross sections.

        gnd_dem
            The ground DEM along the channel
        veg_dem
            The vegetation DEM along the channel
        cross_section_spacing
            The spacing down channel of the cross sections.
        resolution
            The resolution to sample at.
        """

        self.gnd_dem = gnd_dem
        self.veg_dem = veg_dem
        self.cross_section_spacing = cross_section_spacing
        self.resolution = resolution
        self.transect_radius = None
        self.debug = debug

    @property
    def number_of_samples(self) -> int:
        """Return the number of samples to take along cross_sections. This should
        be an odd number. Subtract 1 instead of adding to ensure within the
        generated DEM."""

        assert (
            self.transect_radius is not None
        ), "Transect radius must be set before this is called"
        return int(self.transect_radius / self.resolution) * 2 - 1

    def calculate_min_z_start_i(self, min_z_search_radius) -> int:
        """Return the starting index of samples along each transect to begin
        looking for the minimum z.

        Parameters
        ----------

        min_z_search_radius
            The distance to search from the centre.
        """

        number_min_z_samples = int(min_z_search_radius / self.resolution) * 2 - 1

        return int((self.number_of_samples - number_min_z_samples) / 2)

    def calculate_min_z_stop_i(self, min_z_search_radius) -> int:
        """Return the stopping index of samples along each transect to begin
        looking for the minimum z.

        Parameters
        ----------

        min_z_search_radius
            The distance to search from the centre.
        """

        return int(
            self.number_of_samples - self.calculate_min_z_start_i(min_z_search_radius)
        )

    @property
    def centre_index(self) -> int:
        """Return the centre index for samples taken along a transect."""
        return int(numpy.floor(self.number_of_samples / 2))

    def _segment_slope(self, x_array, y_array, index):
        """Return the slope and length characteristics of a line segment.

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
            + (y_array[index + 1] - y_array[index]) ** 2
        )
        dx = (x_array[index + 1] - x_array[index]) / length
        dy = (y_array[index + 1] - y_array[index]) / length
        return dx, dy, length

    def node_centred_reach_cross_section(self, sampled_channel: geopandas.GeoDataFrame):
        """Calculate cross_sections along a channel at the midpoint of each
        segment.

        Parameters
        ----------

        sampled_channel
            The sampled channel defined as a single polyline. Any branches described
            separately.
        """

        cross_sections_dict = {
            "geometry": [],
            "nx": [],
            "ny": [],
            "length": [],
            "mid_x": [],
            "mid_y": [],
        }

        assert len(sampled_channel) == 1, (
            "Expect only one polyline "
            f"geometry per channel. Instead got {len(sampled_channel)}"
        )

        (x_array, y_array) = sampled_channel.iloc[0].geometry.xy
        for i in range(len(x_array)):
            # calculate slope along segment
            if i == 0:
                # first segment - slope of next segment
                dx, dy, length = self._segment_slope(x_array, y_array, i)
            elif i == len(x_array) - 1:
                # last segment - slope of previous segment
                dx, dy, length = self._segment_slope(x_array, y_array, i - 1)
            else:
                # slope of the length weighted mean of both segments
                dx_prev, dy_prev, l_prev = self._segment_slope(x_array, y_array, i)
                dx_next, dy_next, l_next = self._segment_slope(x_array, y_array, i - 1)
                dx = (dx_prev * l_prev + dx_next * l_next) / (l_prev + l_next)
                dy = (dy_prev * l_prev + dy_next * l_next) / (l_prev + l_next)
                length = (l_prev + l_next) / 2
            normal_x = -dy
            normal_y = dx

            # record normal to a segment nx and ny
            cross_sections_dict["nx"].append(normal_x)
            cross_sections_dict["ny"].append(normal_y)

            # calculate transect - using effectively nx and ny
            cross_sections_dict["geometry"].append(
                shapely.geometry.LineString(
                    [
                        [
                            x_array[i] - self.transect_radius * normal_x,
                            y_array[i] - self.transect_radius * normal_y,
                        ],
                        [x_array[i], y_array[i]],
                        [
                            x_array[i] + self.transect_radius * normal_x,
                            y_array[i] + self.transect_radius * normal_y,
                        ],
                    ]
                )
            )
            cross_sections_dict["mid_x"].append(x_array[i])
            cross_sections_dict["mid_y"].append(y_array[i])

            # record the length of the line segment
            cross_sections_dict["length"].append(length)
        cross_sections = geopandas.GeoDataFrame(
            cross_sections_dict, crs=sampled_channel.crs
        )
        return cross_sections

    def _estimate_water_level_and_slope(self, cross_sections: geopandas.GeoDataFrame):
        """Estimate the water level and slope from the minimum z heights along
        the sampled cross_sections after applying appropiate smoothing and
        constraints to ensure it is monotonically increasing. Values are stored
        in the cross_sections.

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines.
        """

        # Min z values as the water surface. Ensure no NaN
        cross_sections["min_z_centre"] = cross_sections["min_z_centre"].interpolate(
            "index", limit_direction="both"
        )
        # Unimodal enforaces monotonically increasing
        cross_sections["min_z_centre_unimodal"] = self._unimodal_smoothing(
            cross_sections["min_z_centre"]
        )

        # Set the water z value to use for width thresholding
        cross_sections["min_z_water"] = cross_sections["min_z_centre_unimodal"]

        # Slope from the water surface - interpolate to fill any Nan
        cross_sections["slope"] = (
            cross_sections["min_z_water"].diff() / self.cross_section_spacing
        )
        cross_sections["slope"] = cross_sections["slope"].interpolate(
            "index", limit_direction="both"
        )

        # Slopes for a range of smoothings
        for smoothing_distance in [50, 250, 500, 1000]:
            # ensure odd number of samples so array length preserved
            smoothing_samples = int(
                numpy.ceil(smoothing_distance / self.cross_section_spacing)
            )
            smoothing_samples = int(smoothing_samples / 2) * 2 + 1
            label = f"{smoothing_distance/1000}km"

            # Smoothed min_z_centre_unimodal
            cross_sections[
                f"min_z_centre_unimodal_mean_{label}"
            ] = self._rolling_mean_with_padding(
                cross_sections["min_z_centre_unimodal"], smoothing_samples
            )

            # Smoothed slope
            cross_sections[f"slope_mean_{label}"] = self._rolling_mean_with_padding(
                cross_sections["slope"], smoothing_samples
            )

    def _smooth_widths_and_thresholds(self, cross_sections: geopandas.GeoDataFrame):
        """Record the valid and reolling mean of the calculated thresholds and widths

        Parameters
        ----------

        cross_sections
            The elevations and calculated widths and thresholds for each sampled cross
            section
        """

        invalid_mask = numpy.logical_not(cross_sections["valid"])

        # Tidy up widths - pull out the valid widths
        cross_sections["valid_widths"] = cross_sections["widths"]
        cross_sections.loc[invalid_mask, "valid_widths"] = numpy.nan
        widths_no_nan = cross_sections["valid_widths"].interpolate(
            "index", limit_direction="both"
        )

        # Flat widths
        cross_sections["valid_flat_widths"] = cross_sections["flat_widths"]
        cross_sections.loc[invalid_mask, "valid_flat_widths"] = numpy.nan
        flat_widths_no_nan = cross_sections["valid_flat_widths"].interpolate(
            "index", limit_direction="both"
        )

        # Tidy up thresholds - pull out the valid thresholds
        cross_sections["valid_threhold"] = cross_sections["threshold"]
        cross_sections.loc[invalid_mask, "valid_threhold"] = numpy.nan
        thresholds_no_nan = cross_sections["valid_threhold"].interpolate(
            "index", limit_direction="both"
        )

        # Cycle through and caluclate the rolling mean
        for smoothing_distance in [50, 150, 200, 250]:
            # ensure odd number of samples so array length preserved
            smoothing_samples = int(
                numpy.ceil(smoothing_distance / self.cross_section_spacing)
            )
            smoothing_samples = int(smoothing_samples / 2) * 2 + 1
            label = f"{smoothing_distance/1000}km"

            # Apply the rolling mean to each
            cross_sections[f"widths_mean_{label}"] = self._rolling_mean_with_padding(
                widths_no_nan, smoothing_samples
            )
            cross_sections[
                f"flat_widths_mean_{label}"
            ] = self._rolling_mean_with_padding(flat_widths_no_nan, smoothing_samples)
            cross_sections[
                f"thresholds_mean_{label}"
            ] = self._rolling_mean_with_padding(thresholds_no_nan, smoothing_samples)

            """cross_sections[f'widths_Savgol_{label}'] = scipy.signal.savgol_filter(
                cross_sections['widths'].interpolate('index', limit_direction='both'),
                smoothing_samples,  # Ensure odd. number of samples included
                3)  # Polynomial order"""

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

    def sample_cross_sections(
        self, cross_sections: geopandas.GeoDataFrame, min_z_search_radius: float
    ):
        """Return the elevations along the cross_section sampled at the
        sampling resolution. Also add the measured 'min_z_centre' values to
        the cross_sections.

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines.
        min_z_search_radius
            The distance to search from the centre.

        """

        # The number of transect samples - ensure odd - defined from the first
        sample_index_array = numpy.arange(
            -numpy.floor(self.number_of_samples / 2),
            numpy.floor(self.number_of_samples / 2) + 1,
            1,
        )

        min_z_start_i = self.calculate_min_z_start_i(min_z_search_radius)
        min_z_stop_i = self.calculate_min_z_stop_i(min_z_search_radius)

        cross_section_elevations = {"gnd_elevations": [], "veg_elevations": []}
        min_z_centre = []

        # create tree of ground values to sample from
        grid_x, grid_y = numpy.meshgrid(self.gnd_dem.x, self.gnd_dem.y)
        xy_in = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()
        gnd_tree = scipy.spatial.KDTree(xy_in)

        # create tree of vegetation values to sample from
        grid_x, grid_y = numpy.meshgrid(self.veg_dem.x, self.veg_dem.y)
        xy_in = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()
        veg_tree = scipy.spatial.KDTree(xy_in)

        # cycle through each transect - calculate sample points then look up
        for index, row in cross_sections.iterrows():
            # Calculate xx, and yy points to sample at
            xx = row["mid_x"] + sample_index_array * self.resolution * row["nx"]
            yy = row["mid_y"] + sample_index_array * self.resolution * row["ny"]
            xy_points = numpy.concatenate([[xx], [yy]], axis=0).transpose()

            # Sample the vegetation elevations at along the transect
            distances, indices = veg_tree.query(xy_points)
            elevations = self.veg_dem.z.data.flatten()[indices]
            cross_section_elevations["veg_elevations"].append(elevations)

            # Sample the ground elevations at along the transect
            distances, indices = gnd_tree.query(xy_points)
            elevations = self.gnd_dem.z.data.flatten()[indices]
            cross_section_elevations["gnd_elevations"].append(elevations)

            # Find the min elevation along the middle of each cross section
            if (
                len(elevations[min_z_start_i:min_z_stop_i])
                - numpy.sum(numpy.isnan(elevations[min_z_start_i:min_z_stop_i]))
                > 0
            ):
                min_index = numpy.nanargmin(elevations[min_z_start_i:min_z_stop_i])
                min_z_centre.append(elevations[min_z_start_i + min_index])
            else:
                min_z_centre.append(numpy.nan)
        # Set min_z in the cross sections
        cross_sections["min_z_centre"] = min_z_centre

        return cross_section_elevations

    def fixed_thresholded_widths_from_centre_within_radius(
        self,
        cross_sections: geopandas.GeoDataFrame,
        cross_section_elevations: dict,
        threshold: float,
        resolution: float,
        search_radius: float,
        min_channel_width: float,
    ):
        """Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected. Takes the widest channel within
        the radius.'

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines.
        cross_section_elevations
            The sampled elevations along the cross_sections.
        threshold
            The height above the water level to detect as a bank.
        resolution
            The resolution to sample at.
        search_radius
            The distance to search side to side from the centre index.
        min_channel_width
            The minimum width of a 'valid' channel.
        """

        search_radius_index = int(search_radius / self.resolution)
        widths = {
            "widths": [],
            "first_bank_i": [],
            "last_bank_i": [],
            "channel_count": [],
        }

        for j in range(len(cross_section_elevations["gnd_elevations"])):
            assert (
                len(cross_section_elevations["gnd_elevations"][j])
                == self.number_of_samples
            ), "Expect fixed length"

            gnd_samples = cross_section_elevations["gnd_elevations"][j]
            veg_samples = cross_section_elevations["veg_elevations"][j]
            start_index = self.centre_index
            z_water = cross_sections.iloc[j]["min_z_water"]

            start_i, stop_i, channel_count = self.fixed_threshold_width(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=start_index,
                z_water=z_water,
                threshold=threshold,
                search_radius_index=search_radius_index,
                min_channel_width=min_channel_width,
            )

            # assign the longest width
            widths["first_bank_i"].append(start_i)
            widths["last_bank_i"].append(stop_i)
            widths["widths"].append((stop_i - start_i) * resolution)
            widths["channel_count"].append(channel_count)
        for key in widths.keys():
            cross_sections[key] = widths[key]
        # Record if the width is valid - only one possible channel that starts and ends
        # within the samples
        valid_mask = cross_sections["channel_count"] == 1
        valid_mask &= cross_sections["first_bank_i"] > 0
        valid_mask &= cross_sections["last_bank_i"] < self.number_of_samples - 1
        cross_sections["valid"] = valid_mask

    def variable_thresholded_widths_from_centre_within_radius(
        self,
        cross_sections: geopandas.GeoDataFrame,
        cross_section_elevations: dict,
        threshold: float,
        resolution: float,
        search_radius: float,
        maximum_threshold: float,
        min_channel_width=float,
    ):
        """Estimate width based on a thresbold of bank height above water level.
        Start in the centre and work out. Doesn't detect banks until a value
        less than the threshold has been detected. Takes the widest channel within
        the radius.'

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines.
        cross_section_elevations
            The sampled elevations along the cross_sections.
        threshold
            The height above the water level to detect as a bank.
        resolution
            The resolution to sample at.
        search_radius
            The distance to search side to side from the centre index.
        maximum_threshold
            The maximum amount to increase the bank height before stopping.
        min_channel_width
            The minimum width of a 'valid' channel.
        """

        search_radius_index = int(search_radius / self.resolution)

        widths = {
            "widths": [],
            "first_bank_i": [],
            "last_bank_i": [],
            "threshold": [],
            "channel_count": [],
            "flat_widths": [],
            "first_flat_bank_i": [],
            "last_flat_bank_i": [],
        }

        for j in range(len(cross_section_elevations["gnd_elevations"])):
            logging.info(
                f"Variable thresholding cross section {j} out of "
                f"{len(cross_section_elevations['gnd_elevations'])}"
            )
            assert (
                len(cross_section_elevations["gnd_elevations"][j])
                == self.number_of_samples
            ), "Expect fixed length"

            gnd_samples = cross_section_elevations["gnd_elevations"][j]
            veg_samples = cross_section_elevations["veg_elevations"][j]
            start_index = self.centre_index
            z_water = cross_sections.iloc[j]["min_z_water"]

            # Get width based on fixed threshold
            start_i, stop_i, channel_count = self.fixed_threshold_width(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=start_index,
                z_water=z_water,
                threshold=threshold,
                search_radius_index=search_radius_index,
                min_channel_width=min_channel_width,
            )

            widths["first_flat_bank_i"].append(start_i)
            widths["last_flat_bank_i"].append(stop_i)
            widths["flat_widths"].append((stop_i - start_i) * resolution)

            # Iterate out from the fixed threshold width until the banks go down, or the
            # max threshold is reached
            maximum_z = z_water + maximum_threshold
            if numpy.isnan(start_i) or numpy.isnan(stop_i):
                # No valid width to begin with
                dz_bankfull = numpy.nan
            else:
                # Iterate out from the fixed threshold width until the banks go down,
                # or the max threshold is reached
                z_bankfull = numpy.nanmin(gnd_samples[[start_i, stop_i]])
                start_i_bf = start_i
                stop_i_bf = stop_i
                dwidth = 1  # Change in width this iteration

                while (
                    start_i_bf > 0
                    and stop_i_bf < self.number_of_samples - 1
                    and dwidth > 0
                ):
                    dwidth = 0

                    # break if going down - TODO do after other side finished increasing
                    if gnd_samples[start_i_bf - 1] < numpy.nanmax(
                        [gnd_samples[start_i_bf], z_bankfull]
                    ) or gnd_samples[stop_i_bf + 1] < numpy.nanmax(
                        [gnd_samples[stop_i_bf], z_bankfull]
                    ):
                        break
                    # if not, extend whichever bank is lower
                    if gnd_samples[start_i_bf - 1] > gnd_samples[stop_i_bf + 1]:
                        stop_i_bf += 1
                        dwidth += 1
                    elif gnd_samples[start_i_bf - 1] < gnd_samples[stop_i_bf + 1]:
                        start_i_bf -= 1
                        dwidth += 1
                    elif gnd_samples[start_i_bf - 1] == gnd_samples[stop_i_bf + 1]:
                        start_i_bf -= 1
                        stop_i_bf += 1
                        dwidth += 2
                    else:
                        # extend if value is nan and not vegetated
                        if numpy.isnan(gnd_samples[start_i_bf - 1]) and numpy.isnan(
                            veg_samples[start_i_bf - 1]
                        ):
                            start_i_bf -= 1
                            dwidth += 1
                        if numpy.isnan(gnd_samples[stop_i_bf + 1]) and numpy.isnan(
                            veg_samples[start_i_bf + 1]
                        ):
                            stop_i_bf += 1
                            dwidth += 1
                    # Break if the threshold has been meet before updating maz_z
                    if (
                        gnd_samples[start_i_bf] >= maximum_z
                        or gnd_samples[stop_i_bf] >= maximum_z
                    ):
                        break  # todo - set to the max specified threshold
                    # Break if ground is nan, but there are vegatation returns
                    if numpy.isnan(gnd_samples[start_i_bf]) and not numpy.isnan(
                        veg_samples[start_i_bf]
                    ):
                        break  # todo - look at waikanae and consider checking the vegetation hights and only saying bank fill if over threshold or going down
                    if numpy.isnan(gnd_samples[stop_i_bf]) and not numpy.isnan(
                        veg_samples[stop_i_bf]
                    ):
                        break
                    # update maximum value so far
                    if not numpy.isnan(
                        [gnd_samples[start_i_bf], gnd_samples[stop_i_bf]]
                    ).all():
                        z_bankfull = max(
                            z_bankfull,
                            numpy.nanmin(
                                [gnd_samples[start_i_bf], gnd_samples[stop_i_bf]]
                            ),
                        )
                # Set the detected bankful values
                dz_bankfull = z_bankfull - z_water
                start_i = start_i_bf
                stop_i = stop_i_bf
            # assign the longest width
            widths["first_bank_i"].append(start_i)
            widths["last_bank_i"].append(stop_i)
            widths["widths"].append((stop_i - start_i) * resolution)
            widths["threshold"].append(dz_bankfull)
            widths["channel_count"].append(channel_count)
        for key in widths.keys():
            cross_sections[key] = widths[key]
        # Record if the width is valid - only one possible channel that starts and ends
        # within the samples
        valid_mask = cross_sections["channel_count"] == 1
        valid_mask &= cross_sections["first_bank_i"] > 0
        valid_mask &= cross_sections["last_bank_i"] < self.number_of_samples - 1
        valid_mask &= cross_sections["threshold"] < maximum_threshold
        valid_mask &= numpy.logical_not(numpy.isnan(cross_sections["threshold"]))
        cross_sections["valid"] = valid_mask

    def fixed_threshold_width(
        self,
        gnd_samples: numpy.ndarray,
        veg_samples: numpy.ndarray,
        start_index: int,
        z_water: float,
        threshold: float,
        search_radius_index: int,
        min_channel_width: float,
    ):
        """Calculate the maximum width for a cross section given a fixed
        threshold - checking outwards, forwards and backwards within the
        search radius.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        z_water
            The elevation of the water.
        threshold
            The height above the water level to detect as a bank.
        search_radius_index
            The distance in indices to search for the start of a channel away
            from the start_index
        min_channel_width
            The minimum width of a 'valid' channel.
        """

        start_i_list = []
        stop_i_list = []

        forwards_index = start_index
        backwards_index = start_index

        # check outwards
        start_i, stop_i = self.fixed_threshold_width_outwards(
            gnd_samples=gnd_samples,
            veg_samples=veg_samples,
            start_index=start_index,
            z_water=z_water,
            threshold=threshold,
        )
        if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
            start_i_list.append(start_i)
            stop_i_list.append(stop_i)
            forwards_index = stop_i + 1
            backwards_index = start_i - 1
        # check forwards
        while forwards_index - start_index < search_radius_index:
            start_i, stop_i = self.fixed_threshold_width_forwards(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=forwards_index,
                z_water=z_water,
                threshold=threshold,
                stop_index=start_index + search_radius_index,
            )
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                start_i_list.append(start_i)
                stop_i_list.append(stop_i)
                forwards_index = stop_i + 1
            else:
                break
        # check backwards
        while start_index - backwards_index < search_radius_index:
            start_i, stop_i = self.fixed_threshold_width_backwards(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=backwards_index,
                z_water=z_water,
                threshold=threshold,
                stop_index=start_index - search_radius_index,
            )
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
        channel_count = 0
        for i in range(len(start_i_list)):
            channel_width = stop_i_list[i] - start_i_list[i]
            if channel_width > longest_width:
                longest_width = stop_i_list[i] - start_i_list[i]
                start_i = start_i_list[i]
                stop_i = stop_i_list[i]
            if channel_width >= min_channel_width:
                channel_count += 1
        return start_i, stop_i, channel_count

    def fixed_threshold_width_outwards(
        self,
        gnd_samples: numpy.ndarray,
        veg_samples: numpy.ndarray,
        start_index: int,
        z_water: float,
        threshold: float,
    ):
        """If the start_index is nan or less than the threshold, then cycle
        outwards until each side has gone above the threshold.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        z_water
            The elevation of the water.
        threshold
            The height above the water level to detect as a bank.

        """

        start_i = numpy.nan
        stop_i = numpy.nan

        if veg_samples[start_index] - z_water < threshold or (
            numpy.isnan(veg_samples[start_index])
        ):
            for i in numpy.arange(0, self.number_of_samples + 1, 1):
                # work forward checking height
                if start_index + i < self.number_of_samples and numpy.isnan(stop_i):
                    gnd_elevation_over_minimum = veg_samples[start_index + i] - z_water

                    # Detect banks - either ground above threshold, or no ground with
                    # vegetation over threshold
                    if numpy.isnan(stop_i) and gnd_elevation_over_minimum > threshold:
                        # Leaving the channel
                        stop_i = start_index + i
                # work backward checking height
                if start_index - i >= 0 and numpy.isnan(start_i):
                    gnd_elevation_over_minimum = veg_samples[start_index - i] - z_water

                    # Detect bank
                    if numpy.isnan(start_i) and gnd_elevation_over_minimum > threshold:
                        # Leaving the channel
                        start_i = start_index - i
                # break if both edges detected
                if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                    break
                # break if both ends of the sampled cross section reached
                if (
                    start_index + i >= self.number_of_samples - 1
                    and start_index - i <= 0
                ):
                    if numpy.isnan(start_i):
                        start_i = 0
                    if numpy.isnan(stop_i):
                        stop_i = self.number_of_samples - 1
                    break
        return start_i, stop_i

    def fixed_threshold_width_forwards(
        self,
        gnd_samples: numpy.ndarray,
        veg_samples: numpy.ndarray,
        start_index: int,
        z_water: float,
        threshold: float,
        stop_index: int,
    ):
        """Check for channels approaching forward.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        z_water
            The elevation of the water.
        threshold
            The height above the water level to detect as a bank.
        stop_index
            The maximum index to search through
        """

        for i in numpy.arange(start_index, stop_index + 1, 1):
            # check if in channel
            start_i, stop_i = self.fixed_threshold_width_outwards(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=i,
                z_water=z_water,
                threshold=threshold,
            )

            # break if both edges detected
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                break
        return start_i, stop_i

    def fixed_threshold_width_backwards(
        self,
        gnd_samples: numpy.ndarray,
        veg_samples: numpy.ndarray,
        start_index: int,
        z_water: float,
        threshold: float,
        stop_index: int,
    ):
        """Check for channels approaching backwards.

        Parameters
        ----------

        gnd_samples
            The ground elevations for a single cross section.
        veg_samples
            The vegrtation elevations for the same cross section.
        start_index
            The index to start the outward search from.
        z_water
            The elevation of the water.
        threshold
            The height above the water level to detect as a bank.
        stop_index
            The minimum index to search through
        """

        start_i = numpy.nan
        stop_i = numpy.nan

        for i in numpy.arange(start_index, stop_index - 1, -1):
            # Check if in channel
            start_i, stop_i = self.fixed_threshold_width_outwards(
                gnd_samples=gnd_samples,
                veg_samples=veg_samples,
                start_index=i,
                z_water=z_water,
                threshold=threshold,
            )

            # break if both edges detected
            if not numpy.isnan(start_i) and not numpy.isnan(stop_i):
                break
        return start_i, stop_i

    def _plot_results(
        self,
        cross_sections: geopandas.GeoDataFrame,
        threshold: float,
        aligned_channel: geopandas.GeoDataFrame = None,
        initial_spline: geopandas.GeoDataFrame = None,
        plot_cross_sections: bool = True,
    ):
        """Function used for debugging or interactively to visualised the
        samples and widths

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines with width
            estimates.
        threshold
            The bank detection threshold.
        aligned_channel
            The aligned channel generated from the cross_sections
        initial_spline
            Channel centre spline at the start of the current operation.
        plot_cross_sections
            Plot the cross_sections or not.
        """

        # Plot cross_sections, widths, and centrelines on the DEM
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
        self.gnd_dem.z.plot(ax=ax, label="DEM")
        if plot_cross_sections:
            cross_sections.plot(
                ax=ax, color="aqua", linewidth=1, label="cross_sections"
            )
        cross_sections[cross_sections["valid"]].set_geometry("width_line").plot(
            ax=ax, color="red", linewidth=1.5, label="Valid widths"
        )
        cross_sections[numpy.logical_not(cross_sections["valid"])].set_geometry(
            "width_line"
        ).plot(ax=ax, color="salmon", linewidth=1.5, label="Invalid widths")
        if aligned_channel is not None:
            aligned_channel.plot(
                ax=ax, linewidth=2, color="green", zorder=4, label="Aligned channel"
            )
        if initial_spline is not None:
            initial_spline.plot(
                ax=ax, linewidth=2, color="blue", zorder=3, label="REC smooth splne"
            )
        if "perturbed_midpoints" in cross_sections.columns:
            cross_sections.set_geometry("perturbed_midpoints").plot(
                ax=ax, color="aqua", zorder=5, markersize=5, label="Perturbed midpoints"
            )
        ax.set(title=f"Raster Layer with Vector Overlay. Thresh {threshold}")
        ax.axis("off")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()

        # Plot the various min_z values if they have been added to the cross_sections
        f, ax = matplotlib.pyplot.subplots(figsize=(40, 20))
        min_z_columns = [
            column_name
            for column_name in cross_sections.columns
            if "min_z" in column_name
        ]
        if len(min_z_columns) > 0:
            cross_sections[min_z_columns].plot(ax=ax)
        # Plot the widths
        f, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
        width_columns = [
            column_name
            for column_name in cross_sections.columns
            if "widths" in column_name
        ]
        if len(width_columns) > 0:
            cross_sections[width_columns].plot(ax=ax)
        # Plot the slopes
        f, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
        slope_columns = [
            column_name
            for column_name in cross_sections.columns
            if "slope" in column_name
        ]
        if len(slope_columns) > 0:
            cross_sections[slope_columns].plot(ax=ax)
        matplotlib.pyplot.ylim((0, None))

    def _create_flat_water_polygon(self, cross_sections: geopandas.GeoDataFrame):
        """Create a polygon of the flat water from spline's of each bank.

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines with width
            estimates.
        """

        # Only use the valid widths
        channel_mask = cross_sections["valid"]

        # Get the 'flat water' first bank - +1 to move just inwards
        bank_offset = self.resolution * (
            cross_sections.loc[channel_mask, "first_flat_bank_i"]
            + 1
            - self.centre_index
        )
        start_xy = numpy.vstack(
            [
                (
                    cross_sections.loc[channel_mask, "mid_x"]
                    + cross_sections.loc[channel_mask, "nx"] * bank_offset
                ).array,
                (
                    cross_sections.loc[channel_mask, "mid_y"]
                    + cross_sections.loc[channel_mask, "ny"] * bank_offset
                ).array,
            ]
        ).T
        start_xy = start_xy[numpy.logical_not(numpy.isnan(start_xy).any(axis=1))]
        start_xy = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.LineString(start_xy)], crs=cross_sections.crs
        )
        start_xy = Channel(start_xy, resolution=self.cross_section_spacing)
        start_xy_spline = start_xy.get_parametric_spline_fit_points()

        # Get the 'flat water' last bank - -1 to move just inwards
        bank_offset = self.resolution * (
            cross_sections.loc[channel_mask, "last_flat_bank_i"] - 1 - self.centre_index
        )
        stop_xy = numpy.vstack(
            [
                (
                    cross_sections.loc[channel_mask, "mid_x"]
                    + bank_offset * cross_sections.loc[channel_mask, "nx"]
                ).array,
                (
                    cross_sections.loc[channel_mask, "mid_y"]
                    + bank_offset * cross_sections.loc[channel_mask, "ny"]
                ).array,
            ]
        ).T
        stop_xy = stop_xy[numpy.logical_not(numpy.isnan(stop_xy).any(axis=1))]
        stop_xy = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.LineString(stop_xy)], crs=cross_sections.crs
        )
        stop_xy = Channel(stop_xy, resolution=self.cross_section_spacing)
        stop_xy_spline = stop_xy.get_parametric_spline_fit_points()

        flat_water_polygon = shapely.geometry.Polygon(
            numpy.concatenate((start_xy_spline, stop_xy_spline[:, ::-1]), axis=1).T
        )
        flat_water_polygon = geopandas.GeoDataFrame(
            geometry=[flat_water_polygon], crs=cross_sections.crs
        )
        return flat_water_polygon

    def _centreline_from_width_spline(
        self, cross_sections: geopandas.GeoDataFrame, smoothing_multiplier
    ):
        """Fit a spline through the width centres with a healthy dose of
        smoothing.

        Parameters
        ----------

        cross_sections
            The cross_sections with geometry defined as polylines with width
            estimates.
        smoothing_multiplier
            The smoothing multiplier to apply to the spline fit.
        """

        # Only use the valid widths
        channel_mask = cross_sections["valid"]

        # Calculate the offset distance between the transect and width centres
        widths_centre_offset = self.resolution * (
            (
                cross_sections.loc[channel_mask, "first_bank_i"]
                + cross_sections.loc[channel_mask, "last_bank_i"]
            )
            / 2
            - self.centre_index
        )

        # Calculate the location of the centre point between the banks
        widths_centre_xy = numpy.vstack(
            [
                (
                    cross_sections.loc[channel_mask, "mid_x"]
                    + widths_centre_offset * cross_sections.loc[channel_mask, "nx"]
                ).array,
                (
                    cross_sections.loc[channel_mask, "mid_y"]
                    + widths_centre_offset * cross_sections.loc[channel_mask, "ny"]
                ).array,
            ]
        ).T
        widths_centre_xy = widths_centre_xy[
            numpy.logical_not(numpy.isnan(widths_centre_xy).any(axis=1))
        ]

        # Fit a spline to the centre points
        widths_centre_line = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.LineString(widths_centre_xy)],
            crs=cross_sections.crs,
        )
        widths_centre_line = Channel(
            widths_centre_line, resolution=self.cross_section_spacing
        )

        aligned_spline = widths_centre_line.get_b_spline_fit(smoothing_multiplier)
        return aligned_spline

    def _unimodal_smoothing(self, y: numpy.ndarray):
        """Fit a monotonically increasing cublic spline to the data.

        Monotonically increasing cublic splines
        - https://stats.stackexchange.com/questions/467126/monotonic-splines-in-python
        - https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/epdf/10.1002/cem.935

        At end could fit non-monotonic fit. unconstrained_polynomial_fit = numpy.linalg
        .solve(E + la * D3.T @ D3, y)

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
        E = numpy.eye(len(x))
        D3 = numpy.diff(E, n=dd, axis=0)
        D1 = numpy.diff(E, n=1, axis=0)

        # Monotone smoothing
        ws = numpy.zeros(len(x) - 1)
        # Iterative process to improve the monotonic fit
        max_iterations = 30
        for it in range(max_iterations):
            Ws = numpy.diag(ws * kp)

            # Polynomial fit, monotonically constrained
            mon_cof = numpy.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, y)
            ws_new = (D1 @ mon_cof < 0.0) * 1

            # Break criteria for the monotonic fit - break if no change
            if numpy.sum(ws != ws_new) == 0:
                break
            ws = ws_new
        return mon_cof

    def _apply_bank_width(
        self,
        mid_x: float,
        mid_y: float,
        nx: float,
        ny: float,
        first_bank_i: int,
        last_bank_i: int,
    ):
        """Generate a line for each width for visualisation.

        Parameters
        ----------

        mid_x
            The x centre of the transect.
        mid_x
            The y centre of the transect.
        nx
            Transect normal x-component.
        ny
            Transect normal y-component.
        first_bank_i
            The index of the first bank along the transect.
        last_bank_i
            The index of the last bank along the transect.
        """
        return shapely.geometry.LineString(
            [
                [
                    mid_x + (first_bank_i - self.centre_index) * nx * self.resolution,
                    mid_y + (first_bank_i - self.centre_index) * ny * self.resolution,
                ],
                [
                    mid_x + (last_bank_i - self.centre_index) * nx * self.resolution,
                    mid_y + (last_bank_i - self.centre_index) * ny * self.resolution,
                ],
            ]
        )

    def _apply_midpoint(
        self,
        mid_x: float,
        mid_y: float,
        nx: float,
        ny: float,
        first_bank_i: int,
        last_bank_i: int,
    ):
        """Generate a line for each width for visualisation.

        Parameters
        ----------

        mid_x
            The x centre of the transect.
        mid_x
            The y centre of the transect.
        nx
            Transect normal x-component.
        ny
            Transect normal y-component.
        first_bank_i
            The index of the first bank along the transect.
        last_bank_i
            The index of the last bank along the transect.
        """
        mid_i = (first_bank_i + last_bank_i) / 2
        return shapely.geometry.Point(
            [
                mid_x + (mid_i - self.centre_index) * nx * self.resolution,
                mid_y + (mid_i - self.centre_index) * ny * self.resolution,
            ]
        )

    def align_channel(
        self,
        threshold: float,
        search_radius: float,
        min_channel_width: float,
        initial_channel: Channel,
        width_centre_smoothing_multiplier: float,
        cross_section_radius: float,
    ):
        """Estimate the channel centre from transect samples

        Parameters
        ----------

        threshold
            The height above the water level to detect as a bank.
        search_radius
            The distance to search side to side from the centre index.
        min_channel_width
            The minimum width of a 'valid' channel.
        initial_channel
            The initial channel centreline to align.
        width_centre_smoothing_multiplier
            The number of cross_sections to include in the downstream spline
            smoothing.
        cross_section_radius
            The radius (or 1/2 length) of the cross sections along which to
            sample.
        """

        assert (
            cross_section_radius >= search_radius
        ), "The transect radius must be >= the min_z_radius"
        self.transect_radius = cross_section_radius

        # Sample channel
        sampled_channel = initial_channel.get_parametric_spline_fit()

        # Create cross_sections
        cross_sections = self.node_centred_reach_cross_section(
            sampled_channel=sampled_channel
        )

        # Sample along cross_sections
        cross_section_elevations = self.sample_cross_sections(
            cross_sections=cross_sections, min_z_search_radius=search_radius
        )

        # Estimate water surface level and slope - Smooth slope upstream over 1km
        self._estimate_water_level_and_slope(cross_sections=cross_sections)

        # Bank estimates - outside in
        self.fixed_thresholded_widths_from_centre_within_radius(
            cross_sections=cross_sections,
            cross_section_elevations=cross_section_elevations,
            threshold=threshold,
            search_radius=search_radius,
            min_channel_width=min_channel_width,
            resolution=self.resolution,
        )

        # Separate out valid and invalid widths
        cross_sections["valid_widths"] = cross_sections["widths"]
        cross_sections.loc[
            numpy.logical_not(cross_sections["valid"]), "valid_widths"
        ] = numpy.nan

        # Create channel polygon with erosion and dilation to reduce sensitivity to poor
        # width measurements
        aligned_channel = self._centreline_from_width_spline(
            cross_sections=cross_sections,
            smoothing_multiplier=width_centre_smoothing_multiplier,
        )

        # Optoinal outputs
        if self.debug:
            # Add width linestring to the cross_sections
            cross_sections["width_line"] = cross_sections.apply(
                lambda x: self._apply_bank_width(
                    x["mid_x"],
                    x["mid_y"],
                    x["nx"],
                    x["ny"],
                    x["first_bank_i"],
                    x["last_bank_i"],
                ),
                axis=1,
            )
            # Plot results
            self._plot_results(
                cross_sections=cross_sections,
                threshold=threshold,
                plot_cross_sections=False,
                aligned_channel=aligned_channel,
                initial_spline=sampled_channel,
            )
        return aligned_channel, cross_sections

    def estimate_width_and_slope(
        self,
        aligned_channel: geopandas.GeoDataFrame,
        threshold: float,
        max_threshold: float,
        cross_section_radius: float,
        search_radius: float,
        min_channel_width: float,
    ):
        """Estimate the channel centre from transect samples

        Parameters
        ----------

        aligned_channel
            The channel centre line. Should be in the channel bed.
        threshold
            The height height above the water level to detect as a bank.
        max_threshold
            The maximum height above water level to detect as a bank (i.e. not
            a cliff)
        cross_section_radius
            The radius (or 1/2 length) of the cross sections along which to
            sample.
        search_radius
            The distance to search side to side from the centre index.
        min_channel_width
            The minimum width of a 'valid' channel.

        """

        assert (
            cross_section_radius >= search_radius
        ), "The transect radius must be >= the min_z_radius"
        assert (
            max_threshold > threshold
        ), "The max threshold must be greater than the threshold"
        self.transect_radius = cross_section_radius

        # Create cross_sections
        cross_sections = self.node_centred_reach_cross_section(
            sampled_channel=aligned_channel
        )

        # Sample along cross_sections
        cross_section_elevations = self.sample_cross_sections(
            cross_sections=cross_sections, min_z_search_radius=search_radius
        )

        # Estimate water surface level and slope
        self._estimate_water_level_and_slope(cross_sections=cross_sections)

        # Estimate widths
        self.variable_thresholded_widths_from_centre_within_radius(
            cross_sections=cross_sections,
            cross_section_elevations=cross_section_elevations,
            threshold=threshold,
            resolution=self.resolution,
            search_radius=min_channel_width,
            maximum_threshold=max_threshold,
            min_channel_width=min_channel_width,
        )

        # generate a flat water polygon
        river_polygon = self._create_flat_water_polygon(
            cross_sections=cross_sections,
        )

        # Midpoints of the river polygon - buffer slightly to ensure intersection at the
        # start and end
        cross_sections["river_polygon_midpoint"] = cross_sections.apply(
            lambda row: row.geometry.intersection(
                river_polygon.buffer(self.resolution / 10).iloc[0]
            ).centroid,
            axis=1,
        )

        # Width and threshod smoothing - rolling mean
        self._smooth_widths_and_thresholds(cross_sections=cross_sections)

        # Optional outputs
        if self.debug:
            # A line defining the extents of the bankfull width at that cross section
            cross_sections["width_line"] = cross_sections.apply(
                lambda row: self._apply_bank_width(
                    row["mid_x"],
                    row["mid_y"],
                    row["nx"],
                    row["ny"],
                    row["first_bank_i"],
                    row["last_bank_i"],
                ),
                axis=1,
            )

            # The 'flat water' midpoint
            cross_sections["flat_midpoint"] = cross_sections.apply(
                lambda row: self._apply_midpoint(
                    row["mid_x"],
                    row["mid_y"],
                    row["nx"],
                    row["ny"],
                    row["first_flat_bank_i"],
                    row["last_flat_bank_i"],
                ),
                axis=1,
            )
            # Plot results
            self._plot_results(
                cross_sections=cross_sections,
                threshold=threshold,
                aligned_channel=aligned_channel,
                plot_cross_sections=False,
            )
        # Return results
        return cross_sections, river_polygon
