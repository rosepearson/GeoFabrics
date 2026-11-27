import logging

import numpy
import pytest
from scipy.interpolate import RBFInterpolator

from geofabrics import dem


@pytest.fixture
def point_cloud():
    points = numpy.empty(4, dtype=[("X", float), ("Y", float), ("Z", float)])
    points["X"] = [0, 2, 0, 2]
    points["Y"] = [0, 0, 2, 2]
    points["Z"] = 2 * points["X"] + 3 * points["Y"] + 5
    return points


def test_calculate_rbf_returns_scalar_without_changing_value(point_cloud):
    xy = numpy.column_stack((point_cloud["X"], point_cloud["Y"]))
    query = numpy.array([0.5, 0.5])
    expected = RBFInterpolator(
        xy,
        point_cloud["Z"],
        kernel="linear",
        smoothing=0,
        neighbors=dem.RBF_CACHE_SIZE,
    )([query]).item()

    result = dem.calculate_rbf(xy, point_cloud["Z"], query, kernel="linear")

    assert numpy.isscalar(result)
    assert result == expected


def test_radius_rbf_elevations_can_be_assigned_to_raster(point_cloud):
    queries = numpy.array([[0.5, 0.5], [1.5, 0.5]])
    options = {
        "method": "rbf",
        "kernel": "thin_plate_spline",
        "radius": 10,
        "raster_type": numpy.float32,
    }

    result = dem.elevation_from_points(point_cloud, queries, options)

    numpy.testing.assert_allclose(result, [7.5, 9.5])


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mean", [10, 10, numpy.nan]),
        ("linear", [7.5, 9.5, numpy.nan]),
        ("rbf", [7.5, 9.5, numpy.nan]),
    ],
)
def test_interpolation_does_not_reconfigure_logging(
    point_cloud, monkeypatch, caplog, method, expected
):
    queries = numpy.array([[0.5, 0.5], [1.5, 0.5], [50, 50]])
    options = {
        "method": method,
        "strict": True,
        "kernel": "thin_plate_spline",
        "radius": 10,
        "raster_type": numpy.float32,
    }

    def unexpected_logging_configuration(*args, **kwargs):
        pytest.fail("Interpolation must not look up loggers or change their levels")

    with caplog.at_level(logging.WARNING, logger=dem.__name__):
        with monkeypatch.context() as patch:
            patch.setattr(logging, "getLogger", unexpected_logging_configuration)
            patch.setattr(logging.Logger, "setLevel", unexpected_logging_configuration)
            result = dem.elevation_from_points(point_cloud, queries, options)

    numpy.testing.assert_array_equal(result, numpy.array(expected, dtype=numpy.float32))


@pytest.mark.parametrize("use_edge", [False, True])
def test_nearest_rbf_elevations_can_be_assigned_to_raster(point_cloud, use_edge):
    edge_points = point_cloud.copy()
    edge_points["X"] += 4
    edge_points["Y"] += 4
    edge_points["Z"] = 2 * edge_points["X"] + 3 * edge_points["Y"] + 5
    queries = numpy.array([[0.5, 0.5], [1.5, 0.5]])
    options = {
        "method": "rbf",
        "kernel": "thin_plate_spline",
        "k_nearest_neighbours": 4,
        "use_edge": use_edge,
        "raster_type": numpy.float32,
    }

    result = dem.elevation_from_nearest_points(
        point_cloud, edge_points, queries, options
    )

    numpy.testing.assert_allclose(result, [7.5, 9.5])
