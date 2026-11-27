from unittest.mock import patch, sentinel

import dask
import pytest

from geofabrics import processor


@pytest.mark.parametrize(
    ("configured", "argv", "expected", "expected_argv"),
    [
        ([], [], ["pdal"], None),
        (["custom"], [], ["pdal", "custom"], None),
        (
            ["custom", "pdal"],
            [["--custom"], []],
            ["pdal", "custom"],
            [[], ["--custom"]],
        ),
        (["custom"], [["--custom"]], ["pdal", "custom"], [[], ["--custom"]]),
        (["pdal"], [], ["pdal"], None),
    ],
)
def test_windows_preloads_pdal_first(configured, argv, expected, expected_argv):
    settings = {
        "distributed.worker.preload": configured,
        "distributed.worker.preload-argv": argv,
    }
    with (
        dask.config.set(settings),
        patch.object(processor.sys, "platform", "win32"),
        patch.object(
            processor.distributed, "LocalCluster", return_value=sentinel.cluster
        ) as cluster,
    ):
        result = processor._create_local_cluster(n_workers=2, processes=True)

    expected_kwargs = {"n_workers": 2, "processes": True, "preload": expected}
    if expected_argv is not None:
        expected_kwargs["preload_argv"] = expected_argv
    cluster.assert_called_once_with(**expected_kwargs)
    assert result is sentinel.cluster


def test_other_platforms_leave_preload_configuration_to_dask():
    with (
        patch.object(processor.sys, "platform", "linux"),
        patch.object(processor.distributed, "LocalCluster") as cluster,
    ):
        processor._create_local_cluster(n_workers=2, processes=True)

    cluster.assert_called_once_with(n_workers=2, processes=True)


def test_mismatched_preload_arguments_fail_explicitly():
    settings = {
        "distributed.worker.preload": ["first", "second"],
        "distributed.worker.preload-argv": [["--first"]],
    }
    with (
        dask.config.set(settings),
        patch.object(processor.sys, "platform", "win32"),
        patch.object(processor.distributed, "LocalCluster") as cluster,
    ):
        with pytest.raises(ValueError, match="preload and preload-argv"):
            processor._create_local_cluster()
    cluster.assert_not_called()
