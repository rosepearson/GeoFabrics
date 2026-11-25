# -*- coding: utf-8 -*-
"""
pytest configuration file for GeoFabrics tests.

IMPORTANT: Import pdal FIRST on Windows to avoid DLL loading conflicts
with GDAL-dependent libraries. This must run before any test modules are loaded.
See: https://github.com/PDAL/PDAL/issues/4408
"""

import pdal  # noqa: F401

import gc
from collections import defaultdict

import pytest

from tests import base_test


_reports_key = pytest.StashKey()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    reports = item.stash.setdefault(_reports_key, {})
    reports[report.when] = report


def pytest_sessionfinish(session, exitstatus):
    classes = defaultdict(list)
    for item in session.items:
        cls = item.cls
        if cls is not None and issubclass(cls, base_test.Test):
            classes[cls].append(item.stash.get(_reports_key, {}))

    for cls, reports_by_item in classes.items():
        if (
            all("teardown" in reports for reports in reports_by_item)
            and all(
                not report.failed
                for reports in reports_by_item
                for report in reports.values()
            )
            and any(
                reports.get("call") is not None and reports["call"].passed
                for reports in reports_by_item
            )
        ):
            gc.collect()
            cls.clean_data_folder()
