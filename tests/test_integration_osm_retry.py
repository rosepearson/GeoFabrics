from pathlib import Path

pytest_plugins = ("pytester",)


def test_integration_reruns_only_exhausted_osm_downloads(pytester):
    repo_root = Path(__file__).resolve().parent.parent
    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(repo_root)!r})\n"
        "pytest_plugins = ['tests.conftest']\n"
    )
    pytester.makepyfile("""
        import json
        from pathlib import Path
        import urllib.error
        from tests.base_test import Test as Base

        # This subprocess exercises synthetic failures, so skip its retry waits.
        for mark in Base.pytestmark:
            if mark.name == "flaky":
                mark.kwargs["reruns_delay"] = 0

        class Case(Base):
            @classmethod
            def setUpClass(cls):
                cls.test_path = Path(__file__).parent / cls.__name__
                (cls.test_path / "data").mkdir(parents=True, exist_ok=True)
                (cls.test_path / "instruction.json").write_text(json.dumps({}))
                super().setUpClass()
                counter = cls.test_path / "attempts.txt"
                attempt = int(counter.read_text()) + 1 if counter.exists() else 1
                counter.write_text(str(attempt))
                (cls.results_dir / "partial.txt").write_text("result")
                if cls.__name__ == "TestPersistent" or (
                    cls.__name__ == "TestTransient" and attempt == 1
                ):
                    raise ConnectionError(
                        "Did not successfully download the OSM data in 3 attempts."
                    ) from urllib.error.URLError("temporary outage")
                if cls.__name__ == "TestPermanent":
                    raise ValueError("invalid OSM query")
                if cls.__name__ == "TestUnrelatedConnection":
                    raise ConnectionError("unrelated download failure")

        class TestTransient(Case):
            def test_example(self):
                pass

        class TestPersistent(Case):
            def test_example(self):
                pass

        class TestPermanent(Case):
            def test_example(self):
                pass

        class TestUnrelatedConnection(Case):
            def test_example(self):
                pass

        class TestAssertion(Case):
            def test_example(self):
                self.fail(
                    "Did not successfully download the OSM data in 3 attempts."
                )

        class TestSuccess(Case):
            def test_example(self):
                pass
        """)

    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=2, failed=1, errors=3)

    for name in ("TestTransient", "TestPersistent"):
        assert (pytester.path / name / "attempts.txt").read_text() == "2"
    for name in (
        "TestPermanent",
        "TestUnrelatedConnection",
        "TestAssertion",
        "TestSuccess",
    ):
        assert (pytester.path / name / "attempts.txt").read_text() == "1"
    for name in ("TestTransient", "TestSuccess"):
        assert not (pytester.path / name / "data" / "results").exists()
    for name in (
        "TestPersistent",
        "TestPermanent",
        "TestUnrelatedConnection",
        "TestAssertion",
    ):
        assert (pytester.path / name / "data" / "results" / "partial.txt").exists()
