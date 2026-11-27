from pathlib import Path

pytest_plugins = ("pytester",)


def test_cleanup_depends_on_class_outcomes(pytester):
    repo_root = Path(__file__).resolve().parent.parent
    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(repo_root)!r})\n"
        "pytest_plugins = ['tests.conftest']\n"
    )
    pytester.makepyfile("""
        import json
        from pathlib import Path
        import pytest
        from tests.base_test import Test as Base

        class Case(Base):
            @classmethod
            def setUpClass(cls):
                cls.test_path = Path(__file__).parent / cls.__name__
                (cls.test_path / "data").mkdir(parents=True, exist_ok=True)
                (cls.test_path / "instruction.json").write_text(json.dumps({}))
                super().setUpClass()
                (cls.results_dir / "partial.txt").write_text("result")

        class TestPassed(Case):
            def test_passes(self):
                pass

            @pytest.mark.skip(reason="not applicable")
            def test_skipped(self):
                pass

        class TestFailed(Case):
            def test_fails(self):
                self.fail("expected failure")

        class TestSetupFailed(Case):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                raise RuntimeError("expected setup error")

            def test_example(self):
                pass

        class TestTeardownFailed(Case):
            @classmethod
            def tearDownClass(cls):
                raise RuntimeError("expected teardown error")

            def test_example(self):
                pass

        class TestOverride(Case):
            @classmethod
            def clean_data_folder(cls):
                super().clean_data_folder()
                (cls.test_path / "cleanup.txt").write_text("called")

            def test_example(self):
                (self.test_path / "cleanup.txt").unlink()
        """)

    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=3, failed=1, skipped=1, errors=2)

    assert not (pytester.path / "TestPassed" / "data" / "results").exists()
    assert not (pytester.path / "TestOverride" / "data" / "results").exists()
    assert (pytester.path / "TestOverride" / "cleanup.txt").exists()
    for name in ("TestFailed", "TestSetupFailed", "TestTeardownFailed"):
        assert (pytester.path / name / "data" / "results" / "partial.txt").exists()
