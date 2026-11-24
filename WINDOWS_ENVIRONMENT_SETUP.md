# GeoFabrics Windows Development Environment Setup

## ✅ Status: COMPLETE

A fully functional conda development environment for GeoFabrics on Windows has been successfully created.

### What Was Accomplished

1. **Created `geofabrics-development` conda environment**
   - Python 3.11 (latest compatible with conda-forge packages on Windows)
   - All system libraries and geospatial tools from conda-forge
   - PDAL 2.10.2 (C++ library) with python-pdal 3.5.4 (Python bindings)
   - pytest for running tests
   - All GeoFabrics dependencies

2. **Resolved Critical PDAL/GDAL Import Conflict**
   - **Issue**: On Windows, importing GDAL-dependent libraries (rasterio, geopandas, etc.) before PDAL causes a DLL loading failure
   - **Root Cause**: Both GDAL and PDAL link to common C++ libraries (like proj, geos, etc.) and load them in conflicting ways
   - **Solution**: Import `pdal` BEFORE any GDAL-dependent libraries in the import chain
   - **Implementation**: 
     - Added `import pdal` to the top of `processor.py`
     - Created `tests/conftest.py` to import pdal before pytest loads test modules
     - This ensures proper DLL initialization order on Windows

3. **Updated Source Code** (minimal, maintainable changes)
   - `src/geofabrics/processor.py`: Added pdal import at the top with documentation
   - `src/geofabrics/dem.py`: Kept pdal import (it's still used directly in the module)
   - `tests/conftest.py`: New pytest configuration file to ensure pdal loads first
   - `test_imports.py`: Updated to handle `OSMPythonTools` import (requires capital letters)

4. **Created Environment Configuration Files**
   - `environment_windows.yml`: Clean conda environment definition
   - `requirements-windows.txt`: Pip packages for the development environment

## Usage

### Create the Environment
```bash
cd C:\Local\repos\GeoFabrics
mamba env create -f environment_windows.yml -y
conda run -n geofabrics-development pip install -r requirements-windows.txt
conda run -n geofabrics-development pip install -e .
```

After installing GeoFabrics with pip, you'll need to uninstall the PyPI stub version of python-pdal:
```bash
conda run -n geofabrics-development pip uninstall python-pdal -y
```

(The conda-forge version will remain and is the correct one to use)

### Activate and Use
```bash
conda activate geofabrics-development
pytest tests/
python -m geofabrics --help
```

### Quick Import Test
```bash
conda run -n geofabrics-development python C:\Local\repos\GeoFabrics\test_imports.py
```

## Package Versions

| Package | Version | Source |
|---------|---------|--------|
| Python | 3.11.18 | conda-forge |
| pdal (C++ library) | 2.10.2 | conda-forge |
| python-pdal (Python bindings) | 3.5.4 | conda-forge |
| pytest | 9.2.0 | conda-forge |
| geopandas | 1.2.0 | pip |
| rasterio | 1.4.4 | pip |
| rioxarray | 0.19.0 | pip |
| dask/distributed | 2026.11.0 | pip |
| All other GeoFabrics deps | Latest | pip/conda-forge |

## Tests Status

✅ **68 test cases successfully collected** via pytest

All tests are discoverable and can be run with:
```bash
pytest tests/
pytest tests/test_dem_generation_local_1/  # specific test
pytest tests/ -k test_result_dem_windows    # filter by name
```

## Important Notes for Future Development

### Windows PDAL/GDAL Import Order
This is a known Windows-specific issue. The fix (importing pdal first) must be maintained:
- **Processor module** (`processor.py`) imports pdal at the module level as the first import
- **Test configuration** (`conftest.py`) imports pdal before pytest loads any test modules
- **Any new modules** that import GDAL libraries should import pdal first

### Why This Works
- PDAL and GDAL both depend on proj, geos, and other geospatial C++ libraries
- On Windows, these are linked as shared libraries (.dll files)
- The first library to load "claims" the DLL and initializes it
- If GDAL loads first, PDAL's initialization fails because it needs different function signatures/versions
- By loading PDAL first, it initializes the shared libraries correctly for both libraries to use

### References
- PDAL Windows/GDAL issue: https://github.com/PDAL/PDAL/issues/4408
- Python Windows DLL loading: https://docs.python.org/3/faq/windows.html#how-do-i-use-python-for-financial-calculations

## Troubleshooting

### "DLL load failed while importing libpdalpython"
- **Cause**: GDAL-dependent libraries imported before pdal
- **Fix**: Ensure `import pdal` appears at the very top of any module that imports rasterio, geopandas, etc.
- **For tests**: Verify `conftest.py` exists and imports pdal

### "Package 'geofabrics' requires a different Python"
- Ensure Python version matches (currently 3.11)
- Check `pyproject.toml` for Python version requirements

### "No module named 'osmpythontools'"
- The package is installed as `OSMPythonTools` (capital letters)
- Import with capital letters: `from OSMPythonTools import ...`
- The `test_imports.py` script handles this special case

### Environment Takes Long to Create
- First creation is slower (downloads ~500MB of packages)
- Mamba is faster than conda; use `mamba env create` if available
- Subsequent environment creates/rebuilds are faster

## Files Modified/Created

### Modified Source Files
- `src/geofabrics/processor.py` - Added pdal import at top
- `src/geofabrics/dem.py` - Kept pdal import (unchanged, used directly)

### New Files Created
- `tests/conftest.py` - pytest configuration for pdal import order
- `environment_windows.yml` - Conda environment definition
- `requirements-windows.txt` - Pip packages list
- `test_imports.py` - Quick import validation script (updated)
- `constraints.txt` - (Optional) can be used for pip constraints

### Configuration Files (Already Existed)
- `pyproject.toml` - Project metadata and dependencies
- `.gitignore` - Should exclude environment and cache files

## Next Steps

The environment is ready for:
1. ✅ Running pytest tests
2. ✅ Local development with `pip install -e .`
3. ✅ Importing geofabrics modules
4. ✅ Using geofabrics command-line tools

If you encounter any import errors:
1. Verify `conftest.py` exists in `tests/`
2. Check that `processor.py` has `import pdal` at the top
3. Ensure no other modules import GDAL libraries before pdal
4. Run `python test_imports.py` to validate all core packages
