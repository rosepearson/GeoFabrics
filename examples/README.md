# GeoFabrics usage examples
This folder contains various real-word examples for using GeoFabrics to created Hydrologically conditioned DEMs. Please note that some examples contain multiple instruction files for producing DEMs with different levels of hydrological conditoining.

## Setup
There are several steps you will need to undertake **before** you can run these examples.
1. Install or clone GeoFabrics and ensure you have all dependencies installed.
2. Create a LINZ Data Service API Key with all manual permissions selected. Follow [link](https://github.com/niwa/geoapis/wiki/Package-Requirements) for detailed API key setup instructions.
3. Use this key value in place all 'INSERT_LINZ_KEY' in the example instruction files.
4. Review the example instruction files `processing` values (e.g. `chunk_size`, `number_of_cores`, and `memory_limit`). These were set for use on a supercomputer. You will need to ensure these values do not exceed your machines avalible hardware. See [link](https://github.com/rosepearson/GeoFabrics/wiki/Performance-and-benchmarking) for more information.

## Running GeoFabrics
Once you've run through the steps in the setup section, you can run GeoFabrics using the `JSON` files in each example folder as instruction file inputs. Follow [link](https://github.com/rosepearson/GeoFabrics/wiki/Basic-Usage-instructions) for detailed usage instructions.

Please note that the file paths in the instruction files are all setup for if the working directory at execution is the folder containing the instruction files. You may need to update the relative paths to absolute paths if running from another location.

## Examples
The following table contains information about the different tests.

| Name | Instruction Files | Description |
| --- | ----------- | ----------- |
| case_study_1 | `8m_unconditioned.json`, `8m_measured_river.json`, `8m_rupp_and_smart.json`, `8m_neal_et_al.json` | Includes four different instructions for producing a DEM over the lower Buller River (NZ) floodplain from LiDAR (where avalaible), raster (elsewhere on land) and ocean bathymetry contours. The instruction files range from no additional conditioning, to including riverbed values (either interpolated from surveyed cross sections, or estimated using the two avaliable approaches). |
| case_study_2 | `4m_river.json`, `8m_river_and_waterways.json` | Includes two different instructions for producing a DEM over the lower Waikanae River (NZ) floodplain from LiDAR, ocean bathymetry contours, and river centrelines for estimating riverbed elevations. One instruction file also includes waterway bed elevations. |

