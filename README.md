# GeoFrabrics

The GeoFrabrics and associated sub-packages include routines and classes for combining point (i.e. LiDAR), vector (i.e. catchment of interest, infrastructure) and raster (i.e. reference DEM) to generate a hydrologically conditioned raster.

## Instructions for use
The GeoFabricsGenerator class is the main entry point. This can either be accessed by importing the package then using the class directly, or through the main.py script in the root src directory. Pass in the argument 'instructions' with a file path to your JSON instructions file (see instructions.json for an exampled in the root directory of the repository). I do this in Spyder by setting up the run configuration for main.py. 

![image](https://user-images.githubusercontent.com/22883860/123566757-97a43a00-d814-11eb-9e3e-1d2468145e3d.png)

## Tests
Tests exist in the root tests directory.
* test1 - is still being created and will test the basic processing chain over a fabricated dataset.

## Related material
Scripts are included in separate repos. See https://github.com/rosepearson/Hydrologic-DEMs-scripts for an example.
