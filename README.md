# GeoFrabrics

The GeoFrabrics and associated sub-packages include routines and classes for combining point (i.e. LiDAR), vector (i.e. catchment of interest, infrastructure) and raster (i.e. reference DEM) to generate a hydrologically conditioned raster.

## Basic instructions for use
The GeoFabricsGenerator class is the main entry point. This can either be accessed by importing the package then using the class directly, or through the main.py script in the root src directory. 

### importing GeoFabrics
Basic code stub looks like:
`
import GeoFabrics
import json
with open(r'path\to\file\', 'r') as file_pointer:
            instructions = json.load(file_pointer)
runner = processor.GeoFabricsGenerator(instructions)
runner.run()
`
### main.py script
In the conda environment defined in the root\environment.yml, run the following:

`python src\main.py --instructions full\path\to\instruction.json`

## Running tests
In the conda environment defined in the root\environment.yml, run the following in the repository root folder:

1. to run only test1
`python -m tests.test1.test1`

2. to run all tests
`python -m unittest`

### Instructions for use in spyder
The GeoFabricsGenerator class is the main entry point. This can either be accessed by importing the package then using the class directly, or through the main.py script in the root src directory. Pass in the argument 'instructions' with a file path to your JSON instructions file (see instructions.json for an exampled in the root directory of the repository). I do this in Spyder by setting up the run configuration for main.py. 

![image](https://user-images.githubusercontent.com/22883860/123566757-97a43a00-d814-11eb-9e3e-1d2468145e3d.png)

## Related material
Scripts are included in separate repos. See https://github.com/rosepearson/Hydrologic-DEMs-scripts for an example.
