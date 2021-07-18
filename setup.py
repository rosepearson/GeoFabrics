# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:22:08 2021

@author: pearsonra
"""
import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='GeoFabrics',
    author='Rose pearson',
    author_email='rose.pearson@niwa.co.nz',
    description='A package for creating hydrologically conditioned geo-fabrics (i.e. DEMs and roughness maps)',
    keywords='GeoFabrics, DEM, Roughness, Hydrologically conditioned',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rosepearson/GeoFabrics',
    project_urls={
        'Documentation': 'https://github.com/rosepearson/GeoFabrics',
        'Bug Reports':
        'https://github.com/rosepearson/GeoFabrics/issues',
        'Source Code': 'https://github.com/rosepearson/GeoFabrics',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',

        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['pyproj==3.0.1',
                      'numpy==1.21.0',
                      'rioxarray==0.4.2',
                      'pdal==2.2.0',
                      'python-pdal==2.4.0',
                      'geopandas==0.9.0',
                      'shapely==1.7.1',
                      'scipy==1.6.3',
                      'requests==2.25.1',
                      'boto3==1.17.105',
                      'matplotlib==3.4.2'],
    extras_require={
        'dev': ['check-manifest'],
    },

)
