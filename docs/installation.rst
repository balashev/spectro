Installation
============

.. _installation:

This section describe how to download, install dependencies and config ``Spectro`` package for the first use

Downloading
-----------

Since the ``Spectro`` package is still developing the most straightforward way is to install spectro from github, which provides you access to the latest features and bugfixes:

1. Clone ``Spectro`` from github::

    $ git clone https://github.com/balashev/spectro
 
   
2. You can simply place the ``Spectro`` folder someplace importable, such as
   inside the root of another project. Spectro does not need to be "built" or
   compiled in any way.

Dependencies
------------
   
``Spectro`` heavily depends on the list of python packages, that you will need to install::

* Python 3.12+ (the current stable version works on 3.12.7, and it seems that python 3.14.x has conflicted packages)

and::

* adjustText==1.3.0
* astroplan==0.10.1
* astropy==7.0.1
* astroquery==0.4.9.post1
* ccdproc==2.4.3
* chainconsumer==1.1.3
* corner==2.2.3
* dust_extinction==1.5
* dynesty==2.1.5
* emcee==3.1.6
* h5py==3.13.0
* julia==0.6.2
* juliacall==0.9.24
* lmfit==1.3.2
* matplotlib==3.10.1
* mendeleev==0.20.1
* numba==0.61.0
* numpy==2.2.3
* pandas==2.2.3
* pipreqs==0.4.13
* pyGPs==1.3.5
* PyQt6==6.8.1
* PyQt6_sip==13.10.0
* pyqtgraph==0.13.7
* pytz==2024.1
* scikit_learn==1.6.1
* scipy==1.15.2
* spectro==0.2.0
* statsmodels==0.14.4
* ultranest==4.4.0

For convenience, this package list is automatically stored in ``requirements.txt``, therefore you can simply use::

    $ pip install -r /path/to/requirements.txt

The exact versions of the packages listed here are not obligatory (it was generated from the working build), and the code can work with some old/new versions. However, there are sometimes inconsistencies with previous verisons of the packages, e.g. for ``matplotlib`` and ``chainconsumer``. Therefore we highly recommended to install and use the code within virtual environment (either venv or conda), e.g. run following before package installation:

    $ conda create spectro
    $ conda activate spectro

Config
------

(This step can be skipped)

Before first use it can be useful to config some variable inside the code which provide the path to the external databases, e.g. IGMspec. The paths are kept in the file ``sviewer\config\config.ini``. In this file some dependencies and variables automatically store during the working progress, but you can also edit it by hands on your own risk.  

Julia
-----

Nevertheless that ``Spectro`` can work in the pure python installation, the performance of line profile fitting routines can be significantly enhanced by using ``Julia`` language. For this you need to install ``Julia`` (https://julialang.org/) and ``juliacall`` package in Python::

* Julia 1.11+

The  ``Spectro`` automatically check that Julia is installed and used it. Once ``Julia`` is installed you can choose between ``Python`` and ``Julia`` to perform line profile fitting using settings avaliable in  :ref:`preferences` menu (``F11`` or ``View/Preferences`` in the main menu). To run Julia you will also need to install the following packages in Julia (for the plain version of the list check requirements_julia.txt file):

::

* DataStructures
* Interpolations
* ImageFiltering
* LeastSquaresOptim
* LinearAlgebra
* LsqFit
* PeriodicTable
* Polynomials
* PythonCall
* Roots
* SpecialFunctions
* AdvancedHMC
* ClusterManagers
* Combinatorics
* DelimitedFiles
* Distributed
* FileIO
* ForwardDiff
* JLD2
* Measures
* Plots
* Random
* Serialization
* Statistics
* VectorizedStatistics
