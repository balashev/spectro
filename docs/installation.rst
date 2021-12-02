Installation
============

.. _installation:

This section describe how to download, install dependecies and config ``Spectro`` package for the first use  

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

* Python 3.7+

and::

* adjustText==0.7.3
* astroplan==0.8
* astropy==4.2.1
* astroquery==0.4.3
* ccdproc==2.1.1
* ChainConsumer==0.33.0
* corner==2.2.1
* emcee==3.0.2
* h5py==2.10.0
* julia==0.5.6
* lmfit==1.0.2
* lxml==4.6.3
* matplotlib==3.4.3
* mendeleev==0.9.0
* numba==0.53.1
* numdifftools==0.9.39
* numpy==1.20.1
* pyGPs==1.3.5
* PyQt5==5.15.6
* pyqtgraph==0.12.1
* pytz==2021.1
* scikit_learn==1.0.1
* scipy==1.6.2
* seaborn==0.11.1
* statsmodels==0.12.2
* ultranest==3.3.3

The exact versions of the packages listed here are not obligatory (it was generated from the working build), and the code can work with some old/new versions. However, there are sometimes inconsistencies with previous verisons of the packages, e.g. for ``matplotlib`` and ``chainconsumer``. For convinience, this package list is automatically stored in ``requirements.txt``, therefore you can simply use:: 

    $ pip install -r /path/to/requirements.txt    


Config
------

(This step can be skipped)

Before first use it can be useful to config some variable inside the code which provide the path to the external databases, e.g. IGMspec. The paths are kept in the file ``sviewer\config\config.ini``. In this file some dependencies and variables automatically store during the working progress, but you can also edit it by hands on your own risk.  

Julia
-----

Nevertheless that ``Spectro`` can work in the pure python installation, the performance of line profile fitting routines can be significantly enhanced by using ``Julia`` language. For this you need to install ``Julia`` (https://julialang.org/) and PyJulia package in Python. The  ``Spectro`` automatically check and used that Julia is installed. Once ``Julia`` is installed you can choose between ``Python`` and ``Julia`` in  ..Preferences:: menu to perform line profile fitting. 