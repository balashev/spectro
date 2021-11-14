Installation
============

.. _installation:


Downloading
-----------

Since the ``Spectro`` package is still developing the most straightforward way is to install spectro from github, which provides you access to the latest features and bugfixes:

1. Clone ``Spectro`` from github::

    $ git clone https://github.com/balashev/spectro
    $ cd pyqtgraph

   Now you can install ``Spectro`` from the source::

    $ pip install .

2. Directly install from GitHub repo::

    $ pip install git+git://github.com/balashev/spectro

   You can change ``master`` of the above command to the branch name or the
   commit you prefer.

3. You can simply place the ``Spectro`` folder someplace importable, such as
   inside the root of another project. Spectro does not need to be "built" or
   compiled in any way.

Dependencies
------------
   
``Spectro`` is heavily depends on the various python packages. Before use you will need to install following packages::

* Python 3.7+
* A Qt library such as PyQt5, or PySide2
* pyqtgraph
* numpy
* scipy
* astropy

Julia
-----

Nevertheless that ``Spectro`` can work in the pure python installation, the performance of line profile fitting routines can be significantly enhanced by using ``Julia`` language. For this you need to install ``Julia`` (https://julialang.org/) and PyJulia package in Python. The  ``Spectro`` automatically check and used that Julia is installed. Once ``Julia`` is installed you can choose between ``Python`` and ``Julia`` in  ..Preferences:: menu to perform line profile fitting. 