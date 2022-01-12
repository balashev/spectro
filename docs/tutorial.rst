.. _tutorial:

Tutorial
========

Here we provide a brief tutorial to start fitting with the ``sviewer`` GUI.

Loading the spectrum
--------------------

The spectrum can be loaded in GUI by several ways:

* **Drag and drop** method: You can load several files in one drop.

* **File/Import spectrum...** in Main menu

* **File/Open...** in Main menu: In that case it load the spectra specified in .spv file, whih contains saved progress of data analysis.

* **File/Import list...** in Main menu: import a list of spectra from the file with list contains of path to spectra.

* **File/Import folder...** in Main menu: import all spectra from the specified folder. 

The basic ``sviewer`` spectral format is plain ``ASCII`` file, which stores spectrum in column-like representation with the following order: **<wavelength> <flux> <uncertainty>**. The spectrum do not necessary need to have <uncertainty> column, as well as can read the <continuum> provided in the additional forth column. The header of ascii file should be commented out by **#**.

Additionally there is possibility to load the spectrum in ``FITS`` format. The program can automatically recognize several FITS format produced by standard reduction routines, such as UVES popler, SDSS, etc. 

.. _constructing-continuum:
Constructing continuum
----------------------
One of the first part of the absorption line analysis is constructing of the unabsorbed continuum. In most partical cases, the continuum is not well defined a priori, and can be considered as some fixed normalization of the spectrum  (fit profile) or as a nuissance parameter. There are several way to construct continuum:

* B-spline:  continuum is constructed using B-spline interpolation between data points that can be created using mouse interaction:
    * **b + LEFT CLICK**: add point for B-spline at the cursor position.
    * **b + RIGHT CLICK**: remove the closest point of B-spline at the cursor position.
    * **b + MOUSE REGION**: remove all points of B-spline within selected region.

    B-spline is automatically recalculated at each change in the data points array.
 
* Iterative Smoothing: for quick constion of smooth continuum press **q**. For more options see **Continuum window**

* Savitsky-Golay interpolation: avaliable in **Continuum window**

* Chebyshev polinomial fit: avaliable in **Continuum window**

Fine tunning options and actions to construct and modify the continuum is avaliable by **1d spec/Continuum...** in Main Menu or pressing **CTRL + c**.

After continuum is constructed one can **normalize/denormalize** the representation of the spectrum by pressing Normalize button in :ref:`control-panel` or using **n** key.

Making fit model
----------------

Fitting
-------

Viewing results
---------------