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

* **B-spline**:  continuum is constructed using B-spline interpolation between data points that can be created using mouse interaction:
    * **b + LEFT CLICK**: add point for B-spline at the cursor position.
    * **b + RIGHT CLICK**: remove the closest point of B-spline at the cursor position.
    * **b + MOUSE REGION**: remove all points of B-spline within selected region.

    B-spline is automatically recalculated at each change in the data points array.
 
* **Iterative Smoothing**: for quick constion of smooth continuum press **q**. For more options see **Continuum widget**

* **Savitsky-Golay interpolation**: avaliable in **Continuum widget**

* **Chebyshev polinomials**: avaliable in **Continuum widget**

Fine tunning options and actions to construct and modify the continuum is avaliable in **Continuum widget** that can be opened iether **1d spec/Continuum...** in Main Menu or pressing **CTRL + c**.

After continuum is constructed one can **normalize/denormalize** the representation of the spectrum by pressing Normalize button in :ref:`control-panel` or using **n** key.

.. _select-fit-regions:
Selecting fit regions
---------------------

To perform the fit it is necessary to select the spectrum pixels in which the spectrum will be compared with the fit model. This can be done by:

* **s + MOUSE REGION**: select points within region drawn by mouse using Left button
* **s + SHIFT + MOUSE REGION**: select points to all exposures within selected region
* **d + MOUSE REGION**: deselect points within region drawn by mouse using Left button
* **d + SHIFT + MOUSE REGION**: deselect points to all exposures within selected region

The representation of selected pixels (regions, points, lines) can be changed in **Preferences widget** (**View/Preferences...** in Main menu or by pressing **F11**)

.. _making-fit-model:
Making fit model
----------------

The fit model should be defined in the **Fit model** widget, which can be opened iether by **Fit/Fit model...** in Main Menu or pressing **CTRL + f**. 

After setting/modification of fit model one can update (if it was not update automatically) the fit by pressing 

* **f**: this will construct the model profiels in the selected regions only. 

* **SHIFT + f**: this will construct the model profiles for all avaliable line in the spectrum.

.. _fitting:
Fitting
-------

There are two avaliable fit routines:

* Minimizing likelihood using Levenberg-Marquard method. This fit is performed by the pressing **Fit** button in :ref:`control-panel`

* Bayessian approach by Monte Carlo Markov Chain technique with a set of Samplers.


.. _viewing-results:
Viewing results
---------------