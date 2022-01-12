.. _objects:

Objects in GUI
==============

There are several interactive objects in the sviewer that can be move by mouse interaction to alleviate and speed up the spectral analysis. This objects are shown in this Figure and their description is given below


.. image:: ./images/objects.png

B-spline continuum
------------------

The continuum that is constructed using B-spline with the point choosed by holding **b** key (see tutorial :ref:`constructing-continuum` and :ref:`keyboard`). It is marked as object 1 on the Figure.

Line labels
-----------

Line labels (objects 2, 3, 4, 5 in the figure) is a basic interactive graphical objects, which indicate a possible positions of the absorption/emission lines corresponding to atomic/molacular transitions. The vertical position of the line label is tight to the spectrum and the horizontal position can be changed either by moving line labels (using mouse and keyboard) or by setting appropriate redshift in the redshift field in the :ref:`control-panel`. There are useful interaction actions with line labels that ease the exploration of the spectrum:

* **SHIFT + DRAG**: move the line label by changing its redshift 

* **SHIFT + LEFT CLICK**: set line label as a reference line (see object 3), the the top axis will be scaled with velocity offset from this line.

* **CRTL + LEFT CLICK**: delete line label.

* **h + LEFT CLICK**: highlight all line labels of the species (including level) that correspond to this line, similar to writing in the :ref:`console`: ``high <species>``. Highlighed line label is marked by object 4.

* **ALT + LEFT CLICK**: show extended information about this line (see object 5).

Regions
-------

Regions (objects 6 and 7) are convinient graphical object to select certain wavelenght range of the spectrum. It can be used in several parts of the spectral analysis. The region can be constructed by holding **r** key and **LEFT MOUSE** button. There are two representation of the regions in extended (object 6) and minimized (object 7) form. The switch between them is performed by **DOUBLE LEFT CLICK**. The regions can be modified (shifted and extended) by mouse with holded **SHIFT** button. The regions can be delete by **LEFT MOUSE CLICK** with holding **CTRL**.

Composite spectrum
------------------

Composite spectrum (object 8) indicate a composite spectrum of QSO and can be shown/hide by pressing **CRTL + Q**. Currently, there are 3 types of the composite spectrum that is shown one after another. Composit spectrum can be draged by holding **SHIFT** button, during this its normalization and redshift is changed.

Doublet indicators
------------------

Doublet indicators (object 9) are separate line labels to mark the doublet absorption lines at the redshifts with distinct redshift as the main line labels (corresponding to the main redshift in the redshift panel). It can be created by *LEFT CLICK* with holded **d** key. One can highlight particular doublet label by **DOUBLE LEFT CLICK**. To shift or delete doublet label use **LEFT CLICK** with holded **SHIFT** or **CTRL** key, respecively.

Line flux residuals (zero levels)
---------------------------------

Line flux residual (LFR, object 10) is the indicator of zero level, that can be a part of the fit profile constuction, to model the partial coverage. It can be created by **LEFT CLICK** with holded **p** key. To shift or delete LFR use **LEFT CLICK** on the LFR text label with holded **SHIFT** or **CTRL** key, respecively.



