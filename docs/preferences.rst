.. _preferences:

Preferences menu
================

Some of the global preference can be accessed/changed using Preferences menu. To open it please use ``F11`` or ``View/Preferences`` in the main menu.   

Appearance
----------

Controls appearrance of the general graphical elements in the Spectrum panel:

* **Spectrum view**: the representation of the spectrum. Can be ``steps``, ``lines``, ``points``. Each one can also include uncertainties ``+ errors``.

* **Fitting pixels view**: the representation of the pixels that is chosen for the analysis. Can be ``region``, ``points`` and ``colors``.

* **Line labels view**: the representation of the regular line labels (see :ref:`objects`). Can be ``short``, ``infinite`` (the same as fixed line labels).

* **show inactive exp**: show inactive exposures in the spectrum panel using gray-out representation.

* **f in line labels**: show oscillator strengths in all line labels.

Fit
---

Provide setting that control the fit plofile calculation and appearence:

*  **Line profile type**: choose the type of the construction of line profiles between:
    
    * **regular**: regular python calculation (this include non-uniform wavelength grid for calculation the sub pixel profiles)

    * **fast**: python calculation with uniform wavelenght grid, where number in adjacent Line Edit panel set the number of sub-pixels within the spectral pixel. This is typically faster, than regular fit, but can be inaccurate if the number of sub=pixels is not enough. 

    * **julia**: line profiles calculated using  ``Julia``.

    * **fft**: line profiles calculated using Fast Fourier Transform (for fast convolution with instrumental function). **Be careful, it is not fully tested yet!**

* **tau limit**: set the characteristic limit until which the optical depth is calculated.

* **Fit method**: choose the minimization method for least-squares estimation. This setting pass directly to ``minimize`` routine in ``optimize`` package within ``lmfit`` package. Works only for fit calculation using python.

* **Fit components**: choose the representation of individual fit component (besides the total fit profile). This also access by ``C + SHIFT``. Can be: 

    * **all**: show all components.

    * **one**: show only current component (use ``C`` and ``C + LEFT/RIGHT ARROW`` to switch between components).

    * **non**: do not show components.

* **show fit points**: show explicitly the points in which fit profile is calculated.   

* **animate fit**: animate fit minimization, i.e. upadate the fit each step. **Does not properly work yet!**   

Colors
------

To manage the colors of the graphical objects. To be available in the future versions...