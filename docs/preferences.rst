.. _preferences:

Preferences menu
================

Some of the global preference can be accessed/changed using Preferences menu. To open it please use ``F11`` or ``View/Preferences`` in the main menu.   

Appearance
----------

Controls appearance of the general graphical elements in the Spectrum panel:

* **Spectrum view**: the representation of the spectrum. Can be ``steps``, ``lines``, ``points``. Each one can also include uncertainties ``+ errors``.

* **Fitting pixels view**: the representation of the pixels that is chosen for the analysis. Can be ``region``, ``points`` and ``colors``.

* **Line labels view**: the representation of the regular line labels (see :ref:`objects`). Can be ``short``, ``infinite`` (the same as fixed line labels).

* **show inactive exp**: show inactive exposures in the spectrum panel using gray-out representation.

* **f in line labels**: show oscillator strengths in all line labels.

Fit
---

Provide setting that control the fit profile calculation and appearance:

*  **Line profile type**: choose the type of the construction of line profiles:
    
    * **julia**: line profiles calculated using  ``Julia``. This type is only developed and maintained for the moment.

    * **py:regular**: regular python calculation (this include non-uniform wavelength grid for calculation the sub pixel profiles)

    * **py:uniform**: python calculation with uniform wavelength grid, where number in adjacent Line Edit panel set the number of sub-pixels within the spectral pixel. This is typically faster, than regular fit, but can be inaccurate if the number of subpixels is not enough.

* **tau limit**: set the characteristic lower limit until which the optical depth is calculated. Set the width od the absorption line.

* **accuracy**: control the number of spectral points to calculate the line. Roughly accuracy=0.1 meaning that at least each dy=0.1 there will be a spectral point, where line profiles is calculated. Usually, the value of 0.1 - 0.05 is a trade of between speed and accuracy.

* **Julia grid method**: set the grid method when **Line profile type** is **julia** used to calculate the grid of spectral points. This impact on the accuracy and speed (since the larger number of points increase the calculation time). Can be

    * **uniform**: uniform grid with <n> points, set in **number of fitting points** field. The most simple method.

    * **adaptive**: number of equidistant points with each spectral bin decided by the maximum of derivative of line profile within the bin and desired accuracy. The tradeoff between stability of the line profile (as a grid is quasi-uniform) and speed of calculation

    * **minimized**: grid related to spectral bins (i.e. bin boundaries + <n> points) and "minimal" set of points (unrelated to bin) for line given accuracy. May provide the fastest calculation time, including the minimized version when <n> set 0m and binned is unchecked.

* **number of fitting points**: - number of points within each bin (<n>), used to calulate the

* **binned** - if is checked, than calculate the binned spectrum (integrate profile within the bin) to compare with observed spectrum. If not than the binned boundaries do not included in **minimized** type of grid method.

* **Minimization method**: choose the minimization method for least-squares estimation.

* **tolerance**: the setting for minimization fitting, i.e. fit until tolerance is satisfied.

* **Fit components**: choose the representation of individual fit component (besides the total fit profile). This also access by ``C + SHIFT``. Can be: 

    * **all**: show all components.

    * **one**: show only current component (use ``C`` and ``C + LEFT/RIGHT ARROW`` to switch between components).

    * **non**: do not show components.

* **Fit view**: the representation of the calculated line profile:

    * **line**: continious line

    * **one**: for the total component it shows the points at which line profile is calculated (note that this may heavily loaded the GUI)

    * **bins**: shows the line profiles integrated within the bins

* **add telluric/accompanying absorption**: multiply fit by telluric absorption, if it is provided in the GUI

* **animate fit**: animate fit minimization, i.e. update the fit each step. **To be implement**

Colors
------

To manage the colors of the graphical objects. To be available in the future versions...