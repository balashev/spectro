.. _keyboard:

Keyboard
========

The keyboard provide the easy access to the GUI instruments. Most of the useful commands can be assessed form the keyboard.

Keyboard bindings
----------------

* **F1**:  show help/Howto.
* **F2**:  show/hide list of exposures widget.
* **F3**:  show/hide panel to construct fit model
* **F4**:  show/hide choose fitted parameters panel.
* **F5**:  show/hide MCMC widget.
* **F6**:  show/hide fit results widget.
* **F7**:  show/hide plot widget.
* **F8**:  show/hide residuals panel.
* **F9**:  show/hide 2d spectrum panel.
* **F11**:  show/hide Preferences menu.
* **F12**:  switch between fullscreen/maximized modes.

* **a**: add component:

   * **a + LEFT CLICK**                 :  add component (same as the current) to the model fit at the position
   * **a + CTRL**                       :  remove the current component from the model fit 

* **b**: b-splain:
   
   * **b**                              :  add b-splain point (if key is quickly pressed)
   * **b + LEFT CLICK**               :  add b-splain point
   * **b + RIGHT CLICK**              :  remove nearest b-splain point
   * **b + <select region>**            :  remove all b-splain points in selected region
  
* **c**: component mode:
   
   * **c** or **c + RIGHT ARROW**      :  show/select next component
   * **c + LEFT ARROW**                :  show/select previous component
   * **c + CTRL**                       :  continuum fit options window
   * **c + SHIFT**                      :  change show component mode (none, one, all)
   * **c + LEFT CLICK**                :  shift component center at the position (line indicator have to be selected)
   * **c + MOUSE WHEEL**               :  increase/decrease b parameter of the component
   * **c + <mouse drag>**               :  increase/decrease column density of the component 

* **d**: deselect data points:
   
   * **d + <select by mouse>**          :  deselect data points
   * **d + SHIFT + <select by mouse>**  :  deselect data points for all exposures
     
* **e**: select exposure:
   
   * **e**                              :  select next exposure
   * **e + KEY_UP / KEY_DOWN**          :  select next/previous exposure
   * **e + KEY_RIGHT / KEY_LEFT**       :  select next/previous exposure
   * **e + <click on exposure>**        :  choose exposure which is clicked
   * **e + CTRL**                       :  remove exposure 
   * **e + <select by mouse>**          :  select area to calculate flux, FWHM and luminosity within selected region (e.g for emission line)

* **f**: construct fit
   
   * **f**                              :  show fit (only lines nearby fitting points)
   * **f + SHIFT**                      :  show full fit (all available lines in the full wavelength range of exposures)
   * **f + CTRL**                       :  <not set yet>
   
* **g**: fit gauss

   * **g + CTRL**                       :  show composite Galaxy spectrum (several templates available)

* **h**: highlight labels
   
   * **h + LEFT CLICK** on line label        :  highlight all the line labels corresponding to species of selected line label 

* **j**: show the dispersion of the model fit from MCMC

* **k**: add line to the fixed line group

   * **k + MOUSE LEFT CLICK** on  line Label         :  add line label to highlighted fixed line group.  

* **l**: choose lya
   
   * **l + LEFT CLICK**         :  set redshift for lya line to be at the position of mouse

* **m**: smooth spectrum

   * **m + MOUSE WHEEL FORWARD**      :  increase smoothness 
   * **m + MOUSE WHEEL BACKWARD**     :  decrease smoothness

* **n**: normalize/unnormalize the spectrum

* **o**: open / change UVES setup
  
   * **o**                              :  change UVES setup 
   * **o + CTRL**                       :  open file

* **p**: partial coverage and polynomial fitting
   
   * **p + two LEFT CLICKs**            :  create partial coverage line
   * **p + KEY_UP / KEY_DOWN**          :  fit selected points with polynomial higher degree
   * **p + KEY_RIGHT / KEY_LEFT**       :  fit selected points with polynomial lower degree
   
* **r**: select region:
   
   * **r + <select by mouse>**          :  add region (how to work with regions see Tutorial)
   * **r + SHIFT**                      :  force top x axis to show restframe wavelength
   
* **s**: select data points:
   
   * **s + <select by mouse>**          :  select data points
   * **s + SHIFT + <select by mouse>**  :  select data points for all exposures
   * **s + CTRL**                       :  save to recent file
   
* **t**: show fit results:
   
   * **t + CTRL**                       :  show/hide fit result window

* **q**: continuum
   
   * **q**                              :  make continuum in window using smoothing
   * **q + CRTL**                       :  show composite QSO spectrum (several templates available)

* **u**: find doublet:
   
   * **u + LEFT CLICK**         :  add line to doublet guess   
   
* **v**: change view of spectra (steps/points/lines + uncertainties)

* **w**: width of region:
   
   * **w + <select by mouse>**          :  select area to calculate equivalent width of absorption line. Continuum should be set for width calculation!
   * **w + SHIFT + <select by mouse>**  :  select area to calculate equivalent width of absorption line, subtracting fit model. (i.e. respective fit model, but no to continuum)
   * **w**                              :  hide w-region
  
* **x**: select bad pixels:
   
   * **s + <select by mouse>**          :  select bad pixels
   * **s + SHIFT + <select by mouse>**  :  unselect bad pixels

* **y**: likelihood region:
   
   * **y + LEFT CLICK** on line label  :  show a region of likelihood in (logN, b) parameter space for selected line label. The grid range is taken from fit model window as .

* z: zoom mode:
   
   * **z + <select by mouse>**          :  zoom into region
   * **z + CTRL**                       :  return to the previous view 
    
shift: 
  1. when shift is pressed you can shift absorption pointers using mouse
 