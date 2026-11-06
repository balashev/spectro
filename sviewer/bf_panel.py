"""BF lines side panel (Fit > BF lines, F10); the BF analysis window is in bf_analysis.py.
User guide: docs/bf.rst.
"""
from pathlib import Path
import csv
import re

import numpy as np
import pyqtgraph as pg
from astropy import units as u
from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtGui import QColor, QFontMetrics
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QStyle,
                             QVBoxLayout, QWidget)

from ..bf import experiment
from ..bf.data import Spectrum
from ..bf.measure import estimate, measure, window_run
from ..bf.profiles import C_KMS, KTAU
from .utils import Worker

N_DRAWS = 2000
DEBOUNCE_MS = 300
COMPONENT_KMS = 300.                    # fit-model components within this velocity of z set the automatic window
AUTO_FIT, AUTO_SYM = 'auto: from fit components', 'auto: symmetric, &plusmn;2 FWHM'
NOT_NORMALIZED = 'Build a continuum and normalize (N) before measuring BF'
COLUMNS = ('level', 'lambda [A]', 'f', 'log f*lambda', 'obs [A]', 'BF', 'sigma', 'flag', 'use')
FLAGS = ('clear', 'unclear', 'blend')
ESTIMATORS = (('median3', 'median of 3 (default)'), ('min', 'minimum pixel'), ('median', 'median of N'),
              ('mean', 'mean of N'))
SIGMAS = (('mc', 'Monte Carlo'), ('catalogue', 'median pixel error'))
GREY = QColor(150, 150, 150)
AMBER = QColor(255, 226, 168)
HIGHLIGHT = dict(size=11, brush=pg.mkBrush(255, 0, 200), pen=pg.mkPen(40, 40, 40, width=1))   # magenta, not fit-mask green
WINDOW_BRUSH = (255, 165, 0, 50)
AUTO_STYLE = ('QPushButton {background-color: #3c3c3c; color: #e6e6e6; border: 1px solid #5a5a5a; border-radius: 3px; '
              'padding: 1px 8px;} QPushButton:disabled {background-color: #2e2e2e; color: #9a9a9a;}')
COLORS = [tuple(int(255*c) for c in rgb) for rgb in
          ((.122, .467, .706), (1., .498, .055), (.173, .627, .173), (.839, .153, .157), (.580, .404, .741),
           (.549, .337, .294), (.890, .467, .761), (.737, .741, .133), (.090, .745, .812))]   # tab10 without grey
NO_LEVEL = (127, 127, 127)
# Preferences > BF: (options.ini key, label, kind, tooltip); kind 'folder' is a path ('' = default), else a number type.
PREFERENCES = (
    ('bf_b_default', 'b of a fixed common width (km/s)', float, 'Start value of the BF analysis b box'),
    ('bf_r_default', 'R when an exposure has none', float, 'Resolving power assumed for lines without exposure resolution'),
    ('bf_data_folder', 'Data folder (holds atomic/cache)', 'folder',
     'Empty: the folder of the last imported line file, else of the spectrum'),
    ('bf_atomic_cache', 'Atomic grid cache', 'folder', 'Pyratio and CHIANTI population grids. Empty: <data folder>/atomic/cache'),
    ('bf_import_levels', 'Levels on at import', str,
     'Import lines, catalogue files: lines of these levels start with use on (empty: all levels)'),
    ('bf_check_nsigma', 'Check lines: flag |z| above', float, 'z = (BF_obs - BF_pred)/sqrt(sigma_obs^2 + sigma_pred^2)'),
    ('bf_tau0_min', 'Check lines: uninformative below tau0', float, 'Line-centre optical depth at the fitted column'),
    ('bf_agn_filter', 'P2: AGN normalization filter', str, 'Pyratio AGN SED normalization band (e.g. r)'),
    ('bf_agn_mag', 'P2: AGN magnitude in that filter', float, 'Apparent magnitude normalizing the Pyratio AGN SED (P2 grids)'),
    ('bf_contour_n', 'Contours: grid nodes per axis', int, '(log N/g, C) grid of Covering per level'),
    ('bf_cover_depth', 'Implied covering: minimum 1 - F', float, 'Lines with a shallower full-coverage profile are hidden'),
    ('bf_session_vrange', 'Session export: pixels within +- (km/s)', float, 'Velocity range of the pixel stamps'),
    ('bf_match_vrange', 'Import: pixel search within +- (km/s)', float,
     'Imported BF without a window: estimator pixel sets centred within this velocity of the line are searched'))


class Cancelled(Exception):
    """Raised inside a worker job whose run was cancelled."""


def option(parent, key, kind=str):
    """Value of `key` in config/options.ini through the viewer (Preferences > BF); KeyError if the key is missing."""
    value = parent.options(key)
    if value is None: raise KeyError(f'{key} is missing in config/options.ini')
    return kind(value)


def font_pt():
    """Application font size in points (follows the system font and screen scaling)."""
    size = QApplication.font().pointSizeF()
    return size if size > 0 else 12.


def fg():
    """pyqtgraph foreground color (axes, neutral lines and legend samples)."""
    return pg.mkColor(pg.getConfigOption('foreground'))


def style_plot(plot, left=None, bottom=None, title=None):
    """Axis labels, ticks and title in the application font and the foreground color."""
    font, pt = QApplication.font(), f'{font_pt():g}pt'
    for name, text in (('left', left), ('bottom', bottom)):
        axis = plot.getAxis(name)
        axis.setStyle(tickFont=font)
        if text is not None: axis.setLabel(text, **{'font-size': pt, 'color': fg().name()})
    if title is not None: plot.setTitle(title, size=pt)
    return plot


def legend(plot, entries, offset=(5, 5)):
    """Legend of `plot` with exactly `entries`: (sample item, text), on a translucent background box."""
    if plot.legend is None:
        back = pg.mkColor(pg.getConfigOption('background')); back.setAlpha(200)
        plot.addLegend(offset=offset, labelTextSize=f'{font_pt():g}pt', brush=pg.mkBrush(back), pen=pg.mkPen(fg()))
    plot.legend.clear(); plot.legend.setOffset(offset)
    for item, text in entries: plot.legend.addItem(item, text)
    plot.legend.setVisible(bool(entries))


def symbol(color, shape='o'):
    return pg.ScatterPlotItem(symbol=shape, brush=pg.mkBrush(color), pen=pg.mkPen(None), size=9)


def number(value, digits=5):
    """Table text of a BF or sigma (float or imported string); '' stays ''."""
    return '' if value == '' else f'{float(value):.{digits}f}'


def run_at(mask, i):
    """Indexes of the contiguous True run of `mask` containing pixel i, or None."""
    if i < 0 or i >= len(mask) or not mask[i]: return None
    lo, hi = i, i
    while lo > 0 and mask[lo-1]: lo -= 1
    while hi < len(mask)-1 and mask[hi+1]: hi += 1
    return np.arange(lo, hi+1)


def bf_tolerance(text):
    """Precision of a BF printed in a file: half of its last printed digit (at least a few float ulp)."""
    mantissa, _, exponent = str(text).strip().lower().partition('e')
    digits = len(mantissa.split('.')[1]) if '.' in mantissa else 0
    return max(.5*10.**(int(exponent or 0)-digits), 4*np.spacing(abs(float(text))))


def species_of(level):
    """Ion of a level name: 'FeIIj3' -> 'FeII'."""
    return re.sub(r'j\d+$', '', level)


def level_key(level):
    """(ion, j) of a level name: 'FeII' -> ('FeII', 0), 'FeIIj3' -> ('FeII', 3); the sort key of levels."""
    m = re.fullmatch(r'(.*?)(?:j(\d+))?', level)
    return m.group(1), int(m.group(2) or 0)


def no_level_data(level):
    """Guard text of a level without level data (E_j, g_j) in SPECTRO, '' if it has them."""
    if experiment.level_data(level) is not None: return ''
    return (f'no level data (E_j, g_j) for {level} in SPECTRO: used with free per-level columns, excluded from the '
            'excitation laws and the equilibrium columns')


def exclusion_text(excluded):
    """{line id: reason} as text grouped by reason: 'FeIIj9:1234.5, ... (reason)'."""
    by = {}
    for i, why in excluded.items(): by.setdefault(why, []).append(i)
    return '; '.join(f"{', '.join(ids)} ({why})" for why, ids in by.items())


def warning_icon(widget):
    size = widget.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
    return widget.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(size, size)


def start_worker(owner, fn, done, progress=None):
    """Run fn(progress_callback, cancelled) in QThreadPool; errors go to owner.failed.
    Incrementing owner._run (cancel_run) cancels the job: its result is discarded."""
    owner._run = run = getattr(owner, '_run', 0)+1

    def cancelled(): return owner._run != run
    worker = Worker(lambda progress_callback=None, thread_id=0: fn(progress_callback, cancelled))
    worker.signals.result.connect(lambda out: None if cancelled() else done(out))
    worker.signals.error.connect(lambda e: None if cancelled() else owner.failed(str(e[1])))
    if progress is not None: worker.signals.progress.connect(lambda p: None if cancelled() else progress(p))
    owner._worker = worker
    QThreadPool.globalInstance().start(worker)


def progress_row(owner):
    """Progress bar and Cancel button, visible only while a run is active."""
    owner.progress = QProgressBar(); owner.progress.setRange(0, 100); owner.progress.setValue(0)
    owner.cancel_button = QPushButton('Cancel'); owner.cancel_button.clicked.connect(owner.cancel_run)
    owner.progress.hide(); owner.cancel_button.hide()
    row = QHBoxLayout(); row.addWidget(owner.progress, 1); row.addWidget(owner.cancel_button)
    return row


def show_busy(owner, on, determinate=False):
    owner.progress.setRange(0, 100 if determinate or not on else 0)
    owner.progress.setValue(0)
    owner.progress.setVisible(on); owner.cancel_button.setVisible(on)


class BFLinesPanel(QWidget):
    """BF line list of the viewer, inserted into splitter_fit like chooseFitParsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.rows, self.items, self.analysis, self._spec, self._pix = [], [], None, {}, {}
        self.auto, self._ref, self._z, self.import_dir, self.excluded = True, None, None, None, ('', {})
        self._shown = None
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.z_changed)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('<b>BF lines</b>'))
        layout.addLayout(self._controls())
        self.table = pg.TableWidget(editable=False, sortable=False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self.show_line)
        self.table.cellDoubleClicked.connect(self.select_line)
        layout.addWidget(self.table, 1)
        self.info = QLabel(''); self.info.setWordWrap(True)
        layout.addWidget(self.info)
        buttons = QHBoxLayout()
        for text, slot in (('Import lines', lambda: self.import_lines()), ('Export CSV', lambda: self.export_csv()),
                           ('BF analysis', self.open_analysis), ('Close', self.close_panel)):
            b = QPushButton(text); b.clicked.connect(slot); buttons.addWidget(b)
        layout.addLayout(buttons)
        self.fill_table()
        self.update_window()

    def _controls(self):
        form = QFormLayout()
        self.estimator = QComboBox()
        for code, text in ESTIMATORS: self.estimator.addItem(text, code)
        self.n = QSpinBox(); self.n.setRange(3, 15); self.n.setSingleStep(2); self.n.setValue(3); self.n.setPrefix('N = ')
        h = QHBoxLayout(); h.addWidget(self.estimator); h.addWidget(self.n)
        form.addRow('Estimator', h)
        self.sigma = QComboBox()
        for code, text in SIGMAS: self.sigma.addItem(text, code)
        form.addRow('Sigma', self.sigma)
        self.vmin, self.vmax = QDoubleSpinBox(), QDoubleSpinBox()
        for s in (self.vmin, self.vmax):
            s.setRange(-500, 500); s.setDecimals(1); s.setSuffix(' km/s'); s.valueChanged.connect(self._manual)
        self.auto_button = QPushButton('Auto'); self.auto_button.clicked.connect(self.set_auto)
        self.auto_button.setToolTip('Automatic default window: fit-model components at this z, else +-2 LSF FWHM')
        self.auto_button.setStyleSheet(AUTO_STYLE)
        self.window_source = QLabel('')
        h = QHBoxLayout(); h.addWidget(self.vmin); h.addWidget(self.vmax); h.addWidget(self.auto_button)
        h.addWidget(self.window_source, 1)
        form.addRow('Default window', h)
        self.estimator.currentIndexChanged.connect(self._estimator_changed)
        self.n.valueChanged.connect(self.refresh)
        self.sigma.currentIndexChanged.connect(self.refresh)
        self._estimator_changed()
        return form

    def _estimator_changed(self):
        self.n.setEnabled(self.estimator.currentData() in ('median', 'mean'))
        self.refresh()

    def spectrum_file(self):
        """Path of the active exposure's file, or None."""
        s = self.exposures()
        name = getattr(s[self.parent.s.ind], 'filename', '') if s and self.parent.s.ind is not None else ''
        return Path(name) if name else None

    def spectrum_dir(self):
        p = self.spectrum_file()
        return p.parent if p is not None and p.parent.is_dir() and str(p.parent) != '.' else None

    @property
    def data_dir(self):
        """Data folder (parent of the default atomic cache): Preferences > BF, else the folder of the last imported line
        file, else the spectrum's folder (None if none is known)."""
        folder = option(self.parent, 'bf_data_folder')
        return Path(folder) if folder else self.import_dir or self.spectrum_dir()

    def atomic_cache(self):
        """Population-grid cache folder: Preferences > BF, else <data folder>/atomic/cache (None if unknown)."""
        folder = option(self.parent, 'bf_atomic_cache')
        if folder: return Path(folder)
        return None if self.data_dir is None else self.data_dir/'atomic'/'cache'

    @staticmethod
    def color(level):
        """GUI color of a level: its index j in the level name (FeII -> 0, FeIIjN -> N)."""
        return COLORS[level_key(level)[1] % len(COLORS)]

    def color_hex(self, level): return '#%02x%02x%02x' % self.color(level)

    def model_b(self):
        """{level: (b km/s, z)} from the fit model: per level, the component nearest to z (within COMPONENT_KMS) that has
        that level."""
        out = {}
        for s in getattr(getattr(self.parent, 'fit', None), 'sys', []):
            v = abs(C_KMS*((1+s.z.val)/(1+self.z())-1))
            if v >= COMPONENT_KMS: continue
            for name, sp in s.sp.items():
                if name not in out or v < out[name][2]: out[name] = (float(sp.b.val), float(s.z.val), v)
        return {k: x[:2] for k, x in out.items()}

    def _manual(self):
        self.auto = False
        self.update_window()

    def set_auto(self):
        self.auto = True
        self.update_window()

    def components(self):
        """Sorted (velocity, b) in km/s of the fit-model components within COMPONENT_KMS of z (b of the BF lines' ion if
        present, else the largest)."""
        ions = {species_of(r['species']) for r in self.rows}
        out = []
        for s in getattr(getattr(self.parent, 'fit', None), 'sys', []):
            v = C_KMS*((1+s.z.val)/(1+self.z())-1)
            b = [sp.b.val for name, sp in s.sp.items() if name in ions] or [sp.b.val for sp in s.sp.values()]
            if abs(v) < COMPONENT_KMS: out.append((v, max(b, default=0.)))
        return sorted(out)

    def auto_window(self, lam, k):
        """(vmin, vmax, source) of the automatic window for a line at observed lam (Angstrom) in exposure k (or None)."""
        comps, fwhm = self.components(), C_KMS/self.resolution(k, lam)[0]
        v = [c[0] for c in comps]
        if len(v) > 1 and v[-1] > v[0]:
            half = (v[-1]-v[0])/(len(v)-1)/2
            return v[0]-half, v[-1]+half, AUTO_FIT
        if comps:
            h = max(max(c[1] for c in comps), fwhm)
            return v[0]-h, v[0]+h, AUTO_FIT
        h = 2*fwhm
        if k is not None:
            dv = C_KMS*(np.asarray(self.exposures()[k].spec.raw.x)/lam-1)
            for side in (np.sort(dv[dv > 0])[:4], np.sort(-dv[dv < 0])[:4]):   # >= 3 pixels on each side of the centre
                if len(side) > 3: h = max(h, (side[2]+side[3])/2)
                elif len(side) == 3: h = max(h, side[2]*(1+1e-9))
        return -h, h, AUTO_SYM

    def reference(self):
        """(observed lam, exposure) the automatic window is shown for: the last clicked row, else the plot centre."""
        lam = self._ref
        if lam is None:
            try: lam = float(np.mean(self.parent.plot.vb.viewRange()[0]))
            except (AttributeError, TypeError, IndexError): lam = 0.
        return lam, next(iter(self.covering(lam)), None)

    def velocity_window(self, lam=None, k=None):
        """Default window (vmin, vmax) km/s for a line at observed lam in exposure k (automatic or the two spin boxes)."""
        if not self.auto: return (self.vmin.value(), self.vmax.value())
        if lam is None: lam, k = self.reference()
        return self.auto_window(lam, k)[:2]

    def update_window(self):
        """Show the automatic window (for reference()) in the spin boxes, or 'manual'."""
        if self.auto:
            lo, hi, source = self.auto_window(*self.reference())
            for s, v in ((self.vmin, lo), (self.vmax, hi)):
                s.blockSignals(True); s.setValue(v); s.blockSignals(False)
        else: source = 'manual'
        self.window_source.setText(f'<i>{source}</i>')
        self.auto_button.setEnabled(not self.auto)

    def z(self): return float(getattr(self.parent, 'z_abs', 0.))

    def message(self, text):
        bar = getattr(self.parent, 'statusBar', None)
        if bar is not None: bar.setText(text)

    def exposures(self): return list(getattr(self.parent, 's', []))

    def covering(self, lam):
        """Exposure indexes whose wavelength range covers lam, active exposure first."""
        s = self.exposures()
        order = [self.parent.s.ind] + [k for k in range(len(s)) if k != self.parent.s.ind] if s else []
        return [k for k in order if len(s[k].spec.raw.x) and s[k].spec.raw.x[0] <= lam <= s[k].spec.raw.x[-1]]

    def normalized(self, k=None):
        p = self.parent
        if not self.exposures() or not getattr(p, 'normview', False) or not p.panel.normalize.isChecked(): return False
        s = self.exposures()[p.s.ind if k is None else k]
        return s.cont_mask is not None and len(s.spec.norm.x) > 0 and len(s.fit_mask.norm.x) == len(s.spec.norm.x)

    def spectrum(self, k):
        if k not in self._spec:
            n = self.exposures()[k].spec.norm
            self._spec[k] = Spectrum(np.asarray(n.x)*u.AA, np.asarray(n.y), np.asarray(n.err), frame='observed',
                                     normalized=True)
        return self._spec[k]

    def pixel_spectrum(self, k):
        """Flux of exposure k for finding the pixels of an imported BF: normalized if it has a continuum, else as loaded
        (spectro.bf Spectrum; grid 'norm' or 'raw')."""
        if k not in self._pix:
            s = self.exposures()[k]
            if self.normalized(k): x, y, e, grid = s.spec.norm.x, s.spec.norm.y, s.spec.norm.err, 'norm'
            elif s.cont_mask is not None and len(s.cont.y) == int(np.sum(s.cont_mask)):
                cm, c = np.asarray(s.cont_mask, bool), np.asarray(s.cont.y)
                x, y, e, grid = np.asarray(s.spec.raw.x)[cm], np.asarray(s.spec.raw.y)[cm]/c, np.asarray(s.spec.raw.err)[cm]/c, 'norm'
            else: x, y, e, grid = s.spec.raw.x, s.spec.raw.y, s.spec.raw.err, 'raw'
            self._pix[k] = (Spectrum(np.asarray(x)*u.AA, np.asarray(y), np.asarray(e), frame='observed',
                                     normalized=grid == 'norm'), grid)
        return self._pix[k]

    def view_flux(self, k):
        """(x, flux) of exposure k in the current view (normalized or as loaded) and whether the view is normalized."""
        s = self.exposures()[k]
        if self.normalized(k): return np.asarray(s.spec.norm.x), np.asarray(s.spec.norm.y), True
        return np.asarray(s.spec.raw.x), np.asarray(s.spec.raw.y), False

    def resolution(self, k, lam):
        """(R, note) of exposure k at lam; Preferences > BF > R when there is no exposure or it has no resolution."""
        r = float(self.exposures()[k].resolution(x=lam)) if k is not None else 0.
        if np.isfinite(r) and r > 0: return r, ''
        r = option(self.parent, 'bf_r_default', float)
        return r, f'R = {r:.0f} assumed (no exposure resolution)'

    def inside(self, r):
        x = self.exposures()[r['exp']].spec.norm.x
        return len(x) > 0 and x[0] <= r['anchor'] <= x[-1]

    def run(self, r):
        """Selected-pixel run of row r in its exposure (norm pixels), or None."""
        if not self.inside(r): return None
        s = self.exposures()[r['exp']]
        return run_at(np.asarray(s.fit_mask.norm.x, bool), int(np.argmin(np.abs(np.asarray(s.spec.norm.x)-r['anchor']))))

    def find(self, species, lam):
        return next((r for r in self.rows if r['source'] != 'imported' and r['species'] == species
                     and abs(float(r['lambda_rest'])-lam) < 1e-4), None)

    def add_line(self, line, k, source):
        r = self.find(line.name, line.l())
        if r is None:
            r = dict(species=line.name, lambda_rest=f'{line.l()}', f=f'{line.f()}', clarity='clear', use=True,
                     source=source, exp=k, anchor=line.l()*(1+self.z()), BF='', sigma='', reason='')
            self.rows.append(r)
        return r

    def selection_changed(self, x1, x2, remove=False):
        """After s/d + region: shown labels with centre in [x1, x2] and in a selected run become BF lines;
        marked lines whose run disappeared leave the list."""
        if not self.normalized():
            if not any(r['source'] == 'imported' for r in self.rows): return self.message(NOT_NORMALIZED)
            self.refresh()
            return self.reshow()
        lo, hi = min(x1, x2), max(x1, x2)
        if not remove:
            for label in getattr(self.parent.abs, 'lines', []):
                lam = label.line.l()*(1+self.z())
                if self.imported(label.line.name, label.line.l()): continue
                if lo < lam < hi:
                    k = next((k for k in self.covering(lam) if self.normalized(k) and
                              self.run(dict(exp=k, anchor=lam)) is not None), None)
                    if k is not None: self.add_line(label.line, k, 'selection')
        self.rows = [r for r in self.rows if not self.lost(r)]
        self.refresh()
        self.reshow()

    def imported(self, species, lam):
        return any(r['source'] == 'imported' and r['species'] == species and abs(float(r['lambda_rest'])-lam) < 1e-4
                   for r in self.rows)

    def reshow(self):
        """Redraw the highlight of the last shown row (after a pixel selection), keeping the plot range."""
        row = next((i for i, r in enumerate(self.rows) if r is self._shown), None)
        if row is not None: self.show_line(row, centre=False)

    def selected_run(self, k, lam):
        """(lo, hi) wavelengths of the selected-pixel run of exposure k (current view) containing lam, or None."""
        s = self.exposures()[k]
        x, m = np.asarray(s.spec.x()), np.asarray(s.fit_mask.x(), bool)
        if len(x) == 0 or len(m) != len(x) or not x[0] <= lam <= x[-1]: return None
        run = run_at(m, int(np.argmin(np.abs(x-lam))))
        return None if run is None else (x[run[0]], x[run[-1]])

    def lost(self, r):
        """A marked line inside the normalized data whose selected-pixel run has disappeared."""
        k = r['exp']
        return r['source'] != 'imported' and k is not None and self.normalized(k) and self.inside(r) and self.run(r) is None

    def toggle_label(self, label):
        """Double-click on a line label: remove the BF line, or add it and select its default window pixels."""
        r = self.find(label.line.name, label.line.l())
        if r is not None:
            self.rows.remove(r)
            if r['source'] == 'double-click' and r['exp'] is not None: self._select(r['exp'], r['anchor'], remove=True)
            self.refresh()
            return self.message(f'{label.line} removed from BF lines')
        lam = label.line.l()*(1+self.z())
        k = next(iter(self.covering(lam)), None)
        self.add_line(label.line, k, 'double-click')
        if k is not None: self._select(k, lam)
        self.refresh()
        self.message(f'{label.line} added to BF lines')

    def _select(self, k, lam, remove=False):
        """Select (or deselect) the default window around lam with the viewer's s/d pixel-selection code."""
        self._select_range(k, *(lam*(1+v/C_KMS) for v in self.velocity_window(lam, k)), remove=remove)

    def _select_range(self, k, lo, hi, remove=False):
        s = self.exposures()[k]
        s.add_points(lo, -np.inf, hi, np.inf, remove=remove, redraw=False)
        s.set_fit_mask()
        s.update_points()
        s.set_res()
        self.parent.s.chi2()

    def settings(self):
        return dict(estimator=self.estimator.currentData(), n=self.n.value(), sigma=self.sigma.currentData())

    def measure_row(self, r, z):
        r['reason'], r['note'] = '', ''
        lam = float(r['lambda_rest'])*(1+z)
        imported = r['source'] == 'imported'
        k = next(iter(self.covering(lam)), None) if imported else r['exp']
        r['R'], r['note'] = self.resolution(k, lam)
        if imported: return self._imported_pixels(r, lam, z)
        if k is None: r['reason'] = 'outside the data'
        elif not self.normalized(k): r['reason'] = 'not normalized'
        elif not self.inside(r): r['reason'] = 'outside the data'
        else:
            run = self.run(r)
            if run is None: r['reason'] = 'no pixels'
            else: self._measure(r, run, z)
        if r['reason']: r.update(BF='', sigma='')

    def _imported_pixels(self, r, lam, z):
        """Pixels of an imported BF: the file's window, else the default window or a matching pixel set (_match);
        a different selected run at the line replaces the window and BF, sigma are remeasured there."""
        k = next(iter(self.covering(lam)), None)
        r.update(pixels=None, run=None, pix_exp=None, pix_note='', pix_status='', meas=None, remeasured=False,
                 BF=r['BF_imported'], sigma=r['sigma_imported'], method=r['method_imported'],
                 window=tuple(r.get('file_window') or self.velocity_window(lam, k)))
        if r['BF'] == '' or r['sigma'] == '':
            r['reason'] = 'no BF in the imported file'
            return
        if k is None: return
        est, n = (r['estimator'], r['n']) if r.get('estimator') else (self.settings()['estimator'], self.settings()['n'])
        n = {'median3': 3, 'min': 1}.get(est, int(n))
        bf, tol = float(r['BF']), bf_tolerance(r['BF'])
        status, window = ('file', r['file_window']) if r.get('file_window') else ('default', self.velocity_window(lam, k))
        try:
            sp, grid = self.pixel_spectrum(k)
            v = sp.velocity(experiment.transition(r), z)
            run = window_run(sp, experiment.transition(r), z, tuple(window))
        except ValueError as exc:
            r['pix_note'] = str(exc)
            return
        value, used = self._estimate(sp, run, est, n)
        ok = abs(value-bf) <= tol
        if not ok and status == 'default':
            found = self._match(sp, v, bf, tol, est, n)
            if found is not None: (run, used, value, window), status, ok = found, 'matched', True
        x = sp.wavelength.to_value(u.AA)
        sel = self.selected_run(k, lam)
        if sel is not None:
            chosen = np.flatnonzero((x >= sel[0]) & (x <= sel[1]))
            if chosen.size and not (chosen[0] == run[0] and chosen[-1] == run[-1]):
                set_ = self.settings()
                try: m = measure(sp if grid == 'norm' else self._as_loaded(sp), experiment.transition(r), z, pixels=chosen,
                                 estimator=set_['estimator'], n=set_['n'], sigma=set_['sigma'], n_draws=N_DRAWS, seed=0)
                except ValueError as exc:
                    r['pix_note'] = str(exc)
                    return
                run, status, ok, est, n, value = chosen, 'selected', True, m['estimator'], m['n'], m['BF']
                used = np.asarray(m['pixel_indexes'], int)
                i, j = int(run[0]), int(run[-1])
                window = ((v[i-1]+v[i])/2 if i > 0 else v[i], (v[j]+v[j+1])/2 if j < len(v)-1 else v[j])
                flux = '' if grid == 'norm' else ', on unnormalized flux'
                r.update(BF=m['BF'], sigma=m['sigma'], meas=m, remeasured=True,
                         method=f"remeasured in the selected window ({m['estimator']} N={m['n']}, sigma {m['sigma_method']}"
                                f"{flux}); imported BF {r['BF_imported']}")
        i, j = int(run[0]), int(run[-1])
        r.update(pixels=used.tolist(), run=(i, j), pix_exp=k, pix_grid=grid, pix_est=(est, n), pix_value=value,
                 window=tuple(map(float, window)), pix_status=status if ok else f'{status} not reproduced',
                 pix_x=x[used], run_x=(x[i], x[j]),
                 sel_x=((x[i-1]+x[i])/2 if i > 0 else x[i]-1e-6, (x[j]+x[j+1])/2 if j < len(x)-1 else x[j]+1e-6))
        if not ok: r['pix_note'] = (f"this spectrum gives BF = {value:.5f} with {self.pixel_text(r)} in the "
                                    f"{status} window")

    @staticmethod
    def _as_loaded(sp):
        """The spectrum sp (flux as loaded, no continuum) flagged as normalized for measurement."""
        return Spectrum(sp.wavelength, sp.flux, sp.uncertainty, frame='observed', normalized=True, mask=sp.mask)

    @staticmethod
    def _estimate(sp, run, estimator, n):
        """(value, used pixel indexes) of `estimator` on the pixel run of spectrum sp (masked pixels excluded), as
        spectro.bf.measure.measure computes them."""
        flux = np.where(sp.mask[run], np.nan, sp.flux[run])
        if np.isnan(flux).all(): return np.nan, run[:0]
        value, ix = estimate(flux, estimator, n)
        return float(value), run[ix[ix >= 0]]

    def _match(self, sp, v, bf, tol, estimator, n):
        """(run, used, value, window) of the estimator pixel set within +- bf_match_vrange km/s that reproduces bf
        within tol, nearest to the line centre (then the shortest window); None if there is none."""
        idx = np.flatnonzero(np.abs(v) <= option(self.parent, 'bf_match_vrange', float))
        best = None
        for a in range(len(idx)):
            for b in range(a+n-1, len(idx)):
                run = idx[a:b+1]
                value, used = self._estimate(sp, run, estimator, n)
                if used.size == 0 or not abs(value-bf) <= tol: continue
                key = (abs(float(np.mean(v[used]))), len(run))
                if best is None or key < best[0]: best = (key, run, used, value)
        if best is None: return None
        _, run, used, value = best
        i, j = run[0], run[-1]
        window = ((v[i-1]+v[i])/2 if i > 0 else v[i], (v[j]+v[j+1])/2 if j < len(v)-1 else v[j])
        return run, used, value, window

    def _measure(self, r, run, z):
        s = self.settings()
        try:
            m = measure(self.spectrum(r['exp']), experiment.transition(r), z, pixels=run, estimator=s['estimator'], n=s['n'],
                        sigma=s['sigma'], n_draws=N_DRAWS, seed=0)
        except ValueError as exc:
            r['reason'] = str(exc)
            return
        x = self.spectrum(r['exp']).wavelength.to_value(u.AA)
        i, j = int(run[0]), int(run[-1])
        r.update(BF=m['BF'], sigma=m['sigma'], meas=m, pixels=m['pixel_indexes'], window=m['window'], pix_exp=r['exp'],
                 run=(i, j), method=f"{m['estimator']} N={m['n']}, sigma {m['sigma_method']}", pix_grid='norm',
                 pix_x=x[m['pixel_indexes']], run_x=(x[i], x[j]),
                 sel_x=((x[i-1]+x[i])/2 if i > 0 else x[i]-1e-6, (x[j]+x[j+1])/2 if j < len(x)-1 else x[j]+1e-6))

    def norm_key(self):
        return tuple(hash(np.asarray(s.spec.norm.y).tobytes()) for s in self.exposures())

    def data_changed(self):
        """Re-measure if the normalized flux of any exposure changed (e.g. continuum edited in normalized view)."""
        if self.normalized() and self.norm_key() != getattr(self, '_norm_key', None): self.refresh()

    def schedule_refresh(self):
        """z is changing (drag, arrow key, z panel): re-measure DEBOUNCE_MS after the last change."""
        self._timer.start()

    def z_changed(self):
        """End of a z drag or debounce timeout: re-measure if z differs from the last measurement."""
        self._timer.stop()
        if self.z() != self._z: self.refresh()

    def refresh(self, *args):
        if not hasattr(self, 'table'): return
        self._timer.stop()
        self._spec, self._pix, self._norm_key, self._z = {}, {}, self.norm_key(), self.z()
        if any(r['source'] != 'imported' for r in self.rows) and not self.normalized():
            self.message(NOT_NORMALIZED)
        z, seen = self.z(), set()
        for r in self.rows:
            self.measure_row(r, z)
            key = f"{r['species']}:{float(r['lambda_rest']):.4f}"
            if not r['reason'] and key in seen: r['reason'] = 'duplicate of an earlier row'
            seen.add(key)
        self.find_overlaps()
        self.fill_table()
        self.update_window()
        self.changed()

    def used_rows(self):
        """Ticked rows with a valid BF."""
        return [r for r in self.rows if r['use'] and not r.get('reason')]

    def data(self):
        """BFData of the used rows; levels ordered by ion and index j; E_j from the package level data (None without)."""
        from .bf_analysis import bf_data
        return bf_data(self.used_rows())

    def resolving_power(self):
        return {f"{r['species']}:{r['lambda_rest']}": r['R'] for r in self.used_rows()}

    def transitions(self):
        """(name, lambda_rest, z) of the shown line labels (at z_abs) and of every line of every fit-model component with
        line-centre tau0 above the viewer's tau limit (Preferences > Fit)."""
        out = [(l.line.name, l.line.l(), self.z()) for l in getattr(getattr(self.parent, 'abs', None), 'lines', [])]
        tau_min = float(getattr(self.parent, 'tau_limit', 0.) or 0.)
        atomic = getattr(self.parent, 'atomic', None)
        for s in getattr(getattr(self.parent, 'fit', None), 'sys', []):
            for name, sp in s.sp.items():
                lines = atomic.list(name) if atomic is not None and atomic.find(name) else []
                out += [(l.name, l.l(), s.z.val) for l in lines if KTAU*10**sp.N.val/sp.b.val*l.f()*l.l() > tau_min]
        return out

    def find_overlaps(self):
        """r['overlap']: other transitions whose centre lies in the row's BF window (measured, else the default window)."""
        if not self.rows: return
        cand = self.transitions()
        names = np.array([c[0] for c in cand], dtype=object)
        rest_c, lam = np.array([c[1] for c in cand], float), np.array([c[1]*(1+c[2]) for c in cand], float)
        z = self.z()
        for r in self.rows:
            rest = float(r['lambda_rest']); obs = rest*(1+z)
            w = r.get('window') or self.velocity_window(obs, next(iter(self.covering(obs)), None))
            v = C_KMS*(lam/obs-1)
            own = (names == r['species']) & (np.abs(rest_c-rest) < 1e-3)   # the line itself, at any component z
            hit = (v >= w[0]) & (v <= w[1]) & ~own
            r['overlap'] = sorted({f'{cand[i][0]} {cand[i][1]:.2f} at z = {cand[i][2]:.5f}' for i in np.flatnonzero(hit)})

    @staticmethod
    def warnings(r):
        """Guard texts of a row: blend switched on, window overlaps, Check lines flags, no level data."""
        w = ['flagged blend, included manually'] if r['clarity'] == 'blend' and r['use'] else []
        w += [f'window overlaps {t}' for t in r.get('overlap', ())]
        if r.get('check'): w.append(f"check lines: {r['check']}")
        if no_level_data(r['species']): w.append(no_level_data(r['species']))
        return w

    def fill_table(self):
        z, t = self.z(), self.table
        t.clear()
        t.setColumnCount(len(COLUMNS)); t.setRowCount(len(self.rows))
        t.setHorizontalHeaderLabels(COLUMNS)
        for i, r in enumerate(self.rows):
            lam, f = float(r['lambda_rest']), float(r['f'])
            bf = number(r.get('BF', '')) + ((' (remeasured)' if r.get('remeasured') else ' (imported)')
                                            if r['source'] == 'imported' else '')
            vals = (r['species'], f'{lam:.4f}', f'{f:.4g}', f'{np.log10(f*lam):.3f}' if f > 0 else '',
                    f'{lam*(1+z):.3f}', bf, number(r.get('sigma', '')))
            warn = self.warnings(r)
            tip = r.get('reason') and f"Excluded: {r['reason']}"
            tip = tip or r.get('note') or r.get('method', '')
            if r.get('pix_note'): tip = f"{tip}; {r['pix_note']}"
            if warn: tip = '\n'.join(['Warning: '+w for w in warn] + ([tip] if tip else []))
            for c, val in enumerate(vals):
                it = t.itemClass(str(val), i); it.setEditable(False)
                if warn: it.setBackground(AMBER); it.setForeground(QColor(0, 0, 0))
                if r.get('reason'): it.setForeground(GREY)
                it.setToolTip(tip)
                t.setItem(i, c, it)
            t.setCellWidget(i, 7, self._flag_cell(r, warn, tip))
            t.setCellWidget(i, 8, self._use_box(r, tip, warn))
        t.resizeColumnsToContents()
        font = t.horizontalHeader().font(); font.setBold(True)
        for c, name in enumerate(COLUMNS):
            t.setColumnWidth(c, max(t.columnWidth(c), QFontMetrics(font).horizontalAdvance(name)+16))
        if self.rows: t.setColumnWidth(7, max(t.columnWidth(7), t.cellWidget(0, 7).sizeHint().width()))
        self.summary()

    @staticmethod
    def _cell_background(widget, warn):
        if warn:
            widget.setAutoFillBackground(True)
            pal = widget.palette(); pal.setColor(widget.backgroundRole(), AMBER); widget.setPalette(pal)

    def _flag_cell(self, r, warn, tip):
        """Flag combo, with the standard warning icon in front of it when the row has guard warnings."""
        cell = QWidget(); h = QHBoxLayout(cell); h.setContentsMargins(2, 0, 2, 0); h.setSpacing(3)
        icon = QLabel()
        if warn: icon.setPixmap(warning_icon(self))
        icon.setToolTip(tip)
        box = QComboBox(); box.addItems(FLAGS); box.setCurrentText(r['clarity'])
        box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        text = max(QFontMetrics(box.font()).horizontalAdvance(f) for f in FLAGS)
        box.view().setMinimumWidth(text+48)   # macOS popups add a check-mark column: keep 'unclear' whole
        box.currentTextChanged.connect(lambda text, r=r: (r.update(clarity=text), self.changed(), self._refill()))
        box.setToolTip(tip)
        h.addWidget(icon); h.addWidget(box, 1)
        icon.setFixedWidth(warning_icon(self).width())
        self._cell_background(cell, warn)
        return cell

    def _use_box(self, r, tip, warn):
        box = QCheckBox(); box.setChecked(bool(r['use'])); box.setToolTip(tip)
        box.toggled.connect(lambda on, r=r: (r.update(use=on), self.summary(), self.changed(), self._refill()))
        self._cell_background(box, warn)
        return box

    def _refill(self):
        """Redraw the table after the current event (a cell widget must not be deleted inside its own signal)."""
        QTimer.singleShot(0, self.fill_table)

    def summary(self):
        """Footer: lines used in fits, the lines the last run left out (and why), assumed resolving powers."""
        notes = sorted({r['note'] for r in self.used_rows() if r.get('note')})
        text = f'{len(self.used_rows())} of {len(self.rows)} lines used in fits'
        imported = sum(r['source'] == 'imported' for r in self.rows)
        if imported and imported < len(self.rows):
            text += f' ({imported} imported, {len(self.rows)-imported} marked in the panel)'
        name, excluded = self.excluded
        if excluded: text += f'; last run ({name}) excluded {exclusion_text(excluded)}'
        self.info.setText('; '.join([text] + notes))

    def set_excluded(self, name, excluded):
        """Lines the last run `name` left out, {line id: reason} (empty: none), for the footer."""
        self.excluded = (name, dict(excluded))
        self.summary()

    def set_check_flags(self, flags):
        """Check lines result {line id: flag text} into the rows (replaces earlier flags)."""
        for r in self.rows: r['check'] = flags.get(f"{r['species']}:{r['lambda_rest']}", '')
        self.fill_table()

    def changed(self):
        if self.analysis is not None and self.analysis.isVisible(): self.analysis.show_data()

    @staticmethod
    def pixel_text(r):
        """Estimator behind a row's BF pixels, e.g. 'median of 3'."""
        if r['source'] == 'imported' and not r.get('remeasured'):
            est, n = r.get('pix_est') or (r.get('estimator') or 'median3', r.get('n') or 3)
            m = dict(estimator=est, n=n)
        else: m = r.get('meas')
        if m is None: return 'median of 3'
        return {'min': 'minimum pixel', 'mean': f"mean of {m['n']}"}.get(m['estimator'], f"median of {m['n']}")

    def pixel_message(self, r):
        """Status-bar text of a row with BF pixels; for imported rows whether the pixels reproduce the imported BF."""
        n, text, (a, b), status = len(r['pixels']), self.pixel_text(r), r['window'], r.get('pix_status', '')
        head = f'BF pixels: {n} ({text})'
        if status.endswith('not reproduced'):
            digits = min(max(len(str(r['BF']).split('.')[-1]), 1), 6)
            values = f"recomputed {r['pix_value']:.{digits}f} vs imported {float(r['BF']):.{digits}f} (not reproduced)"
            if status.startswith('file'): return f'{head}, window {a:+.1f}..{b:+.1f} km/s from the file: {values}'
            if status.startswith('selected'): return f'{head} in the selected window {a:+.1f}..{b:+.1f} km/s: {values}'
            return f'{head} in the default window: {values}'
        tail = {'file': ', reproduces imported BF', 'default': ', reproduces imported BF',
                'matched': ', found by matching the imported BF'}.get(status, '')
        if status == 'selected':
            text = (f"{head}, selected window {a:+.1f}..{b:+.1f} km/s: BF remeasured {float(r['BF']):.5f} "
                    f"(imported {float(r['BF_imported']):.5f})")
            return text + (' (measured on unnormalized flux)' if r.get('pix_grid') == 'raw' else '')
        return f'{head}, window {a:+.1f}..{b:+.1f} km/s{tail}'

    def show_line(self, row, column=0, centre=True):
        """Centre the main plot on the line (unless centre is False), draw its BF window and highlight the pixels the
        estimator used, at the flux of the current view."""
        r, p = self.rows[row], self.parent
        lam = float(r['lambda_rest'])*(1+self.z())
        self._ref, self._shown = lam, r
        self.clear_highlight()
        k = r.get('pix_exp')
        if r.get('pixels') and not r.get('reason') and k is not None:
            x, y, normalized = self.view_flux(k)
            px = np.asarray(r['pix_x'])
            i = np.minimum(np.searchsorted(x, px), len(x)-1)
            on = x[i] == px
            self.items = [pg.LinearRegionItem(r['run_x'], movable=False, brush=pg.mkBrush(*WINDOW_BRUSH)),
                          pg.ScatterPlotItem(x=px[on], y=y[i[on]], **HIGHLIGHT)]
            text = self.pixel_message(r)
            if not normalized and not text.endswith('(measured on unnormalized flux)'): text += ' (shown on unnormalized flux)'
        else:
            w = self.velocity_window(lam, next(iter(self.covering(lam)), None))
            self.items = [pg.LinearRegionItem([lam*(1+v/C_KMS) for v in w], movable=False, brush=pg.mkBrush(*WINDOW_BRUSH))]
            why = r.get('reason') or r.get('pix_note') or 'no exposure covers the line'
            text = f'BF pixels: none ({why}); default window {w[0]:+.1f}..{w[1]:+.1f} km/s'
        for it in self.items:
            it.setZValue(100)
            p.plot.vb.addItem(it)
        if centre:
            try: p.plot.set_range(lam*(1-150/C_KMS), lam*(1+150/C_KMS))
            except (ValueError, IndexError): p.plot.vb.setXRange(lam*(1-150/C_KMS), lam*(1+150/C_KMS))
        self.update_window()
        self.message(text)
        return text

    def select_line(self, row, column=0):
        """Row double-click: show the line and select its BF window pixels (the window of the shown pixels, else the default
        window) with the viewer's pixel-selection code, for Fit and Compare coverage models."""
        r = self.rows[row]
        lam = float(r['lambda_rest'])*(1+self.z())
        k = r.get('pix_exp')
        if r.get('run') and not r.get('reason') and k is not None:
            self._select_range(k, *r['sel_x'])
            n = r['run'][1]-r['run'][0]+1
        else:
            k = next(iter(self.covering(lam)), None)
            if k is None: return self.show_line(row)
            self._select(k, lam)
            n = None
        self.refresh()
        self.table.selectRow(row)
        text = self.show_line(row) + ('' if n is None else f'; {n} window pixels selected')
        self.message(text)

    def clear_highlight(self):
        for it in self.items:
            if it in self.parent.plot.vb.addedItems: self.parent.plot.vb.removeItem(it)
        self.items = []

    def hideEvent(self, event):
        self.clear_highlight()
        super().hideEvent(event)

    def close_panel(self):
        """Close button: the same as F10."""
        self.parent.showBFLines(False)

    def import_lines(self, path=None):
        """Import lines with read-only BF and sigma from a catalogue CSV or a file written by Export CSV.
        Earlier imported rows are replaced; lines measured in the panel stay."""
        if path is None:
            start = option(self.parent, 'bf_catalogue_dir') or str(self.spectrum_dir() or '')
            path = QFileDialog.getOpenFileName(self, 'Import lines', start, 'CSV (*.csv);;All files (*)')[0]
            if not path: return None
        path = Path(path)
        self.parent.options('bf_catalogue_dir', str(path.parent))
        self.import_dir = path.parent
        on = option(self.parent, 'bf_import_levels').split()
        new = []
        for r in experiment.rows(path.parent, path.name):
            row = dict(species=r['species'], lambda_rest=r['lambda_rest'], f=r['f'], clarity=r['clarity'], BF=r['BF'],
                       sigma=r['sigma'], source='imported', exp=None, estimator=None, n=None, sigma_method='catalogue',
                       file_window=None)
            if 'use' in r:
                if r.get('estimator'): row.update(estimator=r['estimator'], n=int(r['n']), sigma_method=r['sigma_method'])
                if r.get('vmin_kms') and r.get('vmax_kms'): row['file_window'] = (float(r['vmin_kms']), float(r['vmax_kms']))
                row['use'] = r['use'] == 'True'
            else: row['use'] = r['use_in_fit'] == 'True' and r['clarity'] != 'blend' and (not on or r['species'] in on)
            est = f"{row['estimator']} N={row['n']}, " if row['estimator'] else ''
            row.update(window=row['file_window'], method=f"{path.name} ({est}sigma {row['sigma_method']})",
                       BF_imported=row['BF'], sigma_imported=row['sigma'])
            row['method_imported'] = row['method']
            new.append(row)
        self.rows = [r for r in self.rows if r['source'] != 'imported'] + new
        self.refresh()
        self.message(f'Import lines: {len(new)} lines from {path.name}')
        return path

    def export_csv(self, path=None):
        path = path or QFileDialog.getSaveFileName(self, 'Export BF table', 'bf_lines.csv', 'CSV (*.csv)')[0]
        if not path: return
        keys = ('species', 'lambda_rest', 'f', 'lambda_obs', 'BF', 'sigma', 'sigma_mc', 'mc_bias', 'sigma_catalogue',
                'clarity', 'use', 'used_in_fit', 'excluded_reason', 'warnings', 'source', 'estimator', 'n', 'sigma_method',
                'n_draws', 'seed', 'vmin_kms', 'vmax_kms', 'pixel_first', 'pixel_last', 'exposure', 'R', 'z')
        used = {id(r) for r in self.used_rows()}
        with open(path, 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n'); w.writerow(keys)
            for r in self.rows: w.writerow(self._export_row(r, id(r) in used))

    def _export_row(self, r, used):
        imported = r['source'] == 'imported' and not r.get('remeasured')
        m = {} if imported or r.get('reason') else r.get('meas', {})
        z = self.z()
        win = r.get('window', ('', '')) if r['source'] == 'imported' else m.get('window', r.get('window', ('', '')))
        run = m.get('window_pixels', ('', ''))
        return [r['species'], r['lambda_rest'], r['f'], f"{float(r['lambda_rest'])*(1+z):.4f}", r.get('BF', ''),
                r.get('sigma', ''), m.get('sigma_mc', ''), m.get('mc_bias', ''), m.get('sigma_catalogue', ''), r['clarity'],
                r['use'], used, r.get('reason', ''), '; '.join(self.warnings(r)), r['source'],
                m.get('estimator', (r.get('pix_est') or (r['estimator'], r['n']))[0] if imported else ''),
                m.get('n', (r.get('pix_est') or (r['estimator'], r['n']))[1] if imported else ''),
                m.get('sigma_method', r['sigma_method'] if imported else ''), m.get('n_draws', ''), m.get('seed', ''),
                *win, *run, '' if r.get('exp') is None else r['exp'], r.get('R', ''), z]

    def open_analysis(self):
        from .bf_analysis import BFAnalysisWindow
        if self.analysis is None: self.analysis = BFAnalysisWindow(self)
        self.parent._showToolWindow(self.analysis)
        self.analysis.show_data()
        return self.analysis
