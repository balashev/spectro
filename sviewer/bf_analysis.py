"""BF analysis window (Fit > BF analysis), Compare BF models dialog and the BF session export (HDF5).
User guide: docs/bf.rst.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import dataclasses
import subprocess

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QByteArray, QMarginsF, QRectF, QSizeF, Qt, QTimer
from PyQt6.QtGui import QColor, QFontMetrics, QPageSize, QPainter, QPdfWriter
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMenu, QMessageBox,
                             QPlainTextEdit, QPushButton, QStyle, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
                             QVBoxLayout, QWidget)

from ..bf import analysis as an
from ..bf import diagnostics as dg
from ..bf import experiment, session
from ..bf.atomic import PopulationGrid, fit_population_grid, population_problem
from ..bf.covering import LEVEL_LAWS, MODEL_FORMULAS, WAVELENGTH_LAWS, display_name, reference_wavelength
from ..bf.data import BFData
from ..bf.profiles import C_KMS
from .bf_panel import (NO_LEVEL, Cancelled, exclusion_text, fg, legend, level_key, option, progress_row, show_busy,
                       species_of, start_worker, style_plot, symbol)

MODELS = ('M0', 'M1', 'M2', 'M3', 'M4')
QUADRATURE = 32                         # Gauss-Hermite 32 nodes; None is adaptive
INTEGRATIONS = (('Gauss-Hermite 32 (default, fast)', 32, '1.09'), ('Gauss-Hermite 64', 64, '0.45'),
                ('Gauss-Hermite 128', 128, '0.15'), ('Gauss-Hermite 256', 256, '0.05'),
                ('Adaptive (slow)', None, '4e-12'))   # sampled max |Delta BF|/sigma of each quadrature
NARROW_B = 8.                           # km/s: below this the narrow-line switch uses adaptive integration
CHIANTI_FILE = 'chianti_P0.npz'
CHIANTI_TIP = (f'No {CHIANTI_FILE} in the atomic cache folder (Preferences > BF): a CHIANTI electron-collision '
               'population grid (built with ChiantiPy) is needed there.')
COLUMN_MODES = ((None, 'Free per level', 'free column density per level'),
                ('P0', 'Equilibrium: electron collisions', 'Pyratio P0'),
                ('P1', 'Equilibrium: collisions + CMB', 'Pyratio P1'),
                ('P2', 'Equilibrium: collisions + CMB + quasar radiation', 'Pyratio P2'),
                ('chianti_P0', 'Equilibrium: electron collisions (CHIANTI)', f'cached CHIANTI grid {CHIANTI_FILE}'))
COLUMN_NAMES = {code: text for code, text, _ in COLUMN_MODES}
COLUMN_SHORT = {None: 'free', 'P0': 'P0', 'P1': 'P1', 'P2': 'P2', 'chianti_P0': 'CHIANTI'}
WIDTHS = (('common_fixed', 'fixed common b'), ('species_fixed', 'fixed per level'), ('common_free', 'free common b'),
          ('individual_free', 'free per level'))
WIDTH_NAMES = dict(WIDTHS)
WIDTH_SHORT = {'common_fixed': 'b fixed', 'species_fixed': 'b per level', 'common_free': 'b free',
               'individual_free': 'b free per level'}
FIXED = ('common_fixed', 'species_fixed')
# likelihood-ratio significance: Delta chi2 follows chi2 with k dof (Wilks 1938, Ann. Math. Stat. 9, 60)
LR_TIP = ('<html>sqrt(&Delta;&chi;<sup>2</sup>) for 1 extra parameter; for k &gt; 1, the &chi;<sup>2</sup> probability '
          'with k dof converted to Gaussian sigma</html>')
MAP_CLIP = 25.
MAP_LEVELS = ((2.30, Qt.PenStyle.SolidLine, '68.3 %'), (6.18, Qt.PenStyle.DashLine, '95.4 %'),
              (11.83, Qt.PenStyle.DotLine, '99.73 %'))   # Delta chi2 for 2 parameters (Avni 1976, ApJ 210, 642)
CONTOUR_LEVELS = MAP_LEVELS[:2]
CONTOUR_NSIGMA = 5.
LOO_TOP, LOO_LINE = 15, 3.84                 # Delta chi2 = 3.84: 95 % for one parameter
FLAG_SYMBOLS = {'clear': 'o', 'unclear': 't', 'blend': 's'}
RUN_SYMBOLS = ('s', 't', 'd', 'p', 'h', 'star', 't1', 't2')
BLUE, RED = (31, 119, 180), (214, 39, 40)
BAR_HEIGHT = .6
LAMBDA_TIP = ('Reference wavelength lambda_0 of the wavelength term B_lambda*ln(lambda/lambda_0) (laws with a wavelength '
              'term). Auto: geometric mean of the rest wavelengths of the lines in the fit, rounded to 10 A. lambda_0 only '
              'reparametrizes the law: chi2, BIC and B_lambda do not change; the intercepts, i.e. C_j at lambda_0, do.')
LEVEL_B_TIP = ('Fixed b per level (km/s): from the fit model where it has the level (its component nearest to z), else the '
               'fixed common b. Edit a value to override it; the source column says where each value came from.')


def bf_data(rows):
    """BFData from panel rows (dicts: species, lambda_rest, f, BF, sigma, clarity, window); levels ordered by ion and index j;
    Transition.energy from the package level data (experiment.level_data; None for a level without data)."""
    if not rows: raise ValueError('No usable BF lines')
    present = tuple(sorted({r['species'] for r in rows}, key=level_key))
    energy = {l: (experiment.level_data(l) or {}).get('energy_cm1') for l in present}
    tr = tuple(experiment.transition(r, energy[r['species']]) for r in rows)
    return BFData(tr, [float(r['BF']) for r in rows], [float(r['sigma']) for r in rows],
                  tuple(r['clarity'] for r in rows), tuple(tuple(r['window']) for r in rows),
                  '; '.join(sorted({r.get('method', '') for r in rows})), dict(source='SPECTRO BF lines panel'),
                  tuple(r['clarity'] == 'blend' for r in rows), (None,)*len(rows), present)


def select_lines(data, keep):
    """BFData of the lines where `keep` is True; levels left without lines are dropped."""
    d = data.subset(np.flatnonzero(keep))
    present = {t.lower for t in d.transitions}
    return dataclasses.replace(d, levels=tuple(l for l in data.levels if l in present))


def fit_lines(data, models, coordinate='energy', grid=None):
    """(data of a run of the covering laws `models`, {line id: reason} of the lines it leaves out): lines without E_j
    for energy laws or equilibrium columns, lines whose level is not in `grid`. ValueError if no line is left."""
    energy = grid is not None or (coordinate == 'energy' and any(m in LEVEL_LAWS for m in models))
    ids = None if grid is None else {str(l['id']) for l in grid.metadata['levels']}
    out = {}
    for t in data.transitions:
        if energy and t.energy is None: out[t.id] = 'no level data (E_j) in SPECTRO'
        elif ids is not None and t.lower not in ids: out[t.id] = 'level not in the population grid'
    if not out: return data, out
    if len(out) == len(data.transitions): raise ValueError(f'No line left for this run: {exclusion_text(out)}')
    return select_lines(data, [t.id not in out for t in data.transitions]), out


def lambda_attr(lambda_0):
    """Session attribute of a reference wavelength: the value in Angstrom, NaN for laws without a wavelength term."""
    return np.nan if lambda_0 is None else float(lambda_0)


def cj_label(lambda_0):
    """Axis label of C_j: 'C_j at lambda_0 = <value> A' for a law with a wavelength term (lambda_0 not None), else 'C_j'."""
    return 'C<sub>j</sub>' if lambda_0 is None else f'C<sub>j</sub> at &lambda;<sub>0</sub> = {lambda_0:g} &Aring;'


def integration_name(quadrature):
    return 'adaptive' if quadrature is None else f'GH{quadrature}'


def run_fit(data, model, width, b, grid=None, resolving_power=None, coordinate='energy', quadrature=QUADRATURE,
            level_b=None, lambda_0='auto', reference=True):
    """Empirical fit, or a population-grid fit (fixed widths only) if `grid` is given: forward, problem, result.
    reference: also fit M0 for the likelihood-ratio significance; quadrature: Gauss-Hermite nodes, None for adaptive."""
    kw = dict(b=b, quadrature=quadrature, resolving_power=resolving_power, level_b=level_b, lambda_0=lambda_0)
    if grid is None:
        forward, problem, result = an.fit_case(model, width, coordinate, data=data, **kw)
        out = dict(kind='empirical', forward=forward, problem=problem, result=result, data=data, model=model, width=width)
        if reference and model != 'M0':
            r0 = an.fit_case('M0', width, coordinate, data=data, **kw)[2]
            dof = result['k']-r0['k']
            out.update(lr_sigma=an.lr_sigma(r0['chi2']-result['chi2'], dof), lr_dof=dof)
        return out
    if width not in FIXED:
        raise ValueError('Equilibrium columns need a fixed width (common or per level)')
    forward, base = experiment.make_problem(model, width, data=data, coordinate=coordinate, **kw)
    fixed_b = base.metadata['fixed_b_kms']
    a = fit_population_grid(forward, grid, fixed_b=fixed_b)
    problem = population_problem(forward, grid.populations[tuple(a['best_index'])], [l['id'] for l in grid.metadata['levels']],
                                 fixed_b=fixed_b)
    return dict(kind='grid', forward=forward, problem=problem, result=a['best_fit'], grid_fit=a, grid=grid, data=data,
                model=model, width=width)


def refit(data, out):
    """(forward, problem, result) of the model of fit output `out` on `data` with the same settings out['kw'], including
    the numeric lambda_0 of the full fit (equilibrium columns: at its best node)."""
    grid = node_grid(out['grid'], out['grid_fit']) if out['kind'] == 'grid' else None
    o = run_fit(data, out['model'], out['width'], grid=grid, reference=False, **out['kw'])
    return o['forward'], o['problem'], o['result']


def fitted_b(out):
    """Fitted widths (km/s) of a fit output, or None for fixed widths."""
    r = out['result']
    i = [k for k, n in enumerate(r['names']) if n.startswith('log_b')]
    return 10**np.asarray(r['parameters'])[i] if i else None


def node_grid(grid, afit):
    """One-node PopulationGrid at the best node of a grid fit."""
    idx = tuple(afit['best_index'])
    return PopulationGrid({k: v[[i]] for (k, v), i in zip(grid.axes.items(), idx)},
                          grid.populations[tuple(slice(i, i+1) for i in idx)], grid.metadata)


def load_grid(stage, paths, progress, cancelled):
    """Population grid of `stage`: CHIANTI from the cache file, Pyratio from (or into) the cache.
    paths: BFAnalysisWindow.grid_paths()."""
    cache, redshift, agn = paths
    if cache is None: raise ValueError('No atomic cache folder: set Preferences > BF')
    if stage == 'chianti_P0':
        path = Path(cache)/CHIANTI_FILE
        if not path.is_file(): raise ValueError(CHIANTI_TIP)
        return PopulationGrid.load(path)
    from ..bf.pyratio_grid import grid, levels

    def tick(i, n):
        if cancelled(): raise Cancelled('cancelled')
        progress.emit((0, int(100*i/n)))
    return grid(stage, redshift=redshift, agn=agn, progress=tick, cache=Path(cache), levs=levels())


def errors(result):
    cov = result.get('covariance')
    return np.full(len(result['parameters']), np.nan) if cov is None else np.sqrt(np.maximum(np.diag(cov), 0.))


def parameter_value(result, name):
    """(value, 1 sigma) of a named parameter."""
    i = list(result['names']).index(name)
    return float(result['parameters'][i]), float(errors(result)[i])


def contour_lines(x, y, z, level):
    """Contour lines of z[iy, ix] at `level` in data coordinates: list of (N, 2) arrays."""
    import contourpy
    return contourpy.contour_generator(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)).lines(level)


def residual_bands(plot):
    """Solid line at 0 and light bands at +-1 and +-2."""
    for half, alpha in ((2, 28), (1, 40)):
        band = pg.LinearRegionItem((-half, half), orientation='horizontal', movable=False,
                                   brush=pg.mkBrush(120, 120, 120, alpha), pen=pg.mkPen(None))
        band.setZValue(-10); plot.addItem(band)
    plot.addLine(y=0, pen=pg.mkPen(fg(), width=1))


def note(plot, text):
    """Grey hint in the middle of an otherwise empty plot."""
    t = pg.TextItem(text, color=(128, 128, 128), anchor=(0.5, 0.5))
    plot.addItem(t, ignoreBounds=True)
    (x0, x1), (y0, y1) = plot.viewRange()
    t.setPos((x0+x1)/2, (y0+y1)/2)


def auto_range(*plots):
    for p in plots: p.enableAutoRange(); p.autoRange()


def save_figure(parent, item, name, path=None):
    """PNG (pyqtgraph ImageExporter, 2x size) or vector PDF (pyqtgraph SVGExporter rendered with QtSvg) of a plot item or
    of a whole GraphicsLayoutWidget scene. Returns the path, or None if the dialog was cancelled."""
    from pyqtgraph.exporters import ImageExporter, SVGExporter
    path = path or QFileDialog.getSaveFileName(parent, 'Save figure', f'{name}.png', 'PNG (*.png);;PDF (*.pdf)')[0]
    if not path: return None
    path = str(path)
    if path.lower().endswith('.pdf'):
        from PyQt6.QtSvg import QSvgRenderer
        e = SVGExporter(item)
        background = QColor(e.parameters()['background'])
        e.parameters()['background'] = QColor(0, 0, 0, 0)   # QtSvg reads the exporter's width="100%" background as 100 px
        svg = QSvgRenderer(QByteArray(e.export(toBytes=True)))
        pdf = QPdfWriter(path); pdf.setResolution(72)
        pdf.setPageSize(QPageSize(QSizeF(svg.defaultSize()), QPageSize.Unit.Point)); pdf.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(pdf)
        page = QRectF(0, 0, pdf.width(), pdf.height())
        painter.fillRect(page, background); svg.render(painter, page); painter.end()
    else:
        e = ImageExporter(item)
        e.parameters()['width'] = int(2*e.parameters()['width'])
        e.export(path)
    return path


def save_button(owner, entries):
    """'Save figure' button; entries: (menu text, item getter, default file name), the first one for all plots."""
    button = QPushButton('Save figure'); menu = QMenu(button)
    for text, item, name in entries:
        menu.addAction(text).triggered.connect(lambda _=False, item=item, name=name: save_figure(owner, item(), name))
    button.setMenu(menu)
    return button


def columns_box():
    """Columns selector: free per level or equilibrium populations (Pyratio P0-P2, CHIANTI)."""
    box = QComboBox()
    for i, (code, text, tip) in enumerate(COLUMN_MODES):
        box.addItem(text, code); box.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
    box.setToolTip('Equilibrium columns: population grids from the atomic cache (Preferences > BF), fixed widths only')
    return box


def enable_chianti(box, cache):
    """Enable the CHIANTI item only if its grid file is in the cache folder; otherwise say how to build it."""
    i = box.findData('chianti_P0')
    ok = cache is not None and (Path(cache)/CHIANTI_FILE).is_file()
    box.model().item(i).setEnabled(ok)
    box.setItemData(i, COLUMN_MODES[i][2] if ok else CHIANTI_TIP, Qt.ItemDataRole.ToolTipRole)
    if not ok and box.currentIndex() == i: box.setCurrentIndex(0)


def lambda_widgets(window, auto_value):
    """Reference wavelength row [Auto | Manual] [value] A bound to the one lambda_0 setting of the analysis window `window`
    (BFAnalysisWindow.set_lambda_0); auto_value() is the Auto value shown (None without lines)."""
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
    w.mode = QComboBox(); w.mode.addItems(['Auto', 'Manual'])
    w.value = QDoubleSpinBox(); w.value.setRange(100., 100000.); w.value.setDecimals(1); w.value.setSingleStep(10.)
    w.auto_value = auto_value
    for x in (w.mode, w.value): h.addWidget(x)
    unit = QLabel('&Aring;'); unit.setTextFormat(Qt.TextFormat.RichText)
    h.addWidget(unit); h.addStretch(1)
    w.setToolTip(LAMBDA_TIP)
    w.mode.currentIndexChanged.connect(lambda i: window.set_lambda_0(None if i == 0 else w.value.value()))
    w.value.valueChanged.connect(lambda x: window.set_lambda_0(x) if w.mode.currentIndex() == 1 else None)
    window.lambda_rows.append(w)
    return w


def text_table(table, rows, header):
    """Fill a QTableWidget with strings."""
    table.clear(); table.setColumnCount(len(header)); table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels(header)
    for i, row in enumerate(rows):
        for c, v in enumerate(row): table.setItem(i, c, QTableWidgetItem(str(v)))
    table.resizeColumnsToContents()


class BFAnalysisWindow(QWidget):
    """Fit > BF analysis: covering law, width, columns and integration; fit result, plots, session."""

    def __init__(self, lines):
        super().__init__()
        self.lines, self.output, self.compare, self._worker, self._run = lines, None, None, None, 0
        self.runs, self._count, self.map_bar = [], 0, None
        self.lambda_rows, self._lambda_0, self._level_b_edit = [], None, {}
        self.setWindowTitle('BF analysis')
        h = QHBoxLayout(self); left = QVBoxLayout()
        left.addLayout(self._inputs())
        buttons = QHBoxLayout()
        self.fit_button = QPushButton('Fit'); self.fit_button.clicked.connect(lambda: self.start_fit())
        self.compare_button = QPushButton('Compare BF models'); self.compare_button.clicked.connect(self.open_compare)
        self.check_button = QPushButton('Check lines'); self.check_button.clicked.connect(lambda: self.run_check())
        self.check_button.setToolTip('Refit without each used line (same model) and flag lines the others do not predict')
        self.tabs = QTabWidget()
        self.save_button = save_button(self, (('Current tab', self.tab_scene, 'bf_analysis'), ('BF', lambda: self.p_bf, 'bf_fit'),
            ('Residuals', lambda: self.p_res, 'bf_residuals'), ('Delta chi2 map', lambda: self.p_map, 'bf_chi2_map')))
        for b in (self.fit_button, self.compare_button, self.check_button, self.save_button): buttons.addWidget(b)
        left.addLayout(buttons)
        left.addLayout(progress_row(self))
        self.results = QTableWidget(0, 4); self.results.setHorizontalHeaderLabels(['parameter', 'value', '1 sigma', 'bound'])
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results.verticalHeader().setVisible(False)
        left.addWidget(self.results, 2)
        self.summary = QLabel(''); self.summary.setWordWrap(True); left.addWidget(self.summary)
        self.status = QLabel(''); self.status.setWordWrap(True); left.addWidget(self.status)
        left.addWidget(self._session_box(), 2)
        for name, widget in (('Fit', self._fit_tab()), ('Curve of growth', self._cog_tab()),
                             ('Columns and covering', self._columns_tab()), ('Covering vs wavelength', self._covering_tab()),
                             ('Check lines', self._check_tab())):
            self.tabs.addTab(widget, name)
        self.tabs.currentChanged.connect(lambda i: self.redraw())
        h.addLayout(left, 2); h.addWidget(self.tabs, 3)
        self.resize(1320, 860)

    def _inputs(self):
        form = QFormLayout()
        self.model = QComboBox()
        for i, code in enumerate(MODELS):
            self.model.addItem(display_name(code), code)
            self.model.setItemData(i, f'{code}: {MODEL_FORMULAS[code]}', Qt.ItemDataRole.ToolTipRole)
        self.model.setCurrentIndex(4)
        form.addRow('Covering', self.model)
        self.lambda_box = lambda_widgets(self, self.auto_lambda_0)
        form.addRow('Reference wavelength', self.lambda_box)
        self.width_box = QComboBox()
        for code, text in WIDTHS: self.width_box.addItem(text, code)
        form.addRow('Width', self.width_box)
        self._b = option(self.lines.parent, 'bf_b_default', float)
        self.b = QDoubleSpinBox(); self.b.setRange(3, 100); self.b.setValue(self._b); self.b.setSuffix(' km/s')
        self.b.valueChanged.connect(self._b_changed)
        form.addRow('b (fixed common)', self.b)
        self.level_b_table = QTableWidget(0, 3)
        self.level_b_table.setHorizontalHeaderLabels(['level', 'b [km/s]', 'source'])
        self.level_b_table.verticalHeader().setVisible(False)
        self.level_b_table.horizontalHeader().setStretchLastSection(True)
        self.level_b_table.setToolTip(LEVEL_B_TIP)
        self.level_b_table.itemChanged.connect(self._level_b_edited)
        self.level_b_label = QLabel('b per level')
        form.addRow(self.level_b_label, self.level_b_table)
        self.columns = columns_box()
        form.addRow('Columns', self.columns)
        self.integration = QComboBox()
        for i, (text, q, acc) in enumerate(INTEGRATIONS):
            self.integration.addItem(text, q)
            self.integration.setItemData(i, f'<html>Sampled accuracy: max |&Delta;BF| = {acc} '
                                            '&sigma;</html>', Qt.ItemDataRole.ToolTipRole)
        form.addRow('Integration', self.integration)
        self.auto_adaptive = QCheckBox(f'Switch to adaptive for narrow lines (b < {NARROW_B:.0f} km/s)')
        self.auto_adaptive.setChecked(True)
        self.auto_adaptive.setToolTip('Fixed b below 8 km/s: fit with adaptive integration; fitted b below 8 km/s: refit '
                                      'with adaptive integration')
        form.addRow('', self.auto_adaptive)
        self.model.currentIndexChanged.connect(self._enable)
        self.width_box.currentIndexChanged.connect(self._enable)
        self.columns.currentIndexChanged.connect(self._enable)
        self.integration.currentIndexChanged.connect(
            lambda i: self.integration.setToolTip(self.integration.itemData(i, Qt.ItemDataRole.ToolTipRole)))
        self.integration.setToolTip(self.integration.itemData(0, Qt.ItemDataRole.ToolTipRole))
        self._enable()
        return form

    def _b_changed(self, value):
        if self.width_box.currentData() == 'common_fixed': self._b = value

    def common_b(self): return self._b

    def _enable(self):
        """Free widths only with free columns; the b box only for fixed common b ('b fitted' / 'per level' otherwise), the
        b per level table only for fixed per level; the reference wavelength only for laws with a wavelength term."""
        eq = self.columns.currentData() is not None
        for i, (code, _) in enumerate(WIDTHS): self.width_box.model().item(i).setEnabled(not eq or code in FIXED)
        if eq and self.width_box.currentData() not in FIXED: return self.width_box.setCurrentIndex(0)
        w = self.width_box.currentData()
        self.b.blockSignals(True)
        if w == 'common_fixed':
            self.b.setSpecialValueText(''); self.b.setMinimum(3); self.b.setValue(self._b)
            self.b.setToolTip('Common Doppler parameter of all levels')
        else:
            self.b.setMinimum(0); self.b.setSpecialValueText('per level' if w == 'species_fixed' else 'b fitted')
            self.b.setValue(0)
            self.b.setToolTip('Fixed b per level: see the b per level table' if w == 'species_fixed' else
                              'b is a fit parameter for this width')
        self.b.blockSignals(False)
        self.b.setEnabled(w == 'common_fixed')
        for x in (self.level_b_label, self.level_b_table): x.setVisible(w == 'species_fixed')
        if w == 'species_fixed': self.fill_level_b()
        self.lambda_box.setEnabled(self.model.currentData() in WAVELENGTH_LAWS)
        self.show_lambda_0()

    def settings(self):
        return dict(model=self.model.currentData(), width=self.width_box.currentData(), b=self._b)

    def level_b_sources(self, levels):
        """{level: (b km/s, source)}: an edited value, else the fit model's b of the level (BFLinesPanel.model_b), else the
        fixed common b."""
        model, out = self.lines.model_b(), {}
        for l in levels:
            if l in self._level_b_edit: out[l] = (self._level_b_edit[l], 'edited')
            elif l in model: out[l] = (model[l][0], f'fit model, z = {model[l][1]:.6f}')
            else: out[l] = (self._b, 'fixed common b')
        return out

    def level_b(self, levels):
        """{level: b km/s} of the b per level table for `levels`."""
        return {l: b for l, (b, _) in self.level_b_sources(levels).items()}

    def fill_level_b(self):
        """b per level table for the levels of the used lines."""
        try: levels = self.lines.data().levels
        except ValueError: levels = ()
        t = self.level_b_table
        t.blockSignals(True); t.setRowCount(len(levels))
        for i, (l, (b, src)) in enumerate(self.level_b_sources(levels).items()):
            for c, text in enumerate((l, f'{b:g}', src)):
                it = QTableWidgetItem(text); it.setToolTip(f'{l}: b = {b:g} km/s from {src}')
                if c != 1: it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(i, c, it)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + max(len(levels), 1)*t.verticalHeader().defaultSectionSize()
                         + 2*t.frameWidth())

    def _level_b_edited(self, item):
        """A b typed in the table (3-100 km/s, the fitter's width bounds) overrides the level's value; other input reverts."""
        if item.column() != 1: return
        level = self.level_b_table.item(item.row(), 0).text()
        try: b = float(item.text())
        except ValueError: b = np.nan
        if 3. <= b <= 100.: self._level_b_edit[level] = b
        else: self.lines.message(f'b per level: {item.text()!r} is not a width in 3-100 km/s')
        QTimer.singleShot(0, self.fill_level_b)

    def set_lambda_0(self, value):
        """Manual reference wavelength in Angstrom, or None for Auto; shown in every linked control."""
        self._lambda_0 = value
        self.show_lambda_0()

    def show_lambda_0(self):
        for w in self.lambda_rows:
            for x in (w.mode, w.value): x.blockSignals(True)
            w.mode.setCurrentIndex(0 if self._lambda_0 is None else 1)
            v = w.auto_value() if self._lambda_0 is None else self._lambda_0
            if v is not None: w.value.setValue(v)
            w.value.setEnabled(self._lambda_0 is not None)
            for x in (w.mode, w.value): x.blockSignals(False)

    def lambda_0(self, data):
        """Reference wavelength of a fit on `data`: the manual value, else Auto on these lines."""
        return reference_wavelength(data) if self._lambda_0 is None else self._lambda_0

    def auto_lambda_0(self, models=None):
        """Auto reference wavelength of the lines a fit of `models` (default: the Covering box) would use, or None."""
        try: return reference_wavelength(fit_lines(self.lines.data(), models or (self.model.currentData(),))[0])
        except ValueError: return None

    def integration_choice(self, widths, data):
        """(quadrature, note) from the Integration box; adaptive if the narrow-line switch applies to a fixed width."""
        q = self.integration.currentData()
        if q is None or not self.auto_adaptive.isChecked(): return q, ''
        b = [self._b] if 'common_fixed' in widths else []
        if 'species_fixed' in widths: b += list(self.level_b(data.levels).values())
        if b and min(b) < NARROW_B:
            return None, f'b = {min(b):.1f} km/s is below {NARROW_B:.0f} km/s: switched to adaptive integration'
        return q, ''

    def grid_paths(self):
        """(atomic cache folder, absorber z, AGN normalization) for load_grid, read in the GUI thread."""
        p = self.lines.parent
        agn = dict(filter=option(p, 'bf_agn_filter'), mag=option(p, 'bf_agn_mag', float))
        return self.lines.atomic_cache(), self.lines.z(), agn

    def _fit_tab(self):
        plots = pg.GraphicsLayoutWidget()
        self.p_bf = style_plot(plots.addPlot(row=0, col=0), 'BF', 'log f&lambda; [&Aring;]')
        self.p_res = style_plot(plots.addPlot(row=1, col=0), '(BF - model)/&sigma;', 'log f&lambda; [&Aring;]')
        self.p_res.setXLink(self.p_bf)
        self.p_map = style_plot(plots.addPlot(row=2, col=0), 'log T [K]', 'log n<sub>e</sub> [cm<sup>-3</sup>]',
                                '&Delta;&chi;<sup>2</sup>(n<sub>e</sub>, T)')
        return plots

    def _cog_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.cog_view = QComboBox(); self.cog_view.addItems(['BF vs f lambda (curve of growth)', 'BF vs rest wavelength'])
        self.cog_view.currentIndexChanged.connect(lambda i: self.plot_cog())
        self.cog_note = QLabel(''); self.cog_note.setWordWrap(True)
        top.addWidget(self.cog_view); top.addWidget(self.cog_note, 1)
        self.cog_plots = pg.GraphicsLayoutWidget()
        v.addLayout(top); v.addWidget(self.cog_plots, 1)
        return w

    def _columns_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.contour_button = QPushButton('Contours'); self.contour_button.clicked.connect(lambda: self.run_contours())
        self.contour_button.setToolTip('(log N/g, C) Delta chi2 grids of Covering per level with free per-level columns')
        self.columns_note = QLabel(''); self.columns_note.setWordWrap(True)
        top.addWidget(self.contour_button); top.addWidget(self.columns_note, 1)
        plots = pg.GraphicsLayoutWidget()
        self.p_cc = style_plot(plots.addPlot(row=0, col=0), 'C<sub>j</sub>', 'log N<sub>j</sub>/g<sub>j</sub> [cm<sup>-2</sup>]',
                               '&Delta;&chi;<sup>2</sup> = 2.30, 6.18 (Covering per level)')
        self.p_exc = style_plot(plots.addPlot(row=0, col=1), 'log N<sub>j</sub>/g<sub>j</sub> [cm<sup>-2</sup>]',
                                'E<sub>j</sub> [cm<sup>-1</sup>]', 'Excitation diagram')
        v.addLayout(top); v.addWidget(plots, 1)
        return w

    def _covering_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.depth = QDoubleSpinBox(); self.depth.setRange(0., .99); self.depth.setSingleStep(.05); self.depth.setDecimals(2)
        self.depth.setValue(option(self.lines.parent, 'bf_cover_depth', float))
        self.depth.valueChanged.connect(lambda x: self.plot_covering())
        self.m2_button = QPushButton(f"Fit {display_name('M2')}"); self.m2_button.clicked.connect(lambda: self.start_fit('M2'))
        self.m2_button.setToolTip('M2 fit with the current width and free columns (added to the session)')
        self.covering_note = QLabel(''); self.covering_note.setWordWrap(True)
        top.addWidget(QLabel('show lines with 1 - F >')); top.addWidget(self.depth); top.addWidget(self.m2_button)
        top.addWidget(self.covering_note, 1)
        plots = pg.GraphicsLayoutWidget()
        self.p_cv = style_plot(plots.addPlot(row=0, col=0), 'implied C<sub>i</sub> = (1 - BF<sub>i</sub>)/(1 - F<sub>i</sub>)',
                               '&lambda;<sub>rest</sub> [&Aring;]')
        v.addLayout(top); v.addWidget(plots, 1)
        return w

    def _check_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.check_note = QLabel('Fit, then press Check lines.'); self.check_note.setWordWrap(True)
        self.check_table = QTableWidget(0, 0)
        self.check_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.check_table.verticalHeader().setVisible(False)
        v.addWidget(self.check_note); v.addWidget(self.check_table, 1)
        return w

    def tab_scene(self):
        """Plot scene of the current tab (the Fit tab for tabs without plots), for Save figure."""
        plots = self.tabs.currentWidget().findChild(pg.GraphicsLayoutWidget) or self.tabs.widget(0)
        return plots.scene()

    def redraw(self):
        """Redraw the current tab from the current output (or the data alone)."""
        name = self.tabs.tabText(self.tabs.currentIndex())
        try:
            if name == 'Curve of growth': self.plot_cog()
            elif name == 'Columns and covering': self.plot_columns()
            elif name == 'Covering vs wavelength': self.plot_covering()
        except ValueError as exc: self.status.setText(str(exc))

    def _session_box(self):
        box = QGroupBox('Session'); v = QVBoxLayout(box)
        self.session = QTableWidget(0, 3)
        self.session.horizontalHeader().setVisible(False); self.session.verticalHeader().setVisible(False)
        self.session.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.session.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.session.setToolTip('Completed runs; ticked runs are exported')
        h = QHBoxLayout()
        clear = QPushButton('Clear session'); clear.clicked.connect(self.clear_session)
        export = QPushButton('Export BF session'); export.clicked.connect(lambda: self.export_session())
        h.addWidget(clear); h.addStretch(1); h.addWidget(export)
        v.addWidget(self.session, 1); v.addLayout(h)
        return box

    def add_run(self, kind, tail, **payload):
        self._count += 1
        run = dict(name=f'{self._count:02d} {kind} {tail}'.strip(), kind=kind, checked=True, **payload)
        self.runs.append(run)
        self.fill_session()
        return run

    def fill_session(self):
        t = self.session
        t.setRowCount(len(self.runs))
        for i, run in enumerate(self.runs):
            box = QCheckBox(); box.setChecked(run['checked'])
            box.toggled.connect(lambda on, run=run: run.update(checked=on))
            t.setCellWidget(i, 0, box)
            t.setItem(i, 1, QTableWidgetItem(run['name']))
            delete = QToolButton(); delete.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
            delete.setToolTip('Delete this run'); delete.clicked.connect(lambda _=False, run=run: self.delete_run(run))
            t.setCellWidget(i, 2, delete)
        t.resizeColumnToContents(0); t.resizeColumnToContents(2)

    def delete_run(self, run):
        self.runs.remove(run)
        self.fill_session(); self.redraw()

    def clear_session(self):
        self.runs = []
        self.fill_session(); self.redraw()

    def latest(self, kind, model=None):
        return next((r for r in reversed(self.runs) if r['kind'] == kind and (model is None or r['out']['model'] == model)), None)

    @staticmethod
    def level_info(levels):
        """({level: E_j cm^-1}, {level: g_j}) of `levels` from the package level data (NaN for a level without data)."""
        lv = {l: experiment.level_data(l) or dict(energy_cm1=np.nan, g=np.nan) for l in levels}
        return {l: v['energy_cm1'] for l, v in lv.items()}, {l: v['g'] for l, v in lv.items()}

    def report_excluded(self, name, excluded):
        """Lines a run left out ({line id: reason}): footer of the lines panel and the viewer's status bar."""
        self.lines.set_excluded(name, excluded)
        if excluded: self.lines.message(f'{name}: excluded {exclusion_text(excluded)}')

    def add_fit_run(self, out):
        """Session entry of a completed fit (empirical: 'fit'; equilibrium columns: 'equilibrium') with its level table."""
        e, g = self.level_info(out['forward'].data.levels)
        stage = out.get('columns')
        kind = 'equilibrium' if out['kind'] == 'grid' else 'fit'
        tail = f"{out['model']} {WIDTH_SHORT[out['width']]} {COLUMN_SHORT[stage]}"
        run = self.add_run(kind, tail, out=out, levels=dg.level_table(out['forward'], out['problem'], out['result'], e, g))
        self.report_excluded(run['name'], out.get('excluded', {}))
        return run

    def failed(self, text):
        self._busy(False)
        self.status.setText(f'Error: {text}')
        QMessageBox.warning(self, 'BF analysis', text)

    def _busy(self, on, determinate=False):
        for b in (self.fit_button, self.compare_button, self.check_button, self.contour_button, self.m2_button):
            b.setEnabled(not on)
        show_busy(self, on, determinate)

    def cancel_run(self):
        self._run += 1
        self._busy(False)
        self.status.setText('Cancelled')

    def _progress(self, p):
        """Worker progress (thread id, percent); -1: grid ready, fit running."""
        if p[1] < 0: self.progress.setRange(0, 0)
        else: self.progress.setRange(0, 100); self.progress.setValue(p[1])

    def start_fit(self, model=None):
        """Fit in a worker thread with the Integration box and the narrow-line switch. model: covering law (None: the
        Covering box; 'M2' uses free columns)."""
        try: data, R = self.lines.data(), self.lines.resolving_power()
        except ValueError as exc: return self.failed(str(exc))
        s, stage = self.settings(), self.columns.currentData()
        if model is not None: s['model'], stage = model, None
        if stage and s['width'] not in FIXED:
            return self.failed('Equilibrium columns need a fixed width (common or per level)')
        try: data, excluded = fit_lines(data, (s['model'],), 'energy')
        except ValueError as exc: return self.failed(str(exc))
        q, note = self.integration_choice((s['width'],), data)
        refit_adaptive = self.auto_adaptive.isChecked()
        paths, manual = self.grid_paths(), self._lambda_0
        kw = dict(resolving_power=R, coordinate='energy', b=s['b'], level_b=self.level_b(data.levels))

        def job(progress, cancelled):
            grid, d, left_out = None, data, dict(excluded)
            if stage:
                grid = load_grid(stage, paths, progress, cancelled)
                progress.emit((0, -1))
                d, more = fit_lines(data, (s['model'],), 'energy', grid)
                left_out.update(more)
            k = dict(kw, lambda_0=reference_wavelength(d) if manual is None else manual)
            out = dict(run_fit(d, s['model'], s['width'], grid=grid, quadrature=q, **k), integration=integration_name(q),
                       note=note, columns=stage, kw=dict(k, quadrature=q), excluded=left_out)
            b = fitted_b(out)
            if refit_adaptive and q is not None and b is not None and b.min() < NARROW_B:
                if cancelled(): raise Cancelled('cancelled')
                out = dict(run_fit(d, s['model'], s['width'], grid=grid, quadrature=None, **k), integration='adaptive',
                           columns=stage, kw=dict(k, quadrature=None), excluded=left_out,
                           note=f'fitted b = {b.min():.1f} km/s is below {NARROW_B:.0f} km/s: refitted with adaptive '
                                f'integration ({integration_name(q)} first)')
            return out
        self._busy(True, determinate=bool(stage))
        self.status.setText(f"Fitting {display_name(s['model'])}" + (f', {COLUMN_NAMES[stage]}' if stage else '') +
                            f', {integration_name(q)}...' + (f' {note}' if note else ''))
        if note: self.lines.message(note)
        start_worker(self, job, lambda out: (self.add_fit_run(out), self.show_fit(out)), self._progress)

    def show_fit(self, out):
        self._busy(False)
        self.output = out
        r = out['result']
        err = errors(r)
        self.results.setRowCount(len(r['names']))
        for i, (n, x, e, bnd) in enumerate(zip(r['names'], r['parameters'], err, r['boundary'])):
            for c, val in enumerate((n, f'{x:.4f}', f'{e:.4f}' if np.isfinite(e) else '-', 'at bound' if bnd else '')):
                self.results.setItem(i, c, QTableWidgetItem(val))
        self.summary.setText(self._summary(out))
        text = f"{display_name(out['model'])}, {WIDTH_NAMES[out['width']]}"
        if out.get('columns'): text += f", {COLUMN_NAMES[out['columns']]}"
        if out.get('integration'): text += f", {out['integration']}"
        if out.get('excluded'): text += f"; excluded {exclusion_text(out['excluded'])}"
        self.status.setText(text + ': done' + (f"; {out['note']}" if out.get('note') else ''))
        if out.get('note'): self.lines.message(out['note'])
        self.plot_fit(out)
        self.redraw()

    @staticmethod
    def _summary(out):
        r, grid = out['result'], out['kind'] == 'grid'
        k, bic = (out['grid_fit']['k'], out['grid_fit']['bic']) if grid else (r['k'], r['bic'])
        text = f"&chi;<sup>2</sup> = {r['chi2']:.6f}, n = {r['n']}, k = {k}, BIC = {bic:.3f}"
        if grid:
            node = an.best_node(out['grid'], out['grid_fit'])
            text += (f"<br>best node: T = {node['temperature']:.0f} K, n<sub>e</sub> = {node['electron_density']:.3g} "
                     f"cm<sup>-3</sup>" + (f", log rad = {node['log_radiation']:g}" if 'log_radiation' in node else '') +
                     ' (grid axes counted in k)')
        b = fitted_b(out)
        if b is not None: text += '<br>fitted b = ' + ', '.join(f'{x:.1f}' for x in b) + ' km/s'
        lam0 = out['forward'].covering.lambda_0
        if lam0 is not None: text += f'<br>&lambda;<sub>0</sub> = {lam0:g} &Aring; (intercepts: {cj_label(lam0)})'
        if 'lr_sigma' in out:
            text += (f"<br>LR significance vs Single covering: {out['lr_sigma']:.2f} &sigma; "
                     f"(&Delta;&chi;<sup>2</sup> on {out['lr_dof']} dof)")
        return text

    def show_data(self):
        """Data points of the used lines (no model); plots are never empty once lines exist."""
        cache = self.lines.atomic_cache()
        enable_chianti(self.columns, cache)
        if self.compare is not None: enable_chianti(self.compare.columns, cache)
        try: data = self.lines.data()
        except ValueError as exc:
            for p in (self.p_bf, self.p_res, self.p_map): p.clear()
            return self.status.setText(str(exc))
        self.output = None
        self.results.setRowCount(0); self.summary.setText('')
        self.plot_fit(dict(data=data, result=None, kind='data'))
        self.redraw()
        if self.width_box.currentData() == 'species_fixed': self.fill_level_b()
        self.show_lambda_0()
        self.status.setText(f'{len(data.intensity)} lines in {len(data.levels)} levels; press Fit')

    def current(self):
        """The shown fit output, or the data alone."""
        return self.output or dict(data=self.lines.data(), result=None, kind='data')

    def plot_fit(self, out):
        d, r = out['data'], out['result']
        x, y, s = np.log10(d.flambda), d.intensity, d.uncertainty
        pred = None if r is None else r['prediction']
        for p in (self.p_bf, self.p_res, self.p_map): p.clear()
        for j, level in enumerate(d.levels):
            m = d.j == j; o = np.argsort(x[m]); c = self.lines.color(level)
            self.p_bf.addItem(pg.ErrorBarItem(x=x[m], y=y[m], height=2*s[m], pen=c))
            self.p_bf.plot(x[m], y[m], pen=None, symbol='o', symbolBrush=c, symbolPen=None)
            if pred is None: continue
            self.p_bf.plot(x[m][o], pred[m][o], pen=pg.mkPen(c, width=1.5))
            self.p_res.plot(x[m], (y[m]-pred[m])/s[m], pen=None, symbol='o', symbolBrush=c, symbolPen=None)
        residual_bands(self.p_res)
        legend(self.p_bf, [(symbol(self.lines.color(l)), l) for l in d.levels], offset=(5, -5))
        auto_range(self.p_bf, self.p_res)
        if pred is None: note(self.p_res, 'residuals after Fit')
        if out['kind'] == 'grid': self._show_map(out)
        else:
            if self.map_bar is not None: self.map_bar.setVisible(False)
            legend(self.p_map, [])
            note(self.p_map, 'map after a Fit with equilibrium columns')

    def _show_map(self, out):
        m = an.chi2_map(out['grid'], out['grid_fit'])
        lt, ln, d = m['log_T'], m['log_ne'], m['delta_chi2']
        dx = (ln[-1]-ln[0])/(len(ln)-1) if len(ln) > 1 else 1.
        dy = (lt[-1]-lt[0])/(len(lt)-1) if len(lt) > 1 else 1.
        cmap = pg.colormap.get('viridis')
        img = pg.ImageItem(np.clip(d, 0, MAP_CLIP).T, levels=(0, MAP_CLIP))   # pixel centres at the grid nodes
        img.setRect(QRectF(ln[0]-dx/2, lt[0]-dy/2, ln[-1]-ln[0]+dx, lt[-1]-lt[0]+dy))
        img.setColorMap(cmap)
        self.p_map.addItem(img)
        if self.map_bar is None:
            self.map_bar = pg.ColorBarItem(values=(0, MAP_CLIP), colorMap=cmap, interactive=False,
                                           label=f'&Delta;&chi;<sup>2</sup> (clipped at {MAP_CLIP:g})')
            self.map_bar.setImageItem(img, insert_in=self.p_map)
        else: self.map_bar.setImageItem(img)
        self.map_bar.setVisible(True)
        entries = []
        for level, style, text in MAP_LEVELS:
            pen = pg.mkPen('w', width=1.5, style=style)
            for line in contour_lines(ln, lt, d, level): self.p_map.plot(line[:, 0], line[:, 1], pen=pen)
            entries.append((pg.PlotDataItem(pen=pg.mkPen(fg(), width=1.5, style=style)), f'&Delta;&chi;<sup>2</sup> = {level:.2f} ({text})'))
        i, k = m['best']
        self.p_map.plot([ln[k]], [lt[i]], pen=None, symbol='star', symbolSize=16, symbolBrush='r', symbolPen='k')
        legend(self.p_map, entries + [(symbol('r', 'star'), 'best node')], offset=(-5, -5))
        radiation = 'log_radiation' in out['grid'].axes
        style_plot(self.p_map, title=f"&Delta;&chi;<sup>2</sup>(n<sub>e</sub>, T), {COLUMN_NAMES.get(out.get('columns'), '')}" +
                   (', minimum over radiation' if radiation else ''))
        self.p_map.setRange(xRange=(ln[0]-dx/2, ln[-1]+dx/2), yRange=(lt[0]-dy/2, lt[-1]+dy/2), padding=0)

    def plot_cog(self):
        self.cog_plots.clear()
        out = self.current()
        d, r = out['data'], out['result']
        fl_view = self.cog_view.currentIndex() == 0
        grid = np.logspace(np.log10(d.flambda.min())-.5, np.log10(d.flambda.max())+.5, 200)
        cols = 3 if len(d.levels) > 4 else 2
        wavelength_law = r is not None and 'B_lambda' in out['forward'].covering.names
        plots = []
        for j, level in enumerate(d.levels):
            p = self.cog_plots.addPlot(row=j//cols, col=j % cols); plots.append(p)
            m = d.j == j; c = self.lines.color(level)
            x = np.log10(d.flambda[m]) if fl_view else d.wavelength[m]
            p.addItem(pg.ErrorBarItem(x=x, y=d.intensity[m], height=2*d.uncertainty[m], pen=c))
            for flag, shape in FLAG_SYMBOLS.items():
                k = np.array([q == flag for q in np.asarray(d.quality)[m]], bool)
                if k.any(): p.plot(x[k], d.intensity[m][k], pen=None, symbol=shape, symbolBrush=c, symbolPen=fg(), symbolSize=9)
            title = level
            if r is not None and fl_view:
                cg = dg.curve_of_growth(out['forward'], out['problem'], r, level, grid)
                p.plot(np.log10(grid), cg['bf_partial'], pen=pg.mkPen(c, width=2))
                p.plot(np.log10(grid), cg['bf_full'], pen=pg.mkPen(fg(), width=1.5, style=Qt.PenStyle.DashLine))
                title += f", C = {cg['C']:.3f}" + (f" at {cg['lambda_mean']:.0f} &Aring; (mean &lambda;)" if wavelength_law else '')
            elif r is not None:
                p.plot(d.wavelength[m], r['prediction'][m], pen=None, symbol='x', symbolPen=pg.mkPen(fg(), width=1.5),
                       symbolBrush=None, symbolSize=11)
            style_plot(p, 'BF' if j % cols == 0 else None,
                       'log f&lambda; [&Aring;]' if fl_view else '&lambda;<sub>rest</sub> [&Aring;]', title)
            auto_range(p)
        if plots and fl_view:
            flags = [(symbol((150, 150, 150), shape), flag) for flag, shape in FLAG_SYMBOLS.items()]
            model = [(pg.PlotDataItem(pen=pg.mkPen(fg(), width=2)), 'model, fitted covering'),
                     (pg.PlotDataItem(pen=pg.mkPen(fg(), width=1.5, style=Qt.PenStyle.DashLine)), 'full coverage (C = 1)')]
            legend(plots[0], flags + (model if r is not None else []), offset=(5, -5))
        markers = 'markers: circle clear, triangle unclear, square blend'
        if r is None: text = f'Press Fit for the model. {markers}'
        elif fl_view: text = f"{display_name(out['model'])}, {WIDTH_NAMES[out['width']]}: same column and b in both curves"
        else: text = f"{display_name(out['model'])}, {WIDTH_NAMES[out['width']]}; x: model BF of each line; {markers}"
        self.cog_note.setText(text)

    def run_contours(self):
        """Covering per level (M1) with free per-level columns and the current width, then a (log N/g, C) grid per level."""
        try: data, R = self.lines.data(), self.lines.resolving_power()
        except ValueError as exc: return self.failed(str(exc))
        s = self.settings()
        q, note_text = self.integration_choice((s['width'],), data)
        e, g = self.level_info(data.levels)
        n = option(self.lines.parent, 'bf_contour_n', int)
        kw = dict(resolving_power=R, coordinate='energy', b=s['b'], quadrature=q, level_b=self.level_b(data.levels),
                  lambda_0='auto')
        no_g = [l for l in data.levels if not np.isfinite(g[l])]

        def job(progress, cancelled):
            out = dict(run_fit(data, 'M1', s['width'], reference=False, **kw), integration=integration_name(q), columns=None,
                       kw=kw, note=note_text)
            grids = {}
            for i, level in enumerate(data.levels):
                if cancelled(): raise Cancelled('cancelled')
                if level not in no_g:
                    grids[level] = dg.column_covering_grid(out['forward'], out['problem'], out['result'], level, g[level],
                                                           n, CONTOUR_NSIGMA)
                progress.emit((0, int(100*(i+1)/len(data.levels))))
            return out, grids

        def done(res):
            self._busy(False)
            out, grids = res
            b = fitted_b(out)
            b_note = '' if b is None else f"b held at its best value ({', '.join(f'{x:.1f}' for x in b)} km/s)"
            run = self.add_run('contours', WIDTH_SHORT[s['width']], out=out, grids=grids, b_note=b_note,
                               levels=dg.level_table(out['forward'], out['problem'], out['result'], e, g))
            skipped = f"; no grid for {', '.join(no_g)} (no g_j in SPECTRO)" if no_g else ''
            self.report_excluded(run['name'], {})
            if no_g: self.lines.message(f"{run['name']}{skipped}")
            self.status.setText(f'Contours: {len(grids)} levels, {n} x {n} nodes' + (f'; {b_note}' if b_note else '') + skipped)
            self.tabs.setCurrentIndex(2); self.plot_columns()
        self._busy(True, determinate=True)
        self.status.setText(f"Contours: {display_name('M1')}, {WIDTH_NAMES[s['width']]}...")
        start_worker(self, job, done, self._progress)

    def equilibrium_runs(self):
        return [r for r in self.runs if r['kind'] == 'equilibrium']

    def plot_columns(self):
        self.p_cc.clear(); self.p_exc.clear()
        run, eq = self.latest('contours'), self.equilibrium_runs()
        levels = list(run['levels']['level']) if run else sorted({l for r in eq for l in r['levels']['level']}, key=level_key)
        left = [(symbol(self.lines.color(l)), l) for l in levels]
        right = []
        if run:
            lv = run['levels']
            for level, grid in run['grids'].items():
                c = self.lines.color(level)
                for value, style, _ in CONTOUR_LEVELS:
                    for line in contour_lines(grid['logN_g'], grid['C'], grid['delta_chi2'].T, value):
                        self.p_cc.plot(line[:, 0], line[:, 1], pen=pg.mkPen(c, width=2, style=style))
                self.p_cc.plot([grid['best_logN_g']], [grid['best_C']], pen=None, symbol='+', symbolPen=pg.mkPen(c, width=2),
                               symbolSize=12)
            for i, level in enumerate(lv['level']):
                c = self.lines.color(level)
                if not np.isfinite(lv['E_j'][i]): continue
                self.p_exc.addItem(pg.ErrorBarItem(x=lv['E_j'][i:i+1], y=lv['logN_g'][i:i+1],
                                                   height=2*np.nan_to_num(lv['logN_g_err'][i:i+1]), pen=c))
                self.p_exc.plot(lv['E_j'][i:i+1], lv['logN_g'][i:i+1], pen=None, symbol='o', symbolBrush=c, symbolPen=None)
            left += [(pg.PlotDataItem(pen=pg.mkPen(fg(), width=2, style=st)), f'&Delta;&chi;<sup>2</sup> = {v:.2f} ({t})')
                     for v, st, t in CONTOUR_LEVELS]
            left.append((pg.ScatterPlotItem(symbol='+', pen=pg.mkPen(fg(), width=2), size=12), 'best fit'))
            right.append((symbol(fg()), f"{display_name('M1')} (1 &sigma;)"))
            self.columns_note.setText(f"{run['name']}: {display_name('M1')}, {WIDTH_NAMES[run['out']['width']]}" +
                                      (f"; {run['b_note']}" if run['b_note'] else '') +
                                      '. Other markers: best-node predictions of the equilibrium runs (C at the mean '
                                      'wavelength); dashed: LTE at the run\'s best T.')
        else: self.columns_note.setText('Press Contours for the free-per-level grids; equilibrium runs of this session are shown.')
        for n, r in enumerate(eq):
            shape, lv = RUN_SYMBOLS[n % len(RUN_SYMBOLS)], r['levels']
            colors = [pg.mkBrush(self.lines.color(l)) for l in lv['level']]
            self.p_cc.addItem(pg.ScatterPlotItem(x=lv['logN_g'], y=lv['C'], symbol=shape, brush=colors, pen=pg.mkPen(fg()),
                                                 size=11))
            node = an.best_node(r['out']['grid'], r['out']['grid_fit'])
            o = np.argsort(lv['E_j'])
            self.p_exc.plot(lv['E_j'][o], lv['logN_g'][o], pen=pg.mkPen(fg(), width=1), symbol=shape, symbolBrush=colors,
                            symbolPen=fg(), symbolSize=10)
            e = np.linspace(0, lv['E_j'].max()*1.05, 50)
            lte = dg.lte_log_ng(lv['logN_g'][0], np.r_[lv['E_j'][0], e], node['temperature'])[1:]
            self.p_exc.plot(e, lte, pen=pg.mkPen(fg(), width=1, style=Qt.PenStyle.DashLine))
            right.append((pg.ScatterPlotItem(symbol=shape, brush=pg.mkBrush(fg()), pen=pg.mkPen(fg()), size=10),
                          f"{r['name']}, T = {node['temperature']:.0f} K"))
        legend(self.p_cc, left, offset=(-5, 5))
        legend(self.p_exc, right, offset=(-5, 5))
        auto_range(self.p_cc, self.p_exc)
        if not run and not eq: note(self.p_cc, 'press Contours')

    def plot_covering(self):
        self.p_cv.clear()
        run = self.latest('fit', 'M2')
        if run is None:
            legend(self.p_cv, [])
            self.covering_note.setText(f"Press Fit {display_name('M2')}: implied covering uses its columns and b.")
            return note(self.p_cv, f"Fit {display_name('M2')} first")
        out = run['out']; f, p, r = out['forward'], out['problem'], out['result']
        d, ic = f.data, dg.implied_covering(f, p, r)
        show = (1-ic['F']) > self.depth.value()
        lam = np.linspace(d.wavelength.min()*.97, d.wavelength.max()*1.03, 100)
        entries = []
        for j, level in enumerate(d.levels):
            m = (d.j == j) & show; c = self.lines.color(level)
            self.p_cv.addItem(pg.ErrorBarItem(x=d.wavelength[m], y=ic['C'][m], height=2*ic['sigma_C'][m], pen=c))
            self.p_cv.plot(d.wavelength[m], ic['C'][m], pen=None, symbol='o', symbolBrush=c, symbolPen=None)
            self.p_cv.plot(lam, dg.covering_curve(f, p, r, level, lam), pen=pg.mkPen(c, width=1.5))
            entries.append((symbol(c), f'{level} ({int(m.sum())} of {int((d.j == j).sum())} lines)'))
        bl, be = parameter_value(r, 'B_lambda')
        entries.append((pg.PlotDataItem(pen=pg.mkPen(fg(), width=1.5)),
                        f"{display_name('M2')}: B<sub>&lambda;</sub> = {bl:.3f} &plusmn; {be:.3f}"))
        legend(self.p_cv, entries, offset=(-5, -5))
        auto_range(self.p_cv)
        self.covering_note.setText(f"{run['name']}; F<sub>i</sub>: convolved line-centre transmission at full coverage "
                                   f"(fitted column and b); {int(show.sum())} of {len(show)} lines shown")

    def run_check(self):
        out = self.output
        if out is None or 'kw' not in out: return self.failed('Run a fit first: Check lines refits its model')
        n, tau = option(self.lines.parent, 'bf_check_nsigma', float), option(self.lines.parent, 'bf_tau0_min', float)
        data = out['data']

        def job(progress, cancelled):
            res = dg.check_lines(data, lambda d: refit(d, out), n, tau, cancelled,
                                 lambda i, k: progress.emit((0, int(100*i/k))))
            if res is None: raise Cancelled('cancelled')
            return res

        def done(res):
            self._busy(False)
            stage = out.get('columns')
            run = self.add_run('check', f"{out['model']} {WIDTH_SHORT[out['width']]} {COLUMN_SHORT[stage]}", result=res,
                               out=out, n_sigma=n, tau0_min=tau)
            self.report_excluded(run['name'], out.get('excluded', {}))
            self.show_check(res, out, n, tau)
        self._busy(True, determinate=True)
        self.status.setText(f"Check lines: {len(data.intensity)} refits of {display_name(out['model'])}" +
                            (' at the best grid node' if out['kind'] == 'grid' else '') + '...')
        start_worker(self, job, done, self._progress)

    def show_check(self, res, out, n, tau):
        flagged = [i for i, f in enumerate(res['flag']) if f]
        rows = [(res['level'][i], f"{res['lambda_rest'][i]:.3f}", f"{out['data'].intensity[i]:.3f}",
                 f"{res['bf_pred'][i]:.3f} +- {res['sigma_pred'][i]:.3f}" if np.isfinite(res['bf_pred'][i]) else '-',
                 f"{res['z'][i]:+.2f}" if np.isfinite(res['z'][i]) else '-', f"{res['tau0'][i]:.3g}", res['flag'][i])
                for i in flagged]
        text_table(self.check_table, rows, ['level', 'lambda', 'BF obs', 'BF pred', 'z', 'tau0', 'flag'])
        self.check_note.setText(f"{display_name(out['model'])}, {WIDTH_NAMES[out['width']]}: {len(flagged)} of "
                                f"{len(res['flag'])} lines flagged (|z| &gt; {n:g} or tau0 &lt; {tau:g}). "
                                'Nothing is switched off; the flags are also in the BF lines table.')
        self.lines.set_check_flags({i: f for i, f in zip(res['id'], res['flag']) if f})
        self.tabs.setCurrentIndex(4)
        self.status.setText(f'Check lines: {len(flagged)} flagged')

    def open_compare(self, run=True):
        if self.compare is None: self.compare = BFCompareDialog(self)
        enable_chianti(self.compare.columns, self.lines.atomic_cache())
        self.compare.columns.setCurrentIndex(self.columns.currentIndex())
        self.compare.show()
        if run and self.compare.run_button.isEnabled(): self.compare.run()
        return self.compare

    def export_session(self, path=None, note_text=None, vrange=None):
        """Write the ticked runs to an HDF5 session (spectro.bf.session). Without arguments: a dialog for the note and the
        pixel velocity range, then a save dialog starting in the last folder used."""
        runs = [r for r in self.runs if r['checked']]
        if path is None:
            dialog = QDialog(self); dialog.setWindowTitle('Export BF session')
            form = QFormLayout(dialog)
            text = QPlainTextEdit(); text.setPlaceholderText('free-text note stored in the file')
            span = QDoubleSpinBox(); span.setRange(10, 5000); span.setSuffix(' km/s')
            span.setValue(option(self.lines.parent, 'bf_session_vrange', float))
            form.addRow(QLabel(f'{len(runs)} of {len(self.runs)} runs ticked'))
            form.addRow('Note', text); form.addRow('Pixels within +-', span)
            ok = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            ok.accepted.connect(dialog.accept); ok.rejected.connect(dialog.reject); form.addRow(ok)
            if dialog.exec() != QDialog.DialogCode.Accepted: return None
            note_text, vrange = text.toPlainText(), span.value()
            spec = self.lines.spectrum_file()
            name = f"bf_session_{spec.stem if spec else 'spectrum'}_{date.today().isoformat()}.h5"
            start = option(self.lines.parent, 'bf_session_dir') or str(self.lines.spectrum_dir() or '')
            path = QFileDialog.getSaveFileName(self, 'Export BF session', str(Path(start)/name), 'HDF5 (*.h5)')[0]
            if not path: return None
        self.lines.parent.options('bf_session_dir', str(Path(path).parent))
        vrange = option(self.lines.parent, 'bf_session_vrange', float) if vrange is None else vrange
        session.write(path, session_content(self, runs, note_text or '', vrange))
        self.status.setText(f'Session written: {path} ({len(runs)} runs)')
        return path


def git_commit():
    """Short commit of the spectro checkout ('-dirty' with local changes); '' outside git."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                               text=True).stdout.strip()
        return head + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError): return ''


def table_node(table, attrs=None):
    """DataFrame or dict of columns -> session table node."""
    cols = {k: np.asarray(v) for k, v in (table.items() if isinstance(table, dict) else table.to_dict('list').items())}
    return dict(cols, attrs=attrs or {})


def line_node(lines, z):
    """/lines/: one row per panel row; E_j, g_j from the package level data (NaN without)."""
    rows = lines.rows
    lv = {r['species']: experiment.level_data(r['species']) or dict(energy_cm1=np.nan, g=np.nan) for r in rows}
    meas = [(r.get('meas') or {}) if (r['source'] != 'imported' or r.get('remeasured')) and not r.get('reason') else {} for r in rows]
    num = lambda v: float(v) if v not in ('', None) else np.nan
    win = [r.get('window') or (np.nan, np.nan) for r in rows]
    return dict(id=[f"{r['species']}:{r['lambda_rest']}" for r in rows], species=[species_of(r['species']) for r in rows],
                level=[r['species'] for r in rows], E_j=[lv[r['species']]['energy_cm1'] for r in rows],
                g_j=[lv[r['species']]['g'] for r in rows],
                lambda_rest=[float(r['lambda_rest']) for r in rows], f=[float(r['f']) for r in rows],
                flambda=[float(r['f'])*float(r['lambda_rest']) for r in rows],
                lambda_obs=[float(r['lambda_rest'])*(1+z) for r in rows], flag=[r['clarity'] for r in rows],
                use=np.array([bool(r['use']) for r in rows]), used=np.array([r['use'] and not r.get('reason') for r in rows]),
                check=[r.get('check', '') for r in rows], window_lo=[w[0] for w in win], window_hi=[w[1] for w in win],
                BF=[num(r.get('BF')) for r in rows], sigma=[num(r.get('sigma')) for r in rows],
                mc_bias=[m.get('mc_bias', np.nan) for m in meas], n_pix=[len(r.get('pixels') or []) for r in rows],
                R=[r.get('R', np.nan) for r in rows], colour=[lines.color_hex(r['species']) for r in rows])


def pixel_node(lines, z, vrange):
    """/pixels/<key>/ for every row with pixels in a normalized exposure."""
    out, ex = {}, lines.exposures()
    for r in lines.rows:
        k, key = r.get('pix_exp'), session.pixel_key(f"{r['species']}:{r['lambda_rest']}")
        if k is None or key in out or r.get('pix_grid') == 'raw': continue
        s = ex[k]; cm = np.asarray(s.cont_mask, bool)
        x, norm = np.asarray(s.spec.norm.x), np.asarray(s.spec.norm.y)
        flux, err = np.asarray(s.spec.raw.y)[cm], np.asarray(s.spec.raw.err)[cm]
        cont = np.asarray(s.cont.y) if len(s.cont.y) == len(x) else flux/norm
        v = C_KMS*(x/(float(r['lambda_rest'])*(1+z))-1)
        idx = np.flatnonzero(np.abs(v) <= vrange)
        run = r.get('run')
        out[key] = dict(attrs=dict(id=f"{r['species']}:{r['lambda_rest']}", exposure=int(k)), v=v[idx], flux=flux[idx],
                        err=err[idx], cont=cont[idx],
                        in_window=(idx >= run[0]) & (idx <= run[1]) if run else np.zeros(len(idx), bool),
                        used=np.isin(idx, r.get('pixels') or []))
    return out


def fit_node(run):
    """/runs/<run>/ of a fit or equilibrium run."""
    out = run['out']; f, p, r = out['forward'], out['problem'], out['result']
    d, grid = f.data, out['kind'] == 'grid'
    k, bic = (out['grid_fit']['k'], out['grid_fit']['bic']) if grid else (r['k'], r['bic'])
    cov = r['covariance'] if r['covariance'] is not None else np.full((len(r['parameters']),)*2, np.nan)
    node = dict(attrs=dict(kind=run['kind'], model=out['model'], model_name=display_name(out['model']),
                           coordinate=out['kw']['coordinate'], width=out['width'], columns=COLUMN_NAMES[out.get('columns')],
                           integration=out.get('integration', ''), chi2=r['chi2'], k=k, n=r['n'], BIC=bic,
                           note=out.get('note', ''), lambda_0=lambda_attr(f.covering.lambda_0)),
                param_names=list(r['names']), values=r['parameters'], errors=errors(r), covariance=cov,
                line_ids=[t.id for t in d.transitions], bf_model=r['prediction'],
                residual=(d.intensity-r['prediction'])/d.uncertainty,
                levels=table_node(run['levels']), covering=dict(line_ids=[t.id for t in d.transitions],
                                                                **dg.implied_covering(f, p, r)))
    fl = np.logspace(np.log10(d.flambda.min())-.5, np.log10(d.flambda.max())+.5, 200)
    node['cog'] = {}
    for level in d.levels:
        cg = dg.curve_of_growth(f, p, r, level, fl)
        node['cog'][level] = dict(attrs=dict(C=cg['C'], lambda_mean=cg['lambda_mean']), flambda=cg['flambda'],
                                  bf_partial=cg['bf_partial'], bf_full=cg['bf_full'])
    lam = np.linspace(d.wavelength.min()*.97, d.wavelength.max()*1.03, 200)
    node['cover_curve'] = {level: dict(attrs={}, **{'lambda': lam, 'C': dg.covering_curve(f, p, r, level, lam)})
                           for level in d.levels}
    if grid:
        g, a, lv = out['grid'], out['grid_fit'], run['levels']
        best = an.best_node(g, a)
        node['equilibrium'] = dict(attrs=dict(grid=COLUMN_NAMES[out.get('columns')], T_best=best['temperature'],
                                              ne_best=best['electron_density'],
                                              **({'log_radiation_best': best['log_radiation']} if 'log_radiation' in best else {})),
                                   axis_names=list(g.axes), delta_chi2=a['chi2_grid']-a['chi2'],
                                   best_index=np.asarray(a['best_index'], int), logN_g_pred=lv['logN_g'],
                                   logN_g_lte=dg.lte_log_ng(lv['logN_g'][0], lv['E_j'], best['temperature']),
                                   **{f'axis_{k}': v for k, v in g.axes.items()})
    return node


def session_content(window, runs, note_text, vrange):
    """Nested dict for spectro.bf.session.write from the panel rows and the ticked runs."""
    lines, z = window.lines, window.lines.z()
    spec = lines.spectrum_file()
    est = lines.settings()
    s = dict(attrs=dict(spectrum=spec.name if spec else '', z_abs=z,
                        estimator=f"{est['estimator']}, N = {est['n']}, sigma {est['sigma']} (imported rows: the file's values)",
                        integration=window.integration.currentText(), spectro_commit=git_commit(),
                        date=datetime.now(timezone.utc).isoformat(timespec='seconds'), note=note_text,
                        R=np.array([r.get('R', np.nan) for r in lines.rows], float)),
             lines=line_node(lines, z), pixels=pixel_node(lines, z, vrange), runs={}, compare={}, loo={}, check={},
             contours={})
    for run in runs:
        name, kind = run['name'], run['kind']
        if kind in ('fit', 'equilibrium'): s['runs'][name] = fit_node(run)
        elif kind == 'compare':
            s['compare'][name] = table_node(run['table'], dict(columns=run['columns'], integration=run['integration'],
                                                               lambda_0=lambda_attr(run['lambda_0'])))
        elif kind == 'loo':
            s['loo'][name] = table_node(run['table'][['id', 'level', 'lambda_rest', 'delta_chi2']],
                                        dict(model=run['model'], width=run['width'], columns=run['columns'],
                                             lambda_0=lambda_attr(run['lambda_0'])))
        elif kind == 'check':
            out = run['out']
            s['check'][name] = table_node(run['result'], dict(model=out['model'], width=out['width'],
                                                              columns=COLUMN_NAMES[out.get('columns')],
                                                              n_sigma=run['n_sigma'], tau0_min=run['tau0_min'],
                                                              lambda_0=lambda_attr(out['forward'].covering.lambda_0)))
        elif kind == 'contours':
            node = dict(attrs=dict(model='M1', width=run['out']['width'], b_note=run['b_note'],
                                   lambda_0=lambda_attr(run['out']['forward'].covering.lambda_0)),
                        levels=table_node(run['levels']))
            for level, g in run['grids'].items():
                node[level] = dict(attrs=dict(b=g['b'], best_logN_g=g['best_logN_g'], best_C=g['best_C']),
                                   logN_g=g['logN_g'], C=g['C'], delta_chi2=g['delta_chi2'])
            s['contours'][name] = node
    return s


class BFCompareDialog(QDialog):
    """Compare BF models: five covering laws x selected widths on the same lines and columns, sorted by Delta BIC;
    the preferred model is re-checked with adaptive integration."""

    def __init__(self, window):
        super().__init__(window)
        self.analysis, self.table_data, self.fits, self.loo, self._worker, self._run = window, None, {}, None, None, 0
        self.grid = None
        self.setWindowTitle('Compare BF models')
        self.resize(1150, 900)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout(); controls.addWidget(QLabel('Widths:'))
        self.widths = {}
        for code, text in WIDTHS:
            self.widths[code] = QCheckBox(text); self.widths[code].setChecked(code == 'common_fixed')
            controls.addWidget(self.widths[code])
        controls.addWidget(QLabel('Columns:'))
        self.columns = columns_box(); self.columns.currentIndexChanged.connect(self._enable)
        controls.addWidget(self.columns)
        controls.addWidget(QLabel('Reference wavelength:'))
        self.lambda_box = lambda_widgets(window, lambda: window.auto_lambda_0(MODELS))
        controls.addWidget(self.lambda_box)
        window.show_lambda_0()
        self.run_button = QPushButton('Run'); self.run_button.clicked.connect(lambda: self.run())
        controls.addWidget(self.run_button); controls.addStretch(1)
        layout.addLayout(controls)
        self.summary = QLabel(''); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['model', 'width', 'k', 'chi2', 'BIC', 'Delta BIC', 'LR sigma vs Single'])
        self.table.horizontalHeaderItem(6).setToolTip(LR_TIP)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(self.table.horizontalHeader().sizeHint().height()
                                    + 5*self.table.verticalHeader().defaultSectionSize() + 6)   # five laws without scrolling
        self.table.currentCellChanged.connect(lambda row, *_: self.select(row))
        layout.addWidget(self.table, 1)
        plots = self._plots()
        layout.addWidget(plots, 4)
        self.footer = QLabel(''); self.footer.setWordWrap(True); layout.addWidget(self.footer)
        buttons = QHBoxLayout()
        self.loo_button = QPushButton('Leave one out'); self.loo_button.clicked.connect(lambda: self.run_loo())
        export = QPushButton('Export CSV'); export.clicked.connect(lambda: self.export_csv())
        save = save_button(self, (('All plots', plots.scene, 'bf_compare'), ('Delta BIC', lambda: self.p_bic, 'bf_delta_bic'),
                                  ('Covering per level', lambda: self.p_cov, 'bf_covering'),
                                  ('Leave one out', lambda: self.p_loo, 'bf_leave_one_out')))
        close = QPushButton('Close'); close.clicked.connect(self.close)
        for w in (self.loo_button, export, save): buttons.addWidget(w)
        buttons.addLayout(progress_row(self), 1)
        buttons.addStretch(1); buttons.addWidget(close)
        layout.addLayout(buttons)

    def _plots(self):
        plots = pg.GraphicsLayoutWidget()
        self.p_bic = style_plot(plots.addPlot(row=0, col=0), None, '&Delta;BIC', '&Delta;BIC (lowest on top)')
        self.p_cov = style_plot(plots.addPlot(row=0, col=1), cj_label(None), 'E<sub>j</sub> [cm<sup>-1</sup>]')
        self.p_loo = style_plot(plots.addPlot(row=1, col=0, colspan=2), None,
                                '&Delta;&chi;<sup>2</sup> (leave one out)', f'Leave one out: {LOO_TOP} largest')
        for p in (self.p_bic, self.p_loo): p.invertY(True)
        return plots

    def _enable(self):
        eq = self.columns.currentData() is not None
        for code, box in self.widths.items():
            box.setEnabled(not eq or code in FIXED)
            box.setToolTip('' if box.isEnabled() else 'Equilibrium columns need a fixed width')

    def failed(self, text):
        self._busy(False)
        self.summary.setText(f'Error: {text}')
        QMessageBox.warning(self, 'Compare BF models', text)

    def _busy(self, on, determinate=False):
        self.run_button.setEnabled(not on); self.loo_button.setEnabled(not on)
        show_busy(self, on, determinate)

    def cancel_run(self):
        self._run += 1
        self._busy(False)
        self.footer.setText('Cancelled')

    def _progress(self, p):
        if p[1] >= 0: self.progress.setValue(p[1])

    def selected_widths(self):
        return tuple(c for c, box in self.widths.items() if box.isChecked() and box.isEnabled()) or ('common_fixed',)

    def run(self):
        """compare_models (or the same on a population grid) in a worker, all laws on the same lines and the Integration
        box, then the adaptive re-check of the preferred model."""
        a, lines = self.analysis, self.analysis.lines
        try:
            data, R = lines.data(), lines.resolving_power()
            data, excluded = fit_lines(data, MODELS, 'energy')
        except ValueError as exc: return self.failed(str(exc))
        widths, b, stage = self.selected_widths(), a.common_b(), self.columns.currentData()
        q, note_text = a.integration_choice(widths, data)
        manual = a._lambda_0
        self.meta = dict(created=datetime.now(timezone.utc).isoformat(timespec='seconds'), n_lines=len(data.intensity),
                         levels=' '.join(data.levels), widths=' '.join(widths), b_fixed_common_kms=b, coordinate='energy',
                         integration=integration_name(q), z=lines.z(),
                         resolving_power=' '.join(sorted({f'{v:.0f}' for v in R.values()})), **lines.settings(),
                         bf_method=data.method)
        if stage: self.meta['columns'] = f'{stage} ({COLUMN_NAMES[stage]}, cached grid)'
        kw = dict(b=b, level_b=a.level_b(data.levels), quadrature=q, resolving_power=R, coordinate='energy')
        self.stage, self.note = stage, note_text
        self._busy(True, determinate=bool(stage)); self.footer.setText(note_text)
        self.summary.setText(f"Fitting 5 covering laws x {len(widths)} width option(s)" +
                             (f', {COLUMN_NAMES[stage]}' if stage else '') + f', {integration_name(q)}...')
        if note_text: lines.message(note_text)
        paths = a.grid_paths()

        def job(progress, cancelled):
            grid, d, left_out = None, data, dict(excluded)
            if stage:
                grid = load_grid(stage, paths, progress, cancelled)
                d, more = fit_lines(data, MODELS, 'energy', grid)
                left_out.update(more)
            k = dict(kw, lambda_0=reference_wavelength(d) if manual is None else manual)
            if stage is None: t, fits = an.compare_models(d, widths, **k)
            else: t, fits = self._compare_grid(d, widths, grid, k, progress, cancelled)
            return t, fits, grid, d, k, left_out
        start_worker(self, job, self.show_table, self._progress)

    def _compare_grid(self, data, widths, grid, kw, progress, cancelled):
        """As analysis.compare_models, on a population grid: (table, {(code, width): run_fit output})."""
        keys = [(m, w) for w in widths for m in MODELS]
        fits = {}
        for i, (m, w) in enumerate(keys):
            if cancelled(): raise Cancelled('cancelled')
            fits[m, w] = run_fit(data, m, w, grid=grid, **kw)
            progress.emit((0, int(100*(i+1)/len(keys))))
        rows = []
        for (m, w), out in fits.items():
            g, g0 = out['grid_fit'], fits['M0', w]['grid_fit']
            dof = g['k']-g0['k']
            rows.append(dict(code=m, name=display_name(m), width=w, k=g['k'], chi2=g['chi2'], BIC=g['bic'], dof=dof,
                             lr_sigma=an.lr_sigma(g0['chi2']-g['chi2'], dof) if dof > 0 else np.nan))
        t = pd.DataFrame(rows)
        t.insert(6, 'dBIC', t.BIC-t.BIC.min())
        return t.sort_values('dBIC', kind='stable', ignore_index=True), fits

    def show_table(self, out):
        self._busy(False)
        self.table_data, self.fits, self.grid, self.data, self.kw, excluded = out
        self.meta.update(n_lines=len(self.data.intensity), levels=' '.join(self.data.levels), lambda_0=self.kw['lambda_0'],
                         excluded=exclusion_text(excluded))
        t = self.table_data
        run = self.analysis.add_run('compare', f"{' '.join(WIDTH_SHORT[w] for w in dict.fromkeys(t.width))} "
                                               f"{COLUMN_SHORT[self.stage]}", table=t.copy(), lambda_0=self.kw['lambda_0'],
                                    columns=COLUMN_NAMES[self.stage], integration=self.meta['integration'])
        self.analysis.report_excluded(run['name'], excluded)
        self.table.blockSignals(True); self.table.setRowCount(len(t))
        for i, r in enumerate(t.itertuples()):
            vals = (r.name, WIDTH_NAMES[r.width], r.k, f'{r.chi2:.4f}', f'{r.BIC:.3f}', f'{r.dBIC:.3f}',
                    '-' if not np.isfinite(r.lr_sigma) else f'{r.lr_sigma:.2f}')
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v)); it.setToolTip(f'{r.code}: {MODEL_FORMULAS[r.code]}')
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)
        self.summary.setText(f"Preferred by BIC: {t.name[0]} ({WIDTH_NAMES[t.width[0]]}); {self.meta['n_lines']} lines, "
                             + (f'{COLUMN_NAMES[self.stage]}, ' if self.stage else '') +
                             f"integration {self.meta['integration']}, level coordinate energy, "
                             f"&lambda;<sub>0</sub> = {self.kw['lambda_0']:g} &Aring; (M2, M4)" +
                             (f"; excluded {exclusion_text(excluded)}" if excluded else ''))
        self.plot_bic()
        self.table.selectRow(0)
        self.select(0)
        self.recheck()

    def recheck(self):
        """Refit the BIC-preferred model with adaptive integration (equilibrium columns: at its best grid node)."""
        t0 = self.table_data.iloc[0]
        key, chi2, integration = (t0.code, t0.width), float(t0.chi2), self.meta['integration']
        head = f'Best model ({t0["name"]}, {WIDTH_NAMES[key[1]]}): &chi;<sup>2</sup> = {chi2:.6f} with {integration}'
        if self.kw['quadrature'] is None: return self.footer.setText(' '.join(filter(None, (self.note, head))))
        kw, data = dict(self.kw, quadrature=None), self.data
        grid = None if self.grid is None else node_grid(self.grid, self.fits[key]['grid_fit'])
        job = lambda progress, cancelled: run_fit(data, *key, grid=grid, reference=False, **kw)['result']['chi2']

        def done(chi2_adaptive):
            self._busy(False)
            self.footer.setText(f'{head}; best model re-checked with adaptive' + (' at the best grid node' if grid else '') +
                                f': &chi;<sup>2</sup> = {chi2_adaptive:.6f} (difference {chi2_adaptive-chi2:+.1e})')
        self._busy(True); self.footer.setText(f'{head}; re-checking with adaptive integration...')
        start_worker(self, job, done)

    @staticmethod
    def bar_labels(plot, labels):
        """Left-axis tick labels of horizontal bars (one per row), the axis wide enough for the longest label."""
        axis = plot.getAxis('left')
        axis.setTicks([list(enumerate(labels))])
        axis.setWidth(max((QFontMetrics(axis.style['tickFont'] or axis.font()).horizontalAdvance(l) for l in labels),
                          default=20)+18)

    def plot_bic(self):
        t = self.table_data
        self.p_bic.clear()
        y = np.arange(len(t))
        self.p_bic.addItem(pg.BarGraphItem(x0=0, y=y, width=t.dBIC.values, height=BAR_HEIGHT, brush=BLUE, pen=None))
        self.bar_labels(self.p_bic, [f'{display_name(c)} - {WIDTH_SHORT[w]}' for c, w in zip(t.code, t.width)])
        self.p_bic.setYRange(-.6, len(t)-.4, padding=0)
        self.p_bic.enableAutoRange(axis='x'); self.p_bic.autoRange()
        self.p_bic.setYRange(-.6, len(t)-.4, padding=0)

    def selected(self):
        row = self.table.currentRow()
        if self.table_data is None or not 0 <= row < len(self.table_data): return None
        return self.table_data.code[row], self.table_data.width[row]

    def select(self, row):
        """Row click: C_j(E_j) of Covering per level and the selected model; redraw the analysis window plots."""
        key = self.selected()
        if key is None: return
        fit = self.fits[key]
        self.plot_covering(key)
        if isinstance(fit, dict): out = dict(fit)
        else:
            forward, problem, result = fit
            out = dict(kind='empirical', forward=forward, problem=problem, result=result, data=forward.data, model=key[0],
                       width=key[1])
        out.update(integration=self.meta['integration'], columns=self.stage, kw=dict(self.kw), excluded={})
        t = self.table_data.iloc[self.table.currentRow()]
        if np.isfinite(t.lr_sigma): out.update(lr_sigma=t.lr_sigma, lr_dof=int(t.dof))
        self.analysis.show_fit(out)

    def plot_covering(self, key):
        self.p_cov.clear()
        m1 = an.level_covering(*self._fit_pair(('M1', key[1])))
        e = m1.energy_cm1.values
        colors = [pg.mkBrush(self.analysis.lines.color(l)) for l in m1.level]
        self.p_cov.addItem(pg.ErrorBarItem(x=e, y=m1.C.values, height=2*np.nan_to_num(m1.C_err.values), pen=NO_LEVEL))
        self.p_cov.addItem(pg.ScatterPlotItem(x=e, y=m1.C.values, symbol='o', brush=colors, pen=None, size=10))
        forward, result = self._fit_pair(key)
        sel = an.level_covering(forward, result)
        style_plot(self.p_cov, left=cj_label(forward.covering.lambda_0))
        o = np.argsort(sel.energy_cm1.values)
        self.p_cov.plot(sel.energy_cm1.values[o], sel.C.values[o], pen=pg.mkPen(RED, width=2), symbol='s',
                        symbolBrush=RED, symbolPen=None)
        legend(self.p_cov, [(symbol(NO_LEVEL), f"{display_name('M1')} (level colors)"),
                            (pg.PlotDataItem(pen=pg.mkPen(RED, width=2), symbol='s', symbolBrush=RED), display_name(key[0]))],
               offset=(5, -5))
        auto_range(self.p_cov)

    def _fit_pair(self, key):
        fit = self.fits[key]
        return (fit['forward'], fit['result']) if isinstance(fit, dict) else (fit[0], fit[2])

    def run_loo(self):
        key = self.selected()
        if key is None: return self.failed('Run the comparison and select a row first')
        kw, grid, stage = dict(self.kw), self.grid, self.stage
        self._busy(True); self.summary.setText(f'Leave one out: {display_name(key[0])}, {WIDTH_NAMES[key[1]]}...')

        def done(t):
            self.analysis.add_run('loo', f'{key[0]} {WIDTH_SHORT[key[1]]} {COLUMN_SHORT[stage]}', table=t, model=key[0],
                                  width=key[1], columns=COLUMN_NAMES[stage],
                                  lambda_0=kw['lambda_0'] if key[0] in WAVELENGTH_LAWS else None)
            self.show_loo(t, key)
        start_worker(self, lambda progress, cancelled: an.line_loo(self.data, key[0], key[1], grid=grid, **kw), done)

    def show_loo(self, t, key):
        self._busy(False)
        self.loo = t
        self.p_loo.clear()
        top = t.sort_values('delta_chi2', ascending=False).head(LOO_TOP).reset_index(drop=True)
        y = np.arange(len(top))
        brushes = [pg.mkBrush(self.analysis.lines.color(l)) for l in top.level]
        self.p_loo.addItem(pg.BarGraphItem(x0=0, y=y, width=top.delta_chi2.values, height=BAR_HEIGHT, brushes=brushes,
                                           pen=None))
        self.p_loo.addItem(pg.InfiniteLine(pos=LOO_LINE, angle=90, pen=pg.mkPen(fg(), style=Qt.PenStyle.DashLine)))
        self.bar_labels(self.p_loo, [f'{l} {x:.2f}' for l, x in zip(top.level, top.lambda_rest)])
        legend(self.p_loo, [(pg.PlotDataItem(pen=pg.mkPen(fg(), style=Qt.PenStyle.DashLine)),
                             f'&Delta;&chi;<sup>2</sup> = {LOO_LINE:g}')], offset=(-5, -5))
        self.p_loo.enableAutoRange(axis='x'); self.p_loo.autoRange()
        self.p_loo.setXRange(min(0, top.delta_chi2.min()), 1.1*max(top.delta_chi2.max(), LOO_LINE))
        self.p_loo.setYRange(-.6, len(top)-.4, padding=0)
        self.summary.setText(f'Leave one out ({display_name(key[0])}, {WIDTH_NAMES[key[1]]}): '
                             f'{int((t.delta_chi2 > LOO_LINE).sum())} lines with &Delta;&chi;<sup>2</sup> &gt; {LOO_LINE:g}')

    def export_csv(self, path=None):
        if self.table_data is None: return self.failed('Run the comparison first')
        path = path or QFileDialog.getSaveFileName(self, 'Export comparison', 'bf_compare.csv', 'CSV (*.csv)')[0]
        if not path: return
        with open(path, 'w', newline='') as f:
            for k, v in self.meta.items(): f.write(f'# {k}: {v}\n')
            self.table_data.to_csv(f, index=False, lineterminator='\n')
