"""Fit source-coverage models on the same pixels and compare AIC/BIC."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .coverage_models import (
    COVERAGE_MODE_LABELS,
    coverage_mode_index,
    spatial_realization_label,
    spatial_column_scale_map,
)
from .source_models import (
    source_mode_index, source_mode_name, source_model_label, source_from_fit,
    source_wavelengths_from_observed,
    half_light_from_r_lambda,
)


PARAMETER_ATTRIBUTES = (
    "val",
    "min",
    "max",
    "step",
    "vary",
    "fit",
    "fit_w",
    "addinfo",
    "left",
    "right",
    "show",
    "unc",
)


@dataclass
class CoverageFitRecord:
    mode: str
    label: str
    source: str = "achromatic"
    source_label: str = "Achromatic"
    realization: str = "legacy analytic achromatic"
    topology: str = "current"
    beta: float = math.nan
    r_ref: float = math.nan
    r_half: float = math.nan
    scale_mode: str = "legacy_r_lambda"
    offset: float = 0.0
    success: bool = False
    message: str = ""
    chi2: float = math.nan
    ndata: int = 0
    nvarys: int = 0
    dof: int = 0
    redchi: float = math.nan
    aic: float = math.nan
    aicc: float = math.nan
    bic: float = math.nan
    comparable: bool = True
    timed_out: bool = False
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    curves: list[tuple[np.ndarray, np.ndarray] | None] = field(default_factory=list)

    def criterion(self, name: str) -> float:
        key = name.strip().lower()
        return {
            "chi2": self.chi2,
            "reduced chi2": self.redchi,
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
        }.get(key, self.bic)

    @property
    def usable(self) -> bool:
        return bool(self.parameters) and self.comparable and np.isfinite(self.chi2)


def snapshot_fit_parameters(fit) -> dict[str, dict[str, Any]]:
    """Return a parent-free snapshot of all fit parameters."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, parameter in fit.pars().items():
        attrs: dict[str, Any] = {}
        for attr in PARAMETER_ATTRIBUTES:
            if hasattr(parameter, attr):
                value = getattr(parameter, attr)
                try:
                    value = deepcopy(value)
                except Exception:
                    pass
                attrs[attr] = value
        snapshot[name] = attrs
    return snapshot


def restore_fit_parameters(fit, snapshot: dict[str, dict[str, Any]]) -> None:
    """Restore a snapshot without changing the model's component structure."""
    covmode = snapshot.get("covmode", {}).get("val")
    if covmode is not None:
        fit.set_coverage_model(covmode)

    for name, attrs in snapshot.items():
        try:
            parameter = fit.getPar(name)
        except Exception:
            if "val" not in attrs:
                continue
            try:
                fit.setValue(name, attrs["val"], "val", check=False)
                parameter = fit.getPar(name)
            except Exception:
                continue

        for attr, value in attrs.items():
            try:
                setattr(parameter, attr, deepcopy(value))
            except Exception:
                setattr(parameter, attr, value)


def snapshot_model_curves(parent) -> list[tuple[np.ndarray, np.ndarray] | None]:
    curves: list[tuple[np.ndarray, np.ndarray] | None] = []
    for spectrum in parent.s:
        try:
            line = spectrum.fit.line.norm
            x = np.asarray(line.x, dtype=float)
            y = np.asarray(line.y, dtype=float)
            if x.size and y.size:
                curves.append((x.copy(), y.copy()))
            else:
                curves.append(None)
        except Exception:
            curves.append(None)
    return curves


def _metrics_from_result(result, residual=None) -> dict[str, float | int]:
    """Comparison statistics from the final model residual (parent.s.chi() after recalculation)."""
    if residual is None:
        residual = getattr(result, "residual", [])
    residual = np.asarray(residual, dtype=float).ravel()
    residual = residual[np.isfinite(residual)]
    ndata = int(residual.size)
    chi2 = float(np.sum(residual**2)) if ndata else math.nan
    nvarys = int(getattr(result, "nvarys", len(getattr(result, "var_names", []) or [])))
    dof = max(ndata - nvarys, 0)
    redchi = chi2 / dof if dof > 0 and np.isfinite(chi2) else math.nan

    # Akaike 1974; Hurvich & Tsai 1989; Schwarz 1978.
    if ndata > 0 and np.isfinite(chi2):
        aic = chi2 + 2.0 * nvarys
        bic = chi2 + nvarys * math.log(ndata)
        if ndata > nvarys + 1:
            aicc = aic + 2.0 * nvarys * (nvarys + 1) / (ndata - nvarys - 1)
        else:
            aicc = math.inf
    else:
        aic = aicc = bic = math.nan

    return {
        "chi2": chi2,
        "ndata": ndata,
        "nvarys": nvarys,
        "dof": dof,
        "redchi": redchi,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
    }


def _set_parameter_value(fit, name: str, value: float) -> None:
    """Set a fit parameter if it exists, clipping to its declared bounds."""
    try:
        par = fit.getPar(name)
    except Exception:
        return
    if par is None:
        return
    lo = float(getattr(par, "min", -np.inf))
    hi = float(getattr(par, "max", np.inf))
    value = float(np.clip(value, lo, hi))
    fit.setValue(name, value, "val", check=False)


def _cfov_seed_snapshots(parent, original_snapshot, records, source="achromatic"):
    """Seed overlap fits away from Cf1=Cf2 and complete overlap, in both component orderings."""
    ppc = next((record for record in records if record.mode == "ppc" and record.source == source and record.usable), None)
    base = deepcopy(ppc.parameters if ppc is not None else original_snapshot)
    seeds = []

    for direction in (1, -1):
        snap = deepcopy(base)
        snap.setdefault("covmode", {})["val"] = coverage_mode_index("cfov")
        seeds.append((snap, direction))
    return seeds


def _initialize_cfov_from_ppc(parent, direction: int) -> None:
    """Break the symmetric Cf1=Cf2 overlap starting point safely."""
    fit = parent.fit
    for region in range(fit.cf_num):
        try:
            cf = float(fit.getValue(f"cf_{region}"))
        except Exception:
            cf = 0.5
        cf = float(np.clip(cf, 0.05, 0.95))
        low = max(0.02, 0.78 * cf)
        high = min(0.98, 1.12 * cf + 0.04)
        if high <= low + 0.02:
            high = min(0.98, low + 0.08)
        cf1, cf2 = (low, high) if direction > 0 else (high, low)
        _set_parameter_value(fit, f"cov1_{region}", cf1)
        _set_parameter_value(fit, f"cov2_{region}", cf2)
        _set_parameter_value(fit, f"covov_{region}", 0.90)


def _masked_segments(x: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Indices of contiguous fitted wavelength segments, split at masked pixels and large wavelength jumps."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(indices) > 1) + 1

    if indices.size > 1:
        dx = np.diff(x[indices])
        local = dx[(np.diff(indices) == 1) & np.isfinite(dx) & (dx > 0)]
        if local.size:
            typical = float(np.median(local))
            if np.isfinite(typical) and typical > 0:
                gap_breaks = np.flatnonzero(dx > 8.0 * typical) + 1
                if gap_breaks.size:
                    breaks = np.unique(np.concatenate([breaks, gap_breaks]))

    return [segment for segment in np.split(indices, breaks) if segment.size]


def available_comparison_modes(parent) -> list[str]:
    modes = [
        "full",
        "ppc",
        "powerlaw",
        "powerlaw_ppc_sysa",
        "powerlaw_sysa_global",
        "powerlaw_radial",
        "powerlaw_ppc",
        "powerlaw_radial_ppc",
        "two_zone",
    ]
    if len(parent.fit.sys) == 2:
        modes.append("cfov")
    return modes


class CoverageModelSelectionDialog(QDialog):
    """Choose which coverage laws are included in the next comparison."""

    def __init__(self, parent, available_modes: list[str], selected_modes=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.available_modes = list(available_modes)
        previous = set(selected_modes or self.available_modes)
        self.checks = {}

        self.setWindowTitle("Select coverage models")
        self.resize(470, 340)
        layout = QVBoxLayout(self)

        for mode in self.available_modes:
            label = COVERAGE_MODE_LABELS[coverage_mode_index(mode)]
            box = QCheckBox(label)
            box.setChecked(mode in previous)
            self.checks[mode] = box
            layout.addWidget(box)

        presets = QHBoxLayout()
        all_button = QPushButton("Select all")
        none_button = QPushButton("Clear")
        all_button.clicked.connect(lambda: self._set_checked(self.available_modes))
        none_button.clicked.connect(lambda: self._set_checked([]))
        presets.addWidget(all_button)
        presets.addWidget(none_button)
        presets.addStretch(1)
        layout.addLayout(presets)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('Source:'))
        self.source = QComboBox()
        self.source.addItem('achromatic', 'achromatic')
        self.source.addItem('Power-law temperature disk (beta=0.75 is SS73)', 'ss73')
        self.source.addItem('both', 'both')
        default_source = getattr(parent, 'coverageComparisonSource', None)
        if default_source not in {'achromatic', 'ss73', 'both'}:
            default_source = parent.fit.source_model() if hasattr(parent.fit, 'source_model') else 'achromatic'
        source_index = self.source.findData(default_source)
        self.source.setCurrentIndex(source_index if source_index >= 0 else 0)
        source_row.addWidget(self.source)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        topology_row = QHBoxLayout()
        topology_row.addWidget(QLabel('LFR setup:'))
        self.topology = QComboBox()
        self.topology.addItem('common screen', 'common')
        if len(parent.fit.sys) > 1:
            self.topology.addItem('one screen per system', 'by_system')
        self.topology.addItem('keep current LFRs', 'current')
        default_topology = getattr(parent, 'coverageComparisonTopology', None)
        if default_topology is None:
            default_topology = 'by_system' if len(parent.fit.sys) > 1 else 'common'
        index = self.topology.findData(default_topology)
        self.topology.setCurrentIndex(index if index >= 0 else 0)
        self.topology.currentIndexChanged.connect(self._topology_changed)
        topology_row.addWidget(self.topology)
        topology_row.addStretch(1)
        layout.addLayout(topology_row)
        self._topology_changed()

        layout.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        run = QPushButton("Run selected")
        cancel.clicked.connect(self.reject)
        run.clicked.connect(self._accept_if_valid)
        buttons.addWidget(cancel)
        buttons.addWidget(run)
        layout.addLayout(buttons)

    def _set_checked(self, modes):
        selected = set(modes)
        for mode, box in self.checks.items():
            box.setChecked(mode in selected and box.isEnabled())

    def selected_modes(self) -> list[str]:
        return [mode for mode in self.available_modes if self.checks[mode].isChecked()]

    def selected_topology(self) -> str:
        return str(self.topology.currentData())

    def selected_sources(self) -> list[str]:
        value = str(self.source.currentData())
        return ['achromatic', 'ss73'] if value == 'both' else [value]

    def _topology_changed(self):
        by_system = self.selected_topology() == 'by_system'
        box = self.checks.get('cfov')
        if box is not None:
            box.setEnabled(not by_system)
            if by_system:
                box.setChecked(False)
                box.setToolTip('Overlap Cf uses one common LFR containing both systems.')
            else:
                box.setToolTip('')
        for mode, equivalent in [('powerlaw_ppc_sysa', 'Power-law + uncovered (x)'),
                                 ('powerlaw_sysa_global', 'Power-law inhomogeneous (x)')]:
            box = self.checks.get(mode)
            if box is None:
                continue
            box.setEnabled(not by_system)
            if by_system:
                box.setChecked(False)
                box.setToolTip(f'Redundant for one LFR per system; identical to {equivalent}.')
            else:
                box.setToolTip('')

    def _accept_if_valid(self):
        if not self.selected_modes():
            QMessageBox.information(
                self, "Coverage-model comparison",
                "Select at least one coverage model to fit.",
            )
            return
        self.accept()


def select_comparison_modes(parent):
    """Dialog for choosing the applicable models to run; return None on cancel.

    Not opened without absorption systems: with tau == 0 all coverage laws give unit transmission."""
    if len(parent.fit.sys) == 0:
        raise ValueError(
            "No absorption systems are present in the current Fit model. "
            "Add or restore at least one system (sys0, sys1, ...) before "
            "running Compare coverage models."
        )
    available = available_comparison_modes(parent)
    previous = getattr(parent, "coverageComparisonSelectedModes", None)
    if previous is not None:
        previous = [mode for mode in previous if mode in available]
    dialog = CoverageModelSelectionDialog(parent, available, previous)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    selected = dialog.selected_modes()
    parent.coverageComparisonSelectedModes = list(selected)
    topology = dialog.selected_topology()
    parent.coverageComparisonTopology = topology
    sources = dialog.selected_sources()
    parent.coverageComparisonSource = str(dialog.source.currentData())
    return selected, topology, sources


def best_record(records: list[CoverageFitRecord], criterion: str = "BIC") -> CoverageFitRecord | None:
    candidates = [
        r for r in records
        if r.usable and r.success and not r.timed_out and np.isfinite(r.criterion(criterion))
    ]
    if not candidates:
        candidates = [
            r for r in records
            if r.usable and not r.timed_out and np.isfinite(r.criterion(criterion))
        ]
    return min(candidates, key=lambda r: r.criterion(criterion)) if candidates else None


def diagnostic_best_record(records: list[CoverageFitRecord], criterion: str = "BIC") -> CoverageFitRecord | None:
    candidates = [
        r for r in records
        if r.usable and np.isfinite(r.criterion(criterion))
    ]
    return min(candidates, key=lambda r: r.criterion(criterion)) if candidates else None


def record_identity(record: CoverageFitRecord):
    beta = None if not np.isfinite(record.beta) else round(float(record.beta), 12)
    rref = None if not np.isfinite(record.r_ref) else round(float(record.r_ref), 12)
    rhalf = None if not np.isfinite(record.r_half) else round(float(record.r_half), 12)
    offset = None if not np.isfinite(record.offset) else round(float(record.offset), 12)
    return (record.source, record.mode, record.topology, beta, rref, rhalf, record.scale_mode, offset)


def record_display_name(record: CoverageFitRecord) -> str:
    topology = {
        'common': 'common',
        'by_system': 'by system',
        'current': 'current LFR',
    }.get(record.topology, record.topology)
    source = record.source_label
    if np.isfinite(record.beta) and np.isfinite(record.r_ref):
        if np.isfinite(record.r_half):
            source += f" beta={record.beta:g}, R1/2={record.r_half:g} (Rlambda={record.r_ref:g})"
        else:
            source += f" beta={record.beta:g}, Rlambda={record.r_ref:g}"
    if np.isfinite(record.offset) and abs(float(record.offset)) > 1e-15:
        source += f", d={record.offset:g}"
    return f"{topology} | {source} | {record.label}"


def _parameter_values_for_export(record: CoverageFitRecord) -> dict[str, Any]:
    values = {}
    for name, attrs in (record.parameters or {}).items():
        if not isinstance(attrs, dict) or "val" not in attrs:
            continue
        value = attrs.get("val")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (float, int, bool, str)) or value is None:
            values[str(name)] = value
        else:
            try:
                values[str(name)] = float(value)
            except Exception:
                values[str(name)] = str(value)
    return values


def _free_parameter_values_for_export(record: CoverageFitRecord) -> dict[str, Any]:
    all_values = _parameter_values_for_export(record)
    free = {}
    for name, attrs in (record.parameters or {}).items():
        if not isinstance(attrs, dict):
            continue
        active = bool(attrs.get("vary", False) and attrs.get("fit", True))
        if active and name in all_values:
            free[str(name)] = all_values[name]
    return free


def _parameter_boundary_flags_for_export(record: CoverageFitRecord, fraction=1e-3) -> dict[str, str]:
    """Flag active parameters numerically close to their configured bounds."""
    flags = {}
    fraction = float(fraction)
    for name, attrs in (record.parameters or {}).items():
        if not isinstance(attrs, dict) or not bool(attrs.get('vary', False) and attrs.get('fit', True)):
            continue
        try:
            value = float(attrs.get('val'))
            lower = float(attrs.get('min'))
            upper = float(attrs.get('max'))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper) and upper > lower):
            continue
        tol = max(1e-10, fraction * (upper - lower))
        if value <= lower + tol:
            flags[str(name)] = 'lower'
        elif value >= upper - tol:
            flags[str(name)] = 'upper'
    return flags


def _feii_excitation_diagnostics_for_export(record: CoverageFitRecord) -> dict[str, Any]:
    """Return FeII excited/ground column ratios from fitted log10 columns."""
    values = _parameter_values_for_export(record)
    systems = set()
    for name in values:
        match = re.match(r'^N_(\d+)_FeII(?:j[12])?$', str(name))
        if match:
            systems.add(int(match.group(1)))
    out = {}
    for system in sorted(systems):
        ground_name = f'N_{system}_FeII'
        if ground_name not in values:
            continue
        try:
            ground = float(values[ground_name])
        except (TypeError, ValueError):
            continue
        row = {'logN_ground': ground}
        for level in ('FeIIj1', 'FeIIj2'):
            name = f'N_{system}_{level}'
            if name not in values:
                continue
            try:
                excited = float(values[name])
            except (TypeError, ValueError):
                continue
            delta = excited - ground
            row[f'logN_{level}'] = excited
            row[f'log10_{level}_over_FeII'] = delta
            if -300.0 < delta < 300.0:
                row[f'{level}_over_FeII'] = 10.0 ** delta
        out[f'system_{system}'] = row
    return out


def _source_family_for_export(record: CoverageFitRecord) -> str:
    if record.source == "achromatic":
        return "achromatic"
    if record.source == "ss73":
        return "powerlaw_temperature_disk"
    return str(record.source)


def export_coverage_records_csv(records, filename) -> str:
    filename = str(filename)
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "source_family", "source_label", "mode", "label", "realization", "topology",
            "beta", "R_ref_over_R_abs", "R_lambda_ref_over_R_abs", "R_half_ref_over_R_abs", "scale_parameterization", "offset_over_R_abs", "success", "timed_out", "chi2", "ndata",
            "nvarys", "dof", "redchi", "AIC", "AICc", "BIC",
            "parameter_values_json", "free_parameter_values_json",
            "parameter_bound_flags_json", "feii_excitation_diagnostics_json", "message",
        ])
        for record in records:
            writer.writerow([
                record.source, _source_family_for_export(record), record.source_label, record.mode, record.label, record.realization, record.topology,
                record.beta, record.r_ref, record.r_ref, record.r_half, record.scale_mode, record.offset, record.success, record.timed_out, record.chi2, record.ndata, record.nvarys,
                record.dof, record.redchi, record.aic, record.aicc, record.bic,
                json.dumps(_parameter_values_for_export(record), sort_keys=True, separators=(",", ":")),
                json.dumps(_free_parameter_values_for_export(record), sort_keys=True, separators=(",", ":")),
                json.dumps(_parameter_boundary_flags_for_export(record), sort_keys=True, separators=(",", ":")),
                json.dumps(_feii_excitation_diagnostics_for_export(record), sort_keys=True, separators=(",", ":")),
                record.message,
            ])
    return filename


def _matching_achromatic_record(parent, records, mode, topology):
    history = getattr(parent, "coverageComparisonHistory", {}) or {}
    history_records = list(history.values()) if isinstance(history, dict) else []
    candidates = (
        list(records)
        + list(getattr(parent, "coverageComparisonResults", []) or [])
        + history_records
    )
    current_ndata = None
    try:
        residual = np.asarray(parent.s.chi(), dtype=float).ravel()
        current_ndata = int(np.count_nonzero(np.isfinite(residual)))
    except Exception:
        pass
    for record in reversed(candidates):
        if (
            record.source == "achromatic"
            and record.mode == mode
            and (record.topology == topology or mode == "full")
            and record.usable
            and record.success
            and not record.timed_out
            and (current_ndata in {None, 0} or record.ndata == current_ndata)
        ):
            return record
    return None


def _remember_record(parent, record: CoverageFitRecord) -> None:
    """Keep the best completed result for subsequent comparison fits."""
    if not record.usable:
        return
    history = getattr(parent, "coverageComparisonHistory", None)
    if history is None or not isinstance(history, dict):
        history = {}
        parent.coverageComparisonHistory = history
    beta_key = None if not np.isfinite(record.beta) else round(float(record.beta), 12)
    rref_key = None if not np.isfinite(record.r_ref) else round(float(record.r_ref), 12)
    offset_key = None if not np.isfinite(record.offset) else round(float(record.offset), 12)
    key = (record.topology, record.source, record.mode, beta_key, rref_key, offset_key)
    old = history.get(key)
    if old is None:
        history[key] = deepcopy(record)
        return
    old_rank = (bool(old.timed_out), not bool(old.success), old.chi2 if np.isfinite(old.chi2) else math.inf)
    new_rank = (bool(record.timed_out), not bool(record.success), record.chi2 if np.isfinite(record.chi2) else math.inf)
    if new_rank < old_rank:
        history[key] = deepcopy(record)


def _matching_achromatic_seed(parent, records, mode, topology):
    record = _matching_achromatic_record(parent, records, mode, topology)
    return deepcopy(record.parameters) if record is not None else None


def _ss73_scale_diagnostic(parent):
    try:
        spectrum = parent.s[parent.s.ind]
        mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
        observed = np.asarray(spectrum.spec.x(), dtype=float)
        observed = observed[mask & np.isfinite(observed) & (observed > 0)]
        if not observed.size:
            return None
        source = source_from_fit(parent.fit)
        src_wave, _ = source_wavelengths_from_observed(
            np.array([np.min(observed), np.max(observed)], dtype=float), parent.fit
        )
        fractions = [float(source.flux_fraction_inside_radius(1.0, w)) for w in src_wave]
        return min(fractions), max(fractions)
    except Exception:
        return None


def apply_record(parent, record: CoverageFitRecord, redraw: bool = True) -> None:
    """Activate a stored fit without rerunning the minimizer."""
    restore_fit_parameters(parent.fit, record.parameters)
    parent.fitType = "regular"
    try:
        parent.options("fitType", "regular")
    except Exception:
        pass
    parent.fit.update(redraw=False)

    missing_curve = len(record.curves) < len(parent.s)
    for spectrum, curve in zip(parent.s, record.curves):
        if curve is None:
            missing_curve = True
            continue
        x, y = curve
        spectrum.set_fit(x=x, y=y)
        if redraw:
            spectrum.update_fit()
            spectrum.set_gfit()

    if missing_curve:
        parent.s.prepareFit(all=False)
        parent.s.calcFit(recalc=True, redraw=redraw)

    if getattr(parent, "fitModel", None) is not None:
        try:
            parent.fitModel.refresh()
        except Exception:
            pass

    try:
        parent.chiSquare.setText(
            "  coverage model: {0}; chi2 / dof / AICc / BIC = "
            "{1:.2f} / {2:d} / {3:.2f} / {4:.2f}".format(
                record.label, record.chi2, record.dof, record.aicc, record.bic
            )
        )
    except Exception:
        pass

    if redraw:
        try:
            parent.s.calcFitComps(recalc=True)
        except Exception:
            pass
        try:
            parent.s.redraw()
        except Exception:
            pass


def _redshift_drift(snapshot, fit):
    c = 299792.458
    out = []
    for i, system in enumerate(fit.sys):
        name = f'z_{i}'
        if name not in snapshot:
            continue
        z0 = float(snapshot[name].get('val', system.z.val))
        dv = c * (float(system.z.val) - z0) / (1.0 + z0)
        out.append((i, dv))
    return out


def fit_all_coverage_models(parent, modes=None, topology='current', sources=None) -> list[CoverageFitRecord]:
    """Fit selected source/coverage models without mutating the live fit at the end."""
    if len(parent.fit.sys) == 0:
        raise ValueError(
            "No absorption systems are present in the current Fit model. "
            "A coverage comparison without any system has tau=0 everywhere "
            "and is not physically meaningful."
        )
    topology = str(topology or 'current')
    if topology != 'current':
        parent.configureCoverageLFRs(topology, reset_vary=True)
    elif not getattr(parent.fit, "cf_fit", False) or parent.fit.cf_num < 1:
        raise ValueError("Create an LFR or choose an automatic LFR setup.")
    if len(parent.fit.list_fit()) == 0:
        raise ValueError("No parameters are selected for fitting.")

    available_modes = available_comparison_modes(parent)
    if modes is None:
        modes = list(available_modes)
    else:
        modes = [mode for mode in modes if mode in available_modes]
    if not modes:
        raise ValueError("No applicable coverage models were selected.")
    if sources is None:
        sources = [parent.fit.source_model() if hasattr(parent.fit, 'source_model') else 'achromatic']
    sources = [source_mode_name(source) for source in sources]
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError("No source models were selected.")
    if topology == 'by_system' and 'cfov' in modes:
        modes = [mode for mode in modes if mode != 'cfov']
        print('[coverage] overlap Cf skipped: it requires a common LFR containing both systems.', flush=True)
    if not modes:
        raise ValueError('No selected coverage model is compatible with this LFR setup.')

    parent.fit.update(redraw=False)
    parent.s.prepareFit(all=False)
    audit = parent.coverageAssignmentAudit(prepared=True)
    local_modes = {
        'ppc', 'powerlaw', 'powerlaw_ppc', 'powerlaw_ppc_sysa',
        'powerlaw_radial', 'powerlaw_radial_ppc', 'two_zone', 'cfov'
    }
    if audit['unassigned'] and any(mode in local_modes for mode in modes):
        raise ValueError(
            'Some fitted line/system entries are outside the LFR assignment: '
            + ', '.join(audit['unassigned'][:8])
            + '. Choose common/by-system automatic LFRs or fix the manual LFRs.'
        )
    print(f"[coverage] LFR topology={topology}", flush=True)
    for row in audit['topology']:
        print(
            f"[coverage] LFR{row['region']} {row['left']:.3f}-{row['right']:.3f} "
            f"{row['assignment']}",
            flush=True,
        )
    original_snapshot = snapshot_fit_parameters(parent.fit)
    original_curves = snapshot_model_curves(parent)
    original_fit_type = parent.fitType
    original_animate = getattr(parent, "animateFit", False)
    original_option_fit_type = None
    try:
        original_option_fit_type = parent.options("fitType")
    except Exception:
        pass

    jobs = [(source, mode) for source in sources for mode in modes]
    progress = QProgressDialog(
        "Preparing coverage-model comparison...", "Cancel", 0, len(jobs), parent
    )
    progress.setWindowTitle("Coverage-model comparison")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    parent.fitType = "regular"
    parent.animateFit = False
    try:
        parent.options("fitType", "regular")
    except Exception:
        pass

    records: list[CoverageFitRecord] = []
    fatal_error: Exception | None = None
    comparison_started = time.perf_counter()
    print(
        "[coverage] START comparison | "
        f"jobs={len(jobs)} sources={','.join(sources)} models={len(modes)} systems={len(parent.fit.sys)} "
        f"fit_parameters={len(parent.fit.list_fit())}",
        flush=True,
    )
    try:
        for index, (source, mode) in enumerate(jobs):
            if progress.wasCanceled():
                break
            label = COVERAGE_MODE_LABELS[coverage_mode_index(mode)]
            beta_for_label = float(parent.fit.srcbeta.val) if source == "ss73" else math.nan
            source_label = source_model_label(source, beta_for_label)
            job_label = f"{source_label} | {label}"
            progress.setLabelText(f"Fitting {job_label} ({index + 1}/{len(jobs)})")
            progress.setValue(index)
            QApplication.processEvents()

            record = CoverageFitRecord(
                mode=mode,
                label=label,
                source=source,
                source_label=source_label,
                realization=(
                    "legacy analytic achromatic"
                    if source == "achromatic"
                    else spatial_realization_label(mode)
                ),
                topology=topology,
                beta=(float(parent.fit.srcbeta.val) if source == 'ss73' else math.nan),
                r_ref=(float(parent.fit.srcrref.val) if source == 'ss73' else math.nan),
                r_half=(
                    float(half_light_from_r_lambda(parent.fit.srcrref.val, parent.fit.srcbeta.val))
                    if source == 'ss73' else math.nan
                ),
                scale_mode=('legacy_r_lambda' if source == 'ss73' else 'none'),
            )
            model_started = time.perf_counter()
            print(
                f"[coverage] MODEL {index + 1}/{len(jobs)} START: {job_label}",
                flush=True,
            )

            if source == "ss73" and mode == "full":
                achromatic_full = _matching_achromatic_record(parent, records, mode, topology)
                if achromatic_full is not None:
                    for name in ("chi2", "ndata", "nvarys", "dof", "redchi", "aic", "aicc", "bic"):
                        setattr(record, name, getattr(achromatic_full, name))
                    record.success = True
                    record.timed_out = False
                    record.parameters = deepcopy(achromatic_full.parameters)
                    if "srcmode" in record.parameters:
                        record.parameters["srcmode"]["val"] = source_mode_index("ss73")
                    record.curves = deepcopy(achromatic_full.curves)
                    record.message = (
                        "source-invariant full coverage; reused matching completed achromatic fit"
                    )
                    records.append(record)
                    _remember_record(parent, record)
                    restore_fit_parameters(parent.fit, record.parameters)
                    parent.fit.set_source_model("ss73")
                    parent.fit.update(redraw=False)
                    parent.s.prepareFit(all=False)
                    parent.s.calcFit(recalc=True, redraw=False)
                    model_elapsed = time.perf_counter() - model_started
                    print(
                        f"[coverage] MODEL {index + 1}/{len(jobs)} END: {job_label} | "
                        f"chi2={record.chi2:.6g} redchi={record.redchi:.6g} "
                        f"BIC={record.bic:.6g} elapsed={model_elapsed/60.0:.1f} min "
                        "[source-invariant reuse]",
                        flush=True,
                    )
                    continue

            seed_snapshot = deepcopy(original_snapshot)
            seeded_from_achromatic = False
            if source == "ss73":
                achromatic_seed = _matching_achromatic_seed(parent, records, mode, topology)
                if achromatic_seed is not None:
                    seed_snapshot = achromatic_seed
                    seeded_from_achromatic = True
            attempts = [(deepcopy(seed_snapshot), None)]
            if mode == "cfov":
                attempts = _cfov_seed_snapshots(parent, seed_snapshot, records, source=source)

            best_attempt = None
            attempt_messages = []
            source_invariant_full = source == 'ss73' and mode == 'full'
            for attempt_index, (attempt_snapshot, cfov_direction) in enumerate(attempts):
                try:
                    restore_fit_parameters(parent.fit, attempt_snapshot)
                    parent.fit.set_source_model('achromatic' if source_invariant_full else source)
                    parent.fit.set_coverage_model(mode)
                    parent.fit.configure_coverage_parameters(reset_vary=True)
                    if mode == "cfov" and cfov_direction is not None:
                        _initialize_cfov_from_ppc(parent, cfov_direction)

                    progress_label = (
                        f"{job_label} [{index + 1}/{len(jobs)}]"
                        if len(attempts) == 1
                        else
                        f"{job_label} [{index + 1}/{len(jobs)}] "
                        f"start {attempt_index + 1}/{len(attempts)}"
                    )
                    result = parent.fitAbs(
                        timer=False,
                        redraw=False,
                        return_result=True,
                        quiet=True,
                        progress_label=progress_label,
                        progress_every=50,
                        timeout_seconds=600.0,
                        stagnation_patience=max(220, 14 * max(1, len(parent.fit.list_fit()))),
                        stagnation_min_evals=max(120, 7 * max(1, len(parent.fit.list_fit()))),
                        stagnation_rel_tol=2e-5,
                        stagnation_abs_tol=0.05,
                    )

                    final_residual = np.asarray(parent.s.chi(), dtype=float)
                    metrics = _metrics_from_result(result, residual=final_residual)
                    message = str(getattr(result, "message", ""))
                    if any(name.startswith('FeIIj') for name in parent.fit.list_species()):
                        large = [(i, dv) for i, dv in _redshift_drift(original_snapshot, parent.fit) if abs(dv) > 200.0]
                        if large:
                            warning = 'large z drift: ' + ', '.join(f'sys{i} {dv:+.0f} km/s' for i, dv in large)
                            message = f'{message}; {warning}' if message else warning
                    candidate = {
                        "metrics": metrics,
                        "success": bool(getattr(result, "success", True)),
                        "timed_out": bool(getattr(result, "timed_out", False)),
                        "message": message,
                        "parameters": snapshot_fit_parameters(parent.fit),
                        "curves": snapshot_model_curves(parent),
                        "attempt": attempt_index + 1,
                    }
                    if source_invariant_full and "srcmode" in candidate["parameters"]:
                        candidate["parameters"]["srcmode"]["val"] = source_mode_index("ss73")
                        note = "source-invariant full coverage fitted through achromatic evaluator"
                        candidate["message"] = (
                            f"{candidate['message']}; {note}" if candidate["message"] else note
                        )
                    if best_attempt is None:
                        best_attempt = candidate
                    else:
                        candidate_rank = (
                            bool(candidate["timed_out"]),
                            not bool(candidate["success"]),
                            candidate["metrics"]["chi2"] if np.isfinite(candidate["metrics"]["chi2"]) else math.inf,
                        )
                        best_rank = (
                            bool(best_attempt["timed_out"]),
                            not bool(best_attempt["success"]),
                            best_attempt["metrics"]["chi2"] if np.isfinite(best_attempt["metrics"]["chi2"]) else math.inf,
                        )
                        if candidate_rank < best_rank:
                            best_attempt = candidate
                    attempt_messages.append(
                        f"start {attempt_index + 1}: chi2={metrics['chi2']:.3f}"
                    )
                except Exception as exc:
                    attempt_messages.append(
                        f"start {attempt_index + 1}: {type(exc).__name__}: {exc}"
                    )

            if best_attempt is None:
                record.success = False
                record.message = "; ".join(attempt_messages) or "fit failed"
            else:
                for name, value in best_attempt["metrics"].items():
                    setattr(record, name, value)
                record.success = best_attempt["success"]
                record.timed_out = best_attempt["timed_out"]
                record.message = best_attempt["message"]
                if source == "ss73":
                    spatial_note = f"power-law disk spatial realization: {record.realization}"
                    record.message = f"{record.message}; {spatial_note}" if record.message else spatial_note
                    if seeded_from_achromatic:
                        record.message += "; initialized from matching completed achromatic fit"
                    scale_diag = _ss73_scale_diagnostic(parent)
                    if scale_diag is not None:
                        lo, hi = scale_diag
                        record.message += f"; circular flux inside R_abs={lo:.3f}-{hi:.3f}"
                        if lo < 0.9:
                            record.message += "; source/absorber scale is strongly influential"
                if len(attempts) > 1:
                    detail = (
                        f"overlap multi-start chose {best_attempt['attempt']}/{len(attempts)}; "
                        + ", ".join(attempt_messages)
                    )
                    record.message = f"{record.message}; {detail}" if record.message else detail
                record.parameters = best_attempt["parameters"]
                record.curves = best_attempt["curves"]

                restore_fit_parameters(parent.fit, record.parameters)
                parent.fit.update(redraw=False)
                parent.s.prepareFit(all=False)
                parent.s.calcFit(recalc=True, redraw=False)

            records.append(record)
            _remember_record(parent, record)
            model_elapsed = time.perf_counter() - model_started
            if np.isfinite(record.chi2):
                print(
                    f"[coverage] MODEL {index + 1}/{len(jobs)} END: {job_label} | "
                    f"chi2={record.chi2:.6g} redchi={record.redchi:.6g} "
                    f"BIC={record.bic:.6g} elapsed={model_elapsed/60.0:.1f} min",
                    flush=True,
                )
            else:
                print(
                    f"[coverage] MODEL {index + 1}/{len(jobs)} END: {job_label} | "
                    f"FAILED elapsed={model_elapsed/60.0:.1f} min | {record.message}",
                    flush=True,
                )

        progress.setValue(len(jobs))
    except Exception as exc:
        fatal_error = exc
    finally:
        progress.close()
        parent.animateFit = original_animate

    if fatal_error is not None:
        restore_fit_parameters(parent.fit, original_snapshot)
        parent.fitType = original_fit_type
        if original_option_fit_type is not None:
            try:
                parent.options("fitType", original_option_fit_type)
            except Exception:
                pass
        raise fatal_error

    counts = [r.ndata for r in records if r.parameters and np.isfinite(r.chi2)]
    if counts:
        reference_ndata = max(set(counts), key=counts.count)
        for record in records:
            if record.parameters and record.ndata != reference_ndata:
                record.comparable = False
                suffix = f"different fitted-pixel count ({record.ndata} vs {reference_ndata})"
                record.message = f"{record.message}; {suffix}" if record.message else suffix

    preferred = best_record(records, "BIC")
    diagnostic_best = diagnostic_best_record(records, "BIC")
    # Restore the pre-run state; results are applied only from the dialog.
    restore_fit_parameters(parent.fit, original_snapshot)
    for spectrum, curve in zip(parent.s, original_curves):
        if curve is not None:
            spectrum.set_fit(x=curve[0], y=curve[1])
            spectrum.update_fit()
    parent.fitType = original_fit_type
    if original_option_fit_type is not None:
        try:
            parent.options("fitType", original_option_fit_type)
        except Exception:
            pass
    try:
        if getattr(parent, 'fitModel', None) is not None:
            parent.fitModel.refresh()
    except Exception:
        pass

    total_elapsed = time.perf_counter() - comparison_started
    preferred_label = (
        f"{preferred.source_label} | {preferred.label}" if preferred is not None else "none"
    )
    diagnostic_label = "none"
    if diagnostic_best is not None:
        diagnostic_label = f"{diagnostic_best.source_label} | {diagnostic_best.label}"
        if diagnostic_best.timed_out:
            diagnostic_label += " [timed out]"
        elif not diagnostic_best.success:
            diagnostic_label += " [incomplete]"
    print(
        f"[coverage] END comparison | completed={len(records)}/{len(jobs)} "
        f"preferred_completed(BIC)={preferred_label} best_diagnostic(BIC)={diagnostic_label} "
        f"elapsed={total_elapsed/60.0:.1f} min",
        flush=True,
    )

    parent.coverageComparisonResults = records
    return records


def _infer_auto_topology(parent) -> str:
    try:
        if getattr(parent, 'fitModel', None) is not None:
            value = str(parent.fitModel.inferCoverageTopology()).lower().replace(' ', '_')
            if value in {'common', 'by_system'}:
                return value
    except Exception:
        pass
    try:
        audit = parent.coverageAssignmentAudit()
        rows = audit.get('topology', [])
        if len(rows) == 1 and rows[0].get('assignment') == 'all_all':
            return 'common'
        if len(rows) == len(parent.fit.sys) and rows:
            expected = [f'sys{i}_all' for i in range(len(parent.fit.sys))]
            actual = [row.get('assignment') for row in rows]
            if actual == expected:
                return 'by_system'
    except Exception:
        pass
    return 'current'


def export_chromatic_record_diagnostics(parent, record: CoverageFitRecord, prefix: str) -> dict[str, str]:
    """Export spatial maps and the exact Spectro model for one chromatic record."""
    if record is None or not record.usable or record.source != 'ss73':
        raise ValueError('Select a usable chromatic power-law-disk grid result.')
    prefix_path = Path(prefix).expanduser()
    if prefix_path.suffix:
        prefix_path = prefix_path.with_suffix('')
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    original_snapshot = snapshot_fit_parameters(parent.fit)
    original_curves = snapshot_model_curves(parent)
    original_topology = _infer_auto_topology(parent)
    outputs = {}
    try:
        if record.topology in {'common', 'by_system'}:
            parent.configureCoverageLFRs(record.topology, reset_vary=False)
        restore_fit_parameters(parent.fit, record.parameters)
        parent.fit.update(redraw=False)
        parent.s.prepareFit(all=False)
        parent.s.calcFit(recalc=True, redraw=False)

        with __import__('tempfile').TemporaryDirectory() as tmpdir:
            bridge = Path(tmpdir) / 'chromatic_bridge.csv'
            bridge_outputs = parent.exportCoverageProfiles(str(bridge))
            highres = Path(bridge_outputs['highres'])
            rows = []
            with highres.open('r', newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError('No high-resolution tau rows were produced for the selected result.')

            nsys = len(parent.fit.sys)
            region_ids = list(range(int(getattr(parent.fit, 'cf_num', 0))))
            by_window = {}
            for row in rows:
                tau_strength = 0.0
                for k in range(nsys):
                    try:
                        tau_strength += float(row.get(f'tau{k}', 0.0) or 0.0)
                    except Exception:
                        pass
                name = str(row.get('line', 'window'))
                old = by_window.get(name)
                if old is None or tau_strength > old[0]:
                    by_window[name] = (tau_strength, row)
            chosen = [v[1] for v in by_window.values()]
            chosen.sort(key=lambda row: float(row.get('lambda_src_A', 0.0) or 0.0))
            if len(chosen) > 3:
                chosen = [chosen[0], chosen[len(chosen)//2], chosen[-1]]

            source = source_from_fit(parent.fit)
            mode = parent.fit.coverage_model()
            wavelengths = []
            map_payload = {}
            max_offset = max(
                [float(parent.fit.getValue(f'covd_{r}')) for r in region_ids if hasattr(parent.fit, f'covd_{r}')]
                or [0.0]
            )
            source_obs = np.array([float(row['lambda_src_A']) for row in chosen], dtype=float)
            source_rest, _ = source_wavelengths_from_observed(source_obs, parent.fit)
            wavelengths = [float(v) for v in source_rest]
            max_r_lambda = max(float(source.radius(w)) for w in wavelengths)
            extent = max(1.35 + max_offset, 2.5 * max_r_lambda)
            nmap = 220
            axis = np.linspace(-extent, extent, nmap, endpoint=False) + extent / nmap
            xx, yy = np.meshgrid(axis, axis, indexing='xy')

            fig, axes = plt.subplots(
                len(chosen), 2 + max(1, len(region_ids)),
                figsize=(4.0 * (2 + max(1, len(region_ids))), 3.7 * len(chosen)),
                squeeze=False,
            )
            metadata_rows = []
            for iw, (row, lam) in enumerate(zip(chosen, wavelengths)):
                radius = np.hypot(xx, yy)
                brightness = np.asarray(source.brightness(radius, lam), dtype=float)
                finite = np.isfinite(brightness)
                if not np.any(finite):
                    raise ValueError('Source brightness map is non-finite.')
                cap = float(np.nanmax(brightness[finite]))
                brightness = np.where(finite, brightness, cap)
                full_tau = sum(float(row.get(f'tau_full_sys{k}', 0.0) or 0.0) for k in range(nsys))
                transmission_factor = np.full_like(brightness, np.exp(-max(full_tau, 0.0)))
                tau_map = np.full_like(brightness, max(full_tau, 0.0))

                region_maps = {}
                for region in region_ids:
                    cf_default = parent.fit.getValue(f'cf_{region}') if hasattr(parent.fit, f'cf_{region}') else 1.0
                    params = {
                        'cf': cf_default,
                        'log10_a': parent.fit.getValue(f'cova_{region}') if hasattr(parent.fit, f'cova_{region}') else 0.0,
                        'offset': parent.fit.getValue(f'covd_{region}') if hasattr(parent.fit, f'covd_{region}') else 0.0,
                        'log10_a_by_system': {
                            k: parent.fit.getValue(f'covasys_{k}') for k in range(nsys)
                            if hasattr(parent.fit, f'covasys_{k}')
                        },
                        'tau_ratio': parent.fit.getValue(f'covr_{region}') if hasattr(parent.fit, f'covr_{region}') else 0.1,
                    }
                    systems_here = [k for k in range(nsys) if float(row.get(f'tau_lfr{region}_sys{k}', 0.0) or 0.0) > 0]
                    if not systems_here:
                        systems_here = list(range(nsys))
                    # For system-dependent xPL maps a separate G_N exists per system.
                    gn_display = np.zeros_like(brightness)
                    weight_sum = 0.0
                    region_tau_map = np.zeros_like(brightness)
                    for order, k in enumerate(sorted(systems_here)):
                        gn = spatial_column_scale_map(mode, params, xx, yy, system_id=k, system_order=order)
                        tau_ref = float(row.get(f'tau_lfr{region}_sys{k}', 0.0) or 0.0)
                        region_tau_map += tau_ref * gn
                        w = max(tau_ref, 0.0)
                        gn_display += w * gn
                        weight_sum += w
                        # Export physical column maps whenever the dominant-window
                        # species can be resolved in this system.
                        window_text = str(row.get('line', ''))
                        species_guess = window_text.split('_', 2)[1] if '_' in window_text else ''
                        try:
                            logn_ref = float(parent.fit.sys[k].sp[species_guess].N.val)
                        except Exception:
                            logn_ref = None
                        if logn_ref is not None:
                            map_payload[f'lambda{iw}_N_cm2_lfr{region}_sys{k}'] = (10.0 ** logn_ref) * gn
                    if weight_sum > 0:
                        gn_display /= weight_sum
                    else:
                        gn_display = spatial_column_scale_map(mode, params, xx, yy)
                    region_maps[region] = gn_display
                    tau_map += region_tau_map
                    if mode == 'powerlaw_radial_ppc':
                        cf = float(params.get('cf', 1.0))
                        transmission_factor *= (1.0 - cf) + cf * np.exp(-np.clip(region_tau_map, 0.0, 1e6))
                    else:
                        transmission_factor *= np.exp(-np.clip(region_tau_map, 0.0, 1e6))

                transmitted = brightness * transmission_factor

                # Display only: the disk has a bright central cusp, so cap at the 99.5th
                # percentile and use an asinh stretch; exported arrays stay linear.
                positive = brightness[np.isfinite(brightness) & (brightness > 0.0)]
                display_cap = (
                    float(np.nanpercentile(positive, 99.5)) if positive.size
                    else max(float(np.nanmax(brightness)), 1e-300)
                )
                display_cap = max(display_cap, 1e-300)
                stretch = 10.0
                stretch_norm = np.arcsinh(stretch)
                brightness_display = np.arcsinh(
                    stretch * np.clip(brightness, 0.0, display_cap) / display_cap
                ) / stretch_norm
                transmitted_display = np.arcsinh(
                    stretch * np.clip(transmitted, 0.0, display_cap) / display_cap
                ) / stretch_norm

                axes[iw, 0].imshow(brightness_display, origin='lower', extent=[-extent, extent, -extent, extent])
                axes[iw, 0].set_title(rf'$I_\lambda$, $\lambda_{{\rm rest}}$ = {lam:.1f} $\AA$ (asinh display)')
                for j in range(max(1, len(region_ids))):
                    ax = axes[iw, 1 + j]
                    if j < len(region_ids):
                        region = region_ids[j]
                        ax.imshow(region_maps[region], origin='lower', extent=[-extent, extent, -extent, extent])
                        ax.set_title(f'N/Nref, LFR {region}')
                        map_payload[f'lambda{iw}_gn_lfr{region}'] = region_maps[region]
                    else:
                        ax.axis('off')
                axes[iw, -1].imshow(transmitted_display, origin='lower', extent=[-extent, extent, -extent, extent])
                axes[iw, -1].set_title(f'Transmitted (asinh); model={float(row.get("model_pre_lsf", np.nan)):.3f}')
                for ax in axes[iw, :]:
                    if ax.axison:
                        ax.set_xlabel('x / R_abs')
                        ax.set_ylabel('y / R_abs')
                map_payload[f'lambda{iw}_brightness'] = brightness
                map_payload[f'lambda{iw}_transmitted'] = transmitted
                map_payload[f'lambda{iw}_tau_map'] = tau_map
                metadata_rows.append({
                    'window': row.get('line'),
                    'lambda_source_rest_A': lam,
                    'wavelength_highres_A': float(row.get('wavelength_A', np.nan)),
                    'velocity_kms': float(row.get('velocity_kms', np.nan)),
                    'model_pre_lsf': float(row.get('model_pre_lsf', np.nan)),
                    'model_post_lsf_raw': float(row.get('model_post_lsf_raw', np.nan)),
                    'model_final_spectro': float(row.get('model_final_spectro', np.nan)),
                })

            fig.suptitle(record_display_name(record))
            fig.tight_layout()
            map_png = str(prefix_path) + '_chromatic_maps.png'
            fig.savefig(map_png, dpi=180)
            plt.close(fig)
            outputs['maps_png'] = map_png

            map_npz = str(prefix_path) + '_chromatic_maps.npz'
            np.savez_compressed(map_npz, x=axis, y=axis, source_rest_wavelength_A=np.asarray(wavelengths), **map_payload)
            outputs['maps_npz'] = map_npz

            # Companion data/model plot uses the exact currently recalculated Spectro model.
            spectrum = parent.s[parent.s.ind]
            try:
                xobs = np.asarray(spectrum.spec.norm.x, dtype=float)
                yobs = np.asarray(spectrum.spec.norm.y, dtype=float)
                err = np.asarray(spectrum.spec.norm.err, dtype=float)
                mask = np.asarray(spectrum.fit_mask.norm.x, dtype=bool)
            except Exception:
                xobs = np.asarray(spectrum.spec.x(), dtype=float)
                yobs = np.asarray(spectrum.spec.y(), dtype=float)
                err = np.asarray(spectrum.spec.err(), dtype=float)
                mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
            model_x = np.asarray(spectrum.fit.line.norm.x, dtype=float)
            model_y = np.asarray(spectrum.fit.line.norm.y, dtype=float)
            model = np.interp(xobs, model_x, model_y, left=np.nan, right=np.nan)
            good = mask & np.isfinite(xobs) & np.isfinite(yobs) & np.isfinite(err) & (err > 0) & np.isfinite(model)
            fig, (ax, resax) = plt.subplots(2, 1, sharex=True, figsize=(11, 7), gridspec_kw={'height_ratios':[3,1]})
            ax.plot(xobs[good], yobs[good], '.', ms=2.5, label='data')

            # Split curves at large wavelength gaps so separate fit windows are not joined.
            good_idx = np.flatnonzero(good)
            if good_idx.size:
                dx_all = np.diff(xobs[np.isfinite(xobs)])
                dx_pos = dx_all[np.isfinite(dx_all) & (dx_all > 0.0)]
                typical_dx = float(np.nanmedian(dx_pos)) if dx_pos.size else 0.0
                gap_threshold = max(0.5, 10.0 * typical_dx)
                break_at = np.flatnonzero(np.diff(xobs[good_idx]) > gap_threshold) + 1
                segments = np.split(good_idx, break_at)
            else:
                segments = []
            for iseg, seg in enumerate(segments):
                if seg.size == 0:
                    continue
                ax.plot(xobs[seg], model[seg], lw=1.2,
                        label='Spectro model' if iseg == 0 else None)
                resax.plot(xobs[seg], (yobs[seg] - model[seg]) / err[seg], lw=0.8)
            ax.set_ylabel('Normalized flux')
            ax.legend(loc='best')
            ax.set_title(record_display_name(record))
            resax.axhline(0.0, lw=0.8)
            resax.set_ylabel('Residual / sigma')
            resax.set_xlabel('Observed wavelength')
            fig.tight_layout()
            spectrum_png = str(prefix_path) + '_spectrum.png'
            fig.savefig(spectrum_png, dpi=180)
            plt.close(fig)
            outputs['spectrum_png'] = spectrum_png

            meta = {
                'record': {
                    'mode': record.mode, 'topology': record.topology, 'beta': record.beta,
                    'R_ref_over_R_abs': record.r_ref, 'offset_over_R_abs': record.offset,
                    'chi2': record.chi2, 'BIC': record.bic,
                },
                'source_model': source_model_label(record.source, record.beta),
                'spatial_realization': record.realization,
                'map_extent_R_abs': extent,
                'selected_intrinsic_pixels': metadata_rows,
                'note': 'Transmitted maps use the exact intrinsic tau channels at the selected high-resolution pixel, before LSF. N/Nref panels show the fitted dimensionless column field; NPZ also stores N_cm2 maps where the dominant-window species can be resolved. PNG source/transmitted panels use a display-only capped asinh stretch; NPZ brightness/transmitted arrays remain exact linear values. Spectrum plot uses the recalculated Spectro model after coverage and instrumental processing and does not connect independent fit windows.',
            }
            meta_path = str(prefix_path) + '_chromatic_meta.json'
            Path(meta_path).write_text(json.dumps(meta, indent=2), encoding='utf-8')
            outputs['metadata'] = meta_path
    finally:
        try:
            if original_topology in {'common', 'by_system'}:
                parent.configureCoverageLFRs(original_topology, reset_vary=False)
        except Exception:
            pass
        restore_fit_parameters(parent.fit, original_snapshot)
        for spectrum, curve in zip(parent.s, original_curves):
            if curve is not None:
                spectrum.set_fit(x=curve[0], y=curve[1])
                spectrum.update_fit()
        try:
            if getattr(parent, 'fitModel', None) is not None:
                parent.fitModel.refresh()
        except Exception:
            pass
    return outputs


class CoverageComparisonDialog(QDialog):
    """Results table, model switcher, and comparison plot."""

    CRITERIA = ("BIC", "AICc", "AIC", "chi2", "Reduced chi2")

    def __init__(self, parent, records: list[CoverageFitRecord] | None = None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.records = list(records if records is not None else getattr(parent, "coverageComparisonResults", []))
        self.setWindowTitle("Coverage-model comparison")
        self.resize(1050, 520)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Preferred-model criterion:"))
        self.criterion = QComboBox()
        self.criterion.addItems(self.CRITERIA)
        self.criterion.setCurrentText("BIC")
        self.criterion.currentTextChanged.connect(self.refresh_table)
        controls.addWidget(self.criterion)

        controls.addWidget(QLabel("Plot exposure:"))
        self.exposure = QComboBox()
        self.exposure.addItems([f"exp {i}" for i in range(len(parent.s))])
        if len(parent.s):
            active_index = getattr(parent.s, "ind", 0)
            active_index = 0 if active_index is None else int(active_index)
            self.exposure.setCurrentIndex(min(len(parent.s) - 1, max(0, active_index)))
        controls.addWidget(self.exposure)

        self.preview = QCheckBox("Preview selected row in main window (changes live fit)")
        self.preview.setChecked(False)
        controls.addWidget(self.preview)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.summary = QLabel("")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Status", "chi2", "dof", "red. chi2", "k", "AIC", "AICc", "BIC", "Delta", "Message"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.apply_selected())
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply selected")
        self.apply_button.clicked.connect(self.apply_selected)
        buttons.addWidget(self.apply_button)

        self.best_button = QPushButton("Apply preferred")
        self.best_button.clicked.connect(self.apply_preferred)
        buttons.addWidget(self.best_button)

        self.plot_button = QPushButton("Plot comparison")
        self.plot_button.clicked.connect(self.plot_comparison)
        buttons.addWidget(self.plot_button)

        self.export_button = QPushButton("Export table...")
        self.export_button.clicked.connect(self.export_table)
        buttons.addWidget(self.export_button)

        self.export_selected_button = QPushButton("Export selected bundle...")
        self.export_selected_button.setToolTip(
            'Temporarily activates the selected comparison snapshot, exports a complete science bundle, '
            'then restores the live Spectro state.'
        )
        self.export_selected_button.clicked.connect(self.export_selected_bundle)
        buttons.addWidget(self.export_selected_button)

        self.chromatic_export_button = QPushButton("Chromatic maps / export...")
        self.chromatic_export_button.clicked.connect(self.export_chromatic_maps)
        buttons.addWidget(self.chromatic_export_button)

        buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.refresh_table()

    def set_records(self, records: list[CoverageFitRecord]) -> None:
        self.records = list(records)
        self.refresh_table()

    def _selected_record(self) -> CoverageFitRecord | None:
        row = self.table.currentRow()
        return self.records[row] if 0 <= row < len(self.records) else None

    def _selection_changed(self, *_args) -> None:
        if self.preview.isChecked():
            record = self._selected_record()
            if record is not None and record.usable:
                apply_record(self.parent_viewer, record, redraw=True)

    def preferred(self) -> CoverageFitRecord | None:
        return best_record(self.records, self.criterion.currentText())

    @staticmethod
    def _format(value: float) -> str:
        return "-" if not np.isfinite(value) else f"{value:.3f}"

    def refresh_table(self) -> None:
        current_key = record_identity(self._selected_record()) if self._selected_record() else None
        criterion = self.criterion.currentText()
        preferred = self.preferred()
        finite_values = [
            r.criterion(criterion) for r in self.records
            if r.usable and r.success and not r.timed_out and np.isfinite(r.criterion(criterion))
        ]
        minimum = min(finite_values) if finite_values else math.nan

        self.table.blockSignals(True)
        self.table.setRowCount(len(self.records))
        preferred_row = -1
        selected_row = -1
        for row, record in enumerate(self.records):
            if record_identity(record) == current_key:
                selected_row = row
            if preferred is record:
                preferred_row = row
            delta = record.criterion(criterion) - minimum if np.isfinite(minimum) and np.isfinite(record.criterion(criterion)) else math.nan
            if not record.comparable and record.parameters:
                status = "different pixels"
            else:
                if record.timed_out:
                    status = "timed out"
                else:
                    status = "OK" if record.success else ("usable" if record.usable else "failed")
            values = [
                record_display_name(record) + ("  [preferred]" if preferred is record else ""),
                status,
                self._format(record.chi2),
                str(record.dof) if record.dof else "-",
                self._format(record.redchi),
                str(record.nvarys),
                self._format(record.aic),
                self._format(record.aicc),
                self._format(record.bic),
                self._format(delta),
                record.message,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column in range(2, 10):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)

        row_to_select = selected_row if selected_row >= 0 else preferred_row
        if row_to_select >= 0:
            self.table.selectRow(row_to_select)
        self.table.blockSignals(False)

        if preferred is None:
            self.summary.setText("No usable comparison result.")
        else:
            excluded = sum(1 for record in self.records if not record.comparable)
            note = (
                f" {excluded} result(s) were excluded because their fitted-pixel count differed."
                if excluded else ""
            )
            diagnostic = diagnostic_best_record(self.records, criterion)
            extra = ""
            if diagnostic is not None and diagnostic is not preferred and diagnostic.criterion(criterion) < preferred.criterion(criterion):
                status = "timed out" if diagnostic.timed_out else "incomplete"
                extra = (
                    f" Lower diagnostic {criterion}: {record_display_name(diagnostic)} "
                    f"({status}); it is not eligible for automatic scientific selection."
                )
            self.summary.setText(
                f"Preferred completed fit by {criterion}: {record_display_name(preferred)}. "
                "All compared fits use the same masks, pixels, and uncertainties." + note + extra
            )

    def apply_selected(self) -> None:
        record = self._selected_record()
        if record is None or not record.usable:
            QMessageBox.warning(self, "Coverage comparison", "The selected fit is not usable.")
            return
        apply_record(self.parent_viewer, record, redraw=True)

    def apply_preferred(self) -> None:
        record = self.preferred()
        if record is None:
            QMessageBox.warning(self, "Coverage comparison", "No preferred fit is available.")
            return
        apply_record(self.parent_viewer, record, redraw=True)
        self.table.selectRow(self.records.index(record))

    def _spectrum_arrays(self, index: int):
        spectrum = self.parent_viewer.s[index]
        try:
            x = np.asarray(spectrum.spec.norm.x, dtype=float)
            y = np.asarray(spectrum.spec.norm.y, dtype=float)
            err = np.asarray(spectrum.spec.norm.err, dtype=float)
            mask = np.asarray(spectrum.fit_mask.norm.x, dtype=bool)
        except Exception:
            x = np.asarray(spectrum.spec.x(), dtype=float)
            y = np.asarray(spectrum.spec.y(), dtype=float)
            err = np.asarray(spectrum.spec.err(), dtype=float)
            mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
        if mask.size != x.size:
            mask = np.ones(x.size, dtype=bool)
        mask &= np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
        return x, y, err, mask

    def export_selected_bundle(self) -> None:
        record = self._selected_record()
        if record is None or not record.usable:
            QMessageBox.warning(self, 'Coverage comparison', 'Select a usable comparison result first.')
            return
        default = Path(getattr(self.parent_viewer, 'work_folder', '.')) / 'selected_comparison_bundle.zip'
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export selected comparison bundle', str(default), 'ZIP archives (*.zip)'
        )
        if not filename:
            return
        if not filename.lower().endswith('.zip'):
            filename += '.zip'

        original_snapshot = snapshot_fit_parameters(self.parent_viewer.fit)
        original_curves = snapshot_model_curves(self.parent_viewer)
        try:
            apply_record(self.parent_viewer, record, redraw=False)
            self.parent_viewer.exportCoverageBundle(filename, comparison_record=record)
        except Exception as exc:
            QMessageBox.warning(self, 'Export selected comparison failed', str(exc))
            return
        finally:
            restore_fit_parameters(self.parent_viewer.fit, original_snapshot)
            self.parent_viewer.fit.update(redraw=False)
            for spectrum, curve in zip(self.parent_viewer.s, original_curves):
                if curve is not None:
                    spectrum.set_fit(x=curve[0], y=curve[1])
                    spectrum.update_fit()
            try:
                if getattr(self.parent_viewer, 'fitModel', None) is not None:
                    self.parent_viewer.fitModel.refresh()
            except Exception:
                pass
        QMessageBox.information(self, 'Coverage comparison', f'Exported selected comparison state to:\n{filename}')

    def plot_comparison(self) -> None:
        if not self.records or len(self.parent_viewer.s) == 0:
            return
        exp_index = max(0, self.exposure.currentIndex())
        x, y, err, mask = self._spectrum_arrays(exp_index)
        if not np.any(mask):
            QMessageBox.warning(self, "Coverage comparison", "The selected exposure has no fitted pixels.")
            return

        preferred = self.preferred()
        segments = _masked_segments(x, mask)
        if not segments:
            QMessageBox.warning(self, "Coverage comparison", "The selected exposure has no fitted segments.")
            return

        fig, (axis, residual_axis) = plt.subplots(
            2, 1, sharex=True, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
        )

        for segment_index, segment in enumerate(segments):
            axis.plot(
                x[segment], y[segment], ".", markersize=2.5, color="black",
                label="data" if segment_index == 0 else "_nolegend_",
            )

        usable_records = [
            record for record in self.records
            if record.usable and exp_index < len(record.curves)
            and record.curves[exp_index] is not None
        ]
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not palette:
            palette = [f"C{i}" for i in range(max(1, len(usable_records)))]
        model_colors = {
            record_identity(record): palette[i % len(palette)]
            for i, record in enumerate(usable_records)
        }

        plotted = 0
        for record in usable_records:
            curve = record.curves[exp_index]
            model_x, model_y = curve
            label = record_display_name(record) + (" (preferred)" if record is preferred else "")
            color = model_colors[record_identity(record)]
            drew_record = False
            for segment_index, segment in enumerate(segments):
                model = np.interp(
                    x[segment], model_x, model_y, left=np.nan, right=np.nan
                )
                finite = np.isfinite(model)
                if not np.any(finite):
                    continue
                seg = segment[finite]
                mod = model[finite]
                axis.plot(
                    x[seg], mod, linewidth=1.2, color=color,
                    label=label if not drew_record else "_nolegend_",
                )
                residual_axis.plot(
                    x[seg], (y[seg] - mod) / err[seg], linewidth=0.8,
                    color=color,
                    label=record_display_name(record) if not drew_record else "_nolegend_",
                )
                drew_record = True
            if drew_record:
                plotted += 1

        if plotted == 0:
            plt.close(fig)
            QMessageBox.warning(self, "Coverage comparison", "No stored model curves are available.")
            return

        axis.set_ylabel("Normalized flux")
        axis.legend(loc="best", fontsize="small", ncol=2)
        axis.set_title(f"Coverage-model comparison: exposure {exp_index}")
        residual_axis.axhline(0.0, linewidth=0.8)
        residual_axis.set_ylabel("Residual / sigma")
        residual_axis.set_xlabel("Wavelength")
        fig.tight_layout()
        plt.show()

    def export_chromatic_maps(self) -> None:
        record = self._selected_record()
        if record is None or record.source != 'ss73' or not record.usable:
            QMessageBox.warning(self, 'Chromatic maps', 'Select a usable chromatic grid result first.')
            return
        default = Path(getattr(self.parent_viewer, 'work_folder', '.')) / 'chromatic_model'
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Chromatic model export prefix', str(default), 'PNG files (*.png);;All files (*)'
        )
        if not filename:
            return
        try:
            outputs = export_chromatic_record_diagnostics(self.parent_viewer, record, filename)
        except Exception as exc:
            QMessageBox.warning(self, 'Chromatic maps', str(exc))
            raise
        QMessageBox.information(
            self, 'Chromatic maps',
            'Exported:\n' + '\n'.join(str(v) for v in outputs.values())
        )

    def export_table(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export coverage comparison",
            str(Path(getattr(self.parent_viewer, "work_folder", ".")) / "coverage_model_comparison.csv"),
            "CSV files (*.csv)",
        )
        if not filename:
            return
        export_coverage_records_csv(self.records, filename)

    def closeEvent(self, event) -> None:
        self.parent_viewer.coverageComparison = None
        event.accept()
