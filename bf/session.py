"""BF session file written by the BF analysis window (HDF5, h5py): /lines, /pixels/<line>, /runs/<run>, /compare, /loo,
/check, /contours; strings UTF-8, booleans 0/1, missing numbers NaN."""
import h5py
import numpy as np

STRING = h5py.string_dtype('utf-8')


def _dataset(group, name, value):
    a = np.asarray(value)
    if a.dtype.kind in 'USO':
        group.create_dataset(name, data=np.array([str(v) for v in a.ravel()], dtype=object).reshape(a.shape), dtype=STRING)
    elif a.dtype.kind == 'b': group.create_dataset(name, data=a.astype(np.uint8))
    else: group.create_dataset(name, data=a)


def _attr(group, name, value):
    if isinstance(value, str): group.attrs.create(name, value, dtype=STRING)
    elif isinstance(value, (bool, np.bool_)): group.attrs[name] = np.uint8(value)
    elif value is None: group.attrs.create(name, '', dtype=STRING)
    else: group.attrs[name] = value


def _write(group, node):
    """node: dict; 'attrs' holds the group's attributes, dict values become subgroups, everything else a dataset."""
    for k, v in node.get('attrs', {}).items(): _attr(group, k, v)
    for k, v in node.items():
        if k == 'attrs': continue
        if isinstance(v, dict): _write(group.create_group(k), v)
        else: _dataset(group, k, v)


def pixel_key(line_id):
    """Group name of a line under /pixels/."""
    return str(line_id).replace(':', '_').replace('/', '_')


def write(path, session):
    """Write a session (nested dict, see _write) to `path`; an existing file is replaced."""
    with h5py.File(path, 'w') as f: _write(f, session)
    return path
