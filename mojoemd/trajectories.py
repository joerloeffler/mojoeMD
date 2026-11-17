"""Resolve and concatenate restarted production trajectories in numeric order."""
from glob import glob, has_magic
import logging
from pathlib import Path
import re

import mdtraj as md

_SEGMENT = re.compile(r'^production_(\d+)\.dcd$')


def resolve_trajectories(value):
    """Accept a directory, glob, filename, or explicitly ordered filename list.

    A numbered production filename selects all siblings. An explicit list keeps
    exactly the supplied selection/order, including a one-item list.
    """
    if isinstance(value, (list, tuple)):
        paths = [Path(p).expanduser() for p in value]
    else:
        path = Path(value).expanduser()
        if path.is_dir():
            paths = list(path.glob('production_*.dcd'))
            paths = [p for p in paths if _SEGMENT.fullmatch(p.name)]
            if not paths and (path / 'production.dcd').is_file():
                paths = [path / 'production.dcd']
        elif has_magic(str(path)):
            paths = [Path(p) for p in glob(str(path))]
        elif _SEGMENT.fullmatch(path.name):
            if not path.is_file():
                raise FileNotFoundError(path)
            paths = [p for p in path.parent.glob('production_*.dcd') if _SEGMENT.fullmatch(p.name)]
        elif path.name == 'production.dcd' and not path.exists():
            paths = [p for p in path.parent.glob('production_*.dcd') if _SEGMENT.fullmatch(p.name)]
        else:
            paths = [path]
        paths.sort(key=lambda p: (int(_SEGMENT.fullmatch(p.name).group(1)), str(p))
                   if _SEGMENT.fullmatch(p.name) else (float('inf'), str(p)))
    if not paths:
        raise FileNotFoundError(f'No trajectories matched {value!r}')
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate trajectory paths would count frames more than once')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_trajectory_series(topology, value, stride=1):
    """Concatenate all frames, applying stride across the complete frame sequence.

    No boundary frames are discarded: identical coordinates need not represent
    the same time point. Original per-file times are retained; analysis time axes
    should use the configured frame spacing, since DCD times can reset on restart.
    """
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError('stride must be a positive integer')
    stride = int(stride)
    paths = resolve_trajectories(value)
    logging.info('Loading %d trajectory segment(s), global stride=%d: %s', len(paths), stride,
                 ', '.join(str(p) for p in paths))
    parts = []
    offset = 0
    for path in paths:
        frames = 0
        for trajectory in md.iterload(str(path), top=str(topology), chunk=100):
            selected = trajectory[(-offset) % stride::stride]
            offset += trajectory.n_frames
            frames += trajectory.n_frames
            if selected.n_frames:
                parts.append(selected)
        if frames == 0:
            raise ValueError(f'Trajectory has no frames: {path}')
        logging.info('Segment %s: %d frames', path, frames)
    return md.join(parts, check_topology=True, discard_overlapping_frames=False)
