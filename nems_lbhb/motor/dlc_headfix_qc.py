"""Head-aligned DLC marker spread QC.

Measures how tightly each body part clusters once every frame has been virtually
head-fixed, and compares that spread against the project's human training labels.
Rigid head markers should be nearly stationary in head-fixed coordinates, so their
residual scatter is a direct proxy for tracking noise.

This module is deliberately free of NEMS, DeepLabCut, SciPy and Streamlit imports
so it can be used from the Python 3.8 `streamlit_dlc` environment. It carries its
own copy of the head-fix transform from `nems_lbhb.free_tools.dlc_headfix`; a
parity test in tests/test_dlc_headfix_qc.py guards the two against drifting.
"""
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RIGID_ROLES = ('front', 'headpost', 'leftear', 'rightear')

# Alias sets for the rigid head markers. The front/headpost/rightear sets mirror
# nems_lbhb.free_tools.find_dlc_headfix_indices; leftear is not resolved there, so
# its aliases follow the same naming conventions used elsewhere in free_tools.
RIGID_ALIASES = {
    'front': ('front', 'nose'),
    'headpost': ('headpost', 'head', 'back'),
    'leftear': ('leftear', 'left_ear', 'leftheadphone'),
    'rightear': ('rightear', 'right_ear', 'rightheadphone'),
}

# free_tools.find_dlc_headfix_indices resolves these three roles by scanning the
# channel list in order and keeping the LAST match, so a project carrying both a
# 'headpost' and a body 'back' marker resolves the head to whichever comes last.
# The QC must measure the same skeleton the fitting pipeline uses, so it mirrors
# that (incidental) rule exactly. 'leftear' is not resolved by free_tools at all,
# so it keeps alias-priority order; there is nothing to be consistent with.
LAST_MATCH_ROLES = ('front', 'headpost', 'rightear')

RATIO_WARN = 1.5
RATIO_FAIL = 3.0
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 - tracking-rate
# thresholds (TRACK_WARN, TRACK_FAIL) | date: 2026-08-04]
# Tracking-rate thresholds. Inverted sense versus RATIO_*: lower is worse.
TRACK_WARN = 0.80
TRACK_FAIL = 0.50
# [AGENT EDIT END]
MIN_VALID_FRAMES = 100
MIN_REFERENCE_FRAMES = 20
# Below this many pooled training frames the reference's per-marker scale is
# weak enough that ratios should not be read as authoritative; surfaced as a
# SpreadResult warning so the UI (which never reads the log) can show it.
WEAK_REFERENCE_FRAMES = 100
# [AGENT EDIT START | agent: claude | user: wingertj | reason: QC scored at 0.9 while
# baphy_io keeps everything above 0.2 (`options['dlc_threshold']`, baphy_io.py:4602 and
# load_dlc_trace's own default at :5048), so it measured a cleaner subset than any analysis
# actually consumes -- and hid real defects: REI084a10_a_NFB reads worst-rigid 0.86x at 0.9
# and 2.75x at 0.2, on the very frames downstream keeps.
#
# BOTH cuts are the loader's. An earlier revision kept frac_tracked at a strict 0.9 on the
# argument that it saturates at 0.2 and stops discriminating -- that was generalised from
# two sessions and is withdrawn: the ungated tracked error has already been shown to track
# HRTF improvement, so the ungated measure carries real signal. QC describes the data
# analyses actually load; a second, stricter population is a different question and is not
# what this module is for.
#
# track_p_cut survives as a PARAMETER so a caller can probe a stricter bar deliberately,
# but nothing defaults to one. | date: 2026-08-17]
# Mirrors baphy_io's `dlc_threshold` default. Not imported from there on purpose: this
# module is kept free of the heavy baphy/NEMS import graph so it loads in all three envs.
LOADER_P_CUT = 0.2
DEFAULT_P_CUT = LOADER_P_CUT
DEFAULT_TRACK_P_CUT = LOADER_P_CUT
# [AGENT EDIT END]
DEFAULT_MAX_SCATTER = 5000

# Robust scale for 2-D residuals. If the two components are iid N(0, sigma^2)
# the residual norm r is Rayleigh(sigma), with
#     CDF F(r) = 1 - exp(-r^2 / (2 sigma^2))  =>  median = sigma * sqrt(2 ln 2)
#     RMS = sqrt(E[r^2]) = sigma * sqrt(2)
# so RMS = median * sqrt(2) / sqrt(2 ln 2) = median / sqrt(ln 2). The constant is
# computed from that identity rather than hard-coded.
#
# [AGENT EDIT START | agent: claude | user: wingertj | reason: final review - the
# median-to-RMS conversion is exact only for isotropic Gaussian residuals; note the
# bounded bias for other residual shapes instead of leaving that unstated | date: 2026-08-04]
# Caveat: this conversion is exact only for isotropic 2-D Gaussian residuals. For other
# residual shapes it is biased within a bounded envelope -- a shell-like (near-constant-
# norm) distribution reads ~1.201x high, a degenerate near-1-D anisotropic one ~0.809x
# low. Unlike a plain RMS, this bias does not cancel between the numerator and
# denominator of ratio_spread, so a strongly anisotropic marker could shift its ratio by
# up to ~1.48x -- close to RATIO_WARN (1.5), worth knowing when reading a borderline flag.
# [AGENT EDIT END]
RAYLEIGH_MEDIAN_TO_RMS = 1.0 / np.sqrt(np.log(2.0))


def robust_spread(resid):
    """RMS-equivalent scale of 2-D residuals [N, 2], via the median norm.

    Matches a plain RMS for clean isotropic Gaussian residuals (see
    RAYLEIGH_MEDIAN_TO_RMS) but barely moves when a handful of points are gross
    outliers -- which matters because the training reference is built from a few
    dozen hand-placed labels, where one misplaced label would otherwise inflate
    the denominator of every ratio and hide a genuinely bad model.
    """
    resid = np.asarray(resid, dtype=float)
    if resid.ndim != 2 or resid.shape[1] != 2:
        raise ValueError("robust_spread expects [N, 2] residuals, got shape {}"
                         .format(resid.shape))
    norms = np.sqrt((resid ** 2).sum(axis=1))
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return float('nan')
    return float(np.median(norms) * RAYLEIGH_MEDIAN_TO_RMS)


def resolve_rigid_indices(bodyparts):
    """Map each rigid role to its index in `bodyparts` (None when absent).

    front/headpost/rightear reproduce free_tools.find_dlc_headfix_indices: scan in
    order, keep the last matching alias. leftear uses alias priority (free_tools
    does not resolve it).
    """
    lowered = [str(b).lower() for b in bodyparts]
    out = {}
    for role in RIGID_ROLES:
        aliases = RIGID_ALIASES[role]
        found = None
        if role in LAST_MATCH_ROLES:
            for i, name in enumerate(lowered):
                if name in aliases:
                    found = i
        else:
            for alias in aliases:
                if alias in lowered:
                    found = lowered.index(alias)
                    break
        out[role] = found
    return out


def missing_rigid_roles(bodyparts):
    """Rigid roles that have no matching body part in `bodyparts`."""
    idx = resolve_rigid_indices(bodyparts)
    return [role for role in RIGID_ROLES if idx[role] is None]


def load_dlc_h5(path):
    """Load a DLC .h5 into (coords[T, B, 2], likelihood[T, B], bodyparts).

    Handles both analyzed files (x/y/likelihood) and CollectedData label files
    (x/y only, likelihood filled with 1.0). Raises ValueError for multi-animal
    projects, whose column index carries an extra 'individuals' level.
    """
    df = pd.read_hdf(str(path))
    if df.columns.nlevels >= 4:
        raise ValueError(
            "Multi-animal DLC projects are not supported by head-fix spread QC "
            "(column index has {} levels)".format(df.columns.nlevels))
    if df.columns.nlevels != 3:
        raise ValueError(
            "Unexpected DLC column index with {} levels".format(df.columns.nlevels))

    scorer = df.columns.get_level_values(0)[0]
    sub = df[scorer]

    bodyparts = list(dict.fromkeys(sub.columns.get_level_values(0)))
    T = len(sub)
    coords = np.full((T, len(bodyparts), 2), np.nan, dtype=float)
    likelihood = np.ones((T, len(bodyparts)), dtype=float)

    for b, part in enumerate(bodyparts):
        coords[:, b, 0] = np.asarray(sub[(part, 'x')], dtype=float)
        coords[:, b, 1] = np.asarray(sub[(part, 'y')], dtype=float)
        if (part, 'likelihood') in sub.columns:
            likelihood[:, b] = np.asarray(sub[(part, 'likelihood')], dtype=float)

    return coords, likelihood, bodyparts


def _fill_ear_sign(ear_dot):
    """Sign of the ear projection with zero/NaN frames held from the last valid one.

    Reproduces nems_lbhb.free_tools.dlc_headfix at its defaults
    (right_hysteresis=0.0, right_hold_frames=0).
    """
    sign = np.sign(np.asarray(ear_dot, dtype=float))
    sign[sign == 0] = np.nan
    valid = ~np.isnan(sign)
    if not valid.any():
        return np.ones_like(sign)
    idx = np.where(valid, np.arange(sign.shape[0]), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(valid))   # leading gap takes the first valid sign
    return sign[idx]


def headfix_points(coords, rigid_idx):
    """Transform [T, B, 2] world coordinates into head-fixed coordinates.

    Origin is the headpost, +y points toward the nose, +x toward the right ear.
    """
    coords = np.asarray(coords, dtype=float)
    for role in ('front', 'headpost', 'rightear'):
        if rigid_idx.get(role) is None:
            raise ValueError("head-fix transform requires '{}'".format(role))

    nose = coords[:, rigid_idx['front'], :]
    head = coords[:, rigid_idx['headpost'], :]
    ear = coords[:, rigid_idx['rightear'], :]

    nose_t = nose - head
    ear_t = ear - head

    forward_len = np.sqrt((nose_t ** 2).sum(axis=1))
    forward_len[forward_len == 0] = 1e-10
    forward = nose_t / forward_len[:, None]

    perp = np.stack([forward[:, 1], -forward[:, 0]], axis=1)
    ear_dot = (ear_t * perp).sum(axis=1)
    right = perp * _fill_ear_sign(ear_dot)[:, None]

    translated = coords - head[:, None, :]
    hf_x = (translated * right[:, None, :]).sum(axis=2)
    hf_y = (translated * forward[:, None, :]).sum(axis=2)
    return np.stack([hf_x, hf_y], axis=2)


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 3 - Procrustes
# alignment and mean shape (kabsch_align, generalized_procrustes, apply_rigid) | date: 2026-08-04]
def kabsch_align(points, reference):
    """Rigidly align each frame of `points` [T, K, 2] onto `reference` [K, 2].

    Rotation and translation only — no scaling, because the caller has already
    normalized by a per-video scale and per-frame scaling would absorb real
    marker jitter.

    Returns (aligned [T, K, 2], rotation [T, 2, 2], src_centroid [T, 1, 2],
    dst_centroid [1, 2]).
    """
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)

    src_centroid = points.mean(axis=1, keepdims=True)
    dst_centroid = reference.mean(axis=0, keepdims=True)
    src = points - src_centroid
    dst = reference - dst_centroid

    cov = np.einsum('tki,kj->tij', src, dst)
    u, _, vt = np.linalg.svd(cov)
    v = vt.transpose(0, 2, 1)
    ut = u.transpose(0, 2, 1)
    det = np.linalg.det(np.einsum('tij,tjk->tik', v, ut))
    flip = np.zeros((points.shape[0], 2, 2))
    flip[:, 0, 0] = 1.0
    flip[:, 1, 1] = np.sign(det)
    rotation = np.einsum('tij,tjk,tkl->til', v, flip, ut)

    aligned = np.einsum('tij,tkj->tki', rotation, src) + dst_centroid
    return aligned, rotation, src_centroid, dst_centroid


def apply_rigid(points, rotation, src_centroid, dst_centroid):
    """Apply a kabsch_align fit to another set of markers [T, B, 2]."""
    points = np.asarray(points, dtype=float)
    return np.einsum('tij,tkj->tki', rotation, points - src_centroid) + dst_centroid


def generalized_procrustes(shapes, n_iter=3, init=None):
    """Centroid-centred mean shape of [T, K, 2] via iterative Procrustes averaging.

    `init` seeds the iteration and so fixes the orientation of the result. Pass the
    mean head-fixed configuration to get a mean shape in the anatomical frame
    (+y forward, +x right); the default (first shape) leaves it arbitrarily rotated.
    """
    shapes = np.asarray(shapes, dtype=float)
    seed = shapes[0] if init is None else np.asarray(init, dtype=float)
    mean_shape = seed - seed.mean(axis=0, keepdims=True)
    for _ in range(n_iter):
        aligned, _, _, _ = kabsch_align(shapes, mean_shape)
        mean_shape = np.nanmean(aligned, axis=0)
        mean_shape = mean_shape - mean_shape.mean(axis=0, keepdims=True)
    return mean_shape
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 4 - training-label
# head-fix reference (TrainingReference, head_scale, build_reference_from_folders,
# build_training_reference) | date: 2026-08-04]
@dataclass
class TrainingReference:
    """Head-fix reference derived from a DLC project's human training labels."""
    mean_shape: np.ndarray            # [4, 2] Procrustes mean of the rigid markers
    rigid_idx: dict                   # role -> index into bodyparts
    bodyparts: list
    train_rms: dict                   # bodypart -> robust Procrustes-residual scale (head-lengths)
    mean_full: dict                   # bodypart -> mean aligned position [2]
    hf_mean: dict                     # bodypart -> mean head-fixed position [2]
    hf_cov: dict                      # bodypart -> head-fixed covariance [2, 2]
    n_frames: int
    sources: list
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the scatter moves to
    # the Procrustes frame (see SpreadResult.scatter_aligned); its reference ellipses
    # need the training covariance in THAT frame, not the head-fix one. Defaulted so a
    # cached/pickled reference built before this field simply draws no ellipse rather
    # than breaking. | date: 2026-08-18]
    aligned_cov: dict = None          # bodypart -> Procrustes-frame covariance [2, 2]
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 measures spread
    # in the head-fix frame, so ratios need the training labels' scatter in THAT frame.
    # Defaulted so a reference built before this field falls back to sqrt(trace(hf_cov)).
    # | date: 2026-08-18]
    hf_rms: dict = None               # bodypart -> robust head-fix residual scale
    # [AGENT EDIT END]


def head_scale(coords, rigid_idx):
    """Median |front - headpost| distance, the per-video length unit."""
    coords = np.asarray(coords, dtype=float)
    nose = coords[:, rigid_idx['front'], :]
    head = coords[:, rigid_idx['headpost'], :]
    dist = np.sqrt(((nose - head) ** 2).sum(axis=1))
    dist = dist[np.isfinite(dist) & (dist > 0)]
    if dist.size == 0:
        return np.nan
    return float(np.median(dist))


def _rigid_stack(coords, rigid_idx):
    """[T, 4, 2] view of the rigid markers, in RIGID_ROLES order."""
    return np.stack([coords[:, rigid_idx[role], :] for role in RIGID_ROLES], axis=1)


def build_reference_from_folders(folders):
    """Build a TrainingReference from [(coords[T, B, 2], bodyparts), ...].

    Each folder is normalized by its own head_scale, because training frames come
    from different videos and cameras.
    """
    bodyparts = None
    rigid_idx = None
    normalized = []
    sources = []

    for entry in folders:
        if len(entry) == 2:
            coords, parts = entry
            source = None
        else:
            coords, parts, source = entry
        parts = list(parts)
        if bodyparts is None:
            bodyparts = parts
            rigid_idx = resolve_rigid_indices(bodyparts)
            if any(rigid_idx[r] is None for r in RIGID_ROLES):
                log.warning("training labels missing rigid markers %s",
                            missing_rigid_roles(bodyparts))
                return None
        elif parts != bodyparts:
            log.warning("skipping labeled-data folder with mismatched body parts: %s", parts)
            continue

        coords = np.asarray(coords, dtype=float)
        rigid = _rigid_stack(coords, rigid_idx)
        keep = np.isfinite(rigid).all(axis=(1, 2))
        if not keep.any():
            continue
        coords = coords[keep]
        scale = head_scale(coords, rigid_idx)
        if not np.isfinite(scale) or scale <= 0:
            continue
        normalized.append(coords / scale)
        sources.append(source)

    if not normalized:
        return None

    coords = np.concatenate(normalized, axis=0)
    if coords.shape[0] < MIN_REFERENCE_FRAMES:
        log.warning("only %d usable training frames (< %d)",
                    coords.shape[0], MIN_REFERENCE_FRAMES)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: seeding from the
    # head-fix configuration did NOT hold the anatomical orientation: the alignment ran on
    # raw image coordinates (y-down), whose chirality is opposite to the anatomical basis,
    # and rotation-only Kabsch cannot honour a mirrored seed -- the mean settled nose-DOWN
    # (front y=-0.53 on left_chimney-r0001) while every comment claimed +y forward.
    # Aligning the HEAD-FIX coordinates instead keeps chirality consistent, so the mean
    # inherits nose-up and bias_x/bias_y really are anatomical (+y forward, +x toward the
    # right ear). Distances are preserved by the rigid transform, so train_rms and every
    # ratio built on it are unchanged. | date: 2026-08-18]
    hf = headfix_points(coords, rigid_idx)
    rigid = _rigid_stack(hf, rigid_idx)
    mean_shape = generalized_procrustes(rigid, init=np.nanmean(rigid, axis=0))
    _, rot, src_c, dst_c = kabsch_align(rigid, mean_shape)
    aligned_full = apply_rigid(hf, rot, src_c, dst_c)
    # [AGENT EDIT END]

    train_rms = {}
    mean_full = {}
    for b, part in enumerate(bodyparts):
        mean_full[part] = np.nanmean(aligned_full[:, b, :], axis=0)
        resid = aligned_full[:, b, :] - mean_full[part][None, :]
        # Robust (median-norm) scale, not a plain RMS: label sets are small (a few
        # dozen frames is typical) and a single misplaced human label would
        # otherwise inflate this denominator and deflate every downstream ratio.
        train_rms[part] = robust_spread(resid)

    hf_mean = {}
    hf_cov = {}
    hf_rms = {}
    for b, part in enumerate(bodyparts):
        pts = hf[:, b, :]
        good = np.isfinite(pts).all(axis=1)
        if good.sum() >= 2:
            hf_mean[part] = pts[good].mean(axis=0)
            hf_cov[part] = np.cov(pts[good].T)
            hf_rms[part] = robust_spread(pts[good] - hf_mean[part][None, :])
        else:
            hf_mean[part] = np.full(2, np.nan)
            hf_cov[part] = np.full((2, 2), np.nan)
            hf_rms[part] = np.nan

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: covariance of the SAME
    # aligned points train_rms/mean_full come from, so the scatter's reference ellipses
    # can live in the frame ratio_spread is measured in | date: 2026-08-18]
    aligned_cov = {}
    for b, part in enumerate(bodyparts):
        pts = aligned_full[:, b, :]
        good = np.isfinite(pts).all(axis=1)
        if good.sum() >= 2:
            aligned_cov[part] = np.cov(pts[good].T)
        else:
            aligned_cov[part] = np.full((2, 2), np.nan)
    # [AGENT EDIT END]

    return TrainingReference(
        mean_shape=mean_shape, rigid_idx=rigid_idx, bodyparts=bodyparts,
        train_rms=train_rms, mean_full=mean_full, hf_mean=hf_mean, hf_cov=hf_cov,
        n_frames=int(coords.shape[0]), sources=[s for s in sources if s],
        aligned_cov=aligned_cov, hf_rms=hf_rms)


def _project_path_from_config(config_path):
    """Resolve a DLC project's root directory from its config.yaml."""
    import yaml

    with open(str(config_path), 'r') as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg.get('project_path', Path(config_path).parent))


def _label_files(project_path):
    """Sorted list of CollectedData_*.h5 label files under a DLC project or release.

    A mutable project keeps them in ``labeled-data/``; a frozen release keeps a copy in
    ``labels/`` (build_release names it that, deliberately, because it is a snapshot
    rather than a live labelling directory). Looking in only one place meant every
    release-based analysis silently scored with has_reference=False -- ratio_spread all
    NaN, no reference ellipses -- which reads as "this project has no labels" rather than
    "we looked in the wrong directory".
    """
    for subdir in ('labeled-data', 'labels'):
        found = sorted(glob.glob(
            str(Path(project_path) / subdir / '*' / 'CollectedData_*.h5')))
        if found:
            return found
    return []


def build_training_reference(config_path):
    """Build a TrainingReference from a DLC project's CollectedData label files."""
    project_path = _project_path_from_config(config_path)

    label_files = _label_files(project_path)
    folders = []
    for path in label_files:
        try:
            coords, _, parts = load_dlc_h5(path)
        except Exception as exc:
            log.warning("could not read %s: %s", path, exc)
            continue
        folders.append((coords, parts, os.path.basename(os.path.dirname(path))))

    if not folders:
        log.warning("no CollectedData_*.h5 under %s (looked in labeled-data/ and labels/)",
                    project_path)
        return None
    return build_reference_from_folders(folders)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 finding 1 -
# fingerprint of a project's training-label files, used to bust the Streamlit app's
# cached TrainingReference when labels are added/edited (config.yaml mtime alone
# does not change) | date: 2026-08-04]
def label_files_fingerprint(config_path):
    """Short deterministic string that changes whenever a project's
    labeled-data/*/CollectedData_*.h5 set changes (files added, removed, or their
    contents modified).

    Built from each file's path, size and mtime, over the same file set that
    build_training_reference() reads. Never raises: a missing or unreadable
    project path, or a project with no label files yet, both return a stable
    sentinel derived from an empty file list.
    """
    import hashlib

    try:
        project_path = _project_path_from_config(config_path)
        label_files = _label_files(project_path)
    except Exception as exc:
        log.warning("could not resolve project for %s: %s", config_path, exc)
        label_files = []

    parts = []
    for path in label_files:
        try:
            stat = os.stat(path)
            parts.append("{}:{}:{}".format(path, stat.st_size, stat.st_mtime_ns))
        except OSError as exc:
            log.warning("could not stat %s: %s", path, exc)
            parts.append("{}:missing".format(path))

    digest_input = "|".join(parts).encode('utf-8')
    return hashlib.sha1(digest_input).hexdigest()[:16]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 5 - per-video
# spread scoring (SpreadResult, flag_for_ratio, score_video, worst_rigid) | date: 2026-08-04]
@dataclass
class SpreadResult:
    """Per-bodypart head-aligned spread for one analyzed video."""
    table: pd.DataFrame
    scatter_hf: dict
    scale_px: float
    n_valid: int
    n_total: int
    frac_valid: float
    has_reference: bool
    reference: object
    warnings: list
    mean_shape: object = None         # [4, 2] shape the video was Procrustes-fit to
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: scatter_hf pins the
    # headpost to (0,0) BY CONSTRUCTION, so the marker's own spread is invisible there --
    # while ratio_spread is computed from the Procrustes-aligned points, where every rigid
    # marker carries its share of the residual. Plot and statistic now share a frame.
    # | date: 2026-08-18]
    scatter_aligned: dict = None      # bodypart -> [N, 2] Procrustes-frame points
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- the marker
    # serving as the head-fix origin has zero spread by construction and must be
    # excluded from headline means | date: 2026-08-18]
    origin_part: str = None
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 -- the
    # companion self-Procrustes frame: scatter + mean shape for the per-marker
    # diagnostic view, alongside the axis-frame (pipeline) one | date: 2026-08-19]
    scatter_proc: dict = None         # bodypart -> [N, 2] self-Procrustes-frame points
    mean_shape_proc: object = None    # [4, 2] video's own Procrustes mean, RIGID_ROLES order
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 - tracking-rate
# and confidence columns (frac_tracked, p_median, flag_track) | date: 2026-08-04]
TABLE_COLUMNS = ['role', 'n_valid', 'frac_valid', 'frac_tracked', 'p_median',
                 'rms_spread', 'rms_spread_px', 'rms_spread_proc',
                 'ratio_spread', 'ratio_offset',
                 'bias_x', 'bias_y', 'hf_cx', 'hf_cy', 'hf_sd', 'radius',
                 'flag', 'flag_track']
# [AGENT EDIT END]


def flag_for_ratio(ratio):
    """'ok' / 'warn' / 'fail' from a spread ratio; '' when the ratio is unknown."""
    if ratio is None or not np.isfinite(ratio):
        return ''
    if ratio >= RATIO_FAIL:
        return 'fail'
    if ratio >= RATIO_WARN:
        return 'warn'
    return 'ok'


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 -
# flag_for_fraction, inverted sense of flag_for_ratio | date: 2026-08-04]
def flag_for_fraction(frac):
    """'ok' / 'warn' / 'fail' from a tracked-frame fraction; '' when unknown.

    Inverted sense versus flag_for_ratio: for tracking, LOWER is worse. The
    threshold value itself falls in the better band.
    """
    if frac is None or not np.isfinite(frac):
        return ''
    if frac < TRACK_FAIL:
        return 'fail'
    if frac < TRACK_WARN:
        return 'warn'
    return 'ok'
# [AGENT EDIT END]


def score_video(h5_path, reference=None, p_cut=DEFAULT_P_CUT,
                max_scatter=DEFAULT_MAX_SCATTER, seed=0, track_p_cut=None):
    """Score head-aligned marker spread for one analyzed DLC .h5.

    QC v5, self-referential: each frame is aligned by its own front/headpost
    (nose/back-of-head) axis -- the head-fix transform -- and ``rms_spread`` is a
    marker's residual about its own across-frame mean in that frame, in
    head-lengths. No training labels enter the measurement, so the number is
    comparable across models, animals and label sets.

    When a training reference is available, two context quantities are added,
    both in units of the labels' scatter measured in the same head-fix frame:

    ``ratio_spread``
        the video's spread over the labels' spread -- and the only quantity
        `flag` is derived from.
    ``ratio_offset``
        how far the video's own mean position sits from the labels' mean --
        a systematic shape difference (a differently-shaped animal, repositioned
        headphones), which is not a tracking failure.

    By construction the headpost-role marker has zero spread (it is the origin;
    see ``origin_part``) and the front marker's spread is pure head-length
    jitter (it is pinned to the +y axis).
    """
    # the confidence bar frac_tracked reports against, independent of which frames the
    # spread is measured on -- see the LOADER_P_CUT / DEFAULT_TRACK_P_CUT comment
    if track_p_cut is None:
        track_p_cut = p_cut

    warnings = []
    coords, likelihood, bodyparts = load_dlc_h5(h5_path)
    n_total = coords.shape[0]

    rigid_idx = resolve_rigid_indices(bodyparts)
    missing = missing_rigid_roles(bodyparts)
    if missing:
        raise ValueError(
            "head-fix spread QC needs rigid markers {}; missing {}".format(
                list(RIGID_ROLES), missing))

    rigid_cols = [rigid_idx[role] for role in RIGID_ROLES]
    valid = np.isfinite(coords[:, rigid_cols, :]).all(axis=(1, 2))
    # strictly greater, matching baphy_io's own gate (`likelihood.values > dlc_threshold`,
    # baphy_io.py:4769 and :5119) so QC scores exactly the frame set analyses receive
    valid &= (likelihood[:, rigid_cols] > p_cut).all(axis=1)

    nose = coords[:, rigid_idx['front'], :]
    head = coords[:, rigid_idx['headpost'], :]
    sep = np.sqrt(((nose - head) ** 2).sum(axis=1))
    valid &= np.isfinite(sep) & (sep > 0)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(
            "no frames passed the likelihood cut of {} in {}".format(p_cut, h5_path))
    if n_valid < MIN_VALID_FRAMES:
        warnings.append(
            "only {} valid frames (< {}); statistics are noisy".format(
                n_valid, MIN_VALID_FRAMES))

    if reference is not None and reference.n_frames < WEAK_REFERENCE_FRAMES:
        warnings.append(
            "training reference built from only {} labeled frames across {} "
            "folder(s) (< {}); ratios are provisional".format(
                reference.n_frames, len(reference.sources),
                WEAK_REFERENCE_FRAMES))

    coords = coords[valid]
    part_ok = likelihood[valid] > p_cut        # per-body-part mask, same gate as above
    scale_px = head_scale(coords, rigid_idx)
    norm = coords / scale_px

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 (user decision,
    # 2026-08-18) -- the per-frame head-fix transform IS the fit. Each frame is aligned
    # by its own front/headpost (nose/back-of-head) marker axis: translation to the
    # headpost, rotation putting the nose on +y. Every spread statistic is a marker's
    # residual about its own across-frame mean IN THAT FRAME, so the measurement is
    # self-referential and no training labels enter the geometry (the old scheme
    # Procrustes-fit each frame to the training labels' mean shape, which made the
    # statistic depend on which views/animals happened to be labeled). The reference,
    # when present, supplies context only: ratios against the labels' scatter measured
    # in this same head-fix frame. By construction the headpost sits at (0,0) in every
    # frame (zero spread -- excluded from headline means via origin_part) and the nose
    # lies on x=0, so its spread is pure head-length jitter. | date: 2026-08-18]
    hf = headfix_points(norm, rigid_idx)
    aligned_full = hf
    # the video's own mean rigid configuration, rows in RIGID_ROLES order -- what the
    # GUI draws as the mean head shape
    mean_shape = np.stack([
        np.nanmean(hf[:, rigid_idx[role], :], axis=0) for role in RIGID_ROLES])
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 (user decision,
    # 2026-08-19) -- a SECOND, companion spread: per-frame Kabsch of the rigid quad onto
    # the video's OWN Procrustes mean (still reference-free), so the fit uses all four
    # rigid markers and no marker is privileged: front/headpost errors appear in their
    # own clouds instead of being absorbed into the frame and smeared onto the ears.
    # rms_spread (axis frame) stays the pipeline-noise number -- it is measured in the
    # exact head-fix coordinates free_tools hands downstream; rms_spread_proc is the
    # per-marker diagnostic. Keeping both makes their (dis)agreement measurable.
    # | date: 2026-08-19]
    rigid = _rigid_stack(hf, rigid_idx)
    mean_shape_proc = generalized_procrustes(rigid, init=np.nanmean(rigid, axis=0))
    _, _rot, _src_c, _dst_c = kabsch_align(rigid, mean_shape_proc)
    proc_full = apply_rigid(hf, _rot, _src_c, _dst_c)
    # [AGENT EDIT END]

    if max_scatter and n_valid > max_scatter:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n_valid, size=max_scatter, replace=False))
    else:
        pick = np.arange(n_valid)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 - tracking
    # rate and confidence, measured over ALL frames before the joint rigid gate |
    # date: 2026-08-04]
    # Tracking rate and confidence are measured over ALL frames, deliberately
    # NOT through the joint rigid gate. Within surviving frames every rigid
    # marker clears p_cut by construction, so a post-gate rate reads identically
    # for all four and cannot say which marker is costing frames.
    tracked_frac = (likelihood > track_p_cut).mean(axis=0)
    p_median_all = np.median(likelihood, axis=0)
    # [AGENT EDIT END]

    rows = {}
    scatter_hf = {}
    scatter_aligned = {}
    scatter_proc = {}
    for b, part in enumerate(bodyparts):
        role = 'rigid' if b in rigid_cols else 'nonrigid'
        pts = aligned_full[:, b, :]
        # Gate every per-row statistic on THIS part's own likelihood, not just on
        # the frame-level rigid-marker gate. DLC writes coordinates for every part
        # in every frame, so a finite-only mask is all-True and a part the model
        # never learned would otherwise produce a plausible-looking number.
        good = np.isfinite(pts).all(axis=1) & part_ok[:, b]
        any_good = bool(good.any())

        if any_good:
            own_mean = pts[good].mean(axis=0)
            rms_spread = robust_spread(pts[good] - own_mean[None, :])
        else:
            own_mean = np.full(2, np.nan)
            rms_spread = np.nan

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- ratios
        # and offsets compare against the training labels' statistics in the HEAD-FIX
        # frame (hf_rms/hf_mean), the frame the spread is now measured in. hf_rms is
        # absent from references built before v5; sqrt(trace(hf_cov)) is its RMS
        # equivalent. | date: 2026-08-18]
        train_rms = np.nan
        if reference is not None:
            hf_rms = getattr(reference, 'hf_rms', None)
            if hf_rms and part in hf_rms:
                train_rms = hf_rms[part]
            elif part in reference.hf_cov:
                train_rms = float(np.sqrt(np.trace(
                    np.asarray(reference.hf_cov[part], dtype=float))))
        have_train = (reference is not None and np.isfinite(train_rms)
                      and train_rms > 0)

        if reference is not None and part in reference.hf_mean:
            offset = own_mean - np.asarray(reference.hf_mean[part], dtype=float)
        else:
            offset = np.full(2, np.nan)
        # [AGENT EDIT END]

        ratio_spread = rms_spread / train_rms if have_train else np.nan
        if have_train and np.isfinite(offset).all():
            ratio_offset = float(np.sqrt((offset ** 2).sum()) / train_rms)
        else:
            ratio_offset = np.nan

        hf_pts = hf[:, b, :]
        hf_good = np.isfinite(hf_pts).all(axis=1) & part_ok[:, b]
        if hf_good.any():
            hf_c = hf_pts[hf_good].mean(axis=0)
            hf_sd = robust_spread(hf_pts[hf_good] - hf_c[None, :])
            radius = float(np.median(np.sqrt((hf_pts[hf_good] ** 2).sum(axis=1))))
        else:
            hf_c = np.full(2, np.nan)
            hf_sd = np.nan
            radius = np.nan

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 -- the
        # companion self-Procrustes residual, same gating as the axis-frame one
        # | date: 2026-08-19]
        proc_pts = proc_full[:, b, :]
        proc_good = np.isfinite(proc_pts).all(axis=1) & part_ok[:, b]
        if proc_good.any():
            proc_mean = proc_pts[proc_good].mean(axis=0)
            rms_spread_proc = robust_spread(proc_pts[proc_good] - proc_mean[None, :])
        else:
            rms_spread_proc = np.nan
        # [AGENT EDIT END]

        rows[part] = {
            'role': role,
            'n_valid': int(good.sum()),
            'frac_valid': float(good.sum()) / n_total if n_total else np.nan,
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 -
            # frac_tracked / p_median / flag_track | date: 2026-08-04]
            'frac_tracked': float(tracked_frac[b]),
            'p_median': float(p_median_all[b]),
            # [AGENT EDIT END]
            'rms_spread': rms_spread,
            'rms_spread_px': rms_spread * scale_px,
            'rms_spread_proc': rms_spread_proc,
            'ratio_spread': ratio_spread,
            'ratio_offset': ratio_offset,
            'bias_x': float(offset[0]),
            'bias_y': float(offset[1]),
            'hf_cx': hf_c[0],
            'hf_cy': hf_c[1],
            'hf_sd': hf_sd,
            'radius': radius,
            'flag': flag_for_ratio(ratio_spread) if role == 'rigid' else '',
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 -
            # flag_track applies to every marker, not just rigid | date: 2026-08-04]
            'flag_track': flag_for_fraction(float(tracked_frac[b])),
            # [AGENT EDIT END]
        }
        # Frames the part was not tracked in are NaN'd rather than dropped, so the
        # scatter keeps a stable shape and plotly simply skips them.
        picked = np.where(hf_good[pick, None], hf_pts[pick], np.nan)
        scatter_hf[part] = picked
        # same subsample of the ALIGNED points -- what ratio_spread is measured on;
        # `good` is this part's own gate over the aligned coordinates, mirroring hf_good
        scatter_aligned[part] = np.where(good[pick, None], pts[pick], np.nan)
        # and of the self-Procrustes points -- what rms_spread_proc is measured on
        scatter_proc[part] = np.where(proc_good[pick, None], proc_pts[pick], np.nan)

    table = pd.DataFrame.from_dict(rows, orient='index')[TABLE_COLUMNS]
    table = table.loc[bodyparts]

    return SpreadResult(
        table=table, scatter_hf=scatter_hf, scale_px=float(scale_px),
        n_valid=n_valid, n_total=int(n_total),
        frac_valid=float(n_valid) / n_total if n_total else np.nan,
        has_reference=reference is not None, reference=reference,
        warnings=warnings, mean_shape=mean_shape,
        scatter_aligned=scatter_aligned,
        origin_part=bodyparts[rigid_idx['headpost']],
        scatter_proc=scatter_proc, mean_shape_proc=mean_shape_proc)


def worst_rigid(result):
    """(bodypart, ratio_spread) for the rigid marker with the largest finite spread ratio.

    Ranks on ratio_spread (tracking noise) only -- a large ratio_offset means the
    animal's head shape differs from the training animal's, which is not a
    tracking failure and must not drive the headline number.
    """
    rigid = result.table[result.table['role'] == 'rigid']
    ratios = rigid['ratio_spread'].astype(float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.empty:
        return None, float('nan')
    part = ratios.idxmax()
    return part, float(ratios.loc[part])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: only the WORST rigid marker
# was recorded, which throws away three quarters of a measurement the Procrustes fit
# already produces and makes one unlucky marker the whole verdict. Every rigid marker's
# residual is kept, and the MEAN becomes the headline: the worst is a single order
# statistic and jumps around between sessions, the mean is what actually tracks quality.
# | date: 2026-08-17]
_SUMMARY_STR_COLS = ('role', 'flag', 'flag_track')


def _num(v):
    """float, or None when missing/non-finite -- JSON has no NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def summarize(result):
    """The whole SpreadResult as a JSON-safe document of scalars.

    EVERY body part and every column of ``result.table`` is kept, not just the worst
    rigid marker: the Procrustes fit has already computed them, they cost a few hundred
    bytes, and a question asked six months from now ("did the ears drift outward once
    the headphones went on?" -- ``bias_x``/``ratio_offset``) cannot be answered from a
    number nobody wrote down.  Nothing per-FRAME is included, so the document stays
    bounded regardless of video length.

    Headline means and worsts are over RIGID markers only, mirroring :func:`worst_rigid`:
    a legitimately occluded body marker must not drive the verdict.
    """
    parts = {}
    for name, row in result.table.iterrows():
        parts[str(name)] = {
            col: (str(row[col]) if col in _SUMMARY_STR_COLS else _num(row[col]))
            for col in result.table.columns
        }

    rigid = result.table[result.table['role'] == 'rigid']
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- the head-fix
    # origin marker has zero spread by construction; including it would dilute every
    # spread headline by a structural constant. Tracking stats keep it: confidence is
    # independent of the geometry. | date: 2026-08-18]
    origin = getattr(result, 'origin_part', None)
    spread_rigid = (rigid.drop(index=origin, errors='ignore')
                    if origin is not None else rigid)

    def _mean(col, frame=None):
        frame = rigid if frame is None else frame
        if col not in frame.columns:
            return None
        vals = frame[col].astype(float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if not vals.empty else None

    def _worst(col, frame):
        if col not in frame.columns:
            return (None, None)
        vals = frame[col].astype(float)
        vals = vals[np.isfinite(vals)]
        return (None, None) if vals.empty else (vals.idxmax(), float(vals.max()))
    # [AGENT EDIT END]

    part_r, ratio_worst = worst_rigid(result)
    part_t, track_worst = worst_rigid_tracking(result)
    rms_worst_part, rms_worst = _worst('rms_spread', spread_rigid)
    ref = result.reference
    return {
        'parts': parts,
        # headline: the MEAN over rigid markers. The worst is one order statistic and
        # swings with whichever marker got unlucky; the mean is what tracks quality.
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- the
        # self-referential head-fix spread (head-lengths) is the primary, reference-free
        # headline; ratios are context that exists only when labels do | date: 2026-08-18]
        'rms_spread_mean': _mean('rms_spread', spread_rigid),
        'rms_spread_worst': _num(rms_worst),
        'rms_spread_worst_part': rms_worst_part,
        'origin_part': origin,
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 -- the
        # self-Procrustes companion. Means run over ALL rigid markers (no structural
        # zero: the 4-point fit gives every marker its own residual) | date: 2026-08-19]
        'rms_spread_proc_mean': _mean('rms_spread_proc'),
        'rms_spread_proc_worst': _num(_worst('rms_spread_proc', rigid)[1]),
        'rms_spread_proc_worst_part': _worst('rms_spread_proc', rigid)[0],
        # [AGENT EDIT END]
        'ratio_mean': _mean('ratio_spread', spread_rigid),
        'ratio_worst': _num(ratio_worst),
        'ratio_worst_part': part_r,
        # systematic shape difference, kept apart from spread on purpose -- a wider head
        # is not a tracking failure
        'offset_mean': _mean('ratio_offset', spread_rigid),
        'track_mean': _mean('frac_tracked'),
        'track_worst': _num(track_worst),
        'track_worst_part': part_t,
        # head length in px: the unit every ratio is expressed in. Without it a ratio
        # cannot be converted back to pixels or compared across camera zooms.
        'scale_px': _num(result.scale_px),
        'frames': int(result.n_total),
        'frames_scored': int(result.n_valid),
        'frac_valid': _num(result.frac_valid),
        # the animal's rigid head shape in the head-fixed frame -- 8 numbers that make
        # shape drift across sessions answerable later
        'mean_shape': [[_num(x) for x in pt] for pt in np.asarray(result.mean_shape)],
        'has_reference': bool(result.has_reference),
        'reference_frames': int(ref.n_frames) if ref is not None else None,
        'reference_sources': list(ref.sources) if ref is not None else None,
        # "only N valid frames", "reference built from only N labeled frames" -- the
        # caveats that decide whether these numbers should be trusted at all
        'warnings': list(result.warnings),
    }


def rigid_summary(result):
    """Backwards-compatible alias for :func:`summarize`."""
    return summarize(result)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 -
# worst_rigid_tracking, mirrors worst_rigid but for tracking rate | date: 2026-08-04]
def worst_rigid_tracking(result):
    """(bodypart, frac_tracked) for the rigid marker tracked in the FEWEST frames.

    Rigid markers only, mirroring worst_rigid: a legitimately occluded body
    marker (rear behind the animal) must not drive the panel headline.
    """
    rigid = result.table[result.table['role'] == 'rigid']
    fracs = rigid['frac_tracked'].astype(float)
    fracs = fracs[np.isfinite(fracs)]
    if fracs.empty:
        return None, float('nan')
    part = fracs.idxmin()
    return part, float(fracs.loc[part])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 - Streamlit
# panel helpers (match_h5_files, ellipse_path) | date: 2026-08-04]
def match_h5_files(paths, highlight_substr, highlight_tokens):
    """Keep .h5 paths whose filename matches the currently selected DLC model.

    `highlight_substr` and `highlight_tokens` come from the Streamlit app's
    get_dlc_highlight_tokens(config_path) — the same test used to highlight the
    active model's file in the .h5 listings.
    """
    key = highlight_substr.lower() if highlight_substr else None
    tokens = [t.lower() for t in highlight_tokens] if highlight_tokens else []
    out = []
    for p in paths:
        name = Path(p).name.lower()
        if key and key not in name:
            continue
        if tokens and not all(t in name for t in tokens):
            continue
        out.append(p)
    return out


def ellipse_path(mean, cov, n_sd=2.0, n_points=64):
    """Points tracing the n_sd covariance ellipse around `mean`."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if not np.isfinite(cov).all() or not np.isfinite(mean).all():
        return np.full((n_points, 2), np.nan)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    axes = circle * (n_sd * np.sqrt(vals))[None, :]
    return axes.dot(vecs.T) + mean[None, :]
# [AGENT EDIT END]
