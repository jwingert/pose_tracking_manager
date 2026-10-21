# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- resolution-based inference rescaling, motivated by 13 legacy Prince sites recorded at 1280x720 while every labelled training frame in all 17 DLC projects is 640x480 | date: 2026-08-08]
"""Rescale a video to the resolution its DeepLabCut model was trained at.

Why this exists
---------------
Every labelled frame in every free-moving DLC project is **640x480** -- all 17
projects, checked.  But 13 legacy Prince sites (Nov 2022 - Feb 2023) were
recorded at **1280x720**.  Nothing since Feb 2023 is anything but 640x480,
across three animals and four years, so this is a retired camera and not a
coming one.

A ResNet pose model has no scale invariance to spare.  At native 1280x720 a
ferret head spans ~40 px; the models saw heads of ~18 px (Reishi / StinkySquid)
and ~31 px (Prince).  Measured on ``PRN043a05`` with ``right_chimney_merged-r0001``:

    input             head px   worst-marker tracked   p_median
    1280x720 native     40.4          0.334              0.954
    640x360 rescaled    21.9          0.848              0.988

Downscaling recovers it completely -- 0.848 beats the same model on a *native*
640x480 Prince video (0.794) and beats both Prince specialists at native
resolution (0.697, 0.623).

Note the ``p_median`` column: at native resolution the model was already 0.954
confident about the markers it *found*.  It simply failed to find them in most
frames.  This is a scale-detection failure, not a localisation failure, which is
why feeding it a correctly-scaled image fixes it outright.

What "correct" means here
-------------------------
:func:`analysis_scale` matches the **width** to the training width and keeps the
aspect ratio, so 1280x720 becomes 640x360 -- *not* 640x480.  Squashing 16:9 into
4:3 would change the apparent geometry (a head-angle vector, an inter-ear
distance) non-uniformly in x and y, and the head-fixation frame and the arena
ppm calibration both depend on that geometry being an undistorted view.

The scale is clamped to <= 1.0.  Upscaling a video that is already smaller than
the training frames invents no detail; it only makes the head bigger than
anything the model saw, which is the failure we are fixing, mirrored.

Making the output indistinguishable from a native run
-----------------------------------------------------
:func:`rescale_coordinates` multiplies the pose file's x/y back by ``1/scale``
in place, so what lands in ``sorted/`` is in **native pixels**.  Every
downstream consumer -- the arena-config ppm, ``dlc_geometry`` head angle, the
HRTF transforms in ``free_tools`` -- assumes native pixels and none of them know
about this module.  Likelihoods are dimensionless and are never touched.

Because the output is deliberately indistinguishable, the *provenance* cannot
be: ``dlc_fit_script`` writes a ``<stem>.<release_id>.scale.json`` sidecar next
to the results and logs a warning.  Someone reading these coordinates later has
to be able to find out that a re-encode happened.

Contract
--------
* Module scope is **stdlib only**.  ``cv2`` and ``pandas`` are imported inside
  the functions that need them (the DLC labeling env is py3.8 and several
  analysis envs have no OpenCV).  Python 3.8 compatible.
* :func:`analysis_scale` raises on a video it cannot probe -- an unknown native
  resolution must never silently become "no rescale needed".
* :func:`rescaled_copy` verifies the re-encode's frame count against the source
  and raises if they differ.  A short re-encode would produce a pose file with
  fewer rows than the recording, which downstream code aligns against spike
  times by index.

Usage::

    from nems_lbhb.motor import dlc_video_scale as vscale

    info = vscale.analysis_scale(avi, train_frame_size=(640, 480))
    if info['rescale']:
        small = vscale.rescaled_copy(avi, info['analysis_size'], tmpdir)
        ...                                    # analyze `small`
        vscale.rescale_coordinates(h5, info['scale'])
"""

from __future__ import print_function

import json
import logging
import os
import shutil
import subprocess


__all__ = ['analysis_scale', 'rescaled_copy', 'rescale_coordinates',
           'write_scale_sidecar', 'scale_sidecar_name', 'probe_size',
           'frame_count', 'DEFAULT_TRAIN_FRAME_SIZE', 'FFMPEG']

log = logging.getLogger(__name__)

FFMPEG = 'ffmpeg'

#: Every labelled frame in every DLC project in this lab is 640x480.  This is
#: the fallback when an alias does not declare its own ``train_frame_size``.
DEFAULT_TRAIN_FRAME_SIZE = (640, 480)

#: Re-encoding a 30-80k-frame arena video is minutes, not hours.  An explicit
#: ceiling so a wedged ffmpeg cannot hold a GPU queue slot forever.
FFMPEG_TIMEOUT = 7200.0

#: DeepLabCut's HDF5 key, and the format it writes.
DLC_H5_KEY = 'df_with_missing'


# --------------------------------------------------------------- probing

def probe_size(video_path):
    """``(width, height)`` of ``video_path``, via OpenCV.

    Raises ``FileNotFoundError`` if the file is missing and ``RuntimeError`` if
    it cannot be opened or reports a non-positive dimension.  Deliberately
    strict: an unprobeable video that returned a default size would be analyzed
    at the wrong scale and the results would look fine.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError('no such video: %s' % video_path)

    try:
        import cv2
    except Exception as e:                     # ImportError, or a broken build
        raise RuntimeError(
            'cv2 is not importable in this environment (%s), so the native '
            'resolution of %s cannot be determined. Refusing to guess: the '
            'wrong analysis scale produces plausible-looking garbage.'
            % (e, video_path))

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError('cv2.VideoCapture could not open %s' % video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if width <= 0 or height <= 0:
        raise RuntimeError('%s reports a %dx%d frame size, which is unusable'
                           % (video_path, width, height))
    return (width, height)


def frame_count(video_path):
    """Header frame count of ``video_path``, via OpenCV.

    Used only to compare a re-encode against its source, where both files are
    read the same way, so a container header that lies equally about both is
    still a valid comparison.  Returns ``0`` when the header carries no count.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError('no such video: %s' % video_path)
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError('cv2.VideoCapture could not open %s' % video_path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        try:
            cap.release()
        except Exception:
            pass


# ------------------------------------------------------------ the decision

def _even(value):
    """Round to the **nearest even** integer >= 2.

    Every encoder worth using rejects odd dimensions for chroma-subsampled
    formats, and an ffmpeg that refuses to start is a much worse failure here
    than half a pixel of aspect error.

    Nearest-even rather than round-then-bump-up: bumping up is biased, and on a
    height whose ideal lands just under an odd number it can round all the way
    back to the native size (642x362 -> 640x**362**), which is both more aspect
    distortion and, in the degenerate case, a re-encode that does nothing.
    """
    n = 2 * int(round(value / 2.0))
    return max(2, n)


def analysis_scale(video_path, train_frame_size=DEFAULT_TRAIN_FRAME_SIZE):
    """Decide the resolution ``video_path`` should be analyzed at.

    Returns a dict::

        {'native':        (w, h),     # what the file actually is
         'analysis_size': (w, h),     # what DeepLabCut should see
         'scale':         float,      # analysis_width / native_width, <= 1.0
         'rescale':       bool,       # is analysis_size != native?
         'reason':        str}        # one line, for the log and the sidecar

    ``scale`` is ``train_width / native_width``, **clamped to 1.0**: this
    function never upscales.  The height follows the width so the aspect ratio
    is preserved (1280x720 -> 640x360, not 640x480), and both dimensions are
    forced even.

    Raises whatever :func:`probe_size` raises -- a video whose size is unknown
    must not silently be treated as "already the right size".
    """
    native_w, native_h = probe_size(video_path)
    train_w, train_h = int(train_frame_size[0]), int(train_frame_size[1])
    if train_w <= 0 or train_h <= 0:
        raise ValueError('train_frame_size must be positive, got %r'
                         % (train_frame_size,))

    raw_scale = train_w / float(native_w)

    if raw_scale >= 1.0:
        return {
            'native': (native_w, native_h),
            'analysis_size': (native_w, native_h),
            'scale': 1.0,
            'rescale': False,
            'reason': ('native %dx%d is at or below the %dx%d training frame '
                       'size; analyzing as-is (upscaling invents no detail and '
                       'would make the animal larger than anything the model '
                       'was trained on)'
                       % (native_w, native_h, train_w, train_h)),
        }

    analysis_w = _even(native_w * raw_scale)
    analysis_h = _even(native_h * raw_scale)

    if (analysis_w, analysis_h) == (native_w, native_h):
        # Only reachable when rounding to even lands back on the native size,
        # e.g. native width one pixel above the training width.
        return {
            'native': (native_w, native_h),
            'analysis_size': (native_w, native_h),
            'scale': 1.0,
            'rescale': False,
            'reason': ('native %dx%d rounds to itself at the %dx%d training '
                       'frame size; no re-encode is worth doing'
                       % (native_w, native_h, train_w, train_h)),
        }

    # Report the scale actually realised in x, not the requested one, so
    # rescale_coordinates inverts the transform that was really applied.
    scale = analysis_w / float(native_w)

    return {
        'native': (native_w, native_h),
        'analysis_size': (analysis_w, analysis_h),
        'scale': scale,
        'rescale': True,
        'reason': ('native %dx%d is larger than the %dx%d frames this model was '
                   'trained on; analyzing a %dx%d copy (scale %.6f, aspect '
                   'preserved, dimensions forced even) and converting the '
                   'coordinates back to native pixels'
                   % (native_w, native_h, train_w, train_h,
                      analysis_w, analysis_h, scale)),
    }


# ------------------------------------------------------------- the re-encode

def rescaled_copy(video_path, analysis_size, dest_dir, timeout=FFMPEG_TIMEOUT,
                  ffmpeg=None):
    """Write a downscaled copy of ``video_path`` into ``dest_dir``; return its path.

    The copy keeps the **same basename** as the source.  DeepLabCut names its
    output ``<stem><scorer>.h5``, so an identically-named input is what makes a
    rescaled run produce exactly the filename a native run would have -- which
    in turn is what lets the existing alias/collision machinery in
    ``dlc_fit_script`` work unchanged.

    Encoded as MJPEG at ``-q:v 2``.  Intra-only, so a decoder error cannot
    propagate across frames the way it does in the damaged inter-coded files
    ``dlc_video_qc`` exists to catch, and visually lossless enough that the
    re-encode is not itself a source of tracking error.

    The output's frame count is checked against the source's and a mismatch
    raises: a short re-encode yields a pose file with fewer rows than the
    recording, and downstream code aligns pose against spike times by row index,
    so the misalignment would be silent and total.
    """
    width, height = int(analysis_size[0]), int(analysis_size[1])
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError('analysis_size must be positive and even, got %r'
                         % (analysis_size,))

    if not os.path.isfile(video_path):
        raise FileNotFoundError('no such video: %s' % video_path)
    if not os.path.isdir(dest_dir):
        raise NotADirectoryError('destination directory does not exist: %s'
                                 % dest_dir)

    dest = os.path.join(dest_dir, os.path.basename(video_path))
    if os.path.abspath(dest) == os.path.abspath(video_path):
        raise ValueError(
            'rescaled copy would overwrite the source video %s -- pass a '
            'dest_dir that is not the video\'s own directory (this module is '
            'read-only with respect to /auto/data/daq)' % video_path)

    exe = ffmpeg or shutil.which(FFMPEG)
    if exe is None:
        raise RuntimeError(
            '%s not found on PATH, so %s cannot be rescaled for analysis. '
            'Analyzing it at native resolution would silently produce poor '
            'tracking rather than an error.' % (FFMPEG, video_path))

    cmd = [exe, '-nostdin', '-y', '-v', 'error', '-i', video_path,
           '-vf', 'scale=%d:%d' % (width, height),
           '-c:v', 'mjpeg', '-q:v', '2', '-an', dest]
    log.info('rescaling for analysis: %s', ' '.join(cmd))

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout,
                              universal_newlines=True)
    except subprocess.TimeoutExpired:
        _unlink(dest)
        raise RuntimeError('ffmpeg timed out after %gs rescaling %s'
                           % (timeout, video_path))
    if proc.returncode != 0:
        _unlink(dest)
        raise RuntimeError('ffmpeg exited %d rescaling %s: %s'
                           % (proc.returncode, video_path,
                              (proc.stderr or '').strip()[-2000:]))
    if not os.path.isfile(dest):
        raise RuntimeError('ffmpeg reported success but wrote no %s' % dest)

    src_frames = frame_count(video_path)
    dst_frames = frame_count(dest)
    if src_frames and dst_frames != src_frames:
        _unlink(dest)
        raise RuntimeError(
            'rescaled copy of %s has %d frames but the source has %d. A short '
            're-encode would give a pose file with fewer rows than the '
            'recording, and pose is aligned to the neural data by row index, '
            'so the mismatch would be silent. Refusing to analyze it.'
            % (video_path, dst_frames, src_frames))

    log.info('rescaled %s -> %s (%dx%d, %d frames)', video_path, dest,
             width, height, dst_frames)
    return dest


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


# -------------------------------------------------- coordinates back to native

def rescale_coordinates(h5_path, scale):
    """Convert a DeepLabCut pose file to **native** pixels, in place.

    Every ``x`` and ``y`` column is multiplied by ``1 / scale``; ``likelihood``
    columns are dimensionless and are left exactly as they are.  Returns the
    number of rows rewritten.

    This is the step that makes a rescaled analysis indistinguishable from a
    native one for every downstream consumer: the arena-config ppm, the
    head-angle geometry and the HRTF transforms all assume native pixels and
    none of them know this module exists.

    ``scale=1.0`` is a no-op (the row count is still returned), so a caller may
    call this unconditionally.  A non-positive scale raises.
    """
    import pandas as pd

    scale = float(scale)
    if scale <= 0:
        raise ValueError('scale must be positive, got %r' % (scale,))

    df = pd.read_hdf(h5_path, DLC_H5_KEY)
    n_rows = len(df.index)

    if scale == 1.0:
        return n_rows

    if 'coords' not in (df.columns.names or []):
        raise ValueError(
            "%s has no 'coords' level in its column index (levels: %s); this "
            "does not look like a DeepLabCut pose file, and blindly scaling it "
            "could corrupt real data" % (h5_path, list(df.columns.names or [])))

    coords = df.columns.get_level_values('coords')
    xy = [c for c, kind in zip(df.columns, coords) if kind in ('x', 'y')]
    if not xy:
        raise ValueError('%s has no x/y columns to rescale' % h5_path)

    factor = 1.0 / scale
    # .loc on the exact column labels leaves 'likelihood' (and any other level,
    # e.g. multi-animal 'individuals') untouched by construction.
    df.loc[:, xy] = df.loc[:, xy] * factor

    tmp = '%s.tmp-%d' % (h5_path, os.getpid())
    try:
        # DeepLabCut writes 'table' format; keep it so anything that appends to
        # or queries these files keeps working.
        df.to_hdf(tmp, DLC_H5_KEY, format='table', mode='w')
        os.replace(tmp, h5_path)
    finally:
        _unlink(tmp)

    log.info('converted %d x/y columns in %s to native pixels (x %.6f)',
             len(xy), h5_path, factor)
    return n_rows


# --------------------------------------------------------------- provenance

def scale_sidecar_name(video_stem, release_id=None):
    """``<stem>.<release_id>.scale.json``, or ``<stem>.scale.json`` unreleased."""
    if release_id:
        return '%s.%s.scale.json' % (video_stem, release_id)
    return '%s.scale.json' % video_stem


def write_scale_sidecar(dest_dir, video_stem, info, release_id=None, extra=None):
    """Record that this analysis was run on a rescaled copy.  Returns the path.

    Mandatory, not optional.  :func:`rescale_coordinates` deliberately makes the
    pose file indistinguishable from a native run's, so without this sidecar
    there is nothing on disk that says a re-encode ever happened -- and someone
    reading these coordinates in a year has to be able to find that out.
    """
    payload = {
        'video_stem': video_stem,
        'release_id': release_id,
        'native_size': list(info['native']),
        'analysis_size': list(info['analysis_size']),
        'scale': info['scale'],
        'rescaled': bool(info['rescale']),
        'reason': info['reason'],
        'coordinates': 'native pixels (x/y multiplied by 1/scale after analysis)',
    }
    if extra:
        payload.update(extra)

    path = os.path.join(dest_dir, scale_sidecar_name(video_stem, release_id))
    tmp = '%s.tmp-%d' % (path, os.getpid())
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, path)
    return path
# [AGENT EDIT END]
