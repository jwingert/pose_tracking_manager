# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- video-integrity QC for the DLC pipeline, motivated by LMD008a02 (77385-frame header, 52190 decodable, OpenCV dies at frame 2461 while the historical .dlc.h5 looks perfect) | date: 2026-08-07]
"""Video-integrity QC for the DeepLabCut pipeline.

Why this exists
---------------
``/auto/data/daq/LemonDisco/LMD008/LMD008a02_a_NFB/recording.avi`` decoded fully
in November 2023: its production ``.dlc.h5`` tracks well across all 77,385
frames (worst rigid marker 0.710, median likelihood 0.995).  Today the container
header still claims 77,385 frames, ``ffprobe -count_frames`` recovers 52,190
(67%), and **OpenCV -- which is what DeepLabCut actually decodes with -- gives up
at frame 2,461**.  The file's mtime is unchanged and its ten siblings in
LMD005-012 decode cleanly, so this is one rotten file, not a spreading problem.

The failure is invisible from the outputs: the historical pose file is fine, so
nothing flags the session, yet **re-analysing it silently produces garbage for
97% of frames**.  Re-analysis is now central to the release workflow, so the
damage has to be caught before a fit runs, not after.

The two tiers, and why there are two
------------------------------------
A definitive check means decoding every frame: ~12 s of ``ffprobe -count_frames``
plus ~5-25 s of ``ffmpeg`` per video, and there are 550+ sessions.  So:

``quick_check``
    Seconds.  OpenCV only -- the decoder that matters.  Reads frame 0, reads a
    bounded sequential run, and seeks to a grid of positions.  Catches the
    LMD008 mode because OpenCV's seeks fail where the stream is shredded.  When
    anything fails it escalates to a bounded sequential scan to report *where*
    OpenCV stops, which is the number that predicts how much of a re-analysis
    would be garbage.

``full_check``
    Definitive but slow.  ``ffprobe -count_frames`` for decodable-vs-header
    frames, and ``ffmpeg -f null -`` for a decoder error-line count.

**Neither subsumes the other.**  ffmpeg is far more tolerant than OpenCV: it
resynchronises past damage that stops OpenCV dead.  A file can be 100%
"decodable" by ffprobe and still be unusable for DeepLabCut.  ``ok`` therefore
goes False on the OpenCV evidence regardless of what ffprobe says.

Contract
--------
* No function raises on a *bad* file -- damage is reported in the returned dict.
* A **missing** file raises :class:`FileNotFoundError`.  That is a caller bug,
  not a data condition.
* ``cv2`` is imported inside the functions that need it (the DLC labeling env is
  py3.8 and some analysis envs have no OpenCV).  Module scope is stdlib only.
* Missing ``ffprobe``/``ffmpeg`` is reported via ``checked=False``, not raised.

``checked`` vs ``ok``
---------------------
``ok=False, checked=False`` means *we could not look* (no cv2, no ffmpeg,
timeout).  ``ok=False, checked=True`` means *we looked and it is damaged*.
Callers that exclude videos should exclude on ``checked and not ok`` so that a
missing tool does not quietly throw away the whole dataset.

Usage::

    from nems_lbhb.motor import dlc_video_qc as vqc

    r = vqc.quick_check('/auto/data/daq/.../recording.avi')
    if r['checked'] and not r['ok']:
        print('skip:', r['reason'])

    vqc.full_check(path)          # definitive, minutes for a long video
    vqc.check(path, full=True)    # convenience wrapper
"""

from __future__ import print_function

import os
import shutil
import subprocess


__all__ = ['quick_check', 'full_check', 'check', 'video_for_dlc_h5',
           'FFPROBE', 'FFMPEG']

FFPROBE = 'ffprobe'
FFMPEG = 'ffmpeg'

#: Interior positions the quick pass seeks to.  Evenly spaced at k/(n+1) of the
#: header count, so neither endpoint is probed -- the last frame of a truncated
#: file is a known-flaky read even in healthy containers.
DEFAULT_N_SEEKS = 12

#: Frames the quick pass reads sequentially from the top.  Cheap, and catches
#: files that are broken from frame 0 -- but see ``DEEP_SCAN_MAX_FRAMES``: on
#: LMD008 OpenCV survives to 2,461, well past any bounded run we can afford on
#: every video, which is why the seek grid does the real work.
DEFAULT_SEQUENTIAL_FRAMES = 300

#: Escalation budget.  Only spent on a video that already failed, to find out
#: where OpenCV actually stops.  On LMD008 this terminates after 2,461 frames in
#: ~0.7 s; on a healthy file it would be tens of seconds, hence the cap.
DEEP_SCAN_MAX_FRAMES = 20000

#: A deep scan that reaches less than this fraction of the header count is
#: reported as an early stop.  Only used to word ``reason``; it never flips
#: ``ok`` on its own, because the scan may simply have hit its budget.
SEQUENTIAL_REACH_FRACTION = 0.95

#: ``full_check`` thresholds.  Calibrated on LemonDisco: healthy siblings give
#: ratio 1.000 and 0 error lines; LMD008a02 gives 0.674 and 141,376.
MIN_DECODABLE_RATIO = 0.99
MAX_ERROR_LINES = 0

FFPROBE_TIMEOUT = 900.0
FFMPEG_TIMEOUT = 1800.0


# --------------------------------------------------------------- small helpers

def _require_file(path):
    """Missing input is a caller bug -- raise.  Damage is a data condition."""
    if not os.path.exists(path):
        raise FileNotFoundError('no such video: %s' % path)
    if not os.path.isfile(path):
        raise FileNotFoundError('not a regular file: %s' % path)


def _base_result(path, checker):
    return {
        'path': path,
        'checker': checker,
        'ok': False,
        'checked': False,
        'reason': '',
        'header_frames': None,
        'width': None,
        'height': None,
    }


def _run(cmd, timeout):
    """``subprocess.run`` with an explicit timeout, never raising.

    Returns ``(completed_or_None, error_string)``.  ``error_string`` is empty on
    a clean run; it is the thing to put in ``reason`` otherwise.
    """
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout,
                              text=True, errors='replace')
    except subprocess.TimeoutExpired:
        return None, '%s timed out after %gs' % (os.path.basename(cmd[0]),
                                                 timeout)
    except OSError as e:
        return None, 'could not run %s: %s' % (cmd[0], e)
    return proc, ''


def _tail(text, n=3):
    lines = [ln for ln in (text or '').splitlines() if ln.strip()]
    return ' | '.join(lines[-n:])


def video_for_dlc_h5(h5_path):
    """``.../<pen>/sorted/<stem>.dlc.h5`` -> the source ``.avi``, or ``None``.

    Two layouts are in the tree: a ``<pen>/<stem>.avi`` symlink (what the DLC
    inventory reconstructs) pointing at ``<pen>/<stem>/recording.avi``.  Prefer
    the symlink, fall back to the real file, return ``None`` if neither exists
    so the caller can count unresolved rows instead of crashing on them.
    """
    h5_path = str(h5_path)
    sorted_dir = os.path.dirname(h5_path)
    pen_dir = os.path.dirname(sorted_dir)
    stem = os.path.basename(h5_path)
    for suffix in ('.dlc.h5', '.h5'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    for cand in (os.path.join(pen_dir, stem + '.avi'),
                 os.path.join(pen_dir, stem, 'recording.avi')):
        if os.path.exists(cand):
            return cand
    return None


# ------------------------------------------------------------------ quick tier

def quick_check(path, n_seeks=DEFAULT_N_SEEKS,
                sequential_frames=DEFAULT_SEQUENTIAL_FRAMES,
                deep_scan_max=DEEP_SCAN_MAX_FRAMES):
    """Cheap OpenCV probe (~1-4 s on a 30-80k-frame arena video).

    OpenCV is the point: DeepLabCut decodes with it, so its verdict is the one
    that predicts whether re-analysis works.  ffprobe is more forgiving and will
    call LMD008a02 67% decodable while OpenCV manages 3%.

    Steps: open, read frame 0, read ``sequential_frames`` frames from the top,
    then seek+read at ``n_seeks`` evenly spaced interior positions.  If anything
    fails, escalate to a bounded sequential scan (``deep_scan_max``) to find
    where OpenCV actually stops -- cheap precisely because a broken file
    terminates it early.

    Returns a dict with at least ``ok``, ``header_frames``, ``seek_attempts``,
    ``seek_failures``, ``sequential_ok``, ``width``, ``height``, ``reason``.

    Raises ``FileNotFoundError`` if ``path`` does not exist.  Never raises on a
    damaged file.
    """
    _require_file(path)

    r = _base_result(path, 'quick')
    r.update({
        'seek_attempts': 0,
        'seek_failures': 0,
        'first_failed_seek': None,
        'sequential_ok': False,
        'sequential_frames_read': 0,
        'sequential_frames_requested': 0,
        'deep_scan_frames': None,
        'fps': None,
        'file_mb': None,
    })

    # Compare the raw byte count, not file_mb: a 100-byte file rounds to 0.0 MB
    # and is broken, but it is not *empty* and deserves the real reason.
    try:
        n_bytes = os.path.getsize(path)
        r['file_mb'] = round(n_bytes / 1e6, 1)
    except OSError:
        n_bytes = None
    if n_bytes == 0:
        r['checked'] = True
        r['reason'] = 'file is empty (0 bytes)'
        return r

    try:
        import cv2
    except Exception as e:            # ImportError, or a broken cv2 build
        r['reason'] = ('cv2 is not importable in this environment (%s) -- video '
                       'integrity was NOT checked' % e)
        return r

    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            r['checked'] = True
            r['reason'] = 'cv2.VideoCapture could not open the file'
            return r

        r['checked'] = True
        r['header_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        r['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        r['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        try:
            r['fps'] = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 3)
        except (TypeError, ValueError):
            r['fps'] = None

        if r['header_frames'] <= 0:
            r['reason'] = ('container reports %s frames -- the index is unusable '
                           'and DeepLabCut cannot size its output'
                           % r['header_frames'])
            return r

        # --- frame 0 -------------------------------------------------------
        got, frame = cap.read()
        if not got or frame is None:
            r['reason'] = 'OpenCV could not read frame 0'
            r.update(_deep_scan(path, deep_scan_max))
            return r

        # --- bounded sequential run ----------------------------------------
        want = int(min(sequential_frames, r['header_frames']))
        r['sequential_frames_requested'] = want
        n_read = 1
        while n_read < want:
            got, frame = cap.read()
            if not got or frame is None:
                break
            n_read += 1
        r['sequential_frames_read'] = n_read
        r['sequential_ok'] = (n_read >= want)

        # --- seek grid ------------------------------------------------------
        # This is what catches LMD008: 4 of 12 positions fail there, 0 of 12 on
        # every healthy LemonDisco sibling.
        positions = []
        for k in range(1, int(n_seeks) + 1):
            pos = int(r['header_frames'] * k / float(n_seeks + 1))
            if 0 < pos < r['header_frames'] and pos not in positions:
                positions.append(pos)
        for pos in positions:
            r['seek_attempts'] += 1
            got = False
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                got, frame = cap.read()
                got = bool(got) and frame is not None
            except Exception:
                got = False
            if not got:
                r['seek_failures'] += 1
                if r['first_failed_seek'] is None:
                    r['first_failed_seek'] = pos
    except Exception as e:
        # A decoder that segfaults its way into a Python exception is a fail,
        # not a crash of the caller.
        r['reason'] = 'OpenCV raised while probing: %r' % (e,)
        return r
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    r['ok'] = (r['sequential_ok'] and r['seek_failures'] == 0)
    if r['ok']:
        return r

    # Failed: spend the escalation budget to say where OpenCV actually stops.
    r.update(_deep_scan(path, deep_scan_max))
    r['reason'] = _quick_reason(r)
    return r


def _deep_scan(path, max_frames):
    """Sequentially decode until OpenCV refuses, capped at ``max_frames``.

    Only ever called on a video that already failed, so the usual cost is the
    ~0.7 s it takes a broken file to die.
    """
    out = {'deep_scan_frames': None, 'deep_scan_truncated': None}
    try:
        import cv2
    except Exception:
        return out
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            out['deep_scan_frames'] = 0
            out['deep_scan_truncated'] = False
            return out
        n = 0
        while n < max_frames:
            got, frame = cap.read()
            if not got or frame is None:
                break
            n += 1
        out['deep_scan_frames'] = n
        out['deep_scan_truncated'] = (n >= max_frames)
    except Exception:
        pass
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return out


def _quick_reason(r):
    bits = []
    hdr = r.get('header_frames') or 0
    if not r['sequential_ok']:
        bits.append('OpenCV sequential decode stopped after %d of the first %d '
                    'frames' % (r['sequential_frames_read'],
                                r['sequential_frames_requested']))
    if r['seek_failures']:
        bits.append('OpenCV seek+read failed at %d of %d probe positions (first '
                    'at frame %s)'
                    % (r['seek_failures'], r['seek_attempts'],
                       r['first_failed_seek']))
    reach = r.get('deep_scan_frames')
    if reach is not None and hdr > 0:
        frac = reach / float(hdr)
        if r.get('deep_scan_truncated'):
            bits.append('sequential decode still alive at frame %d (scan budget '
                        'reached)' % reach)
        else:
            bits.append('OpenCV gives up at frame %d of the %d the header claims '
                        '(%.1f%%) -- re-analysis would produce garbage for the '
                        'remaining %.1f%%'
                        % (reach, hdr, 100.0 * frac, 100.0 * (1.0 - frac)))
            if frac < SEQUENTIAL_REACH_FRACTION:
                bits.append('below the %.0f%% sequential-reach floor'
                            % (100.0 * SEQUENTIAL_REACH_FRACTION))
    return '; '.join(bits)


# ------------------------------------------------------------------- full tier

def full_check(path, timeout=FFPROBE_TIMEOUT, ffmpeg_timeout=FFMPEG_TIMEOUT,
               min_ratio=MIN_DECODABLE_RATIO, max_error_lines=MAX_ERROR_LINES,
               count_errors=True):
    """Definitive decode-everything check via ffprobe + ffmpeg.

    ``ffprobe -count_frames`` gives ``nb_read_frames`` (frames the decoder
    actually recovered) against ``nb_frames`` (what the container header
    claims); ``ffmpeg -v error -i ... -f null -`` counts decoder error lines.

    Slower than :func:`quick_check` and **more forgiving**: ffmpeg resynchronises
    past damage that stops OpenCV cold, so a ``full_check`` pass is not a
    licence to run DeepLabCut.  Run both.

    Returns a dict with ``header_frames``, ``decodable_frames``, ``ratio``,
    ``error_lines``, ``ok``.  Raises ``FileNotFoundError`` if ``path`` is
    missing; never raises on a damaged file or an absent ffmpeg.
    """
    _require_file(path)

    r = _base_result(path, 'full')
    r.update({
        'decodable_frames': None,
        'ratio': None,
        'error_lines': None,
        'ffprobe_returncode': None,
        'ffmpeg_returncode': None,
    })

    ffprobe = shutil.which(FFPROBE)
    if ffprobe is None:
        r['reason'] = ('%s not found on PATH -- decodable frame count was NOT '
                       'checked (try `module load ffmpeg`, or use quick_check)'
                       % FFPROBE)
        return r

    cmd = [ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_frames,nb_read_frames,width,height',
           '-of', 'default=noprint_wrappers=1', path]
    proc, err = _run(cmd, timeout)
    if proc is None:
        r['reason'] = err
        return r
    r['ffprobe_returncode'] = proc.returncode
    if proc.returncode != 0:
        r['checked'] = True
        r['reason'] = ('ffprobe exited %d: %s'
                       % (proc.returncode, _tail(proc.stderr)))
        return r

    r['checked'] = True
    fields = {}
    for line in (proc.stdout or '').splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            fields[k.strip()] = v.strip()

    r['header_frames'] = _as_int(fields.get('nb_frames'))
    r['decodable_frames'] = _as_int(fields.get('nb_read_frames'))
    r['width'] = _as_int(fields.get('width'))
    r['height'] = _as_int(fields.get('height'))

    if r['header_frames'] and r['decodable_frames'] is not None:
        r['ratio'] = round(r['decodable_frames'] / float(r['header_frames']), 6)

    if count_errors:
        ffmpeg = shutil.which(FFMPEG)
        if ffmpeg is None:
            r['reason'] = ('%s not found on PATH -- error lines not counted; '
                           'frame counts below are still valid' % FFMPEG)
        else:
            cmd = [ffmpeg, '-nostdin', '-v', 'error', '-i', path, '-f', 'null',
                   '-']
            proc2, err2 = _run(cmd, ffmpeg_timeout)
            if proc2 is None:
                r['reason'] = err2
            else:
                r['ffmpeg_returncode'] = proc2.returncode
                r['error_lines'] = _count_error_lines(proc2.stderr)

    return _finish_full(r, min_ratio, max_error_lines)


def _finish_full(r, min_ratio, max_error_lines):
    problems = []
    if r['header_frames'] is None or r['header_frames'] <= 0:
        problems.append('container reports no frame count (nb_frames=%s)'
                        % r['header_frames'])
    if r['decodable_frames'] is None:
        problems.append('ffprobe returned no decodable frame count')
    elif r['ratio'] is not None and r['ratio'] < min_ratio:
        problems.append('only %d of the %d frames the header claims are '
                        'decodable (%.1f%%, floor %.1f%%)'
                        % (r['decodable_frames'], r['header_frames'],
                           100.0 * r['ratio'], 100.0 * min_ratio))
    if r['error_lines'] is not None and r['error_lines'] > max_error_lines:
        problems.append('ffmpeg reported %d decoder error line(s)'
                        % r['error_lines'])
    if r['ffmpeg_returncode']:
        problems.append('ffmpeg exited %d' % r['ffmpeg_returncode'])

    if problems:
        r['ok'] = False
        sep = '; ' if r['reason'] else ''
        r['reason'] = r['reason'] + sep + '; '.join(problems)
    else:
        # A pre-existing reason here means a tool was missing, so we did not
        # actually establish the file is sound.
        r['ok'] = (not r['reason'])
    return r


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_error_lines(stderr):
    """Non-blank ffmpeg error lines, expanding ``Last message repeated N times``."""
    total = 0
    for line in (stderr or '').splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if 'last message repeated' in low:
            for tok in low.replace('last message repeated', '').split():
                if tok.isdigit():
                    total += int(tok)
                    break
            continue
        total += 1
    return total


# --------------------------------------------------------------------- wrapper

def check(path, full=False, **kwargs):
    """``quick_check`` by default, ``full_check`` when ``full=True``."""
    if full:
        return full_check(path, **kwargs)
    return quick_check(path, **kwargs)
# [AGENT EDIT END]
