# [AGENT EDIT START | agent: claude | user: wingertj | reason: luminance sidecar gate -- the lights-out timeout punishment blanks the video, and DLC emits confident-looking garbage on those frames | date: 2026-08-11]
"""Per-frame luminance gate for DLC outputs, stored as a sidecar next to the `.dlc.h5`.

Why
---
In the early free-moving sessions an incorrect trial is punished by switching the room
lights off.  The camera then records pure black for the timeout duration -- up to 45% of a
session.  DLC still emits coordinates for those frames, and the confidences are not always
low enough to be caught by `dlc_threshold`, so blank-frame positions reach downstream
analyses as if they were real.

Validated against the behavioural event log (`to_start`/`to_end` in psi's `event_log.csv`)
on four sessions: a per-frame `std < 5` gate matches the logged timeout intervals with
precision 0.96-0.99 and recall 0.94-0.97 once the video/event clock offset is fitted, and
flags ZERO non-timeout frames on a session that has timeouts without lights-out.  Absolute
one-sided `std` beats every adaptive rule tried (deviation-from-mean, Otsu): with no dark
mode present those split the normal lit variation instead, flagging 6-99% of clean files.

Why a sidecar and not an edit of the `.dlc.h5`
----------------------------------------------
Those `.h5` files are the targets of release-stamped aliases; rewriting likelihoods in
place would break the reproducibility guarantee the release store exists to provide, with
no undo.  The sidecar leaves them byte-identical, is cheap to recompute if the threshold
changes, and records WHY each frame was masked so a post-error analysis can deliberately
re-include the timeout epoch that a pose analysis wants excluded.

Contents of `<video_stem>.lumgate.npz`
--------------------------------------
    usable      bool  [n_frames]   False = do not trust DLC on this frame
    frame_std   f32   [n_frames]   per-frame pixel std of a 1/16 spatial subsample
    frame_mean  f32   [n_frames]   per-frame pixel mean (second half of the gate)
    reason      u8    [n_frames]   0=ok 1=timeout(lights out) 2=dark, not matched to an event
    meta        json  thresholds, code version, fitted clock lag, event-match statistics
"""
import glob
import json
import os
import re

import numpy as np

VERSION = 2
# A frame is masked only when BOTH statistics collapse. std alone is not safe: a session
# recorded at 100 fps has a short exposure and sits at std ~3.95 for its whole duration
# while being perfectly trackable (REI084a10: mean pinned at 25.2, std<5 on 96% of frames,
# animal clearly visible once contrast-stretched). A genuine lights-out frame has mean ~1-3
# AND std ~0, so requiring both keeps the timeout detection (validated at F1 0.95-0.99
# against the behavioural event log) and drops that false-positive mode.
MEAN_THRESH = 20.0
STD_THRESH = 5.0
SUBSAMPLE = 16
REASON = {0: 'ok', 1: 'timeout', 2: 'dark_unmatched'}


def sidecar_path(dlc_h5_path):
    """Sidecar for a `.dlc.h5` (or its alias): `<video_stem>.lumgate.npz` beside it."""
    d, b = os.path.split(str(dlc_h5_path))
    stem = b.split('DLC_')[0].split('.dlc.h5')[0]
    stem = re.sub(r'\.dlc$', '', stem)
    return os.path.join(d, stem + '.lumgate.npz')


def find_video(dlc_h5_path, video_stem=None):
    """Source video for a `.dlc.h5` living in `<penname>/sorted/`."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(str(dlc_h5_path))))
    if video_stem is None:
        video_stem = os.path.basename(str(dlc_h5_path)).split('DLC_')[0]
        video_stem = re.sub(r'\.dlc\.h5$|\.h5$', '', video_stem).rstrip('.')
    for c in (os.path.join(d, video_stem + '.avi'),
              os.path.join(d, video_stem, 'recording.avi'),
              os.path.join(d, video_stem, 'video_side.avi'),
              os.path.join(d, video_stem, 'video_side2.avi')):
        if os.path.exists(c):
            return c
    return None


def _session_dir_for(video):
    """Directory holding psi's event_log.csv / trial_log.csv for this video.

    The video is usually reached as `<penname>/<stem>.avi`, a symlink into
    `<penname>/<stem>/recording.avi` -- the logs live in that subdirectory, not beside the
    symlink, so a plain dirname() finds nothing.
    """
    cands = []
    real = os.path.realpath(video)
    cands.append(os.path.dirname(real))                       # .../<stem>/
    d, b = os.path.split(os.path.abspath(video))
    cands.append(os.path.join(d, os.path.splitext(b)[0]))     # .../<stem>/ from the symlink
    cands.append(d)
    for c in cands:
        if os.path.exists(os.path.join(c, 'event_log.csv')):
            return c
    return None


def _timeout_intervals(session_dir):
    """Merged [start, end) timeout intervals from psi's event_log.csv, in event-clock s."""
    ev = os.path.join(session_dir, 'event_log.csv')
    tl = os.path.join(session_dir, 'trial_log.csv')
    if not os.path.exists(ev):
        return []
    import pandas as pd
    e = pd.read_csv(ev)
    if 'event' not in e.columns:
        return []
    dur = 4.0
    if os.path.exists(tl):
        t = pd.read_csv(tl)
        if 'to_duration' in t.columns and t['to_duration'].notna().any():
            dur = float(t['to_duration'].dropna().iloc[0])
    starts = e.loc[e.event == 'to_start', 'timestamp'].values
    ends = e.loc[e.event == 'to_end', 'timestamp'].values
    iv = []
    for s in starts:
        nxt = ends[ends > s]
        iv.append((float(s), float(nxt[0]) if len(nxt) else float(s) + dur))
    iv.sort()
    merged = []
    for s, en in iv:
        # a repeated to_start extends one blackout rather than starting a new one
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((s, en))
    return merged


def _fit_lag(dark, fps, intervals, span=40.0):
    """Video/event clock offset maximising F1 between the dark mask and timeout intervals.

    psi's `psivideo_frame_ts` implies frame 0 at event-time 0, which is wrong by 1.5-6 s and
    varies per session, so the offset is fitted rather than read.
    """
    if not intervals or dark.sum() == 0:
        return None, None
    t = np.arange(len(dark)) / fps
    best = (None, -1.0, None)
    for lag in np.arange(-span, span, 1.0 / fps):
        m = np.zeros(len(dark), bool)
        for s, e in intervals:
            m[(t + lag >= s) & (t + lag < e)] = True
        tp = int((dark & m).sum())
        if tp == 0:
            continue
        f1 = 2.0 * tp / (dark.sum() + m.sum())
        if f1 > best[1]:
            best = (float(lag), float(f1), m)
    return best[0], best


def build(video, out_path, session_dir=None, std_thresh=STD_THRESH,
          mean_thresh=MEAN_THRESH, overwrite=False):
    """Compute and write the sidecar. Returns a summary dict."""
    if os.path.exists(out_path) and not overwrite:
        return dict(status='exists', path=out_path)
    import cv2
    cv2.setLogLevel(0)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mu, sd = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        s = fr[::SUBSAMPLE, ::SUBSAMPLE]
        mu.append(s.mean()); sd.append(s.std())
    cap.release()
    if not sd:
        return dict(status='unreadable', path=out_path, video=video)
    mu = np.asarray(mu, np.float32); sd = np.asarray(sd, np.float32)
    dark = (sd < std_thresh) & (mu < mean_thresh)
    usable = ~dark
    reason = np.where(dark, 2, 0).astype(np.uint8)

    lag = f1 = prec = rec = None
    if session_dir is None:
        session_dir = _session_dir_for(video)
    iv = _timeout_intervals(session_dir) if session_dir else []
    if iv and dark.any():
        lag, best = _fit_lag(dark, fps, iv)
        if lag is not None:
            _, f1, m = best
            reason[dark & m] = 1                       # blackout explained by a timeout
            tp = int((dark & m).sum())
            prec = tp / float(dark.sum()); rec = tp / float(max(m.sum(), 1))
    meta = dict(version=VERSION, std_thresh=float(std_thresh),
                mean_thresh=float(mean_thresh), subsample=SUBSAMPLE,
                fps=float(fps), n_frames=int(len(sd)), video=video,
                frac_masked=float(dark.mean()), clock_lag_s=lag,
                event_f1=f1, event_precision=prec, event_recall=rec,
                n_timeout_intervals=len(iv), reason_legend=REASON)
    tmp = out_path + '.tmp.npz'
    np.savez_compressed(tmp, usable=usable, frame_std=sd, frame_mean=mu,
                        reason=reason, meta=json.dumps(meta))
    os.replace(tmp, out_path)                          # atomic
    return dict(status='built', path=out_path, **meta)


def load(dlc_h5_path):
    """Sidecar for this `.dlc.h5`, or None. Never raises."""
    p = sidecar_path(dlc_h5_path)
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p, allow_pickle=False)
        return dict(usable=z['usable'], frame_std=z['frame_std'],
                    frame_mean=z['frame_mean'] if 'frame_mean' in z else None,
                    reason=z['reason'], meta=json.loads(str(z['meta'])), path=p)
    except Exception:
        return None


def apply_to_dataframe(dataframe, dlc_h5_path, include_reasons=(), log=None):
    """Zero the likelihood of masked frames in a freshly-loaded DLC dataframe.

    Returns (dataframe, n_masked). A no-op when no sidecar exists, so callers need no
    branch. `include_reasons` re-admits specific reasons -- e.g. `('timeout',)` for an
    analysis that WANTS the timeout epoch -- because masking those frames silently
    conditions every behavioural distribution on "not in timeout".
    """
    g = load(dlc_h5_path)
    if g is None:
        return dataframe, 0
    usable = g['usable']
    if include_reasons:
        codes = {c for c, name in REASON.items() if name in include_reasons}
        for c in codes:
            usable = usable | (g['reason'] == c)
    n = len(dataframe)
    if len(usable) != n:
        # frame-count mismatch: trust the shorter one and leave the tail untouched
        k = min(len(usable), n)
        full = np.ones(n, bool); full[:k] = usable[:k]
        usable = full
        if log is not None:
            log.warning('lumgate length %d != dlc rows %d for %s; gating first %d only',
                        len(g['usable']), n, dlc_h5_path, k)
    bad = ~usable
    if not bad.any():
        return dataframe, 0
    idx = dataframe.index[bad]
    cols = [c for c in dataframe.columns if c[-1] == 'likelihood']
    dataframe.loc[idx, cols] = 0.0
    if log is not None:
        log.info('lumgate: zeroed likelihood on %d/%d frames (%.1f%%) from %s',
                 int(bad.sum()), n, 100.0 * bad.mean(), os.path.basename(g['path']))
    return dataframe, int(bad.sum())
# [AGENT EDIT END]
