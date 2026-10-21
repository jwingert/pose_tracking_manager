"""What every free-moving session has on disk, and what it still needs.

The labeler GUI targets one video at a time, and finding the NEXT video to work on
meant walking celldb's hyperlinks by hand.  This answers that question in bulk: for a
scope (an animal, optionally narrowed to a few runclasses), one row per recording,
with the three facts that decide whether there is anything to do -- is there a video,
is there a ``.dlc.h5`` beside it in ``sorted/``, is there an arena config -- plus what
gDLCFit recorded about the fit that produced it.

Scoping then scanning is the whole performance story.  Runclass is a bad filter on its
own (266 of 14,067 FTC recordings are free-moving; the rest are head-fixed), so the
scope comes from celldb and the free-moving ones are recognised on disk.  The scan is
one ``os.listdir`` per penetration plus one per ``sorted/`` -- NOT one per session, which
does not finish in a usable time over NFS.

What dominates is whether the kernel's NFS attribute cache is warm, not the row count.
Measured end to end 2026-09-15, whole animal, every runclass: StinkySquid (824
recordings, 113 penetrations) 37.5 s cold and 0.22 s warm; Reishi (882, 131) 64.1 s cold
and 0.24 s warm; Prince (574, 76) 34.1 s cold.  A caller that re-reads this
interactively must cache it -- a cold minute is a plausible first look at an animal.

Nothing here imports streamlit or DeepLabCut, so it loads in any environment -- same
rule as :mod:`nems_lbhb.motor.dlc_fit_record`, which supplies the gDLCFit half.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- per-animal status of what each session still needs for DLC | date: 2026-09-15]

import os

# psi names video files after the camera SLOT. `recording.avi` holds the ceiling view on
# all but a handful of sessions; see dlc_camera_override for the ones where a USB
# re-enumeration moved it, which the GUI redirects to at display time.
SLOTS = ('recording.avi', 'video_side.avi', 'video_side2.avi')

CONFIG_SUFFIX = 'dlc_arena_config.toml'

COLUMNS = ('siteid', 'session', 'runclass', 'behavior', 'needs', 'has_dlc',
           'has_config', 'routing_alias', 'release_id', 'video', 'slot', 'rawid',
           'pendate')

_SQL = """
SELECT r.id AS rawid, r.cellid AS siteid, r.runclass, r.behavior,
       r.parmfile, r.resppath, p.pendate
  FROM gDataRaw r
  JOIN gCellMaster m ON r.masterid = m.id
  JOIN gPenetration p ON m.penid = p.id
 WHERE r.bad = 0 AND r.training = 0
"""


def _scope_sql(animal):
    """The scoped query and its bind parameters.

    The animal is a bind parameter, never interpolated -- it arrives from a selectbox
    today, but the module has no way to know that.
    """
    sql, params = _SQL, ()
    if animal:
        sql += " AND m.animal = %s"
        params = (animal,)
    return sql + " ORDER BY r.cellid, r.parmfile", params


def _lister():
    """A directory lister that reads each directory at most once per scan."""
    seen = {}

    def names(path):
        if path not in seen:
            try:
                seen[path] = set(os.listdir(path))
            except OSError:
                seen[path] = set()
        return seen[path]
    return names


def _session_video(resppath, stem, pen_names):
    """``(path, slot)`` for the session's video, or ``(None, None)``.

    A session directory with no video at all is a real state, not an error: the
    headphone rig records no camera, and 40 of Reishi's NFB/NFC sessions look exactly
    like this.  Those sessions are reported, with nothing to do.
    """
    if not stem or stem not in pen_names:
        return None, None
    for slot in SLOTS:
        candidate = os.path.join(resppath, stem, slot)
        if os.path.exists(candidate):
            return candidate, slot
    return None, None


def _needs(video, has_dlc, has_config):
    """The single next action for one session: what is MISSING, nothing else.

    Staleness deliberately does not appear.  A stale fit is a complete one whose alias
    now points at a newer release, so re-running it is a bulk decision about a release,
    not a gap in this session -- and on these animals it outnumbers the real gaps
    roughly four to one, which buries them.  ``release_id`` still says which model
    produced what is there.
    """
    if not video:
        return 'no video'
    if not has_dlc and not has_config:
        return 'dlc+config'
    if not has_dlc:
        return 'dlc'
    if not has_config:
        return 'config'
    return ''


def session_status(animal=None, query=None, fits=None):
    """One row per recording in scope, with what is on disk beside it.

    ``query`` is ``db.pd_query``'s signature; the GUI passes its own bounded-timeout
    reader so a slow database degrades the tab instead of hanging the page.  ``fits``
    is a gDLCFit status frame (:func:`dlc_fit_record.fit_status`); it is joined on
    ``rawid``, which every gDLCFit row carries -- ``video_stem`` is
    ``<stem>.<release_id>`` and is not a key.

    Scoped by animal only.  Narrowing by runclass in SQL would save about 0.2 s of a
    0.9 s scan and cost a re-scan every time the filter changed, so the caller holds one
    cached animal and filters the frame.
    """
    import pandas as pd

    if query is None:
        from nems_lbhb import db
        query = db.pd_query
    if fits is None:
        from nems_lbhb.motor import dlc_fit_record
        fits = dlc_fit_record.fit_status(include_current=False)

    raw = query(*_scope_sql(animal))
    if raw is None or not len(raw):
        return pd.DataFrame(columns=list(COLUMNS))

    fit_cols = ['routing_alias', 'release_id']
    if fits is None or not len(fits):
        by_rawid = {}
    else:
        keep = fits.dropna(subset=['rawid']).drop_duplicates('rawid', keep='first')
        by_rawid = {int(r.rawid): {c: getattr(r, c, None) for c in fit_cols}
                    for r in keep.itertuples(index=False)}

    names = _lister()
    rows = []
    for r in raw.itertuples(index=False):
        resppath = r.resppath or ''
        stem = os.path.splitext(r.parmfile or '')[0]
        video, slot = _session_video(resppath, stem, names(resppath))
        sorted_names = names(os.path.join(resppath, 'sorted'))
        has_dlc = (stem + '.dlc.h5') in sorted_names
        # two naming conventions are in the wild: 235 of 236 configs are
        # `recording-<stem>-dlc_arena_config.toml`, one is `<stem>_dlc_arena_config.toml`
        has_config = any(n.endswith(CONFIG_SUFFIX) and stem in n for n in sorted_names)
        fit = by_rawid.get(int(r.rawid), {})
        rows.append(dict(
            siteid=r.siteid, session=stem, runclass=r.runclass, behavior=r.behavior,
            needs=_needs(video, has_dlc, has_config),
            has_dlc=has_dlc, has_config=has_config,
            routing_alias=fit.get('routing_alias'), release_id=fit.get('release_id'),
            video=video, slot=slot, rawid=int(r.rawid), pendate=r.pendate))

    return pd.DataFrame(rows, columns=list(COLUMNS))
# [AGENT EDIT END]
