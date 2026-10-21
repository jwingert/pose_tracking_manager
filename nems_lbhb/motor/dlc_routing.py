# [AGENT EDIT START | agent: claude | user: wingertj | reason: routing should suggest from what
# the recording IS -- geometry, audio, camera stream -- rather than from site numbers, which
# were only ever a proxy for data nobody had recorded | date: 2026-08-13]
"""Suggest which DLC model fits a recording, from the recording rather than its site number.

Two methods, both available, neither authoritative:

``geometry``
    Observes the recording -- chimney side, audio configuration, camera stream -- and
    matches that against each alias's ``geometry`` declaration in the registry. Knows
    nothing about animals or site numbers. This is the default the GUI preselects.

``legacy``
    ``dlc_registry.resolve_for_video``: the migrated ``if/elif`` chain, keyed on animal
    path fragments and site-number ranges. Kept so a historical file can be re-analysed
    with the model that actually produced it, and so the GUI can offer that explicitly.

**Nothing here constrains anything.** :func:`suggest` never raises and never picks for you:
it returns ranked candidates with the evidence behind each, for a GUI to preselect and a
human to override. What keeps arbitrary routing safe is not a restrictive router -- it is
that every production fit is release-stamped and recorded in ``gDLCFit``, so whatever ran
is auditable after the fact.

Geometry vocabulary (the values an alias may declare, and what is observed):

* ``chimney`` -- ``left`` / ``right`` / ``two`` / ``none``. Observed from
  ``gPenetration.ear``, which is CONTRALATERAL to the chimney: ``ear='R'`` means the
  chimney is on the LEFT. A recording with no penetration row has no implant at all.
* ``audio``   -- ``free-field`` / ``in-ear``. Observed from ``gData`` ``name='audio'``.
  In-ear means headphones are on the animal's head, which changes what the camera sees.
* ``stream``  -- ``arena`` / ``pupil`` / ``face``. Observed from the DLC cam tag. The
  arena body model applies to the ceiling view and to the side cameras alike, so
  ``base``/``camB``/``camC`` all observe as ``arena``; only the close-ups differ.

An omitted key in a declaration means "any": ``pupil`` declares only a stream because a
close-up of an eye is the same close-up whatever is implanted below it.
"""
import os
import re

from nems_lbhb.motor import dlc_alias
from nems_lbhb.motor import dlc_registry

# gPenetration.ear is contralateral to the chimney.
_CHIMNEY_FOR_EAR = {'L': 'right', 'R': 'left'}

_AUDIO_FOR_SVALUE = {'Free-field': 'free-field', 'In-ear': 'in-ear'}

# The arena body model applies to every arena view; only the close-ups are different
# models. So the cam tag collapses to three observable stream classes.
_STREAM_CLASS = {'': 'arena', 'camB': 'arena', 'camC': 'arena',
                 'pupil': 'pupil', 'lick': 'face'}

_GEOMETRY_KEYS = ('chimney', 'audio', 'stream')


# [AGENT EDIT START | agent: claude | user: wingertj | reason: celldb records no chimney
# count, but open-ephys settings.xml records how many probes were running, which is
# ASYMMETRIC evidence for it | date: 2026-08-13]
_NP_PROBE_RE = re.compile(r'<NP_PROBE[^>]*probe_serial_number="([^"]+)"')
_PROBE_STREAM_RE = re.compile(r'<STREAM name="(Probe[A-Z])-AP"')

# how much of a site's raw tree to read before giving up -- this runs behind a GUI
_MAX_SESSIONS = 4
_MAX_RUNS = 2

_PROBE_CACHE = {}


def probe_count(video_path, cache=_PROBE_CACHE):
    """Distinct Neuropixels probes recorded at this site, or ``None`` if unreadable.

    Read from the open-ephys ``settings.xml`` under ``<site>/raw/``. Record-Node COUNT is
    not usable for this -- SLJ017 has three nodes and two probes -- so the probes are
    counted directly, by serial number, falling back to the ``ProbeX-AP`` stream names.

    Cached per site directory: a GUI calls this on every rerun, and the answer cannot
    change for a site that has already been recorded.
    """
    p = str(video_path)
    parts = p.split(os.sep)
    try:
        site_dir = os.sep.join(parts[:parts.index('daq') + 3])
    except ValueError:
        return None
    if site_dir in cache:
        return cache[site_dir]

    raw = os.path.join(site_dir, 'raw')
    best = None
    try:
        for sess in sorted(os.listdir(raw))[:_MAX_SESSIONS]:
            sdir = os.path.join(raw, sess)
            if not os.path.isdir(sdir):
                continue
            for run in sorted(os.listdir(sdir))[:_MAX_RUNS]:
                rdir = os.path.join(sdir, run)
                if not os.path.isdir(rdir):
                    continue
                for node in sorted(os.listdir(rdir)):
                    if not node.startswith('Record Node'):
                        continue
                    try:
                        with open(os.path.join(rdir, node, 'settings.xml'),
                                  encoding='utf-8', errors='replace') as fh:
                            text = fh.read()
                    except OSError:
                        continue
                    n = max(len(set(_NP_PROBE_RE.findall(text))),
                            len(set(_PROBE_STREAM_RE.findall(text))))
                    if n:
                        best = max(best or 0, n)
                if best and best >= 2:
                    break                      # two is all we need to conclude anything
            if best and best >= 2:
                break
    except OSError:
        pass
    cache[site_dir] = best
    return best
# [AGENT EDIT END]


def session_stem(video_path):
    """The session stem for any of the shapes a caller holds (dir, slot file, `<stem>.avi`)."""
    p = str(video_path).rstrip('/')
    base = os.path.basename(p)
    if base in ('recording.avi', 'video_side.avi', 'video_side2.avi') or base.startswith('video_'):
        return os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(base)[0] if base.endswith('.avi') else base
    return dlc_alias.strip_stream_suffix(stem)


def observe(video_path, db=None):
    """What this recording IS: ``{chimney, audio, stream, penname, evidence, unknown}``.

    Every field is ``None`` when it cannot be determined, and the reason lands in
    ``unknown`` -- an absent fact is reported, never guessed. Database failures degrade the
    same way, so this is safe to call from a GUI with no connection.
    """
    stem = session_stem(video_path)
    penname = None
    m = re.match(r'^([A-Za-z]{3}\d+)', stem)
    if m:
        penname = m.group(1)

    tagged = dlc_alias.strip_stream_suffix(stem)
    raw_stream = dlc_alias.stream_for(os.path.basename(str(video_path))) or \
        dlc_alias.stream_for(os.path.splitext(os.path.basename(str(video_path)))[0])
    out = {'stem': stem, 'penname': penname, 'chimney': None, 'audio': None,
           'stream': _STREAM_CLASS.get(raw_stream, 'arena'),
           'probes': None, 'two_chimney_evidence': False,
           # [AGENT EDIT START | agent: Codex | user: wingertj | reason: distinguish unavailable DB evidence from legitimately missing metadata | date: 2026-09-09]
           'evidence': {}, 'unknown': [], 'db_errors': []}
           # [AGENT EDIT END]
    out['evidence']['stream'] = ('cam tag %r -> %s'
                                 % (raw_stream or '(none, ceiling view)', out['stream']))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: probe count is ASYMMETRIC
    # evidence for chimney count -- the only handle on the one fact celldb does not record
    # | date: 2026-08-13]
    # Two probes running means two chimneys: you cannot record from a chimney that is not
    # there. ONE probe means nothing -- LemonDisco carried both chimneys and habitually ran
    # a single probe, and SlipperyJack does the same at SLJ001-003 while running two at
    # SLJ004/005/017. So a count of 1 is recorded and then deliberately not acted on.
    out['probes'] = probe_count(video_path)
    if out['probes'] and out['probes'] >= 2:
        out['two_chimney_evidence'] = True
        out['evidence']['chimney_count'] = (
            '%d probes in settings.xml -> at least two chimneys' % out['probes'])
    elif out['probes'] == 1:
        out['unknown'].append('1 probe recorded, which does not establish chimney count '
                              '(a two-chimney animal may run a single probe)')
    else:
        out['unknown'].append('no open-ephys settings.xml found, so chimney count is unknown')
    # [AGENT EDIT END]

    if db is None:
        try:
            import nems_lbhb.db as db
        except Exception as exc:                          # noqa: BLE001
            out['unknown'].append('no database: %s' % exc)
            # [AGENT EDIT START | agent: Codex | user: wingertj | reason: expose refresh failures to the GUI | date: 2026-09-09]
            out['db_errors'].append('database unavailable: %s' % exc)
            # [AGENT EDIT END]
            return out

    # --- chimney side, from the penetration -------------------------------------
    if penname is None:
        out['unknown'].append('cannot parse a penetration name from %r' % stem)
    else:
        try:
            pen = db.pd_query('SELECT ear FROM gPenetration WHERE penname = %s LIMIT 1',
                              (penname,))
            if not len(pen):
                # no penetration at all -- a training recording, so nothing is implanted
                out['chimney'] = 'none'
                out['evidence']['chimney'] = ('no gPenetration row for %s -> not implanted'
                                              % penname)
            else:
                ear = pen['ear'].iloc[0]
                side = _CHIMNEY_FOR_EAR.get(str(ear).strip().upper() if ear else None)
                if side is None:
                    out['unknown'].append('gPenetration.ear=%r for %s is not L or R'
                                          % (ear, penname))
                else:
                    out['chimney'] = side
                    out['evidence']['chimney'] = (
                        '%s ear=%s -> chimney %s (ear is contralateral)'
                        % (penname, ear, side))
        except Exception as exc:                          # noqa: BLE001
            out['unknown'].append('penetration lookup failed: %s' % exc)
            # [AGENT EDIT START | agent: Codex | user: wingertj | reason: expose refresh failures to the GUI | date: 2026-09-09]
            out['db_errors'].append('penetration lookup failed: %s' % exc)
            # [AGENT EDIT END]

    # --- audio configuration, from the raw-file parameters -----------------------
    try:
        aud = db.pd_query(
            "SELECT d.svalue FROM gData d JOIN gDataRaw r ON d.rawid = r.id "
            "WHERE d.name = 'audio' AND r.parmfile IN (%s, %s) LIMIT 1",
            (tagged, tagged + '.m'))
        if len(aud):
            val = aud['svalue'].iloc[0]
            out['audio'] = _AUDIO_FOR_SVALUE.get(val)
            if out['audio'] is None:
                out['unknown'].append('unrecognised audio value %r' % val)
            else:
                out['evidence']['audio'] = 'audio=%s -> %s' % (val, out['audio'])
        else:
            out['unknown'].append('no audio parameter recorded for %s' % tagged)
    except Exception as exc:                              # noqa: BLE001
        out['unknown'].append('audio lookup failed: %s' % exc)
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: expose refresh failures to the GUI | date: 2026-09-09]
        out['db_errors'].append('audio lookup failed: %s' % exc)
        # [AGENT EDIT END]

    return out


def _match(declared, observed):
    """``True`` (declared admits it), ``False`` (contradicts), or ``None`` (can't tell)."""
    if not declared:
        return True                       # omitted == any
    if observed is None:
        return None                       # unobserved: neither supported nor refuted
    return observed in declared


def _match_chimney(declared, observed, two_evidence=False):
    """As :func:`_match`, but ``ear`` reports SIDE and says nothing about COUNT.

    ``two_evidence`` is positive proof of a second chimney (two probes were recording;
    see :func:`probe_count`). It resolves the one thing ``ear`` cannot: with it, a
    two-chimney model MATCHES and the single-chimney models are contradicted rather than
    merely outranked. Without it nothing changes -- its absence is not evidence of a single
    chimney, which is exactly why it is a separate flag and not a count.

    ``gPenetration.ear`` is the only implant fact celldb records, and it is contralateral
    to *a* chimney -- it cannot distinguish one chimney from two, and for a two-chimney
    implant it is not a geometry signal at all (the corpus bears this out: ``two_chimney``
    ran under both ear values, 48 vs 9). So an alias declaring ``two`` must come back
    UNKNOWN against a side observation, not contradicted: ruling it out would be asserting
    a fact -- the count -- that nothing has recorded. It then ranks below an exact side
    match while staying visible as a candidate, which is what a suggestion should do.
    """
    if not declared:
        return True
    if two_evidence:
        # a second chimney is established, so 'two' is confirmed and 'none' refuted
        if 'two' in declared:
            return True
        if 'none' in declared:
            return False
        # a single-chimney model on a demonstrably two-chimney implant: wrong skeleton
        return False
    if observed is None:
        return None
    if observed == 'none' or 'none' in declared:
        # "not implanted" is decidable: it comes from the absence of a penetration row
        return observed in declared
    sides = [d for d in declared if d in ('left', 'right')]
    if not sides:                          # declares only 'two'
        return None
    return observed in sides


def suggest(video_path, method='geometry', registry=None, db=None):
    """Ranked model suggestions for ``video_path``. Never raises.

    Returns ``{method, alias, candidates, observed, note}`` where ``alias`` is the top
    suggestion (``None`` when nothing matches) and ``candidates`` is a list of
    ``{alias, release, matched, unknown, why}`` ordered best-first.

    A candidate is kept when no declared dimension CONTRADICTS what was observed. A
    dimension that could not be observed neither supports nor rules out an alias, so it is
    reported in ``unknown`` rather than silently treated as a match -- the difference
    between "this fits" and "we could not tell" is exactly what a suggestion should show.

    Ranking is by dimensions positively matched, then by FEWEST unresolved requirements,
    then registry file order. The middle term is what makes a tie meaningful: between two
    aliases matching equally well, prefer the one whose declaration we could actually
    verify rather than the one still resting on facts nobody recorded. On a pre-implant
    training recording, where no penetration exists to read a side from, that is the whole
    difference between suggesting ``behavior_only`` (1 unresolved) and ``left_chimney``
    (2) -- it is worth 70 rows of the corpus. File order breaks the remaining ties, and is
    where the lab already expresses which model it prefers, so reordering the registry
    changes the suggestion, deliberately.

    An earlier version tie-broke on how narrowly an alias was declared, on the theory that
    specific beats general. That is wrong here and the corpus says so: for a right-chimney
    in-ear recording it preferred ``right_chimney_headphones`` (declared for one side, and
    never run on a single file) over ``left_chimney_headphones`` (declared for both, and
    the model that actually produced all 28 such rows). Breadth of declaration measures
    ambition, not fitness.
    """
    registry = registry or dlc_registry.load_registry()

    if method == 'legacy':
        try:
            name = dlc_registry.resolve_for_video(video_path, registry)
            block = registry['aliases'][name]
            return {'method': 'legacy', 'alias': name, 'observed': None,
                    'note': 'matched a legacy path/site rule',
                    'candidates': [{'alias': name, 'release': block.get('current'),
                                    'matched': [], 'unknown': [],
                                    'why': ['legacy rule matched the video path']}]}
        except Exception as exc:                          # noqa: BLE001
            return {'method': 'legacy', 'alias': None, 'observed': None,
                    'note': str(exc), 'candidates': []}

    obs = observe(video_path, db=db)
    candidates = []
    for order, name in enumerate(registry['alias_names']):
        block = registry['aliases'][name]
        geo = block.get('geometry') or {}
        matched, unknown, contradicted = [], [], False
        for key in _GEOMETRY_KEYS:
            if key == 'chimney':
                verdict = _match_chimney(geo.get(key), obs.get(key),
                                         two_evidence=obs.get('two_chimney_evidence'))
            else:
                verdict = _match(geo.get(key), obs.get(key))
            if verdict is False:
                contradicted = True
                break
            if verdict is None:
                unknown.append('%s unknown, %s wants %s' % (key, name, geo[key]))
            elif geo.get(key):
                # when the probe count is what settled the chimney, say so -- the side
                # reading is not what carried the decision
                ev_key = ('chimney_count' if key == 'chimney' and obs.get('two_chimney_evidence')
                          else key)
                matched.append(obs['evidence'].get(ev_key, '%s=%s' % (key, obs.get(key))))
        if contradicted:
            continue
        candidates.append({'alias': name, 'release': block.get('current'),
                           'matched': matched, 'unknown': unknown, 'why': matched,
                           '_rank': (-len(matched), len(unknown), order)})
    candidates.sort(key=lambda c: c.pop('_rank'))

    note = None
    if not candidates:
        note = 'nothing declared fits this recording'
    elif obs['unknown']:
        note = 'incomplete evidence: ' + '; '.join(obs['unknown'])
    return {'method': 'geometry', 'alias': candidates[0]['alias'] if candidates else None,
            'observed': obs, 'candidates': candidates, 'note': note}
# [AGENT EDIT END]
