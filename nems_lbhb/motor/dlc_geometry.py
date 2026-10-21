"""Recording geometry (chimney side, headphones) for DeepLabCut model selection.

A free-moving video's *geometry* is what the camera actually sees: which side of
the animal's head the implant chimney sits on, whether there are two chimneys,
and whether the animal is wearing in-ear headphones.  Those three things decide
which DLC model can track the video, so they are the axis a model-generalization
evaluation has to be stratified along.

Sources of truth -- this module derives, it does not invent
-----------------------------------------------------------
``gPenetration.ear``
    Looked up by **penname**.  ``nems_lbhb/baphy_experiment.py:302`` turns it
    into ``primary_ear`` with the rule ``ear == 'L' -> 1, else 0``, and
    ``primary_ear == 1`` is what sets ``flipchans`` in the HRTF path
    (``nems_lbhb/baphy_io.py:2740``).  :func:`primary_ear` reproduces that rule
    exactly so the two never drift apart.

``gData`` key ``'audio'``
    Headphone use is **not** a column.  ``nems_lbhb/batch_manager.py:470-480``
    derives it (``calibration_gain`` present -> ``In-ear``; else ``level > 40``
    -> ``Free-field``; else ``In-ear``) and caches the answer back into
    ``gData`` as a key-value row.  We read that cache and never re-derive it.
    It covers 304 of the 411 free-moving sessions in the DLC inventory; the
    rest are reported as unresolved.

The contralateral convention
----------------------------
**The chimney is contralateral to the primary ear.**  A right-ear penetration
(``ear == 'R'``) puts the chimney on the animal's **left**; ``ear == 'L'`` puts
it on the **right**.  This is the single most error-prone step in picking a DLC
model, so :func:`chimney_side` is the only place it is written down.

Why blank ``ear`` raises
------------------------
``BAPHYExperiment`` silently treats a blank/absent ``ear`` as right
(``if ear == 'L': 1 else: 0``).  That is a safe-enough default for a channel
flip but a wrong-model-selection bug here: a blank ear would be reported as a
confident ``left_chimney`` and the video would be pushed through a model that
never saw that geometry.  Everything in this module refuses instead.  In
practice only the ``training2023`` / ``training2024`` pennames lack an ``ear``
(70 inventory rows of ungated training video); callers are expected to exclude
them explicitly rather than have them silently mislabelled.

Likewise, unresolved audio yields ``'<chimney>_chimney (audio unknown)'`` --
never a default to free-field.

Two-chimney is observational
----------------------------
SlipperyJack and some LemonDisco animals carry bilateral implants, so both
chimneys are visible no matter which hemisphere was recorded.  That cannot be
read off ``gPenetration.ear`` at all.  It comes from the caller (in practice
from whether the session was historically analyzed with a ``two_chimney*``
model) and is passed in as ``two_chimney=``.

This is an analysis module, not the production fit path, so importing pandas
and ``nems_lbhb.db`` is fine.  Both DB tables are pulled with exactly ONE query
each and cached at module level -- never query per penname in a loop.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- chimney-side / headphone geometry lookup for the DLC model-generalization harness | date: 2026-08-06]

import logging

import pandas as pd

log = logging.getLogger(__name__)


EAR_QUERY = "SELECT penname, ear FROM gPenetration"

AUDIO_QUERY = ("SELECT r.parmfile, d.svalue AS audio "
               "FROM gData d JOIN gDataRaw r ON d.rawid=r.id "
               "WHERE d.name='audio'")

#: ``geometry`` values this module can emit.  ``unknown_audio`` is the *category*
#: for a session whose headphone state could not be resolved; the emitted string
#: is ``'<chimney>_chimney (audio unknown)'`` so it never silently sorts in with
#: the resolved free-field sessions.
#
# [AGENT EDIT START | agent: claude | user: wingertj | reason: consolidation task step 5 --
# a geometry label and the registry alias that serves that geometry must be the SAME
# string, so a label can be handed to dlc_registry.resolve() without a translation table
# | date: 2026-08-07]
#: Every value here is also an alias in ``nems_lbhb/configs/dlc_registry.toml`` (except
#: the ``unknown_audio`` category, which names a gap rather than a rig).  Keep it that
#: way: the no-chimney rig was spelled ``no_chimney`` in ``scripts/dlc_merge_labels.py``
#: and ``behavior_only`` in the registry, and one of the two names had to go.
#: ``behavior_only`` won.  It is absent from this tuple because this module cannot derive
#: it -- ``gPenetration`` always names an ear, so every session it can resolve has a
#: chimney; a no-chimney session is identified by the caller, not by the database.
GEOMETRIES = ('left_chimney', 'right_chimney', 'two_chimney',
              'left_chimney_headphones', 'right_chimney_headphones',
              'unknown_audio')
# [AGENT EDIT END]

#: Values of ``gPenetration.ear`` that name a single side.  Anything else
#: ('B', 'LR', 'L-R', blank, NULL) does not, and :func:`chimney_side` raises.
_LATERAL = {'l': 'L', 'left': 'L', 'r': 'R', 'right': 'R'}

_EAR_TABLE = None
_AUDIO_TABLE = None


class UnknownEarError(ValueError):
    """Raised when ``gPenetration.ear`` is missing, blank, or not a single side.

    Deliberately not a silent default: see the module docstring.
    """


# --------------------------------------------------------------- DB tables

def _clean(value):
    """Return a stripped string, or None for NULL/NaN/blank."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _stem_of(parmfile):
    """``'PRN001a01_a_NTD.m'`` -> ``'PRN001a01_a_NTD'`` (trailing ``.m`` only)."""
    text = _clean(parmfile)
    if text is None:
        return None
    if text.endswith('.m'):
        text = text[:-2]
    return text


def load_ear_table(force=False):
    """penname -> raw ``gPenetration.ear`` string (None where blank/NULL).

    ONE query, cached at module level for the life of the process.
    """
    global _EAR_TABLE
    if _EAR_TABLE is None or force:
        from nems_lbhb import db
        d = db.pd_query(EAR_QUERY)
        _EAR_TABLE = {str(p).strip(): _clean(e)
                      for p, e in zip(d['penname'], d['ear'])}
        log.info('loaded gPenetration ear table: %d pennames', len(_EAR_TABLE))
    return _EAR_TABLE


def load_audio_table(force=False):
    """video stem -> ``'In-ear'`` / ``'Free-field'`` from the ``gData`` cache.

    ONE query, cached at module level.  Keyed on the parmfile stem (trailing
    ``.m`` stripped) so it joins straight onto the DLC inventory's
    ``video_stem``.  Sessions with no ``audio`` row are simply absent from the
    mapping -- absence means *unresolved*, not free-field.
    """
    global _AUDIO_TABLE
    if _AUDIO_TABLE is None or force:
        from nems_lbhb import db
        d = db.pd_query(AUDIO_QUERY)
        table = {}
        for parmfile, audio in zip(d['parmfile'], d['audio']):
            stem = _stem_of(parmfile)
            value = _clean(audio)
            if stem is not None and value is not None and value != 'Unknown':
                table[stem] = value
        _AUDIO_TABLE = table
        log.info('loaded gData audio table: %d sessions', len(_AUDIO_TABLE))
    return _AUDIO_TABLE


def clear_cache():
    """Drop both cached tables (tests, or after a DB write)."""
    global _EAR_TABLE, _AUDIO_TABLE
    _EAR_TABLE = None
    _AUDIO_TABLE = None


# ------------------------------------------------------------ ear / chimney

def raw_ear(penname, ear_table=None):
    """The raw ``gPenetration.ear`` string. Raises if missing or blank."""
    table = load_ear_table() if ear_table is None else ear_table
    name = _clean(penname)
    if name is None:
        raise UnknownEarError('penname is blank')
    if name not in table:
        raise UnknownEarError(
            "penname %r is not in gPenetration -- cannot determine the ear, and "
            "guessing would silently pick a DLC model for the wrong geometry "
            "(the training20XX pennames land here; exclude them explicitly)"
            % name)
    ear = _clean(table[name])
    if ear is None:
        raise UnknownEarError(
            "gPenetration.ear is blank for penname %r -- BAPHYExperiment would "
            "default this to right (primary_ear=0), but a blank ear must not be "
            "reported as a confident chimney side" % name)
    return ear


def primary_ear(penname, ear_table=None):
    """0 or 1, matching ``baphy_experiment.py:302`` (``ear == 'L'`` -> 1, else 0).

    Differs from production in one way only: a blank/absent ear raises
    :class:`UnknownEarError` instead of falling through to 0.  A non-blank but
    non-lateral value ('B', 'LR', ...) keeps the production ``else 0`` so this
    stays a faithful mirror of what the HRTF path will actually do; use
    :func:`chimney_side`, which refuses such values, when you need a side.
    """
    return 1 if raw_ear(penname, ear_table=ear_table) == 'L' else 0


def chimney_side(penname, ear_table=None):
    """``'left'`` or ``'right'`` -- the side the implant chimney is visible on.

    **Contralateral to the primary ear**: ``ear == 'R'`` -> chimney on the
    ``'left'``; ``ear == 'L'`` -> chimney on the ``'right'``.

    Raises :class:`UnknownEarError` for a blank, missing or non-lateral ear
    ('B'/'LR' name no side, and picking one would be a guess).
    """
    ear = raw_ear(penname, ear_table=ear_table)
    side = _LATERAL.get(ear.lower())
    if side is None:
        raise UnknownEarError(
            "gPenetration.ear=%r for penname %r does not name a single side, so "
            "the chimney side cannot be derived" % (ear, penname))
    return 'left' if side == 'R' else 'right'


# --------------------------------------------------------------- geometry

def _audio_for(video_stem, audio_table=None):
    if video_stem is None:
        return None
    table = load_audio_table() if audio_table is None else audio_table
    stem = _stem_of(video_stem)
    if stem is None:
        return None
    value = table.get(stem)
    if value is None:
        value = table.get(str(video_stem).strip())
    return _clean(value)


def geometry_for(penname, video_stem=None, two_chimney=None,
                 ear_table=None, audio_table=None):
    """Full geometry record for one session.

    Parameters
    ----------
    penname : str
        e.g. ``'PRN001'``.  Keys ``gPenetration``.
    video_stem : str, optional
        e.g. ``'PRN001a01_a_NTD'``.  Keys the ``gData`` audio cache.  Omit it
        (or pass a stem with no cached ``audio`` row) and the geometry comes
        back flagged ``(audio unknown)`` rather than assumed free-field.
    two_chimney : bool, optional
        Bilateral implant -- both chimneys visible.  Not derivable from the
        database; pass it in (in practice: the session was historically
        analyzed with a ``two_chimney*`` model).  When true it overrides the
        ear-derived side.
    ear_table, audio_table : dict, optional
        Injected lookups, for tests and for batch callers that already hold the
        tables.  Default: the module-level cached DB queries.

    Returns
    -------
    dict with keys ``penname, ear, primary_ear, chimney, two_chimney, audio,
    headphones, geometry, geometry_source``.  ``headphones`` is None when the
    audio state is unresolved (never False-by-default).

    Raises
    ------
    UnknownEarError
        Unknown penname, or a blank / non-lateral ``ear``.
    """
    ear = raw_ear(penname, ear_table=ear_table)
    pear = 1 if ear == 'L' else 0
    chimney = chimney_side(penname, ear_table=ear_table)
    two = bool(two_chimney)

    audio = _audio_for(video_stem, audio_table=audio_table)
    headphones = None if audio is None else (audio == 'In-ear')

    if two:
        geometry = 'two_chimney'
        source = ("bilateral implant: caller flagged two_chimney (observational "
                  "-- historically analyzed with a two_chimney* model); this "
                  "overrides the ear-derived %s chimney side" % chimney)
    elif audio is None:
        geometry = '%s_chimney (audio unknown)' % chimney
        source = ("gPenetration.ear=%r -> chimney contralateral -> %s; no cached "
                  "gData 'audio' row for %r, so headphone use is UNKNOWN and is "
                  "not defaulted to free-field" % (ear, chimney, video_stem))
    elif headphones:
        geometry = '%s_chimney_headphones' % chimney
        source = ("gPenetration.ear=%r -> chimney contralateral -> %s; "
                  "gData audio='In-ear'" % (ear, chimney))
    else:
        geometry = '%s_chimney' % chimney
        source = ("gPenetration.ear=%r -> chimney contralateral -> %s; "
                  "gData audio=%r" % (ear, chimney, audio))

    return {'penname': _clean(penname),
            'ear': ear,
            'primary_ear': pear,
            'chimney': chimney,
            'two_chimney': two,
            'audio': audio,
            'headphones': headphones,
            'geometry': geometry,
            'geometry_source': source}


def geometry_table(pennames=None, stems=None, two_chimney=None,
                   ear_table=None, audio_table=None, errors='raise'):
    """Batch form of :func:`geometry_for` -> a :class:`pandas.DataFrame`.

    The DB tables are fetched once up front and injected into every row, so a
    10,000-row call still costs exactly two queries.

    Parameters
    ----------
    pennames : sequence of str
        Required.  One entry per session.
    stems : sequence of str, optional
        Parallel to ``pennames``.  None -> every row is audio-unknown.
    two_chimney : bool or sequence of bool, optional
        Scalar broadcasts to every row.
    errors : {'raise', 'skip'}
        ``'skip'`` drops rows whose ear cannot be resolved (and logs them)
        instead of raising.  Callers that legitimately expect ungated rows --
        the training20XX pennames -- should use it *and say so*.
    """
    if pennames is None:
        raise ValueError('geometry_table needs an explicit list of pennames')
    pennames = list(pennames)
    n = len(pennames)

    if stems is None:
        stems = [None] * n
    else:
        stems = list(stems)
        if len(stems) != n:
            raise ValueError('stems has %d entries but pennames has %d'
                             % (len(stems), n))

    if two_chimney is None or isinstance(two_chimney, bool):
        two_list = [bool(two_chimney)] * n
    else:
        two_list = [bool(t) for t in two_chimney]
        if len(two_list) != n:
            raise ValueError('two_chimney has %d entries but pennames has %d'
                             % (len(two_list), n))

    if errors not in ('raise', 'skip'):
        raise ValueError("errors must be 'raise' or 'skip', got %r" % errors)

    # two queries, up front, once
    ear_table = load_ear_table() if ear_table is None else ear_table
    audio_table = load_audio_table() if audio_table is None else audio_table

    rows = []
    for penname, stem, two in zip(pennames, stems, two_list):
        try:
            rows.append(geometry_for(penname, video_stem=stem, two_chimney=two,
                                     ear_table=ear_table,
                                     audio_table=audio_table))
        except UnknownEarError as e:
            if errors == 'raise':
                raise
            log.warning('skipping %s / %s: %s', penname, stem, e)

    columns = ['penname', 'ear', 'primary_ear', 'chimney', 'two_chimney',
               'audio', 'headphones', 'geometry', 'geometry_source']
    return pd.DataFrame(rows, columns=columns)
# [AGENT EDIT END]
