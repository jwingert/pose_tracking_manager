"""Attribute an existing DeepLabCut output to the frozen release that produced it.

552 ``.dlc.h5`` outputs exist on disk from before release stamping
(``nems_lbhb.motor.dlc_release``) existed, and none of them record which model
produced them.  This module is the evidence ladder that answers that question for
one output at a time, given the ``scorer`` string DLC embedded in its filename and
the file's mtime.  It never touches the celldb database and never walks the
``daq`` tree -- its only inputs are the release manifests under
``/auto/data/dlc/releases`` (read-only) and the hand-curated rules in
``nems_lbhb/configs/dlc_attribution.toml``.

A SCORER MATCH IS EVIDENCE, NOT A CONCLUSION.  Several DLC networks in this lab
retrain in place, or regenerate their training set, without changing the scorer
string DLC bakes into an output's filename -- the same string comes out of two (or
three) different sets of weights.  Where that happens the rule file either
separates the outputs by mtime or forces the scorer to ``ambiguous`` rather than
letting a coincidental string match stand in for an unidentifiable set of weights.

The full evidence ladder, weakest to strongest, is:

* ``unknown``         -- no scorer, or a scorer no release manifest emits.
* ``ambiguous``       -- the scorer matches more than one release and nothing
                          (an mtime rule) separates them, or a rule explicitly
                          forces ambiguity (``forced_ambiguous`` in the TOML).
* ``inferred_mtime``  -- the scorer is shared by two releases; a ``cutoff_utc``
                          in the rule file splits which one produced this output.
* ``inferred_scorer`` -- the scorer matches exactly one release manifest.
* ``inferred_hash``   -- NOT produced here.  A later stage (the filesystem
                          sweeper) resolves one otherwise-unresolvable output by
                          content-hashing it against a sibling output whose scorer
                          it borrows, then calls :func:`attribute` with that
                          borrowed scorer and overrides the returned
                          ``attribution`` to ``'inferred_hash'``.
* ``stamped``         -- NOT produced here either.  A release-stamped filename
                          (one written after ``dlc_release`` existed) is
                          recognised by the sweeper before it ever calls
                          :func:`attribute`; this module only ever sees legacy,
                          unstamped outputs.

:func:`attribute` is pure: given a scorer, a target mtime, and (optionally) an
already-loaded ``release_index``/``rules`` pair, it returns a decision and never
reads or writes anything besides the two inputs above.  Dependency-free by
design, like its sibling ``dlc_registry``: standard library plus ``tomli`` only,
no NEMS, no DeepLabCut, no pandas, no numpy, no yaml.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- evidence ladder attributing legacy (pre-release-stamp) DLC outputs to the frozen release that produced them | date: 2026-08-12]
import datetime
import glob
import os

try:                       # python >= 3.11
    import tomllib
except ImportError:        # python <= 3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


DEFAULT_RULES_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'dlc_attribution.toml'))

# Where frozen releases live.  Read-only from here: this module never writes into
# this tree, it only reads RELEASE.toml manifests out of it.
RELEASE_ROOT = '/auto/data/dlc/releases'

# path -> (mtime, rules dict).  Keyed on (path, mtime): editing the TOML busts the
# entry by itself, so a test can force a reload just by touching the file.
_RULES_CACHE = {}

# root -> (sorted [(manifest_path, mtime), ...], index dict).  Keyed on the sorted
# list of every manifest's (path, mtime) so that adding, removing or editing a
# RELEASE.toml under root busts the cache.
_INDEX_CACHE = {}


def _toml_parser():
    """Return the TOML module, or raise an ImportError that says what to install."""
    if tomllib is None:
        raise ImportError(
            "reading %s needs a TOML parser: python>=3.11 has 'tomllib' in the "
            "standard library, on older pythons run `pip install tomli`"
            % DEFAULT_RULES_PATH)
    return tomllib


def load_rules(path=None):
    """Parse the attribution rule TOML and return it as a dict.

    Keys: ``schema_version, mtime_rules, forced_ambiguous, reproducibility``.
    ``mtime_rules`` and ``forced_ambiguous`` default to ``[]`` and
    ``reproducibility`` to ``{}`` when absent from the file, so callers never have
    to guard a missing key.

    Cached on ``(path, mtime)``, so an edit to the file is picked up automatically
    -- a test that rewrites the TOML just has to make sure the mtime actually
    changes.
    """
    if path is None:
        path = DEFAULT_RULES_PATH
    path = os.path.normpath(os.path.abspath(path))

    mtime = os.path.getmtime(path)
    cached = _RULES_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(path, 'rb') as f:
        raw = _toml_parser().load(f)

    rules = {
        'schema_version': raw.get('schema_version'),
        'mtime_rules': raw.get('mtime_rules', []),
        'forced_ambiguous': raw.get('forced_ambiguous', []),
        'reproducibility': raw.get('reproducibility', {}),
    }
    _RULES_CACHE[path] = (mtime, rules)
    return rules


def release_index(root=RELEASE_ROOT):
    """Map every release's output scorer to the release id(s) that emit it.

    Globs ``<root>/*/RELEASE.toml``, reads ``[release].id`` and
    ``[model].output_suffix`` out of each, and returns
    ``{output_suffix: sorted([release_id, ...])}``.  A manifest with an empty or
    missing ``output_suffix`` (or a missing ``[release].id``) is skipped, not
    crashed on -- a malformed manifest must not take down attribution for every
    other release.

    Cached on the sorted ``(path, mtime)`` list of every manifest under ``root``.
    """
    paths = sorted(glob.glob(os.path.join(root, '*', 'RELEASE.toml')))
    key = tuple((p, os.path.getmtime(p)) for p in paths)

    cached = _INDEX_CACHE.get(root)
    if cached is not None and cached[0] == key:
        return cached[1]

    index = {}
    for p in paths:
        with open(p, 'rb') as f:
            data = _toml_parser().load(f)
        suffix = data.get('model', {}).get('output_suffix', '')
        release_id = data.get('release', {}).get('id', '')
        if not suffix or not release_id:
            continue
        index.setdefault(suffix, []).append(release_id)

    for suffix, ids in index.items():
        ids.sort()

    _INDEX_CACHE[root] = (key, index)
    return index


# release_index() the function is shadowed inside attribute() by its own
# release_index= parameter; keep a module-level alias so the default value can
# still call it by name.
_default_release_index = release_index


def _to_utc(dt):
    """Normalise a datetime to UTC-aware.  A naive value is assumed to already be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _parse_cutoff(text):
    """Parse a TOML ``cutoff_utc`` string (RFC 3339, e.g. a trailing ``Z``) to UTC-aware."""
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    return _to_utc(datetime.datetime.fromisoformat(text))


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- attribution_evidence
# is a category error site: it used to carry this sentence too, conflating "which release
# produced this row" (attribution) with "do the weights still exist to reproduce it"
# (reproducible), two orthogonal questions the gDLCFit column split apart on purpose. The
# constant is KEPT -- it is still the true reason `reproducible` defaults to 1 (Stage 2 audited
# every DLC project and found no damage for a release absent from [reproducibility]) -- but as
# of this edit it documents that policy for a docstring/spec reader only. `_finalize` below no
# longer appends it to any row's `evidence`; the rationale for `reproducible=1` lives here and
# in the module docstring of nems_lbhb/motor/dlc_fit_record.py, not duplicated onto 459 rows.
# | date: 2026-08-12]
REPRODUCIBLE_BY_DEFAULT_EVIDENCE = (
    'reproducible=1 by default: this release is absent from the [reproducibility] table, and '
    'absence is not silence -- the Stage 2 provenance audit covered every DLC project (it is '
    'what produced defects 2-9 and the provenance warnings in RELEASES.md) and found no '
    'damage for this one.')
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 2 -- across 551
# real backfill rows, 375 carry an attribution_evidence of exactly this sentence (measured
# 2026-08-12, before the final-fix below stopped appending reproducibility text to evidence
# at all): a bare scorer match with no row-specific argument of its own -- so a caller reading
# a non-null attribution_evidence can trust "there is a real argument here, read it". Named
# here, once, rather than as a repeated literal: `attribute()` returns it verbatim for ladder
# step 4, and `_finalize` strips it (via :func:`_strip_generic_evidence`) in the one case
# where nothing else survives. | date: 2026-08-12]
GENERIC_SCORER_MATCH_EVIDENCE = 'scorer matches exactly one release manifest'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- attribution_evidence
# is attribution-only now (see _finalize below), so the one thing left that can make an
# inferred_scorer row's evidence a bare, row-non-specific sentence is the generic match text
# itself; the REPRODUCIBLE_BY_DEFAULT_EVIDENCE-specific branch this used to have is gone
# because that text is never appended to evidence anymore. | date: 2026-08-12]
def _strip_generic_evidence(evidence):
    """Return ``None`` when ``evidence`` is exactly :data:`GENERIC_SCORER_MATCH_EVIDENCE`,
    otherwise return ``evidence`` unchanged.

    A bare "scorer matches exactly one release manifest" restates what the `attribution`
    column (``inferred_scorer``) already says and argues nothing about the specific row, so
    it is ``NULL`` rather than stored boilerplate. Anything else -- an mtime-rule sentence, a
    forced-ambiguous entry, a content-hash-match note, or the generic sentence followed by a
    DAMAGED PROVENANCE override (the reproducibility half of which :func:`_finalize` no
    longer concatenates in here) -- is left untouched, including the case where the generic
    sentence is a prefix of something else: that combination no longer arises (nothing is
    appended to ``evidence`` after ladder step 4 returns it), but a caller handing this
    function a hand-built string with that shape still gets it back unchanged rather than
    silently truncated, since only an exact match is stripped.
    """
    if evidence is None:
        return None
    if evidence == GENERIC_SCORER_MATCH_EVIDENCE:
        return None
    return evidence
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- attribution_evidence
# and reproducible answer two orthogonal questions (which release produced this row vs. do the
# weights still exist to regenerate it), and appending reproducibility prose into the
# attribution column was a category error: the 64 left_chimney rows are identified with high
# confidence (inferred_scorer) AND simultaneously unreproducible (weights destroyed by an
# in-place retrain) -- two independent facts, not one argument. `_finalize` now only computes
# `reproducible` as a flag; it never touches `evidence` again, regardless of which branch
# below decides the flag's value. Anyone wanting the full story behind a `reproducible = 0`
# row follows `release_id` -> `release_path` -> that release's RELEASE.toml, whose
# `provenance_warning` carries the complete warning verbatim (see left_chimney-r0001's, which
# is the literal source of REPRODUCIBLE_BY_DEFAULT_EVIDENCE's sibling entry's `evidence` key in
# dlc_attribution.toml) -- nothing is lost, it is one hop away and it is the authoritative copy
# rather than a duplicated one. The `reproducible = 1` default's rationale (Stage 2 audit
# coverage) is now policy, documented once at REPRODUCIBLE_BY_DEFAULT_EVIDENCE's definition and
# in the spec, not repeated onto 459 rows. | date: 2026-08-12]
def _finalize(release_id, attribution, evidence, rules):
    """Attach ``reproducible`` (looked up by ``release_id``) and return the result dict.

    ``evidence`` explains ATTRIBUTION ONLY -- which release produced this row -- and this
    function never appends anything to it; the only thing done to ``evidence`` here is
    stripping it down via :func:`_strip_generic_evidence` when it is nothing but the generic
    scorer-match sentence. The rationale for ``reproducible``'s value is deliberately NOT
    stored in this column: for ``1`` it is a policy default (see
    :data:`REPRODUCIBLE_BY_DEFAULT_EVIDENCE`'s own docstring and the spec); for ``0`` it is
    whatever the producing release's ``RELEASE.toml`` ``provenance_warning`` says, one hop
    away via ``release_id`` -> ``release_path``, not duplicated here.

    Absent from ``rules['reproducibility']`` means reproducible.  ``release_id is
    None`` means ``reproducible`` is ``None`` -- there is nothing to look up.
    """
    if release_id is None:
        reproducible = None
    else:
        entry = rules['reproducibility'].get(release_id)
        if entry is None:
            reproducible = True
        else:
            reproducible = entry.get('reproducible', True)
    evidence = _strip_generic_evidence(evidence)
    return {
        'release_id': release_id,
        'attribution': attribution,
        'evidence': evidence,
        'reproducible': reproducible,
    }


def attribute(scorer, target_mtime_utc, alias=None, release_index=None, rules=None):
    """Attribute one legacy DLC output to the release that produced it.

    ``scorer`` is the ``DLC_resnet50_..._<snapshot>`` string embedded in the
    output's filename; ``target_mtime_utc`` is the output file's mtime (naive or
    tz-aware -- naive is assumed UTC, and both sides are normalised to UTC-aware
    before any comparison, since Python 3 raises ``TypeError`` comparing naive to
    aware and that would abort a sweep over hundreds of files on one bad row).

    ``alias`` is accepted for forward compatibility with future callers (e.g. a
    sweeper that also knows which registry alias routed the source video) but is
    not consulted by this ladder: every decision below is made from ``scorer`` and
    ``target_mtime_utc`` alone.

    ``release_index`` and ``rules`` default to :func:`release_index` and
    :func:`load_rules` respectively; pass them explicitly (as the golden test
    does) to avoid re-reading the manifests/TOML on every call in a sweep over
    hundreds of rows.

    Returns a dict with keys ``release_id, attribution, evidence, reproducible``.
    Evaluated in this order; the first answer wins:

    1. no ``scorer`` -> ``unknown``.
    2. ``scorer`` is forced ambiguous by the rule file -> ``ambiguous``.
    3. an ``mtime_rules`` entry names this ``scorer`` -> ``inferred_mtime``,
       split on ``cutoff_utc`` (at-or-after is ``after``, before is ``before``).
    4. the release index maps this ``scorer`` to exactly one release ->
       ``inferred_scorer``.
    5. the index maps it to more than one release and no rule separated them ->
       ``ambiguous``.
    6. otherwise -> ``unknown``.
    """
    if release_index is None:
        release_index = _default_release_index()
    if rules is None:
        rules = load_rules()

    if not scorer:
        return _finalize(None, 'unknown', 'no scorer could be parsed from the filename', rules)

    for entry in rules['forced_ambiguous']:
        if entry['scorer'] == scorer:
            return _finalize(None, 'ambiguous', entry['evidence'], rules)

    for rule in rules['mtime_rules']:
        if rule['scorer'] == scorer:
            cutoff = _parse_cutoff(rule['cutoff_utc'])
            target = _to_utc(target_mtime_utc)
            release_id = rule['after'] if target >= cutoff else rule['before']
            return _finalize(release_id, 'inferred_mtime', rule['evidence'], rules)

    candidates = release_index.get(scorer, [])
    if len(candidates) == 1:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 2 --
        # the module constant, not a repeated literal | date: 2026-08-12]
        return _finalize(candidates[0], 'inferred_scorer', GENERIC_SCORER_MATCH_EVIDENCE, rules)
        # [AGENT EDIT END]
    if len(candidates) > 1:
        evidence = ('scorer matches %d release manifests with no rule to separate them: %s'
                    % (len(candidates), ', '.join(candidates)))
        return _finalize(None, 'ambiguous', evidence, rules)

    return _finalize(None, 'unknown', 'scorer matches no release manifest', rules)
# [AGENT EDIT END]
