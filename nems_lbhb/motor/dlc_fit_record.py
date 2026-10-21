"""Schema, row identity, validation and SQL access for ``gDLCFit``, the celldb
table that records every DeepLabCut tracking output in the lab as one row.

The module has two halves. The PURE half (above) is the DDL (as a string
constant, not executed), the row-identity hash, record construction and
validation, QC-document coercion, and the staleness predicate; it never opens
a database connection and never executes SQL -- :data:`CREATE_TABLE_SQL` is a
string a caller runs, not something this half runs itself. The SQL half
(below, its own ``AGENT EDIT`` block) is the only code in the project that
reads or writes ``gDLCFit`` through the connection helpers in ``nems_lbhb.db``,
and only ever touches the single table it is called with -- never implicitly:
:func:`build_record` and every function in the pure half computes a dict or a
bool and returns it, nothing more, and :func:`create_table` is the only
function anywhere in the module that runs DDL.

:data:`QC_KEYS` is the single place the current QC statistic set is written down;
change the shape of the ``qc`` JSON document there and nowhere else.

:func:`is_stale` is the one function in the pure half with a runtime dependency,
and even that is lazy: it imports ``nems_lbhb.motor.dlc_registry`` (whose
``load_registry()`` shape is documented on that module) only when a caller does
not hand it an already-loaded registry, so importing this module never touches
the registry TOML.

Dependency-free by design, like its siblings ``dlc_registry`` and
``dlc_attribution``: standard library only at module level.  ``nems_lbhb.db``,
``pandas`` and ``json`` are for the SQL half and are imported inside the
functions that need them, not here.

Column-order contract: :data:`COLUMNS` mirrors ``CREATE_TABLE_SQL`` exactly, and
every DataFrame the SQL half returns from the database carries columns in that
fixed order -- :func:`fits_for` and :func:`current_fit` return exactly
:data:`COLUMNS`; :func:`fit_status` and :func:`stale_fits` return
:data:`COLUMNS` followed by :data:`_STATUS_EXTRA_COLUMNS`. This is the contract
the Streamlit app reads and does not change without updating both.

.. _column-alias-vs-release:

``routing_alias`` names which model routes this video **today** (the alias
:func:`nems_lbhb.motor.dlc_registry.resolve_for_video` currently answers for its
path), not which model *produced* it -- that is ``release_id``. The two diverge
whenever a video was analyzed by a model that has since been superseded or
retired; :func:`is_stale` compares ``release_id`` against the registry's
``current`` release *for* ``routing_alias``, since that is the alias promoting a
new release would repoint. ``release_path`` is derived, never independent data:
:func:`build_record` always sets it to
``os.path.join(dlc_attribution.RELEASE_ROOT, release_id)`` when ``release_id``
is set (``None`` otherwise), so it can never disagree with ``release_id``.

.. _column-evidence-vs-reproducible:

``attribution_evidence`` and ``reproducible`` answer two orthogonal questions and
must never be conflated. ``attribution_evidence`` explains ATTRIBUTION ONLY --
which release produced this row, and why -- and nothing else; it is written by
``nems_lbhb.motor.dlc_attribution._finalize`` (and the sweeper's ``stamped``
branch, which mirrors it) and is stripped to ``NULL`` when nothing but a bare,
row-non-specific scorer-match sentence would otherwise be stored (see
:func:`nems_lbhb.motor.dlc_attribution._strip_generic_evidence`).
``reproducible`` is a FLAG, not prose: whether the weights that produced this row
still exist to regenerate it, set regardless of how that was verified. Its
rationale for ``1`` is policy, not row text -- every release absent from the
``[reproducibility]`` table in ``nems_lbhb/configs/dlc_attribution.toml`` defaults
to reproducible because the Stage 2 provenance audit covered every DLC project and
found no damage for it (see
:data:`nems_lbhb.motor.dlc_attribution.REPRODUCIBLE_BY_DEFAULT_EVIDENCE`'s own
docstring, and the design spec, for the full argument). Its rationale for ``0`` is
never duplicated into this table either: follow ``release_id`` -> ``release_path``
-> that release's ``RELEASE.toml``, whose ``provenance_warning`` field carries the
complete warning verbatim -- one hop away, and the authoritative copy rather than
a second one that could drift from it.

.. _column-addedby-vs-requested-by:

``addedby`` and ``requested_by`` are also two different questions, kept
deliberately separate rather than collapsed: ``addedby`` is the ACCOUNT THAT
EXECUTED THE WRITE -- ``'lbhb'`` for a row a queued fit job stamped on its own
output, or the operator's own account for a sweeper backfill run by hand --
while ``requested_by`` is the PERSON WHO ASKED FOR IT, which may be a different
human than whoever's account actually ran the job. Neither is redundant with the
other and neither should be inferred from the other. ``lastmod`` is
``TIMESTAMP ... ON UPDATE CURRENT_TIMESTAMP`` in the DDL, but ``gDLCFit`` is
append-only (see :func:`insert_fit`/:func:`insert_many`'s ``INSERT IGNORE`` --
nothing ever ``UPDATE``s an existing row), so in practice ``lastmod`` is simply
the row's insertion time and never actually gets updated after the fact.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- the module
# docstring above now documents two more separations: attribution_evidence vs reproducible
# (the "column-evidence-vs-reproducible" paragraph -- reproducibility prose was being appended
# into attribution_evidence, a category error the fix in dlc_attribution.py's _finalize
# removes) and addedby vs requested_by (the "column-addedby-vs-requested-by" paragraph, the
# distinction the user asked to have recorded and chose to keep as-is) | date: 2026-08-12]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema changes 1 and 3 -- the
# module docstring above now documents the routing_alias/release_id distinction and the
# release_path derivation (the "column-alias-vs-release" paragraph), since fit_status's
# column order/meaning is the Streamlit app's interface | date: 2026-08-12]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- gDLCFit schema, row key and record validation (pure half of the DB layer, no SQL execution) | date: 2026-08-12]
import datetime
import getpass
import hashlib
import os
import socket


TABLE = 'gDLCFit'

# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review follow-up -- the
# datetime columns are DATETIME(6), not DATETIME. A fractional-second-less DATETIME does not
# truncate a sub-second value, it ROUNDS it, and the mtimes stored here are the evidence
# several attribution decisions rest on: two_chimney splits 42/13 on
# `mtime < 2024-01-12 12:08:00`, so a file at 12:07:59.7 would store as 12:08:00 and land on
# the wrong side of its own cutoff. Anyone re-deriving that split with a SQL WHERE -- the
# obvious thing to do with this table -- would then get a different answer than the sweeper
# did. Truncating the value before binding would fix the flip too, but DATETIME(6) keeps the
# full fidelity for free, and it can only be chosen now: adding it later needs an ALTER.
# NOTE the deliberate asymmetry with row_key(), which stays at SECONDS resolution -- see the
# note in its docstring. That is not an inconsistency to be fixed. | date: 2026-08-12]

# [AGENT EDIT START | agent: claude | user: wingertj | reason: three pre-DDL schema changes,
# the last moment they are free (gDLCFit does not exist yet; afterwards each would need a
# forbidden ALTER). (1) `alias` -> `routing_alias`, KEY alias_idx -> routing_alias_idx: the
# column holds which model routes this video TODAY (dlc_registry.resolve_for_video), not
# which model PRODUCED it (that is release_id) -- the two diverge in 63 of 523 attributed
# rows, and the plain name `alias` invites exactly that misreading. (2) new `release_path`
# column, placed immediately after `release_id`: DERIVED, not independent data --
# os.path.join(dlc_attribution.RELEASE_ROOT, release_id), computed unconditionally in
# build_record() below so it can never drift from release_id. If dlc_attribution.RELEASE_ROOT
# ever moves, every stored release_path must be regenerated (a sweeper re-run does this for
# free, since gDLCFit is append-only and a moved release store would need a fresh backfill
# anyway). | date: 2026-08-12]
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS gDLCFit (
  id                   INT UNSIGNED NOT NULL AUTO_INCREMENT,
  row_key              CHAR(64) NOT NULL,
  rawid                INT UNSIGNED DEFAULT NULL,
  siteid               VARCHAR(32) DEFAULT NULL,
  animal               VARCHAR(64) DEFAULT NULL,
  penname              VARCHAR(64) DEFAULT NULL,
  video_stem           VARCHAR(255) DEFAULT NULL,
  stream               VARCHAR(16) NOT NULL DEFAULT '',
  alias_path           VARCHAR(512) NOT NULL,
  target_path          VARCHAR(512) DEFAULT NULL,
  target_mtime_utc     DATETIME(6) DEFAULT NULL,
  is_link              TINYINT NOT NULL DEFAULT 1,
  routing_alias        VARCHAR(64) DEFAULT NULL,
  release_id           VARCHAR(64) DEFAULT NULL,
  release_path         VARCHAR(512) DEFAULT NULL,
  scorer               VARCHAR(255) DEFAULT NULL,
  snapshot             INT DEFAULT NULL,
  source_project       VARCHAR(512) DEFAULT NULL,
  attribution          VARCHAR(24) NOT NULL,
  attribution_evidence TEXT,
  reproducible         TINYINT DEFAULT NULL,
  source               VARCHAR(16) NOT NULL DEFAULT 'sweeper',
  queueid              INT DEFAULT NULL,
  requested_by         VARCHAR(64) DEFAULT NULL,
  analyzed_utc         DATETIME(6) DEFAULT NULL,
  code_git_hash        VARCHAR(40) DEFAULT NULL,
  host                 VARCHAR(64) DEFAULT NULL,
  qc                   JSON DEFAULT NULL,
  addedby              VARCHAR(50) DEFAULT NULL,
  lastmod              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY row_key_idx (row_key),
  KEY rawid_idx (rawid),
  KEY siteid_idx (siteid),
  KEY video_stem_idx (video_stem),
  KEY routing_alias_idx (routing_alias),
  KEY release_id_idx (release_id),
  KEY attribution_idx (attribution)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# Evidence-ladder rungs a row's `attribution` may hold; see dlc_attribution's
# module docstring for what each one means.  Anything else is an authoring error.
ATTRIBUTIONS = ('stamped', 'inferred_scorer', 'inferred_mtime', 'inferred_hash',
                 'ambiguous', 'unknown')

# Who wrote this row: an enqueued fit job stamping its own output, or the
# filesystem sweeper backfilling a legacy one.
SOURCES = ('job', 'sweeper')

# The complete set of keys the `qc` JSON document may carry.  This tuple is the
# single place the current QC statistic set is written down -- add a new stat
# here first, nowhere else, and coerce_qc() picks it up automatically (as a
# string unless also listed in _QC_INT_KEYS/_QC_FLOAT below).
QC_KEYS = ('ratio_worst', 'ratio_worst_part', 'track_worst', 'track_worst_part',
           # [AGENT EDIT START | agent: claude | user: wingertj | reason: qc v2 -- every
           # per-marker residual the Procrustes fit already computes, the means that
           # become the headline, the second threshold frac_tracked is measured against,
           # and the context that decides whether any of it is trustworthy (head scale,
           # reference size/sources, warnings). An unlisted key is dropped SILENTLY by
           # coerce_qc, so this tuple is the schema. | date: 2026-08-17]
           'parts', 'ratio_mean', 'offset_mean', 'track_mean',
           'track_p_cut', 'frames_scored', 'frac_valid', 'scale_px', 'mean_shape',
           'has_reference', 'reference_frames', 'reference_sources', 'warnings',
           # [AGENT EDIT END]
           # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- the
           # self-referential head-fix spread is the primary reference-free headline,
           # and origin_part names the marker whose spread is structurally zero (the
           # head-fix origin, excluded from the means) | date: 2026-08-18]
           'rms_spread_mean', 'rms_spread_worst', 'rms_spread_worst_part',
           'origin_part',
           # [AGENT EDIT END]
           # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 --
           # self-Procrustes companion spread | date: 2026-08-19]
           'rms_spread_proc_mean', 'rms_spread_proc_worst',
           'rms_spread_proc_worst_part',
           # [AGENT EDIT END]
           'p_median', 'frames', 'p_cut', 'reference_fingerprint',
           'reference_source', 'computed_utc', 'qc_module_version', 'error')

# QC_KEYS coerced to int; everything not in this tuple, _QC_STR_KEYS or
# _QC_STRUCT_KEYS below is coerced to float.
_QC_INT_KEYS = ('frames', 'frames_scored', 'reference_frames', 'qc_module_version')

# QC_KEYS that are nested JSON structures, stored as-is rather than coerced to a
# scalar: 'parts' is {part: {metric: value}}, 'mean_shape' is [[x, y], ...],
# 'reference_sources'/'warnings' are lists, 'has_reference' is a bool.
_QC_STRUCT_KEYS = ('parts', 'mean_shape', 'reference_sources', 'warnings',
                   'has_reference')

# QC_KEYS that are text, not numeric, and so pass through str() rather than
# float()/int().
_QC_STR_KEYS = ('ratio_worst_part', 'track_worst_part', 'reference_fingerprint',
                 'reference_source', 'computed_utc', 'error',
                 'rms_spread_worst_part', 'origin_part',
                 'rms_spread_proc_worst_part')

# Row fields build_record() fills when the caller does not supply them.  Kept as
# a dict (rather than named arguments) so build_record() accepts exactly the
# gDLCFit columns a caller passes and nothing about its signature has to change
# when a new optional column is added -- add it here and it defaults to None.
_DEFAULTS = {
    'rawid': None,
    'siteid': None,
    'animal': None,
    'penname': None,
    'video_stem': None,
    'stream': '',
    'target_path': None,
    'target_mtime_utc': None,
    'is_link': 1,
    'routing_alias': None,
    'release_id': None,
    # release_path is deliberately NOT listed here: build_record() always derives it from
    # release_id (see there), so a default would never be read.
    'scorer': None,
    'snapshot': None,
    'source_project': None,
    'attribution_evidence': None,
    'reproducible': None,
    'source': 'sweeper',
    'queueid': None,
    'requested_by': None,
    'analyzed_utc': None,
    'code_git_hash': None,
    'host': None,
    'qc': None,
    'addedby': None,
}


def _to_utc(dt):
    """Normalise a datetime to UTC-aware.  A naive value is assumed to already be UTC.

    Mirrors ``dlc_attribution._to_utc``: kept as a private copy here rather than a
    cross-import so this module stays a self-contained leaf, like its siblings.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def row_key(alias_path, target_realpath, target_mtime_utc):
    """SHA-256 identity of one tracking output: ``alias_path``, ``target_realpath``
    and ``target_mtime_utc`` joined by newlines and hashed.

    The mtime is formatted at SECONDS resolution with no timezone suffix
    (``%Y-%m-%dT%H:%M:%S`` after normalising to UTC-aware), so a naive datetime and
    a tz-aware datetime naming the same instant hash identically -- callers should
    not have to track which representation a particular source handed them.  A
    re-analysis that rewrites the target file bumps its mtime and so mints a new
    row key, which is the point: gDLCFit accretes one row per analysis, it never
    overwrites one in place.

    The seconds resolution here is DELIBERATE and is not an oversight to be
    reconciled with the ``DATETIME(6)`` ``target_mtime_utc`` column: identity is
    ``(alias_path, target_realpath, second)``, and two rows differing only in the
    sub-second part of one mtime are the same file, not two analyses.  The column
    stores full microseconds because it is evidence, which wants fidelity; the key
    truncates because it is an identity, which wants stability.  Do not couple them.
    """
    iso = _to_utc(target_mtime_utc).strftime('%Y-%m-%dT%H:%M:%S')
    text = '\n'.join([alias_path, target_realpath, iso])
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def coerce_qc(d):
    """Coerce a ``qc`` dict to the types gDLCFit's JSON column expects.

    ``None`` in, ``None`` out.  Keys outside :data:`QC_KEYS` are dropped silently
    -- an unrecognised stat is not this row's problem to carry.  Recognised keys
    are coerced by role: :data:`_QC_INT_KEYS` to ``int``, :data:`_QC_STR_KEYS` to
    ``str``, everything else (the numeric ratios/fractions) to ``float`` -- JSON
    has no column type, so ``'0.5'`` and ``0.5`` must not compare differently once
    they round-trip through it.

    A value that cannot be coerced (e.g. a non-numeric string for a float key) is
    DROPPED, not raised -- a single bad statistic must never cost the whole row --
    and a note is appended to the returned dict's ``error`` key so the failure is
    still visible.
    """
    if d is None:
        return None

    out = {}
    problems = []
    for k, v in d.items():
        if k not in QC_KEYS:
            continue
        if v is None:
            out[k] = None
            continue
        try:
            if k in _QC_INT_KEYS:
                out[k] = int(v)
            elif k in _QC_STR_KEYS:
                out[k] = str(v)
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: qc v2 carries
            # nested structures (per-part table, mean_shape, reference_sources,
            # warnings). float() on a dict raises, which under the rule below would
            # DROP the per-marker measurements and merely note it in 'error' -- the
            # whole point of recording them, lost quietly. | date: 2026-08-17]
            elif k in _QC_STRUCT_KEYS:
                if not isinstance(v, (dict, list, tuple, bool)):
                    raise TypeError('expected a dict/list for %r, got %s'
                                    % (k, type(v).__name__))
                out[k] = list(v) if isinstance(v, tuple) else v
            # [AGENT EDIT END]
            else:
                out[k] = float(v)
        except (TypeError, ValueError):
            problems.append('could not coerce %s=%r' % (k, v))

    if problems:
        note = '; '.join(problems)
        out['error'] = (out['error'] + '; ' + note) if out.get('error') else note

    return out


def build_record(**fields):
    """Build one gDLCFit row dict from keyword fields, filling defaults and the
    computed ``row_key``.

    ``alias_path`` (non-empty) and a valid ``attribution`` (one of
    :data:`ATTRIBUTIONS`) are the only required fields; a missing or empty
    ``alias_path``, or an ``attribution``/``source`` outside their allowed sets,
    raises ``ValueError`` naming the offending value.  Every other gDLCFit column
    not supplied defaults per :data:`_DEFAULTS` (``None``, except ``stream=''``,
    ``is_link=1`` and ``source='sweeper'``, matching the DDL's own defaults).

    ``addedby`` defaults to ``os.environ.get('USER')`` (falling back to
    ``getpass.getuser()``) and ``host`` to ``socket.gethostname()`` when not
    supplied, so a caller building a record for its own analysis run does not have
    to look either up itself.  ``qc`` is passed through :func:`coerce_qc`.

    ``release_path`` is never taken from ``fields`` -- see the note in the function
    body -- it is always derived from ``release_id``, so the two can never disagree.

    Does not touch the database and does not call :func:`create_table` -- building
    a record and persisting it are separate steps, deliberately.
    """
    alias_path = fields.get('alias_path') or ''
    if not alias_path:
        raise ValueError('alias_path is required and must be non-empty')

    attribution = fields.get('attribution')
    if attribution not in ATTRIBUTIONS:
        raise ValueError('unknown attribution %r; must be one of %s'
                          % (attribution, ATTRIBUTIONS))

    source = fields.get('source', _DEFAULTS['source'])
    if source not in SOURCES:
        raise ValueError('unknown source %r; must be one of %s' % (source, SOURCES))

    row = dict(_DEFAULTS)
    row.update(fields)
    row['alias_path'] = alias_path
    row['attribution'] = attribution
    row['source'] = source

    if not row.get('addedby'):
        row['addedby'] = os.environ.get('USER') or getpass.getuser()
    if not row.get('host'):
        row['host'] = socket.gethostname()

    row['qc'] = coerce_qc(row.get('qc'))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 3 --
    # release_path is DERIVED data, computed here unconditionally (never taken from a
    # caller-supplied `fields['release_path']`, which this deliberately overwrites) so it
    # can never independently drift from release_id -- the invariant the user asked to be
    # enforced, not merely tested. dlc_attribution is imported lazily, matching is_stale()'s
    # own lazy import below: this module stays a standard-library-only leaf at import time,
    # and every caller of RELEASE_ROOT already accepts that dlc_attribution is a sibling
    # stdlib+tomli leaf, not a heavy dependency. Regenerate every stored release_path if
    # dlc_attribution.RELEASE_ROOT ever moves -- see the comment above CREATE_TABLE_SQL.
    # | date: 2026-08-12]
    if row.get('release_id'):
        from nems_lbhb.motor import dlc_attribution
        row['release_path'] = os.path.join(dlc_attribution.RELEASE_ROOT, row['release_id'])
    else:
        row['release_path'] = None
    # [AGENT EDIT END]

    row['row_key'] = row_key(row['alias_path'], row.get('target_path') or '',
                              row['target_mtime_utc'])
    return row


# [AGENT EDIT START | agent: claude | user: wingertj | reason: read-time "is this the analysis
# currently being served?", replacing a stored is_current flag that would have required
# UPDATEs on append-only rows | date: 2026-08-12]
def pointer_path(alias_path):
    """The ``<stem>.dlc.h5`` current-pointer path for a row's ``alias_path``.

    For a release-stamped alias this strips the stamp; for a plain alias it is the path
    itself. The cam-tag/stream suffix is deliberately KEPT -- ``<stem>_camB.dlc.h5`` is its
    own pointer, served independently of the base camera's.
    """
    from nems_lbhb.motor import dlc_alias
    d = os.path.dirname(str(alias_path))
    base, _rid = dlc_alias.parse_stamped_alias(alias_path)
    return os.path.join(d, base + '.dlc.h5')


def is_current(row, _cache=None):
    """True when the current pointer for this session resolves to THIS row's output.

    ``None`` when it cannot be decided: no ``target_path`` recorded, or no pointer on disk
    (a temp/scratch analysis that was never given one). ``None`` is not ``False`` -- "not
    being served" and "cannot tell" are different answers and a caller filtering for stale
    work should be able to see which it got.
    """
    target = row.get('target_path')
    alias = row.get('alias_path')
    if not target or not alias:
        return None
    p = pointer_path(alias)
    if _cache is not None and p in _cache:
        resolved = _cache[p]
    else:
        try:
            resolved = os.path.realpath(p) if os.path.lexists(p) else None
        except OSError:
            resolved = None
        if _cache is not None:
            _cache[p] = resolved
    if resolved is None:
        return None
    return resolved == os.path.normpath(str(target))


def _is_current_series(df):
    """:func:`is_current` over a DataFrame, resolving each distinct pointer once."""
    cache = {}
    return df.apply(lambda row: is_current(row, _cache=cache), axis=1)
# [AGENT EDIT END]


def is_stale(row, registry=None):
    """True when ``row`` no longer reflects the alias's currently-served release.

    ``row`` is a gDLCFit row (dict-like, e.g. a DataFrame record or the dict
    :func:`build_record` returns).  ``registry`` defaults to
    ``dlc_registry.load_registry()``, imported lazily so importing this module
    never touches the registry TOML.

    Stale (``True``) when any of:

    * ``attribution`` is ``'ambiguous'`` or ``'unknown'`` -- there is no reliable
      release to compare against in the first place.
    * ``reproducible`` is ``0`` -- re-running the release that produced this
      output would not reproduce it, so nothing can vouch for it.
    * ``release_id`` is ``None`` -- nothing was attributed.
    * ``release_id`` starts with ``'archive_'`` -- a release retired on purpose.
    * the row's ``routing_alias`` is absent from the registry, or the registry's
      ``current`` release for that alias differs from ``release_id``.

    An alias the registry has dropped entirely returns ``True`` rather than
    raising ``KeyError`` -- a report walking thousands of rows must not crash on
    one that names a retired alias; it should just sort with the rest of the
    stale work.
    """
    if registry is None:
        from nems_lbhb.motor import dlc_registry
        registry = dlc_registry.load_registry()

    if row.get('attribution') in ('ambiguous', 'unknown'):
        return True
    if row.get('reproducible') == 0:
        return True

    release_id = row.get('release_id')
    if release_id is None:
        return True
    if release_id.startswith('archive_'):
        return True

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 1 -- gDLCFit
    # column `alias` renamed `routing_alias`; dlc_registry's own 'aliases' vocabulary
    # (registry.get('aliases', ...)) is untouched -- only the gDLCFit-side lookup key changes
    # | date: 2026-08-12]
    block = registry.get('aliases', {}).get(row.get('routing_alias'))
    # [AGENT EDIT END]
    if block is None:
        return True

    return block.get('current') != release_id
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 3 -- SQL half of the DB layer: create_table, insert_fit/insert_many, fits_for/current_fit/fit_status/stale_fits, and a NaN/Inf-safe qc JSON serialization path | date: 2026-08-12]
#
# Everything below this point is the SQL half: it is the only code in the
# project that opens a connection to ``gDLCFit`` (or, in tests, a same-shaped
# sibling table). All statements are parameterized ``sqlalchemy.text()`` bound
# through ``nems_lbhb.db.Engine()`` -- never ``nems_lbhb.db.sqlinsert``, which
# emits the delete-and-reinsert flavor of INSERT (destroying the append-only
# semantics ``row_key`` and this module's tests exist to protect) and builds
# SQL by string-concatenating values while stripping ``"`` from every string it
# inserts (which would corrupt the ``qc`` JSON column). Every write goes
# through ``with engine.begin() as
# conn:``, not ``engine.connect()``: on SQLAlchemy 2.x a bare ``connect()``
# outside a transaction block leaves an InnoDB insert uncommitted.
#
# Column order contract: :data:`COLUMNS` mirrors ``CREATE_TABLE_SQL`` exactly,
# and :func:`fits_for`/:func:`current_fit` return DataFrames with columns in
# that order. :func:`fit_status`/:func:`stale_fits` return those same columns
# followed by :data:`_STATUS_EXTRA_COLUMNS`, in that order. This ordering is a
# contract the Streamlit app reads -- it does not change without updating both.

# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema changes 1 and 3 --
# `alias` renamed `routing_alias` (routes-today, not produced-by; see the comment above
# CREATE_TABLE_SQL); `release_path` added immediately after `release_id`, mirroring the DDL
# | date: 2026-08-12]
COLUMNS = ('id', 'row_key', 'rawid', 'siteid', 'animal', 'penname', 'video_stem',
           'stream', 'alias_path', 'target_path', 'target_mtime_utc', 'is_link',
           'routing_alias', 'release_id', 'release_path', 'scorer', 'snapshot',
           'source_project', 'attribution', 'attribution_evidence', 'reproducible',
           'source', 'queueid', 'requested_by', 'analyzed_utc', 'code_git_hash', 'host',
           'qc', 'addedby', 'lastmod')
# [AGENT EDIT END]

# COLUMNS minus the two columns the database fills in itself (AUTO_INCREMENT
# id, ON UPDATE CURRENT_TIMESTAMP lastmod) -- exactly the keys build_record()
# hands back, and the columns every INSERT below writes.
INSERT_COLUMNS = tuple(c for c in COLUMNS if c not in ('id', 'lastmod'))

# fit_status()/stale_fits() append these, in this order, after COLUMNS: the
# QC scalars a caller most often wants flat (plus their *_part companions),
# then the values computed at read time rather than stored.
# [AGENT EDIT START | agent: claude | user: wingertj | reason: rms_spread_mean is the
# headline QC number the Database tab triages on; it has always been in the qc document
# but was never lifted, so nothing could sort or filter on it | date: 2026-09-16]
_STATUS_EXTRA_COLUMNS = ('ratio_worst', 'ratio_worst_part', 'track_worst',
                          'track_worst_part', 'rms_spread_mean', 'p_median', 'p_cut',
                          'stale', 'flag_track', 'flag_ratio', 'is_current')
# [AGENT EDIT END]


def _validate_identifier(name):
    """Reject a ``table`` argument that is not a plain SQL identifier.

    A table name cannot be bound as a query parameter -- MySQL has no
    placeholder for an identifier -- so every function below splices ``table``
    into SQL text directly rather than through a bound parameter. Every caller
    in this module passes a hardcoded constant (:data:`TABLE`, or a test's own
    table name such as ``gDLCFit_test``), never a value a user typed, but this
    check costs nothing and removes the question.
    """
    import re
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name or ''):
        raise ValueError('not a valid table identifier: %r' % (name,))
    return name


def _qc_to_json(qc):
    """Serialize a ``qc`` dict to a JSON string safe for gDLCFit's JSON column.

    ``None`` in, ``None`` out -- binds as SQL ``NULL``, not the string
    ``'null'``. Every non-finite float (``nan``, ``inf``, ``-inf``) is remapped
    to ``None`` before serialization: ``json.dumps`` would otherwise emit the
    bare tokens ``NaN``/``Infinity``, which are not valid JSON and which
    MySQL's JSON column rejects outright. This is not hypothetical --
    ``dlc_headfix_qc.worst_rigid()`` and ``worst_rigid_tracking()`` both return
    ``float('nan')`` when no rigid marker has a finite value, so a real QC
    document will eventually carry one.

    The sanitization happens HERE, in the serialization path, rather than in
    :func:`coerce_qc` -- so the guarantee holds regardless of what a caller
    passes in, including a hand-built dict that never went through
    :func:`coerce_qc` at all. ``allow_nan=False`` is kept as a backstop: any
    gap in the sanitization becomes a loud ``ValueError`` here, not a silent
    bad write.
    """
    import json
    import math

    if qc is None:
        return None

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: this walked only the
    # TOP level, which was complete while every value was a scalar. qc v2 nests
    # ('parts' is {part: {metric: value}}, 'mean_shape' is [[x, y], ...]), so a
    # non-finite buried one level down reached json.dumps and raised, defeating the
    # guarantee this docstring makes. Recursive now. | date: 2026-08-17]
    def _sanitize(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: _sanitize(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_sanitize(x) for x in v]
        return v
    # [AGENT EDIT END]

    return json.dumps({k: _sanitize(v) for k, v in qc.items()}, allow_nan=False)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review C1/C2 -- a record whose values came out of a pandas DataFrame (which is how the sweeper builds all 551 backfill rows) carries pandas-typed scalars the MySQL driver cannot bind: a float64 column's missing value arrives as float('nan') (pymysql raises "nan can not be used with MySQL", failing the WHOLE executemany), and a datetime64[ns, UTC] column's value arrives as a tz-aware pd.Timestamp, which pymysql's exact-type encoder table misses and stringifies WITH its offset, so MySQL's session time_zone silently converts it out of UTC and into local time -- in a column whose name and contract say UTC. Normalised HERE, in the one function every INSERT binds through, for the same reason _qc_to_json sanitizes there rather than in coerce_qc: the guarantee then holds for any caller, including one that never went near pandas. | date: 2026-08-12]
# gDLCFit columns declared INT/TINYINT in CREATE_TABLE_SQL. A FINITE float
# arriving for one of these (64.0, the shape a pandas float64 column gives an
# integer) is bound as an int, so the value MySQL stores and the value a caller
# handed over are the same number in the same type.
_INT_COLUMNS = ('rawid', 'is_link', 'snapshot', 'reproducible', 'queueid')


def _is_missing(value):
    """True for ``None`` and for every "absent" sentinel a DataFrame can hand over.

    ``float('nan')`` and ``NaT`` are caught by the fact that they are the only
    values not equal to themselves. Pandas' ``NA`` is caught separately: comparing
    it returns ``NA`` rather than a bool, so truth-testing the comparison raises
    ("boolean value of NA is ambiguous") -- it is still a missing value. Written
    without importing pandas: this module is a standard-library-only leaf.
    """
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return type(value).__name__ == 'NAType'


def _bind_value(column, value):
    """Normalise one value to something the MySQL driver binds correctly.

    Four rules, in order:

    * ``None`` binds as SQL ``NULL``, unchanged.
    * A missing value -- ``float('nan')``, pandas' ``NaT`` and pandas' ``NA``, the
      three ways an absent value reaches here from a DataFrame -- becomes ``None``,
      i.e. SQL ``NULL``. Non-finite is not a number this schema can store, and
      ``NULL`` is what "absent" already means in every one of these columns.
    * A datetime-like value (``datetime.datetime``, or anything exposing
      ``to_pydatetime`` such as ``pandas.Timestamp``) becomes a plain naive
      ``datetime.datetime`` in UTC: converted to UTC when it carries a tzinfo,
      then stripped of it. MySQL's ``DATETIME`` has no timezone, and the contract
      of ``target_mtime_utc``/``analyzed_utc`` is that the stored value IS UTC --
      handing the driver an aware value instead lets the session's ``time_zone``
      shift it (by a DST-dependent amount, so not even a constant offset).
      A naive datetime is assumed to be UTC already, matching :func:`_to_utc`.
      Microseconds pass through UNTOUCHED: the columns are ``DATETIME(6)``, which
      stores them exactly, and rounding or truncating here would put a value in
      the database that is not the mtime the file actually has.
    * A number binds as ``int`` for the :data:`_INT_COLUMNS` (and for any integral
      value anywhere, e.g. a numpy ``int64``) and as ``float`` otherwise.

    Anything else -- strings above all -- passes through untouched. Deliberately
    duck-typed rather than importing pandas: this module is a standard-library-only
    leaf, and the normalisation is defined by what a value IS, not by which library
    produced it.
    """
    import math
    import numbers

    if _is_missing(value):
        return None

    to_pydatetime = getattr(value, 'to_pydatetime', None)
    if callable(to_pydatetime):
        value = to_pydatetime()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        return int(as_float) if column in _INT_COLUMNS else as_float

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: a tuple reached this
    # unchallenged and MySQL answered with "Operand should contain 1 column(s)" -- an error
    # naming neither the column nor the value | date: 2026-08-12]
    # Every gDLCFit column is a scalar, so a sequence here is always a caller bug (the one
    # that happened: dlc_release._git_hash returns (hash, warning), and the hash alone was
    # meant). Refuse by column name, at the point the value is known, instead of letting the
    # driver reject the whole INSERT with a message that identifies nothing.
    if isinstance(value, (tuple, list, set, dict)):
        raise TypeError(
            'gDLCFit column %r got a %s, but every column here is a scalar: %r. '
            'If this came from a helper returning a pair, pass the element you meant.'
            % (column, type(value).__name__, value))
    # [AGENT EDIT END]

    return value


def _row_params(record):
    """Build the ``{column: bound value}`` mapping :func:`insert_fit` and
    :func:`insert_many` execute against, covering every :data:`INSERT_COLUMNS`
    entry. Every value goes through :func:`_bind_value` first -- see there for
    what that guarantees and why it is done here rather than at the call site.
    ``qc`` is re-coerced through :func:`coerce_qc` (idempotent for an
    already-coerced value; defensive for a record a caller built by hand) and
    then serialized via :func:`_qc_to_json`.
    """
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: `qc` is the one column
    # that legitimately holds a mapping on the way in, and its _bind_value result was
    # discarded by the next line anyway -- excluding it keeps the scalar guard in
    # _bind_value meaningful for every column where a sequence really is a bug
    # | date: 2026-08-12]
    params = {c: _bind_value(c, record.get(c)) for c in INSERT_COLUMNS if c != 'qc'}
    # [AGENT EDIT END]
    params['qc'] = _qc_to_json(coerce_qc(record.get('qc')))
    return params
# [AGENT EDIT END]


def create_table(table=TABLE):
    """Create ``table`` (default :data:`TABLE`) if it does not already exist.

    Executes :data:`CREATE_TABLE_SQL` verbatim when ``table == TABLE``;
    otherwise the table name is substituted into the DDL string -- the one
    place in this module an identifier, not a value, is spliced into SQL text,
    because MySQL has no bind-parameter syntax for a table name. This is the
    only function in the module that runs DDL, and nothing else here calls it
    (see ``test_create_table_is_never_called_implicitly``): creating the real
    ``gDLCFit`` table is a deliberate, one-time, human-confirmed act, not a
    side effect of any read or write path.
    """
    import logging

    import nems_lbhb.db as db
    from sqlalchemy import text

    _validate_identifier(table)
    sql = CREATE_TABLE_SQL if table == TABLE else CREATE_TABLE_SQL.replace(TABLE, table, 1)
    engine = db.Engine()
    with engine.begin() as conn:
        conn.execute(text(sql))
    logging.getLogger(__name__).info('created table %s (if it did not already exist)', table)


def insert_fit(record, table=TABLE):
    """Insert one gDLCFit row (a dict shaped like :func:`build_record`'s output).

    Uses ``INSERT IGNORE``, so a second insert of a ``record`` whose
    ``row_key`` already exists in ``table`` is silently ignored rather than
    raising or overwriting -- gDLCFit is append-only, a re-analysis mints a new
    ``row_key`` (see :func:`row_key`), and this is not the function that
    decides whether that happened. Returns 1 if the row was inserted, 0 if it
    was ignored as a duplicate ``row_key``.
    """
    import nems_lbhb.db as db
    from sqlalchemy import text

    _validate_identifier(table)
    col_list = ', '.join(INSERT_COLUMNS)
    placeholders = ', '.join(':%s' % c for c in INSERT_COLUMNS)
    stmt = text('INSERT IGNORE INTO %s (%s) VALUES (%s)' % (table, col_list, placeholders))
    engine = db.Engine()
    with engine.begin() as conn:
        res = conn.execute(stmt, _row_params(record))
    return res.rowcount


def insert_many(records, table=TABLE):
    """Insert many gDLCFit rows in one ``executemany`` batch.

    Same ``INSERT IGNORE`` semantics as :func:`insert_fit`, applied per row.
    Returns ``{'inserted': N, 'ignored': M}`` where ``N`` is the number of rows
    actually written and ``M`` is the number skipped as duplicate ``row_key``s,
    with ``N + M == len(records)``.
    """
    import nems_lbhb.db as db
    from sqlalchemy import text

    _validate_identifier(table)
    if not records:
        return {'inserted': 0, 'ignored': 0}
    col_list = ', '.join(INSERT_COLUMNS)
    placeholders = ', '.join(':%s' % c for c in INSERT_COLUMNS)
    stmt = text('INSERT IGNORE INTO %s (%s) VALUES (%s)' % (table, col_list, placeholders))
    engine = db.Engine()
    with engine.begin() as conn:
        res = conn.execute(stmt, [_row_params(r) for r in records])
    inserted = res.rowcount
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: fix review Minor (a) -- sanity-check rowcount so a driver returning e.g. -1 (some DBAPIs use it for "unknown") surfaces as a loud error rather than silently yielding a negative 'inserted' or an 'ignored' count exceeding len(records) | date: 2026-08-12]
    if not (0 <= inserted <= len(records)):
        raise RuntimeError(
            'insert_many: driver returned an unusable rowcount %r for %d records '
            '(expected 0..%d); refusing to report inserted/ignored counts'
            % (inserted, len(records), len(records)))
    # [AGENT EDIT END]
    return {'inserted': inserted, 'ignored': len(records) - inserted}


# [AGENT EDIT START | agent: claude | user: wingertj | reason: qc v2 needs the 553 existing
# rows re-scored, but insert_many is INSERT IGNORE and an unchanged file keeps its row_key,
# so a re-run would silently write NOTHING. Updating qc does not violate append-only: the
# PROVENANCE columns record a historical fact (which network produced this file) and stay
# immutable; qc is a derived MEASUREMENT of that same file, and re-measuring it with a
# better method is an improvement, not a rewrite. | date: 2026-08-17]
def update_qc_many(pairs, table=TABLE):
    """Replace the ``qc`` document of existing rows, keyed by ``row_key``.

    ``pairs`` is an iterable of ``(row_key, qc_dict)``.  Only the ``qc`` column is
    touched -- every provenance column is left exactly as recorded.  Returns
    ``{'updated': N, 'missing': M}``; a ``row_key`` with no row is counted, not raised,
    because a sweep legitimately sees files that were never inserted.
    """
    import nems_lbhb.db as db
    from sqlalchemy import text

    _validate_identifier(table)
    pairs = [(k, q) for k, q in pairs if k]
    if not pairs:
        return {'updated': 0, 'missing': 0}

    stmt = text('UPDATE %s SET qc = :qc WHERE row_key = :row_key' % table)
    engine = db.Engine()
    updated = 0
    with engine.begin() as conn:                  # begin(), not connect(): a bare
        for key, qc in pairs:                     # connect()+execute() silently rolls
            res = conn.execute(stmt, {            # back under SQLAlchemy 2.0
                'row_key': key, 'qc': _qc_to_json(coerce_qc(qc))})
            updated += res.rowcount or 0
    return {'updated': updated, 'missing': len(pairs) - updated}
# [AGENT EDIT END]


def _query_fits(filters, stale_only=False, table=TABLE):
    """Shared SELECT behind :func:`fits_for` and :func:`current_fit`.

    ``filters`` is a ``{column: value}`` mapping; a ``None`` value is omitted
    from the ``WHERE`` clause rather than matched literally. Results are always
    ordered ``target_mtime_utc DESC, id DESC`` (newest first). ``stale_only``
    is applied in Python, after the fetch, via :func:`is_stale` -- staleness
    depends on the release registry, which the database does not know about,
    so it cannot be expressed in SQL.
    """
    import nems_lbhb.db as db
    import pandas as pd
    from sqlalchemy import text

    _validate_identifier(table)
    where_parts = []
    params = {}
    for col, val in filters.items():
        if val is not None:
            where_parts.append('%s = :%s' % (col, col))
            params[col] = val
    where_sql = (' WHERE ' + ' AND '.join(where_parts)) if where_parts else ''
    stmt = text('SELECT %s FROM %s%s ORDER BY target_mtime_utc DESC, id DESC'
                % (', '.join(COLUMNS), table, where_sql))
    engine = db.Engine()
    with engine.begin() as conn:
        df = pd.read_sql_query(stmt, conn, params=params)

    if stale_only and len(df):
        keep = df.apply(lambda row: is_stale(row), axis=1)
        df = df[keep].reset_index(drop=True)
    return df


# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 1 -- the
# `alias=` filter/kwarg renamed `routing_alias=` throughout, matching the gDLCFit column
# rename | date: 2026-08-12]
def fits_for(rawid=None, siteid=None, animal=None, routing_alias=None, stale_only=False,
             table=TABLE):
    """Rows from ``table`` matching whichever of ``rawid``/``siteid``/``animal``/
    ``routing_alias`` are not ``None`` (AND'd together; a filter left ``None`` is not
    applied), newest first (``target_mtime_utc DESC, id DESC``).

    ``stale_only=True`` filters the result in Python via :func:`is_stale`
    rather than in SQL: staleness is relative to the release registry, which
    ``gDLCFit`` itself has no notion of, so it cannot be pushed into the
    ``WHERE`` clause.
    """
    filters = {'rawid': rawid, 'siteid': siteid, 'animal': animal,
               'routing_alias': routing_alias}
    return _query_fits(filters, stale_only=stale_only, table=table)
# [AGENT EDIT END]


def current_fit(alias_path, table=TABLE):
    """:func:`fits_for` scoped to one ``alias_path``, newest row only.

    Returns a DataFrame of 0 or 1 rows -- 0 if ``alias_path`` has never been
    recorded in ``table``.
    """
    df = _query_fits({'alias_path': alias_path}, table=table)
    return df.head(1).reset_index(drop=True)


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: allow database-only views to skip unused filesystem pointer checks | date: 2026-09-09]
def fit_status(include_current=True, **filters):
# [AGENT EDIT END]
    """Read-time QC view over :func:`fits_for`'s columns, for the Streamlit app.

    ``filters`` are forwarded to :func:`fits_for` verbatim (so
    ``rawid``/``siteid``/``animal``/``routing_alias``/``stale_only``/``table`` are all
    valid keywords here). The result is deduplicated to the newest row per
    ``(video_stem, stream)`` for rows that HAVE a ``video_stem`` (rows arrive
    newest-first from ``fits_for``, so ``keep='first'`` is exactly the newest);
    a row with a ``NULL`` ``video_stem`` -- ``DEFAULT NULL`` in the DDL, and
    exactly what an ``attribution='unknown'`` row plausibly has -- is never
    collapsed against another such row, because pandas' ``drop_duplicates``
    treats ``NaN``/``None`` as equal to each other, which would silently hide
    rows from the operator rather than merely show harmless duplicates. Then
    extended with:

    * the ``qc`` JSON document's scalars lifted into flat columns --
      ``ratio_worst``, ``ratio_worst_part``, ``track_worst``,
      ``track_worst_part``, ``rms_spread_mean``, ``p_median``, ``p_cut``;
    * ``stale``, from :func:`is_stale`;
    * ``flag_track`` / ``flag_ratio``, computed from the STORED
      ``track_worst``/``ratio_worst`` numbers by calling
      ``dlc_headfix_qc.flag_for_fraction`` / ``dlc_headfix_qc.flag_for_ratio``
      at read time -- flags are never stored, so retuning a threshold
      reclassifies all of history the next time this function runs.

    ``dlc_headfix_qc`` is imported lazily (it pulls in numpy and pandas), so
    importing this module never pulls them in for a caller that only ever
    calls the pure functions above.
    """
    import json

    import pandas as pd

    df = fits_for(**filters)
    status_cols = list(COLUMNS) + list(_STATUS_EXTRA_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=status_cols)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: fix review Finding 2 -- drop_duplicates(subset=['video_stem','stream']) treats NULL video_stem as equal to NULL video_stem, silently collapsing distinct rows (plausible for attribution='unknown' rows) into one; dedup only the rows that HAVE a stem, and let every NULL-stem row survive on its own | date: 2026-08-12]
    has_stem = df['video_stem'].notna()
    with_stem = df[has_stem].drop_duplicates(subset=['video_stem', 'stream'], keep='first')
    without_stem = df[~has_stem]
    df = pd.concat([with_stem, without_stem], ignore_index=True)
    df = df.sort_values(['target_mtime_utc', 'id'], ascending=[False, False]).reset_index(drop=True)
    # [AGENT EDIT END]

    def _parse_qc(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, (bytes, bytearray)):
            v = v.decode('utf-8')
        if not v:
            return {}
        try:
            return json.loads(v)
        except (TypeError, ValueError):
            return {}

    parsed = df['qc'].apply(_parse_qc)
    for col in ('ratio_worst', 'ratio_worst_part', 'track_worst', 'track_worst_part',
                'rms_spread_mean', 'p_median', 'p_cut'):
        df[col] = parsed.apply(lambda d: d.get(col))

    df['stale'] = df.apply(lambda row: is_stale(row), axis=1)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: gDLCFit now holds one row
    # per ANALYSIS (the release-stamped alias), so "which analysis is currently served?" is no
    # longer answerable from alias_path alone | date: 2026-08-12]
    # Derived at read time, never stored, for the same reason `stale` and the QC flags are:
    # a stored is_current flag would have to be UPDATEd off on an older row every time a
    # session was re-analysed, and this table is append-only evidence (INSERT IGNORE on
    # row_key) precisely so that nothing rewrites history. Which release is in force is a
    # filesystem fact -- whatever `<stem>.dlc.h5` points at right now -- so it is read from
    # there, and a rollback is reflected immediately with no write at all.
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: preserve the default API while making lightweight status views possible | date: 2026-09-09]
    df['is_current'] = _is_current_series(df) if include_current else None
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    from nems_lbhb.motor import dlc_headfix_qc
    df['flag_track'] = df['track_worst'].apply(dlc_headfix_qc.flag_for_fraction)
    df['flag_ratio'] = df['ratio_worst'].apply(dlc_headfix_qc.flag_for_ratio)

    return df[status_cols]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: fix review Finding 3 -- stale_fits had no table kwarg, so it could only hit the real gDLCFit (which deliberately does not exist yet) and was untestable by construction; add table=TABLE to match every sibling function | date: 2026-08-12]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 1 -- alias=
# kwarg renamed routing_alias= | date: 2026-08-12]
def stale_fits(routing_alias=None, table=TABLE):
    """:func:`fit_status` filtered to ``stale`` rows, read from ``table``.

    ``.attrs['counts']`` on the returned DataFrame is a ``{(routing_alias, release_id):
    count}`` dict (from ``groupby(['routing_alias', 'release_id']).size()``) -- a
    quick tally of how much stale work sits behind each release, without a
    caller having to group it again itself.  ``.attrs['n_sites']`` is the number
    of distinct non-NULL ``siteid``s in the same rows, so the sentence this
    function exists to produce -- "promoting release X invalidates N sessions
    across M sites" -- is one call rather than two.
    """
    df = fit_status(routing_alias=routing_alias, table=table)
    # [AGENT EDIT END]
    stale_df = df[df['stale']].reset_index(drop=True) if len(df) else df
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review I1 -- groupby defaults to dropna=True, so every row with a NULL alias or NULL release_id was silently missing from the tally. That is not an edge case here: 'ambiguous' and 'unknown' rows have release_id=NULL BY CONSTRUCTION and are stale BY DEFINITION (is_stale returns True for both), which on the real corpus is 28 of the 92 stale rows -- a counts dict summing to 64 next to a len() of 92 is worse than no tally at all. dropna=False keeps them as their own (alias, NULL) group. | date: 2026-08-12]
    counts = (stale_df.groupby(['routing_alias', 'release_id'], dropna=False).size().to_dict()
              if len(stale_df) else {})
    stale_df.attrs['counts'] = counts
    stale_df.attrs['n_sites'] = int(stale_df['siteid'].nunique()) if len(stale_df) else 0
    # [AGENT EDIT END]
    return stale_df
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: nothing wired recording into the
# fit job, so gDLCFit was a periodic snapshot of the filesystem rather than a log of analyses:
# a fit produced no row at all until someone remembered to run the sweeper | date: 2026-08-12]
def record_job_fit(alias_path, target_path, release_id, routing_alias=None,
                   source_project=None, scorer=None, snapshot=None, queueid=None,
                   requested_by=None, code_git_hash=None, qc=None,
                   rawid=None, siteid=None, table=TABLE):
    """Record ONE gDLCFit row for an analysis this job just produced. Returns the record.

    Called by ``dlc_fit_script`` after the aliases are written, so the row describes an
    output that already exists under a stable name.

    ``alias_path`` must be the RELEASE-STAMPED alias, not the plain current pointer: the
    stamped name is the durable one (never repointed), and recording both would store one
    analysis twice. Which release is currently *served* is derived from the pointer at read
    time by :func:`is_current`, so nothing is lost by not recording it.

    ``attribution`` is ``'stamped'`` and ``reproducible`` is ``True`` unconditionally here,
    and both are simply true by construction: the job ran a verified, frozen, single-snapshot
    release and wrote a filename carrying that release's id. This is the one code path that
    does not have to infer either.

    ``rawid``/``siteid`` are passed IN rather than looked up here: this module is the only
    code that touches :data:`TABLE` and it touches nothing else -- a guard test asserts it
    never so much as names another table -- so the raw-file lookup stays with the caller
    (``dlc_fit_script`` does it, the sweeper does its own bulk join).

    Raises on a failure to build or insert. The caller is responsible for making sure that
    cannot take down the analysis -- see ``dlc_fit_script``, which catches and logs loudly.
    """
    from nems_lbhb.motor import dlc_alias

    alias_path = os.path.abspath(str(alias_path))
    target_path = os.path.abspath(str(target_path))
    p = alias_path.split(os.sep)
    # .../<animal>/<penname>/sorted/<alias>.dlc.h5
    animal = p[-4] if len(p) >= 4 else None
    penname = p[-3] if len(p) >= 3 else None

    video_stem = os.path.basename(alias_path)
    if video_stem.endswith('.dlc.h5'):
        video_stem = video_stem[: -len('.dlc.h5')]
    base_stem, stamped_release = dlc_alias.parse_stamped_alias(video_stem)
    if stamped_release is None:
        raise ValueError(
            'record_job_fit expects the release-stamped alias, got %r. The plain pointer '
            'is not recorded: it is repointed on every re-analysis, and is_current() '
            'derives what it says at read time.' % alias_path)

    stream = dlc_alias.stream_of(base_stem, target_path)

    record = build_record(
        alias_path=alias_path,
        target_path=target_path,
        target_mtime_utc=datetime.datetime.fromtimestamp(
            os.path.getmtime(target_path), tz=datetime.timezone.utc),
        is_link=1 if os.path.islink(alias_path) else 0,
        rawid=rawid, siteid=siteid, animal=animal, penname=penname,
        video_stem=video_stem, stream=stream,
        routing_alias=routing_alias, release_id=stamped_release,
        scorer=scorer, snapshot=snapshot, source_project=source_project,
        attribution='stamped',
        attribution_evidence='written by the fit job that ran release %s' % stamped_release,
        reproducible=True,
        source='job',
        queueid=queueid, requested_by=requested_by,
        analyzed_utc=datetime.datetime.now(datetime.timezone.utc),
        code_git_hash=code_git_hash,
        qc=qc,
    )
    insert_fit(record, table=table)
    return record
# [AGENT EDIT END]
