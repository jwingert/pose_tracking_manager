# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 1 -- gDLCFit's
# `alias` column renamed `routing_alias` everywhere it is used as a column/kwarg in this
# file (build_record kwarg, is_stale row field, fits_for/fit_status/stale_fits filter). The
# REGISTRY fixture's own 'aliases'/'alias_names' keys are dlc_registry's vocabulary, untouched
# per the brief. | date: 2026-08-12]
import datetime
import os
import re

import pytest

from nems_lbhb.motor import dlc_fit_record as fr

UTC = datetime.timezone.utc
MT = datetime.datetime(2024, 1, 11, 2, 47, 49, tzinfo=UTC)
# [AGENT EDIT END]


def _base(**over):
    d = dict(alias_path='/auto/data/daq/A/A1/sorted/A1a01_a_NTD.dlc.h5',
             target_path='/auto/data/daq/A/A1/sorted/A1a01_a_NTDDLC_x.h5',
             target_mtime_utc=MT, is_link=1, animal='A', penname='A1',
             video_stem='A1a01_a_NTD', stream='', routing_alias='two_chimney',
             release_id='two_chimney-r0002', scorer='DLC_x', snapshot=60000,
             source_project='/auto/data/dlc/p', attribution='inferred_mtime',
             attribution_evidence='because', reproducible=1, source='sweeper')
    d.update(over)
    return d


def test_row_key_is_stable_across_calls():
    a = fr.row_key('/x/a.dlc.h5', '/x/t.h5', MT)
    b = fr.row_key('/x/a.dlc.h5', '/x/t.h5', MT)
    assert a == b and len(a) == 64


def test_row_key_changes_with_mtime():
    """A re-analysis writes a new target mtime and must produce a new row."""
    later = MT + datetime.timedelta(seconds=1)
    assert fr.row_key('/x/a.dlc.h5', '/x/t.h5', MT) != fr.row_key('/x/a.dlc.h5', '/x/t.h5', later)


def test_row_key_ignores_timezone_representation():
    naive = MT.replace(tzinfo=None)
    assert fr.row_key('/x/a.dlc.h5', '/x/t.h5', naive) == fr.row_key('/x/a.dlc.h5', '/x/t.h5', MT)


def test_build_record_fills_row_key():
    r = fr.build_record(**_base())
    assert r['row_key'] == fr.row_key(r['alias_path'], r['target_path'], MT)


def test_build_record_rejects_an_unknown_attribution():
    with pytest.raises(ValueError) as e:
        fr.build_record(**_base(attribution='probably'))
    assert 'probably' in str(e.value)


def test_build_record_requires_alias_path():
    with pytest.raises(ValueError):
        fr.build_record(**_base(alias_path=''))


def test_build_record_allows_null_rawid_and_release():
    r = fr.build_record(**_base(release_id=None, attribution='unknown', reproducible=None))
    assert r['release_id'] is None and r['rawid'] is None


# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 3 -- release_path
# is DERIVED from release_id, computed unconditionally in build_record() so the two can never
# independently disagree. A test on one hand-picked value could pass even if the derivation
# were subtly wrong (e.g. only wired up for the common case); check the invariant across a
# whole SET of built records instead -- every shape release_id can actually take (None, a
# plain release, an archive_ release) in one pass, per the brief's own instruction that this
# is worth more than a single-value assertion. | date: 2026-08-12]
def test_release_path_always_agrees_with_release_id_across_a_record_set():
    from nems_lbhb.motor import dlc_attribution

    release_ids = [None, 'two_chimney-r0001', 'two_chimney-r0002',
                   'left_chimney-r0001', 'archive_two_chimney-r0001',
                   'left_chimney_headphones-r0003']
    records = [fr.build_record(**_base(
                   release_id=rid,
                   attribution='unknown' if rid is None else 'inferred_scorer',
                   reproducible=None if rid is None else 1))
               for rid in release_ids]

    assert len(records) == len(release_ids)
    for rec in records:
        if rec['release_id'] is None:
            assert rec['release_path'] is None
        else:
            assert rec['release_path'] == os.path.join(
                dlc_attribution.RELEASE_ROOT, rec['release_id'])


def test_release_path_ignores_a_caller_supplied_value_and_rederives_it():
    """release_path must not be independently settable -- a caller-supplied value that
    disagrees with release_id would silently drift; build_record() always overwrites it."""
    from nems_lbhb.motor import dlc_attribution

    r = fr.build_record(**_base(release_id='two_chimney-r0002',
                                 release_path='/somewhere/else/entirely'))
    assert r['release_path'] == os.path.join(dlc_attribution.RELEASE_ROOT, 'two_chimney-r0002')


def test_release_path_is_immediately_after_release_id_in_columns_and_ddl():
    """Placement contract from the brief: release_path sits right after release_id, both in
    COLUMNS and in CREATE_TABLE_SQL."""
    assert fr.COLUMNS.index('release_path') == fr.COLUMNS.index('release_id') + 1

    ddl_columns = [line.strip().split()[0] for line in fr.CREATE_TABLE_SQL.splitlines()
                   if line.strip() and not line.strip().startswith(('CREATE', 'PRIMARY', 'UNIQUE',
                                                                      'KEY', ')'))]
    assert ddl_columns.index('release_path') == ddl_columns.index('release_id') + 1
# [AGENT EDIT END]


def test_coerce_qc_drops_unknown_keys():
    got = fr.coerce_qc({'track_worst': 0.6, 'nonsense': 1})
    assert 'nonsense' not in got and got['track_worst'] == 0.6


def test_coerce_qc_forces_numeric_strings_to_floats():
    """JSON has no column type; '0.5' and 0.5 must not compare differently."""
    got = fr.coerce_qc({'track_worst': '0.5'})
    assert got['track_worst'] == 0.5
    assert isinstance(got['track_worst'], float)


def test_coerce_qc_passes_an_error_document_through():
    got = fr.coerce_qc({'error': 'could not read h5'})
    assert got['error'] == 'could not read h5'


def test_coerce_qc_of_none_is_none():
    assert fr.coerce_qc(None) is None


REGISTRY = {'aliases': {'two_chimney': {'current': 'two_chimney-r0002'}},
            'alias_names': ['two_chimney']}


@pytest.mark.parametrize('over,expected', [
    (dict(release_id='two_chimney-r0002', attribution='inferred_mtime', reproducible=1), False),
    (dict(release_id='two_chimney-r0001', attribution='inferred_mtime', reproducible=1), True),
    (dict(release_id='archive_two_chimney-r0001', attribution='inferred_scorer', reproducible=1), True),
    (dict(release_id=None, attribution='ambiguous', reproducible=None), True),
    (dict(release_id=None, attribution='unknown', reproducible=None), True),
    (dict(release_id='two_chimney-r0002', attribution='inferred_mtime', reproducible=0), True),
    (dict(release_id='two_chimney-r0002', attribution='stamped', reproducible=1), False),
])
def test_staleness_truth_table(over, expected):
    assert fr.is_stale(_base(**over), registry=REGISTRY) is expected


def test_is_stale_on_an_unknown_alias_is_stale_not_an_error():
    """A row naming an alias the registry dropped must sort with the work, not crash a report."""
    assert fr.is_stale(_base(routing_alias='gone_alias'), registry=REGISTRY) is True


# [AGENT EDIT START | agent: claude | user: wingertj | reason: fix review Finding 1 -- space-padded substring match (' ALTER ', ' DROP ') did not catch the quoted-literal shape real destructive SQL takes in this codebase (e.g. text('DROP TABLE %s' % table), sql = 'ALTER TABLE ...'), where the verb is preceded by a quote or a newline, not a space; replace with a word-boundary regex anchored on quote/newline/start-of-line so it fires on the actual risk shape without false-positiving on legitimate code like df.reset_index(drop=True) | date: 2026-08-12]
# Matches a forbidden SQL verb only where it could plausibly begin a SQL string
# literal or statement -- immediately after a quote character, a newline, or at
# the very start of the source (optionally followed by indentation), with a
# trailing word boundary so "DROPPED"/"dropdown" and pandas kwargs like
# "reset_index(drop=True)" (preceded by "(", not a quote or newline) don't
# false-positive.
_FORBIDDEN_SQL_RE = re.compile(
    r'''(?:^|['"\n])\s*(ALTER|DROP|TRUNCATE|REPLACE\s+INTO)\b''',
    re.IGNORECASE | re.MULTILINE)


def test_module_sql_touches_exactly_one_table():
    """Blast radius. This module must never write to an existing celldb table."""
    import inspect
    src = inspect.getsource(fr)
    m = _FORBIDDEN_SQL_RE.search(src)
    assert m is None, 'forbidden SQL verb %r in dlc_fit_record' % (m.group(1) if m else None)
    assert 'GDATARAW' not in src.upper()


def test_forbidden_sql_detector_actually_fires():
    """Self-check on the blast-radius guard itself: it must catch the exact
    quoted-literal shapes real destructive SQL would take here, or it has
    silently rotted back into the space-padded substring match this fixes.
    """
    assert _FORBIDDEN_SQL_RE.search("conn.execute(text('DROP TABLE %s' % table))")
    assert _FORBIDDEN_SQL_RE.search("sql = 'ALTER TABLE gDLCFit ADD COLUMN y INT'")
    assert _FORBIDDEN_SQL_RE.search("REPLACE INTO gDLCFit (a) VALUES (1)")
    assert _FORBIDDEN_SQL_RE.search("TRUNCATE gDLCFit")
    # and it must NOT false-positive on legitimate, already-in-module pandas code
    assert not _FORBIDDEN_SQL_RE.search("df.reset_index(drop=True)")
    assert not _FORBIDDEN_SQL_RE.search("df.drop_duplicates(subset=['video_stem'])")
# [AGENT EDIT END]


def test_create_table_is_never_called_implicitly():
    """DDL must run only from a deliberate invocation, never on import or first insert.

    Checked by parsing the module: a monkeypatch-and-reload test cannot detect this,
    because reload rebinds the patched attribute before any module-level code runs.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fr))

    def calls_create_table(node):
        return any(isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'create_table'
                   for n in ast.walk(node))

    module_level = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert not any(calls_create_table(n) for n in module_level), \
        'create_table() is called at module level'

    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if fn.name in ('insert_fit', 'insert_many', 'fits_for', 'current_fit',
                       'fit_status', 'stale_fits', 'build_record'):
            assert not calls_create_table(fn), '%s() calls create_table()' % fn.name


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 3 -- SQL half: create/insert/read round-trip tests (gated behind NEMS_DLC_DBTEST) plus a pure unit test pinning the NaN/Inf-safe qc JSON serialization | date: 2026-08-12]
def test_qc_json_serialization_is_null_safe_for_nan_and_inf():
    """dlc_headfix_qc.worst_rigid()/worst_rigid_tracking() legitimately return
    float('nan') when no rigid marker has a finite value, so a real qc document
    will eventually carry a non-finite float. json.dumps would otherwise emit the
    bare tokens NaN/Infinity, which are not valid JSON and which MySQL's JSON
    column rejects -- _qc_to_json must remap them to JSON null instead, and the
    result must actually be parseable JSON (not merely "did not raise").
    """
    import json
    import math

    got = fr._qc_to_json({'track_worst': float('nan'), 'ratio_worst': float('inf'),
                           'ratio_worst_alt': float('-inf'), 'p_cut': 0.9})
    parsed = json.loads(got)  # raises if _qc_to_json emitted bare NaN/Infinity
    assert parsed['track_worst'] is None
    assert parsed['ratio_worst'] is None
    assert parsed['ratio_worst_alt'] is None
    assert parsed['p_cut'] == 0.9


def test_qc_json_serialization_of_none_is_none():
    assert fr._qc_to_json(None) is None


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review C1/C2 -- pin the
# two binding guarantees _row_params now makes. Both defects were pandas-typed scalars
# reaching the driver from build_records (the only path the 551-row backfill takes), and
# both were invisible to every existing test because every existing test hand-builds pure
# Python dicts. These are the pure-Python halves; the seam is covered end-to-end in
# tests/test_dlc_sync_fits.py's gated round-trip. | date: 2026-08-12]
def test_row_params_binds_a_nan_int_column_as_sql_null():
    """C1: pd.DataFrame types a column holding None among ints as float64, so an absent
    `reproducible` arrives as float('nan'); pymysql raises ProgrammingError('nan can not
    be used with MySQL') on it, and because the backfill is ONE executemany that aborts
    all 551 rows.
    """
    params = fr._row_params(_base(reproducible=float('nan'), attribution='ambiguous'))
    assert params['reproducible'] is None


def test_row_params_binds_a_finite_float_to_an_int_column_as_int():
    """Same float64 column, present value: 1.0/64.0 must land in TINYINT/INT as an int."""
    params = fr._row_params(_base(reproducible=1.0, snapshot=60000.0, rawid=123.0, is_link=1.0))
    for col in ('reproducible', 'snapshot', 'rawid', 'is_link'):
        assert isinstance(params[col], int) and not isinstance(params[col], bool), col
    assert params['snapshot'] == 60000 and params['rawid'] == 123


def test_row_params_binds_a_non_finite_float_column_as_null_too():
    """A float column (not just the int ones) must not hand the driver inf/nan either."""
    assert fr._row_params(_base(snapshot=float('inf')))['snapshot'] is None


def test_row_params_binds_an_aware_datetime_as_naive_utc():
    """C2: MySQL DATETIME has no timezone, and with session time_zone=SYSTEM an aware
    value that reaches the driver as a string ('...+00:00') is CONVERTED to session-local
    time -- silently, and by a DST-dependent amount -- in a column whose contract is UTC.
    """
    got = fr._row_params(_base())['target_mtime_utc']
    assert got.tzinfo is None
    assert got == MT.replace(tzinfo=None)      # same wall clock, in UTC

    eastern = datetime.timezone(datetime.timedelta(hours=-5))
    shifted = MT.astimezone(eastern)           # same instant, different representation
    got2 = fr._row_params(_base(target_mtime_utc=shifted))['target_mtime_utc']
    assert got2.tzinfo is None and got2 == MT.replace(tzinfo=None)


def test_row_params_keeps_microseconds():
    """The columns are DATETIME(6); the binder must not round or truncate on the way in."""
    precise = MT.replace(microsecond=398314)
    assert fr._row_params(_base(target_mtime_utc=precise))['target_mtime_utc'].microsecond \
        == 398314


def test_row_params_binds_a_naive_datetime_unchanged():
    """A naive value is assumed to already be UTC, matching row_key()/_to_utc()."""
    naive = MT.replace(tzinfo=None)
    assert fr._row_params(_base(target_mtime_utc=naive))['target_mtime_utc'] == naive


def test_row_params_binds_a_pandas_timestamp_as_naive_utc():
    """The exact type the sweeper's datetime64[ns, UTC] column yields per row."""
    pd = pytest.importorskip('pandas')
    ts = pd.Timestamp('2024-01-11 02:47:49', tz='UTC')
    got = fr._row_params(_base(target_mtime_utc=ts))['target_mtime_utc']
    assert type(got) is datetime.datetime
    assert got.tzinfo is None
    assert got == MT.replace(tzinfo=None)


def test_row_params_binds_pandas_missing_values_as_null():
    """NaT is what a missing datetime looks like coming out of a DataFrame; it is an
    instance of datetime.datetime, so a type check alone would bind it as the string
    'NaT'."""
    pd = pytest.importorskip('pandas')
    params = fr._row_params(_base(target_mtime_utc=pd.NaT, snapshot=pd.NA))
    assert params['target_mtime_utc'] is None
    assert params['snapshot'] is None


def test_row_params_leaves_strings_alone():
    params = fr._row_params(_base())
    assert params['routing_alias'] == 'two_chimney'
    assert params['attribution'] == 'inferred_mtime'
    assert params['source_project'] == '/auto/data/dlc/p'
# [AGENT EDIT END]


DBTEST = pytest.mark.skipif(
    os.environ.get('NEMS_DLC_DBTEST') != '1',
    reason='set NEMS_DLC_DBTEST=1 to run round-trip tests against gDLCFit_test')

TEST_TABLE = 'gDLCFit_test'


@pytest.fixture
def clean_table():
    """Create and drop a SEPARATE table. Never touches gDLCFit itself."""
    import nems_lbhb.db as db
    assert TEST_TABLE != fr.TABLE
    fr.create_table(table=TEST_TABLE)
    db.sql_command('DELETE FROM %s' % TEST_TABLE)
    yield TEST_TABLE
    db.sql_command('DROP TABLE %s' % TEST_TABLE)


@DBTEST
def test_insert_then_read_round_trips(clean_table):
    rec = fr.build_record(**_base())
    assert fr.insert_fit(rec, table=clean_table) == 1
    got = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    assert len(got) == 1
    assert got.iloc[0]['release_id'] == 'two_chimney-r0002'


@DBTEST
def test_second_insert_of_the_same_row_key_is_ignored(clean_table):
    rec = fr.build_record(**_base())
    assert fr.insert_fit(rec, table=clean_table) == 1
    assert fr.insert_fit(rec, table=clean_table) == 0
    assert len(fr.fits_for(routing_alias='two_chimney', table=clean_table)) == 1


@DBTEST
def test_reanalysis_inserts_a_second_row_and_keeps_the_first(clean_table):
    fr.insert_fit(fr.build_record(**_base()), table=clean_table)
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=later,
                                          release_id='two_chimney-r0001')), table=clean_table)
    rows = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    assert len(rows) == 2
    assert list(rows['release_id']) == ['two_chimney-r0001', 'two_chimney-r0002']   # newest first


@DBTEST
def test_current_fit_returns_only_the_newest(clean_table):
    fr.insert_fit(fr.build_record(**_base()), table=clean_table)
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=later,
                                          release_id='two_chimney-r0001')), table=clean_table)
    got = fr.current_fit(_base()['alias_path'], table=clean_table)
    assert len(got) == 1 and got.iloc[0]['release_id'] == 'two_chimney-r0001'


@DBTEST
def test_qc_json_survives_the_round_trip(clean_table):
    qc = {'track_worst': 0.61, 'track_worst_part': 'rear', 'p_cut': 0.9}
    fr.insert_fit(fr.build_record(**_base(qc=qc)), table=clean_table)
    got = fr.fit_status(routing_alias='two_chimney', table=clean_table)
    assert abs(float(got.iloc[0]['track_worst']) - 0.61) < 1e-9
    assert got.iloc[0]['track_worst_part'] == 'rear'


@DBTEST
def test_fit_status_computes_flags_from_stored_numbers(clean_table):
    """Thresholds are applied at read time, so retuning them reclassifies history."""
    fr.insert_fit(fr.build_record(**_base(qc={'track_worst': 0.20})), table=clean_table)
    got = fr.fit_status(routing_alias='two_chimney', table=clean_table)
    assert got.iloc[0]['flag_track'] == 'fail'


@DBTEST
def test_insert_many_reports_inserted_and_ignored(clean_table):
    recs = [fr.build_record(**_base()), fr.build_record(**_base())]
    got = fr.insert_many(recs, table=clean_table)
    assert got == {'inserted': 1, 'ignored': 1}


@DBTEST
def test_nan_qc_value_round_trips_as_sql_null_not_a_write_failure(clean_table):
    """Pins the Task-2-carried-forward defect fix against the real JSON column:
    a non-finite qc value must not raise on insert, and must come back as a
    missing/None value rather than corrupting the row.
    """
    import math

    rec = fr.build_record(**_base(qc={'track_worst': float('nan'), 'p_cut': 0.9}))
    assert fr.insert_fit(rec, table=clean_table) == 1
    got = fr.fit_status(routing_alias='two_chimney', table=clean_table)
    track_worst = got.iloc[0]['track_worst']
    assert track_worst is None or (isinstance(track_worst, float) and math.isnan(track_worst))
    assert abs(float(got.iloc[0]['p_cut']) - 0.9) < 1e-9
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: fix review Findings 2 and 3, and Minor (b) -- NULL video_stem rows must each survive fit_status rather than collapse together; stale_fits/fits_for(stale_only=True) are registry-dependent behaviours the brief singles out and must be covered by a gated DB test now that stale_fits takes a table kwarg; pin the fits_for/current_fit column-order contract | date: 2026-08-12]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review follow-up -- the
# datetime columns are DATETIME(6) precisely so a sub-second mtime is stored as it is rather
# than ROUNDED to the nearest second (MySQL rounds, it does not truncate). This is not
# cosmetic: two_chimney's attribution splits 42/13 on `mtime < 2024-01-12 12:08:00`, so a
# file at 12:07:59.999 stored as 12:08:00 would sit on the wrong side of its own cutoff and
# anyone re-deriving that split with a SQL WHERE would get a different answer than the
# sweeper did. | date: 2026-08-12]
@DBTEST
def test_sub_second_mtime_round_trips_exactly(clean_table):
    precise = MT.replace(microsecond=398314)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=precise)), table=clean_table)
    got = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    stored = got.iloc[0]['target_mtime_utc']
    assert stored.microsecond == 398314
    assert stored.to_pydatetime() == precise.replace(tzinfo=None)


@DBTEST
def test_a_sub_second_mtime_is_not_rounded_across_an_attribution_cutoff(clean_table):
    """The concrete defect DATETIME(6) prevents: two_chimney's cutoff is
    2024-01-12 12:08:00 UTC, and this file is 1 ms BEFORE it -- iteration-0, r0001.
    Stored in a DATETIME(0) it would round UP to 12:08:00 and re-derive as r0002.
    """
    cutoff = datetime.datetime(2024, 1, 12, 12, 8, 0, tzinfo=UTC)
    just_before = cutoff - datetime.timedelta(milliseconds=1)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=just_before,
                                          release_id='two_chimney-r0001')), table=clean_table)
    got = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    stored = got.iloc[0]['target_mtime_utc'].to_pydatetime()
    assert stored == just_before.replace(tzinfo=None)
    assert stored < cutoff.replace(tzinfo=None), (
        'stored mtime %s crossed the attribution cutoff it was 1 ms below' % stored)


@DBTEST
def test_fits_for_and_current_fit_columns_match_the_documented_contract(clean_table):
    """Minor (b): the module docstring calls COLUMNS the contract the Streamlit
    app reads by position; pin that fits_for/current_fit actually return it.
    """
    fr.insert_fit(fr.build_record(**_base()), table=clean_table)
    got = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    assert list(got.columns) == list(fr.COLUMNS)
    cur = fr.current_fit(_base()['alias_path'], table=clean_table)
    assert list(cur.columns) == list(fr.COLUMNS)


@DBTEST
def test_fit_status_keeps_each_null_video_stem_row_separate(clean_table):
    """A row with video_stem=None (e.g. attribution='unknown', where the stem was
    never resolved) must not collapse against another NULL-stem row: pandas'
    drop_duplicates treats None/NaN as equal, so naively deduping on
    (video_stem, stream) would silently hide one of two genuinely distinct rows
    from the operator -- worse than showing a harmless duplicate.
    """
    fr.insert_fit(fr.build_record(**_base(video_stem=None, attribution='unknown',
                                          reproducible=None)), table=clean_table)
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(video_stem=None, attribution='unknown',
                                          reproducible=None, target_mtime_utc=later,
                                          release_id='two_chimney-r0001')), table=clean_table)
    got = fr.fit_status(routing_alias='two_chimney', table=clean_table)
    assert len(got) == 2
    assert got['video_stem'].isna().all()
    assert set(got['release_id']) == {'two_chimney-r0002', 'two_chimney-r0001'}


@DBTEST
def test_fits_for_stale_only_filters_via_registry(clean_table, monkeypatch):
    """The other registry-dependent behaviour the brief singles out: stale_only
    filters in Python via is_stale, which needs the release registry.
    """
    from nems_lbhb.motor import dlc_registry
    monkeypatch.setattr(dlc_registry, 'load_registry', lambda: REGISTRY)

    fr.insert_fit(fr.build_record(**_base()), table=clean_table)  # current release -> not stale
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=later,
                                          release_id='two_chimney-r0001')), table=clean_table)  # stale

    all_rows = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    assert len(all_rows) == 2

    stale_rows = fr.fits_for(routing_alias='two_chimney', stale_only=True, table=clean_table)
    assert len(stale_rows) == 1
    assert stale_rows.iloc[0]['release_id'] == 'two_chimney-r0001'


@DBTEST
def test_stale_fits_returns_stale_rows_with_counts(clean_table, monkeypatch):
    from nems_lbhb.motor import dlc_registry
    monkeypatch.setattr(dlc_registry, 'load_registry', lambda: REGISTRY)

    fr.insert_fit(fr.build_record(**_base()), table=clean_table)  # current release -> not stale
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(target_mtime_utc=later,
                                          release_id='two_chimney-r0001')), table=clean_table)  # stale

    got = fr.stale_fits(routing_alias='two_chimney', table=clean_table)
    assert len(got) == 1
    assert got.iloc[0]['release_id'] == 'two_chimney-r0001'
    assert got.attrs['counts'] == {('two_chimney', 'two_chimney-r0001'): 1}


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review I1 -- the counts
# tally silently omitted every NULL-release_id row (groupby's dropna=True default), which is
# exactly the ambiguous/unknown set: NULL release_id by construction, stale by definition,
# 28 of the 92 stale rows on the real corpus. Also pins the site count the spec's "N sessions
# across M sites" sentence needs. | date: 2026-08-12]
@DBTEST
def test_stale_fits_counts_include_null_release_id_rows(clean_table, monkeypatch):
    from nems_lbhb.motor import dlc_registry
    monkeypatch.setattr(dlc_registry, 'load_registry', lambda: REGISTRY)

    # one ordinary stale row (superseded release), plus two ambiguous rows whose
    # release_id is NULL -- all three are stale, so all three must be tallied.
    later = MT + datetime.timedelta(days=30)
    fr.insert_fit(fr.build_record(**_base(release_id='two_chimney-r0001', siteid='TWO001a')),
                  table=clean_table)
    fr.insert_fit(fr.build_record(**_base(release_id=None, attribution='ambiguous',
                                          reproducible=None, target_mtime_utc=later,
                                          video_stem='A1a02_a_NTD', siteid='TWO001a')),
                  table=clean_table)
    fr.insert_fit(fr.build_record(**_base(release_id=None, attribution='ambiguous',
                                          reproducible=None,
                                          target_mtime_utc=later + datetime.timedelta(days=1),
                                          video_stem='A1a03_a_NTD', siteid='TWO002a')),
                  table=clean_table)

    got = fr.stale_fits(routing_alias='two_chimney', table=clean_table)
    assert len(got) == 3
    counts = got.attrs['counts']
    assert sum(counts.values()) == len(got), (
        'counts must tally every stale row, not just the ones with a release_id: %r' % (counts,))
    assert counts[('two_chimney', 'two_chimney-r0001')] == 1
    null_keys = [k for k in counts if k[1] is None or k[1] != k[1]]
    assert len(null_keys) == 1 and counts[null_keys[0]] == 2
    assert got.attrs['n_sites'] == 2
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: gDLCFit now holds one row per
# ANALYSIS (the release-stamped alias), so "which analysis is currently served?" is derived
# from the pointer at read time rather than stored as a flag -- a stored flag would need an
# UPDATE on an append-only row every time a session was re-analysed | date: 2026-08-12]
def _analysis(tmp_path, stem, release, scorer='DLC_resnet50_two_chimneyJan10shuffle1_60000'):
    """One analysis on disk: the output, its stamped alias, and no pointer yet."""
    out = tmp_path / ('%s%s.h5' % (stem, scorer))
    out.write_bytes(b'stand-in')
    stamped = tmp_path / ('%s.%s.dlc.h5' % (stem, release))
    os.symlink(str(out), str(stamped))
    return str(out), str(stamped)


def test_pointer_path_strips_the_stamp_but_keeps_the_stream(tmp_path):
    assert fr.pointer_path('/d/SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5') == \
        '/d/SLJ017a07_a_NTD.dlc.h5'
    assert fr.pointer_path('/d/SLJ017a07_a_NTD.dlc.h5') == '/d/SLJ017a07_a_NTD.dlc.h5'
    # a cam-tagged stream is served independently of the base camera
    assert fr.pointer_path('/d/SLJ017a07_a_NTD_camB.two_chimney-r0001.dlc.h5') == \
        '/d/SLJ017a07_a_NTD_camB.dlc.h5'


def test_is_current_true_when_the_pointer_resolves_to_this_output(tmp_path):
    out, stamped = _analysis(tmp_path, 'SLJ017a07_a_NTD', 'two_chimney-r0001')
    os.symlink(out, str(tmp_path / 'SLJ017a07_a_NTD.dlc.h5'))
    assert fr.is_current({'alias_path': stamped, 'target_path': out}) is True


def test_is_current_false_after_the_pointer_moves_to_a_newer_release(tmp_path):
    """The older analysis stays in the table as evidence -- nothing is rewritten -- and
    simply stops reading as current."""
    old_out, old_stamped = _analysis(tmp_path, 'SLJ017a07_a_NTD', 'two_chimney-r0001')
    new_out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_90000.h5'
    new_out.write_bytes(b'newer')
    new_stamped = tmp_path / 'SLJ017a07_a_NTD.two_chimney-r0002.dlc.h5'
    os.symlink(str(new_out), str(new_stamped))
    os.symlink(str(new_out), str(tmp_path / 'SLJ017a07_a_NTD.dlc.h5'))

    assert fr.is_current({'alias_path': old_stamped, 'target_path': old_out}) is False
    assert fr.is_current({'alias_path': str(new_stamped),
                          'target_path': str(new_out)}) is True


def test_is_current_follows_a_rollback_with_no_write(tmp_path):
    """Rolling the pointer back must flip the answer immediately: that is the whole reason
    this is derived rather than stored."""
    old_out, old_stamped = _analysis(tmp_path, 'SLJ017a07_a_NTD', 'two_chimney-r0001')
    new_out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_90000.h5'
    new_out.write_bytes(b'newer')
    pointer = str(tmp_path / 'SLJ017a07_a_NTD.dlc.h5')
    os.symlink(str(new_out), pointer)
    assert fr.is_current({'alias_path': old_stamped, 'target_path': old_out}) is False
    os.remove(pointer)
    os.symlink(old_out, pointer)                       # rollback
    assert fr.is_current({'alias_path': old_stamped, 'target_path': old_out}) is True


def test_is_current_is_none_when_undecidable_not_false(tmp_path):
    """"not being served" and "cannot tell" are different answers; a temp analysis that was
    never given a pointer must not masquerade as superseded."""
    out, stamped = _analysis(tmp_path, 'SLJ017a07_a_NTD', 'two_chimney-r0001')
    assert fr.is_current({'alias_path': stamped, 'target_path': out}) is None   # no pointer
    assert fr.is_current({'alias_path': stamped, 'target_path': None}) is None
    assert fr.is_current({'alias_path': None, 'target_path': out}) is None


def test_legacy_plain_alias_reads_as_current(tmp_path):
    """A legacy row's alias IS the pointer, so it must come out current rather than
    undecidable -- otherwise the whole existing corpus would read as not-served."""
    out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_60000.h5'
    out.write_bytes(b'legacy')
    plain = str(tmp_path / 'SLJ017a07_a_NTD.dlc.h5')
    os.symlink(str(out), plain)
    assert fr.is_current({'alias_path': plain, 'target_path': str(out)}) is True
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the fit job now records its own
# row, so gDLCFit is a log of analyses rather than a periodic snapshot of the filesystem
# | date: 2026-08-12]
def test_record_job_fit_refuses_the_plain_pointer(tmp_path):
    """Recording the pointer would store the analysis twice and key the row on a name that
    gets repointed. The stamped alias is the durable one."""
    out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_60000.h5'
    out.write_bytes(b'x')
    plain = tmp_path / 'SLJ017a07_a_NTD.dlc.h5'
    os.symlink(str(out), str(plain))
    with pytest.raises(ValueError, match='release-stamped alias'):
        fr.record_job_fit(alias_path=str(plain), target_path=str(out),
                          release_id='two_chimney-r0001')


@DBTEST
def test_record_job_fit_writes_one_stamped_row(tmp_path, clean_table):
    out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_60000.h5'
    out.write_bytes(b'x')
    stamped = tmp_path / 'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'
    os.symlink(str(out), str(stamped))

    rec = fr.record_job_fit(alias_path=str(stamped), target_path=str(out),
                            release_id='two_chimney-r0001',
                            routing_alias='two_chimney', queueid=4242,
                            requested_by='wingertj', table=clean_table)

    assert rec['attribution'] == 'stamped'      # true by construction on this path
    assert rec['reproducible'] is True          # a frozen, single-snapshot release ran
    assert rec['source'] == 'job'
    assert rec['release_id'] == 'two_chimney-r0001'
    got = fr.fits_for(routing_alias='two_chimney', table=clean_table)
    assert len(got) == 1
    assert got.iloc[0]['queueid'] == 4242
    assert got.iloc[0]['requested_by'] == 'wingertj'


@DBTEST
def test_record_job_fit_is_idempotent_for_one_analysis(tmp_path, clean_table):
    """A job re-run that produces the same output must not add a second row; only a genuine
    re-analysis (which rewrites the target, bumping its mtime) mints one."""
    out = tmp_path / 'SLJ017a07_a_NTDDLC_resnet50_two_chimneyJan10shuffle1_60000.h5'
    out.write_bytes(b'x')
    stamped = tmp_path / 'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'
    os.symlink(str(out), str(stamped))
    kw = dict(alias_path=str(stamped), target_path=str(out),
              release_id='two_chimney-r0001', table=clean_table)
    fr.record_job_fit(**kw)
    fr.record_job_fit(**kw)
    assert len(fr.fits_for(release_id='two_chimney-r0001', table=clean_table)
               if False else fr.fits_for(table=clean_table)) == 1


def test_record_job_fit_reads_the_stream_from_the_target(tmp_path, monkeypatch):
    """A camera override points a SESSION-named stamped alias at a camC target; the row must
    say camC, and must still join gDataRaw on the session stem."""
    monkeypatch.setattr(fr, 'insert_fit', lambda rec, table=None: 1)
    out = tmp_path / 'SLJ017a07_a_NTD_camCDLC_resnet50_two_chimneyJan10shuffle1_60000.h5'
    out.write_bytes(b'x')
    stamped = tmp_path / 'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'
    os.symlink(str(out), str(stamped))

    rec = fr.record_job_fit(alias_path=str(stamped), target_path=str(out),
                            release_id='two_chimney-r0001', rawid=999, siteid='SLJ017a')

    assert rec['stream'] == 'camC'
    assert rec['video_stem'] == 'SLJ017a07_a_NTD.two_chimney-r0001'
    assert rec['rawid'] == 999
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: a tuple reached the driver and
# MySQL answered "Operand should contain 1 column(s)", naming neither column nor value; the
# real cause was dlc_release._git_hash returning (hash, warning) | date: 2026-08-12]
def test_a_sequence_bound_to_a_scalar_column_is_refused_by_name():
    with pytest.raises(TypeError, match='code_git_hash'):
        fr._bind_value('code_git_hash', ('abc123', ''))
    with pytest.raises(TypeError, match='scalar'):
        fr._bind_value('siteid', ['SLJ017a'])


def test_scalars_still_pass_through_untouched():
    assert fr._bind_value('code_git_hash', 'abc123') == 'abc123'
    assert fr._bind_value('rawid', 154176.0) == 154176      # pandas float64 for an int column
    assert fr._bind_value('siteid', None) is None
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: qc v2 -- the per-marker
# structures must survive coerce_qc. They nearly did not: coerce_qc float()s anything not
# explicitly typed, and a failed coercion DROPS the key while only noting it in 'error', so
# a schema addition that forgot _QC_STRUCT_KEYS would have silently discarded exactly the
# measurements the change existed to record. | date: 2026-08-17]
def test_nested_qc_structures_survive_coercion():
    doc = {
        'parts': {'leftear': {'ratio_spread': 0.7, 'role': 'rigid'}},
        'mean_shape': [[0.0, 1.0], [1.0, 0.0]],
        'reference_sources': ['PRN011a01_a_NTD', 'REI096a03_a_NFB'],
        'warnings': ['ratios are provisional'],
        'has_reference': True,
        'ratio_mean': 0.83,
        'frames_scored': 140175,
    }
    out = fr.coerce_qc(doc)

    assert out['parts'] == {'leftear': {'ratio_spread': 0.7, 'role': 'rigid'}}
    assert out['mean_shape'] == [[0.0, 1.0], [1.0, 0.0]]
    assert out['reference_sources'] == ['PRN011a01_a_NTD', 'REI096a03_a_NFB']
    assert out['warnings'] == ['ratios are provisional']
    assert out['has_reference'] is True
    assert out['ratio_mean'] == 0.83
    assert isinstance(out['frames_scored'], int)
    assert out.get('error') is None            # nothing was dropped


def test_qc_v2_document_round_trips_to_json():
    import json
    doc = {
        'parts': {'front': {'ratio_spread': float('nan'), 'role': 'rigid'}},
        'mean_shape': [[0.0, 1.0]],
        'warnings': [],
        'ratio_mean': 0.83,
    }
    s = fr._qc_to_json(fr.coerce_qc(doc))
    back = json.loads(s)                        # MySQL JSON rejects NaN/Infinity
    assert back['parts']['front']['ratio_spread'] is None
    assert back['ratio_mean'] == 0.83


def test_a_rescore_cannot_go_through_the_insert_path():
    # insert_many is INSERT IGNORE and an unchanged file keeps its row_key, so re-running
    # the sweep after a QC change writes NOTHING while reporting success. The qc refresh
    # therefore needs its own UPDATE path -- this asserts it exists and is separate.
    assert hasattr(fr, 'update_qc_many')
    import inspect
    src = inspect.getsource(fr.update_qc_many)
    assert 'UPDATE' in src
    # provenance is a historical fact and must not be rewritten by a re-measurement
    for col in ('release_id', 'scorer', 'target_path', 'attribution', 'analyzed_utc'):
        assert col not in src
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the Database tab triages on
# head-marker spread, which the qc document has always carried but fit_status never lifted
# to a column | date: 2026-09-16]
def test_fit_status_lifts_the_head_spread_headline_out_of_the_qc_document(monkeypatch):
    """``rms_spread_mean`` is the one reference-free quality headline QC produces.

    Every other flat QC column is lifted here; leaving this one inside the JSON meant no
    caller could sort or filter on it without re-parsing the document row by row.
    """
    import json

    import pandas as pd

    row = dict.fromkeys(fr.COLUMNS)
    row.update(id=1, video_stem='A1a01_a_NTD', stream='', attribution='ambiguous',
               target_mtime_utc=MT,
               qc=json.dumps({'rms_spread_mean': 0.41, 'p_median': 0.99}))
    # 'ambiguous' short-circuits is_stale before it reaches the registry, so this stays
    # a pure test of the lift
    monkeypatch.setattr(fr, 'fits_for', lambda **kw: pd.DataFrame([row]))

    got = fr.fit_status(include_current=False)

    assert 'rms_spread_mean' in got.columns
    assert abs(float(got.iloc[0]['rms_spread_mean']) - 0.41) < 1e-9
# [AGENT EDIT END]
