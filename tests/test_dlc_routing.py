# [AGENT EDIT START | agent: claude | user: wingertj | reason: routing suggestions must come
# from what the recording IS, and must never assert a fact nothing recorded | date: 2026-08-13]
"""Tests for nems_lbhb/motor/dlc_routing.py.

The database is stubbed: what is under test is the inference from (chimney side, audio,
stream) to a ranked suggestion, not celldb. One live-data check of the whole corpus lives
in the validation run recorded in the commit message, not here.
"""
import pandas as pd
import pytest

from nems_lbhb.motor import dlc_routing as R
from nems_lbhb.motor import dlc_registry


class FakeDB:
    """Answers the two queries `observe` makes, from dicts keyed by penname/parmfile."""

    def __init__(self, ear=None, audio=None, fail=False):
        self.ear, self.audio, self.fail = ear, audio, fail

    def pd_query(self, sql, params=None):
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: these tests passed
        # against the BROKEN named-parameter form -- a stub that accepts anything cannot
        # catch a param-style regression, and the real failure was silent because observe()
        # catches lookup errors | date: 2026-08-13]
        # db.pd_query binds at the driver level (_exec_driver_sql_to_df), so it takes
        # pymysql's %s with a TUPLE. A dict of named :params reaches MySQL verbatim and is
        # a syntax error -- which observe() would swallow into 'unknown', degrading every
        # suggestion instead of failing.
        assert params is None or isinstance(params, (tuple, list)), (
            'pd_query takes %%s with a tuple, not named parameters; got %r' % (params,))
        assert ':' not in sql.replace('::', ''), (
            'named :params in pd_query SQL reach MySQL verbatim: %r' % sql)
        # [AGENT EDIT END]
        if self.fail:
            raise RuntimeError('database is down')
        if 'gPenetration' in sql:
            return pd.DataFrame([] if self.ear is None else [{'ear': self.ear}])
        return pd.DataFrame([] if self.audio is None else [{'svalue': self.audio}])


ARENA = '/auto/data/daq/Reishi/REI084/REI084a10_a_NFB.avi'


@pytest.fixture(scope='module')
def reg():
    return dlc_registry.load_registry()


def _suggest(path=ARENA, ear='R', audio='Free-field', reg=None, **kw):
    return R.suggest(path, registry=reg, db=FakeDB(ear=ear, audio=audio, **kw))


def test_ear_is_read_as_contralateral_to_the_chimney(reg):
    """The single most reversible fact in the whole scheme: ear=R means the chimney is
    on the LEFT. Verified against six sites with independently known models."""
    assert _suggest(ear='R', reg=reg)['observed']['chimney'] == 'left'
    assert _suggest(ear='L', reg=reg)['observed']['chimney'] == 'right'


def test_free_field_left_chimney_suggests_the_left_arena_model(reg):
    s = _suggest(ear='R', audio='Free-field', reg=reg)
    assert s['alias'] == 'left_chimney'
    assert any('contralateral' in w for w in s['candidates'][0]['why'])


def test_in_ear_suggests_the_headphone_model_on_either_side(reg):
    """left_chimney_headphones is declared for BOTH sides because it demonstrably ran on
    both (28 rows right, 12 left) -- the alias name is historical, not a geometry claim."""
    for ear in ('R', 'L'):
        s = _suggest(ear=ear, audio='In-ear', reg=reg)
        assert s['alias'] == 'left_chimney_headphones'
        assert 'left_chimney' not in [c['alias'] for c in s['candidates']], \
            'the free-field model must be ruled out by In-ear, not merely outranked'


def test_ranking_does_not_prefer_a_narrower_declaration_over_a_proven_one(reg):
    """For a right-chimney in-ear recording both headphone models match. Preferring the
    narrowly-declared one would pick right_chimney_headphones, which has never been run on
    a single file, over the model that produced all 28 such rows."""
    s = _suggest(ear='L', audio='In-ear', reg=reg)
    aliases = [c['alias'] for c in s['candidates']]
    assert aliases.index('left_chimney_headphones') < aliases.index('right_chimney_headphones')


def test_no_penetration_means_no_implant(reg):
    s = _suggest(ear=None, audio=None, reg=reg)
    assert s['observed']['chimney'] == 'none'
    assert s['alias'] == 'behavior_only'


def test_close_up_streams_are_decided_by_stream_alone(reg):
    pupil = _suggest('/auto/data/daq/Reishi/REI084/REI084a09_p_NFB_pupil.avi', reg=reg)
    face = _suggest('/auto/data/daq/Clathrus/CLT001/CLT001a01_a_NTD.lick.avi', reg=reg)
    assert pupil['alias'] == 'pupil'
    assert face['alias'] == 'face'


def test_side_cameras_are_still_the_arena_model(reg):
    """The arena body model applies to the ceiling view and the side cameras alike; only
    the close-ups are a different model."""
    s = _suggest('/auto/data/daq/Reishi/REI084/REI084a10_a_NFB_camC.avi', ear='R', reg=reg)
    assert s['observed']['stream'] == 'arena'
    assert s['alias'] == 'left_chimney'


def test_two_chimney_is_never_ruled_out_by_ear_alone(reg):
    """ear reports SIDE. Nothing in celldb records COUNT, so excluding a two-chimney model
    on the strength of a side observation would assert a fact nobody has recorded. It must
    survive as a lower-ranked candidate instead."""
    s = _suggest(ear='R', audio='Free-field', reg=reg)
    aliases = [c['alias'] for c in s['candidates']]
    assert s['alias'] == 'left_chimney'          # the side match still wins
    assert 'two_chimney' in aliases              # but the alternative stays visible
    two = next(c for c in s['candidates'] if c['alias'] == 'two_chimney')
    assert two['unknown'], 'the reason it is only a candidate must be stated'


def test_a_contradicted_side_is_excluded(reg):
    """Unknown and contradicted are different: a right-chimney model must not be offered
    for a left-chimney recording."""
    s = _suggest(ear='R', audio='Free-field', reg=reg)
    assert 'right_chimney' not in [c['alias'] for c in s['candidates']]


def test_unobserved_audio_keeps_both_options_alive(reg):
    s = _suggest(ear='R', audio=None, reg=reg)
    aliases = [c['alias'] for c in s['candidates']]
    assert 'left_chimney' in aliases and 'left_chimney_headphones' in aliases
    assert 'no audio parameter recorded' in (s['note'] or '')


def test_suggest_never_raises_without_a_database(reg):
    s = _suggest(reg=reg, fail=True)
    assert s['alias'] is not None or s['candidates'] == []
    assert any('failed' in u or 'down' in u for u in s['observed']['unknown'])


def test_legacy_method_still_answers_and_reports_its_own_failure(reg):
    """The legacy router is kept selectable so a historical file can be re-analysed with
    the model that actually produced it."""
    ok = R.suggest('/auto/data/daq/Prince/PRN005/PRN005a01_a_NTD.avi',
                   method='legacy', registry=reg)
    assert ok['alias'] == 'left_chimney'
    # REI084 is one of the sites the legacy chain never covered
    bad = R.suggest(ARENA, method='legacy', registry=reg)
    assert bad['alias'] is None
    assert 'No DLC model specified' in bad['note']


def test_geometry_answers_where_legacy_raises(reg):
    """The whole point: REI084a10 is unroutable by site rules and obvious by geometry."""
    assert R.suggest(ARENA, method='legacy', registry=reg)['alias'] is None
    assert _suggest(ARENA, ear='R', audio='Free-field', reg=reg)['alias'] == 'left_chimney'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: ranking ties by registry file
# order alone silently cost 70 rows -- every pre-implant training recording was suggested
# left_chimney over behavior_only | date: 2026-08-13]
def test_ties_prefer_the_declaration_we_could_actually_verify(reg):
    """A pre-implant training recording has no penetration to read a side from, so chimney
    is UNKNOWN rather than 'none'. behavior_only and left_chimney then match equally well
    on stream alone, and the tie must go to the alias with fewer unresolved requirements --
    behavior_only leaves 1 open, left_chimney leaves 2 (chimney and audio)."""
    s = R.suggest('/auto/data/daq/LemonDisco/training2023/LemonDisco_2023_09_25_NFB_4.avi',
                  registry=reg, db=FakeDB(ear=None, audio=None))
    assert s['observed']['chimney'] is None, 'an unparseable penetration is unknown, not none'
    assert s['alias'] == 'behavior_only'
    ranked = [c['alias'] for c in s['candidates']]
    assert ranked.index('behavior_only') < ranked.index('left_chimney')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: celldb records no chimney count,
# but open-ephys settings.xml records how many probes ran -- which is ASYMMETRIC evidence for
# it, and must be treated as evidence rather than as the count | date: 2026-08-13]
SETTINGS_2 = '''<SETTINGS>
 <PROCESSOR name="Neuropix-PXI">
  <STREAM name="ProbeA-AP" sample_rate="30000.0"/>
  <STREAM name="ProbeB-AP" sample_rate="30000.0"/>
  <EDITOR><NP_PROBE slot="2" probe_serial_number="21120"/>
          <NP_PROBE slot="2" probe_serial_number="21121"/></EDITOR>
 </PROCESSOR></SETTINGS>'''

SETTINGS_1 = '''<SETTINGS>
 <PROCESSOR name="Neuropix-PXI">
  <STREAM name="ProbeA-AP" sample_rate="30000.0"/>
  <EDITOR><NP_PROBE slot="2" probe_serial_number="21120"/></EDITOR>
 </PROCESSOR></SETTINGS>'''


def _site(tmp_path, settings=None, nodes=('Record Node 101',)):
    """A daq-shaped site tree, optionally with an open-ephys run under raw/."""
    site = tmp_path / 'daq' / 'TestAnimal' / 'TAN001'
    site.mkdir(parents=True)
    if settings is not None:
        run = site / 'raw' / 'TAN001a01' / 'TAN001a01_p_FTC_2023-07-17_12-47-47'
        for node in nodes:
            d = run / node
            d.mkdir(parents=True)
            (d / 'settings.xml').write_text(settings)
    return str(site / 'TAN001a01_a_NTD.avi')


def test_probe_count_reads_two_probes(tmp_path):
    assert R.probe_count(_site(tmp_path, SETTINGS_2), cache={}) == 2


def test_probe_count_is_not_record_node_count(tmp_path):
    """SLJ017 has three Record Nodes and two probes, so counting node folders would be
    wrong. The probes themselves are counted."""
    path = _site(tmp_path, SETTINGS_2,
                 nodes=('Record Node 101', 'Record Node 102', 'Record Node 111'))
    assert R.probe_count(path, cache={}) == 2


def test_probe_count_is_none_without_settings(tmp_path):
    assert R.probe_count(_site(tmp_path), cache={}) is None


def test_two_probes_establish_two_chimneys(tmp_path, reg):
    """You cannot record from a chimney that is not there, so two probes settle the count
    and the single-chimney models become wrong rather than merely outranked."""
    path = _site(tmp_path, SETTINGS_2)
    R._PROBE_CACHE.clear()
    s = R.suggest(path, registry=reg, db=FakeDB(ear='R', audio='Free-field'))
    assert s['observed']['two_chimney_evidence'] is True
    assert s['alias'] == 'two_chimney'
    assert 'left_chimney' not in [c['alias'] for c in s['candidates']]
    assert any('probes' in w for w in s['candidates'][0]['why'])


def test_one_probe_establishes_nothing(tmp_path, reg):
    """LemonDisco carried both chimneys and habitually ran a single probe (25 rows of the
    corpus), and SlipperyJack does the same at some sites. So a count of 1 must leave the
    side-based answer standing, with two_chimney still visible."""
    path = _site(tmp_path, SETTINGS_1)
    R._PROBE_CACHE.clear()
    s = R.suggest(path, registry=reg, db=FakeDB(ear='R', audio='Free-field'))
    assert s['observed']['probes'] == 1
    assert s['observed']['two_chimney_evidence'] is False
    assert s['alias'] == 'left_chimney'
    assert 'two_chimney' in [c['alias'] for c in s['candidates']]
    assert any('does not establish chimney count' in u for u in s['observed']['unknown'])


def test_missing_probe_evidence_is_reported_not_assumed(tmp_path, reg):
    path = _site(tmp_path)
    R._PROBE_CACHE.clear()
    s = R.suggest(path, registry=reg, db=FakeDB(ear='R', audio='Free-field'))
    assert s['observed']['probes'] is None
    assert any('chimney count is unknown' in u for u in s['observed']['unknown'])
    assert s['alias'] == 'left_chimney'
# [AGENT EDIT END]
