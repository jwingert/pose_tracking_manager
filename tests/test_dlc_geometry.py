# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for the DLC recording-geometry lookup (chimney side / headphones) | date: 2026-08-06]
"""Tests for :mod:`nems_lbhb.motor.dlc_geometry`.

Every test but the last injects the ear and audio tables, so the suite runs
without a live celldb.  The final test is DB-backed and skips itself when the
database is unreachable.
"""
import pytest

from nems_lbhb.motor import dlc_geometry as gm


# Small fixture tables.  The real pennames carry their real gPenetration.ear
# values, so the fixtures double as a record of the live data: PRN001 is a
# right-ear penetration (chimney on the LEFT), PRN039 a left-ear one (chimney
# on the RIGHT), SQD024 left-ear + in-ear headphones, REI068 right-ear + in-ear.
EARS = {
    'PRN001': 'R',
    'PRN039': 'L',
    'REI043': 'R',
    'REI068': 'R',
    'SQD024': 'L',
    'LMD004': 'R',
    'BLANK01': '',
    'NONE01': None,
    'BOTH01': 'B',
}

AUDIO = {
    'PRN001a01_a_NTD': 'Free-field',
    'PRN039a01_a_NTD': 'Free-field',
    'SQD024a04_a_NTD': 'In-ear',
    'REI068a14_a_NFC': 'In-ear',
    # 'PRN001a99_a_NTD' deliberately absent -> unresolved audio
}


def geom(penname, stem=None, two=None):
    return gm.geometry_for(penname, video_stem=stem, two_chimney=two,
                           ear_table=EARS, audio_table=AUDIO)


# ---------------------------------------------------------------- ear rule

def test_primary_ear_left_is_one():
    """Must match nems_lbhb/baphy_experiment.py:302 -- ear == 'L' -> 1."""
    assert gm.primary_ear('PRN039', ear_table=EARS) == 1


def test_primary_ear_right_is_zero():
    assert gm.primary_ear('PRN001', ear_table=EARS) == 0


def test_primary_ear_non_lateral_follows_else_zero_rule():
    """'B' is not blank, so primary_ear mirrors production (`else 0`) rather than raising."""
    assert gm.primary_ear('BOTH01', ear_table=EARS) == 0


def test_primary_ear_blank_raises():
    with pytest.raises(gm.UnknownEarError):
        gm.primary_ear('BLANK01', ear_table=EARS)


def test_primary_ear_null_raises():
    with pytest.raises(gm.UnknownEarError):
        gm.primary_ear('NONE01', ear_table=EARS)


def test_primary_ear_unknown_penname_raises():
    with pytest.raises(gm.UnknownEarError):
        gm.primary_ear('training2023', ear_table=EARS)


# ------------------------------------------------------- contralateral rule

def test_right_ear_gives_left_chimney():
    assert gm.chimney_side('PRN001', ear_table=EARS) == 'left'


def test_left_ear_gives_right_chimney():
    assert gm.chimney_side('PRN039', ear_table=EARS) == 'right'


def test_chimney_side_blank_raises():
    with pytest.raises(gm.UnknownEarError):
        gm.chimney_side('BLANK01', ear_table=EARS)


def test_chimney_side_unknown_penname_raises():
    with pytest.raises(gm.UnknownEarError):
        gm.chimney_side('training2024', ear_table=EARS)


def test_chimney_side_bilateral_ear_raises_rather_than_guessing():
    """ear == 'B' does not name a side; refusing beats silently picking one."""
    with pytest.raises(gm.UnknownEarError):
        gm.chimney_side('BOTH01', ear_table=EARS)


# ------------------------------------------------------------- geometry_for

def test_geometry_free_field_right_ear():
    g = geom('PRN001', 'PRN001a01_a_NTD')
    assert g['geometry'] == 'left_chimney'
    assert g['chimney'] == 'left'
    assert g['ear'] == 'R'
    assert g['primary_ear'] == 0
    assert g['audio'] == 'Free-field'
    assert g['headphones'] is False
    assert g['two_chimney'] is False


def test_geometry_free_field_left_ear():
    assert geom('PRN039', 'PRN039a01_a_NTD')['geometry'] == 'right_chimney'


def test_geometry_in_ear_gives_headphones_variant():
    g = geom('SQD024', 'SQD024a04_a_NTD')
    assert g['geometry'] == 'right_chimney_headphones'
    assert g['headphones'] is True


def test_geometry_in_ear_left_chimney():
    assert geom('REI068', 'REI068a14_a_NFC')['geometry'] == 'left_chimney_headphones'


def test_two_chimney_override_beats_chimney_side():
    """A bilateral implant shows both chimneys whichever hemisphere was recorded."""
    g = geom('LMD004', 'LMD004a00_a_NFB', two=True)
    assert g['geometry'] == 'two_chimney'
    assert g['two_chimney'] is True
    # the ear-derived side is still reported, it just does not drive the geometry
    assert g['chimney'] == 'left'


def test_two_chimney_override_applies_to_right_ear_too():
    assert geom('PRN001', 'PRN001a01_a_NTD', two=True)['geometry'] == 'two_chimney'


# ------------------------------------------------------------ unknown audio

def test_unresolved_audio_is_flagged_not_defaulted():
    g = geom('PRN001', 'PRN001a99_a_NTD')
    assert g['geometry'] == 'left_chimney (audio unknown)'
    assert g['geometry'] != 'left_chimney'
    assert g['audio'] is None
    assert g['headphones'] is None
    assert 'unknown' in g['geometry_source'].lower()


def test_missing_video_stem_is_unresolved_audio():
    assert geom('PRN039')['geometry'] == 'right_chimney (audio unknown)'


def test_geometry_source_is_populated_for_every_branch():
    for g in (geom('PRN001', 'PRN001a01_a_NTD'),
              geom('SQD024', 'SQD024a04_a_NTD'),
              geom('PRN001', 'PRN001a99_a_NTD'),
              geom('LMD004', 'LMD004a00_a_NFB', two=True)):
        assert isinstance(g['geometry_source'], str) and g['geometry_source']


def test_geometry_for_unknown_penname_raises():
    with pytest.raises(gm.UnknownEarError):
        geom('training2023', 'LemonDisco_2023_09_07_NFB_2')


# ----------------------------------------------------------- geometry_table

def test_geometry_table_batch_form():
    df = gm.geometry_table(
        pennames=['PRN001', 'PRN039', 'SQD024'],
        stems=['PRN001a01_a_NTD', 'PRN039a01_a_NTD', 'SQD024a04_a_NTD'],
        ear_table=EARS, audio_table=AUDIO)
    assert list(df['geometry']) == ['left_chimney', 'right_chimney',
                                    'right_chimney_headphones']
    for col in ('penname', 'ear', 'primary_ear', 'chimney', 'two_chimney',
                'audio', 'headphones', 'geometry', 'geometry_source'):
        assert col in df.columns


def test_geometry_table_honours_two_chimney_list():
    df = gm.geometry_table(pennames=['LMD004', 'PRN001'],
                           stems=['LMD004a00_a_NFB', 'PRN001a01_a_NTD'],
                           two_chimney=[True, False],
                           ear_table=EARS, audio_table=AUDIO)
    assert list(df['geometry']) == ['two_chimney', 'left_chimney']


def test_geometry_table_requires_pennames():
    with pytest.raises(ValueError):
        gm.geometry_table(ear_table=EARS, audio_table=AUDIO)


def test_geometry_table_length_mismatch_raises():
    with pytest.raises(ValueError):
        gm.geometry_table(pennames=['PRN001', 'PRN039'], stems=['PRN001a01_a_NTD'],
                          ear_table=EARS, audio_table=AUDIO)


# --------------------------------------------------------------- DB-backed

def _db_ear_table():
    try:
        return gm.load_ear_table()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip('celldb unreachable: %s' % e)


def test_db_backed_known_chimney_sides():
    """Anchors the contralateral rule against three real penetrations."""
    ear = _db_ear_table()
    for penname, expected in (('PRN001', 'left'), ('PRN039', 'right'),
                              ('REI043', 'left')):
        if penname not in ear:
            pytest.skip('%s not in gPenetration' % penname)
        assert gm.chimney_side(penname) == expected
# [AGENT EDIT END]
