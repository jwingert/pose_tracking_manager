import os
import pytest

from nems_lbhb.motor import dlc_alias as al


def _mk(tmp_path, name, body='x'):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_point_alias_creates_when_absent(tmp_path):
    tgt = _mk(tmp_path, 'a.h5')
    alias = tmp_path / 'v.dlc.h5'
    al.point_alias(str(tgt), str(alias))
    assert os.path.islink(str(alias))
    assert os.path.realpath(str(alias)) == os.path.realpath(str(tgt))


def test_point_alias_repoints_existing_link(tmp_path):
    """The bug this module exists to fix: an existing alias must be replaced, not skipped."""
    old, new = _mk(tmp_path, 'old.h5'), _mk(tmp_path, 'new.h5')
    alias = tmp_path / 'v.dlc.h5'
    al.point_alias(str(old), str(alias))
    al.point_alias(str(new), str(alias))
    assert os.path.realpath(str(alias)) == os.path.realpath(str(new))


def test_point_alias_replaces_dangling_link(tmp_path):
    alias = tmp_path / 'v.dlc.h5'
    os.symlink(str(tmp_path / 'gone.h5'), str(alias))
    assert not os.path.exists(str(alias))       # dangling
    tgt = _mk(tmp_path, 'real.h5')
    al.point_alias(str(tgt), str(alias))
    assert os.path.realpath(str(alias)) == os.path.realpath(str(tgt))


def test_point_alias_refuses_to_clobber_a_regular_file(tmp_path):
    """A real .h5 sitting at the alias path is data, not a pointer. Never silently delete it."""
    tgt = _mk(tmp_path, 'a.h5')
    alias = _mk(tmp_path, 'v.dlc.h5', 'REAL DATA')
    with pytest.raises(FileExistsError):
        al.point_alias(str(tgt), str(alias))
    assert alias.read_text() == 'REAL DATA'


def test_point_alias_raises_when_target_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        al.point_alias(str(tmp_path / 'nope.h5'), str(tmp_path / 'v.dlc.h5'))


def test_point_alias_relative_target_survives_tree_move(tmp_path):
    sub = tmp_path / 'sorted'
    sub.mkdir()
    tgt = _mk(sub, 'a.h5')
    alias = sub / 'v.dlc.h5'
    al.point_alias(str(tgt), str(alias), relative=True)
    assert not os.readlink(str(alias)).startswith('/')
    moved = tmp_path / 'moved'
    os.rename(str(sub), str(moved))
    assert os.path.exists(str(moved / 'v.dlc.h5'))


def test_point_alias_leaves_no_temp_file_behind(tmp_path):
    tgt = _mk(tmp_path, 'a.h5')
    al.point_alias(str(tgt), str(tmp_path / 'v.dlc.h5'))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.h5', 'v.dlc.h5']


def test_find_dlc_output_selects_by_suffix_not_mtime(tmp_path):
    """Two outputs in one folder; the newer one is the WRONG model. mtime must not decide."""
    want = _mk(tmp_path, 'V1DLC_resnet50_two_chimney3fbJan10shuffle1_60000.h5')
    os.utime(str(want), (1, 1))
    _mk(tmp_path, 'V1DLC_resnet50_free_trainSep26shuffle1_60000.h5')   # newer
    got = al.find_dlc_output(str(tmp_path), 'V1',
                             'DLC_resnet50_two_chimney3fbJan10shuffle1_60000')
    assert os.path.basename(got) == os.path.basename(str(want))


def test_find_dlc_output_ignores_dlc_sidecar_files(tmp_path):
    want = _mk(tmp_path, 'V1DLC_resnet50_tcJan10shuffle1_60000.h5')
    _mk(tmp_path, 'V1DLC_resnet50_tcJan10shuffle1_60000_meta.pickle')
    _mk(tmp_path, 'V1DLC_resnet50_tcJan10shuffle1_60000_filtered.h5')
    got = al.find_dlc_output(str(tmp_path), 'V1', 'DLC_resnet50_tcJan10shuffle1_60000')
    assert os.path.basename(got) == os.path.basename(str(want))


def test_find_dlc_output_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        al.find_dlc_output(str(tmp_path), 'V1', 'DLC_resnet50_tcJan10shuffle1_60000')
    assert 'tcJan10shuffle1_60000' in str(e.value)


def test_parse_scorer_round_trips():
    d = al.parse_scorer('/x/SLJ017a07_a_NTDDLC_resnet50_two_chimney3fbJan10shuffle1_60000.h5')
    assert d['net'] == 'resnet50'
    assert d['task'] == 'two_chimney3fb'
    assert d['date'] == 'Jan10'
    assert d['shuffle'] == 1
    assert d['snapshot'] == 60000


def test_parse_scorer_returns_none_fields_for_non_dlc_name():
    d = al.parse_scorer('/x/SLJ017a07_a_NTD.dlc.h5')
    assert d['scorer'] is None


def test_stamped_alias_name():
    assert al.stamped_alias_name('SLJ017a07_a_NTD', 'two_chimney-r0001') == \
        'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'


def test_audit_reports_broken_and_current(tmp_path):
    tgt = _mk(tmp_path, 'V1DLC_resnet50_tcJan10shuffle1_60000.h5')
    good, bad = tmp_path / 'g.dlc.h5', tmp_path / 'b.dlc.h5'
    al.point_alias(str(tgt), str(good))
    os.symlink(str(tmp_path / 'gone.h5'), str(bad))
    rows = {os.path.basename(r['alias']): r for r in al.audit([str(good), str(bad)])}
    assert rows['g.dlc.h5']['broken'] is False
    assert rows['g.dlc.h5']['snapshot'] == 60000
    assert rows['b.dlc.h5']['broken'] is True


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the release id has to reach the
# recording cache hash, so a loader must be able to answer "which model produced the file this
# alias points at?" from the filesystem alone | date: 2026-08-12]
STEM = 'SLJ017a07_a_NTD'


def test_parse_stamped_alias_round_trips_what_stamped_alias_name_writes():
    name = al.stamped_alias_name(STEM, 'two_chimney-r0001')
    assert al.parse_stamped_alias(name) == (STEM, 'two_chimney-r0001')
    assert al.parse_stamped_alias('%s.dlc.h5' % STEM) == (STEM, None)
    # accepts a full path too
    assert al.parse_stamped_alias('/a/b/%s' % name)[1] == 'two_chimney-r0001'


def test_release_for_alias_reads_provenance_off_the_current_pointer(tmp_path):
    """The plain alias is what loaders open and what says which output is CURRENT, but it
    carries no provenance in its name. The release is recovered from the stamped sibling
    that resolves to the same file."""
    tgt = _mk(tmp_path, 'out.h5')
    al.point_alias(str(tgt), str(tmp_path / al.stamped_alias_name(STEM, 'two_chimney-r0001')))
    al.point_alias(str(tgt), str(tmp_path / ('%s.dlc.h5' % STEM)))
    assert al.release_for_alias(str(tmp_path / ('%s.dlc.h5' % STEM))) == 'two_chimney-r0001'


def test_release_for_alias_follows_the_pointer_not_the_newest_stamp(tmp_path):
    """Rolling the current pointer back to an older release must report the OLDER id --
    the question is which model produced the file in force, not which release is latest."""
    old, new = _mk(tmp_path, 'old.h5'), _mk(tmp_path, 'new.h5')
    al.point_alias(str(old), str(tmp_path / al.stamped_alias_name(STEM, 'two_chimney-r0001')))
    al.point_alias(str(new), str(tmp_path / al.stamped_alias_name(STEM, 'two_chimney-r0002')))
    plain = str(tmp_path / ('%s.dlc.h5' % STEM))
    al.point_alias(str(new), plain)
    assert al.release_for_alias(plain) == 'two_chimney-r0002'
    al.point_alias(str(old), plain)          # rollback
    assert al.release_for_alias(plain) == 'two_chimney-r0001'


def test_release_for_alias_is_none_for_legacy_unstamped_output(tmp_path):
    """~550 files on disk predate release-based inference. None is the honest answer, and
    it is what keeps their recording caches from being invalidated to say nothing new."""
    tgt = _mk(tmp_path, 'out.h5')
    plain = str(tmp_path / ('%s.dlc.h5' % STEM))
    al.point_alias(str(tgt), plain)
    assert al.release_for_alias(plain) is None


def test_release_for_alias_is_none_for_a_dangling_link(tmp_path):
    plain = tmp_path / ('%s.dlc.h5' % STEM)
    os.symlink(str(tmp_path / 'gone.h5'), str(plain))
    assert al.release_for_alias(str(plain)) is None


def test_release_for_alias_reports_its_own_stamp_directly(tmp_path):
    tgt = _mk(tmp_path, 'out.h5')
    stamped = str(tmp_path / al.stamped_alias_name(STEM, 'two_chimney-r0001'))
    al.point_alias(str(tgt), stamped)
    assert al.release_for_alias(stamped) == 'two_chimney-r0001'


def test_release_for_alias_surfaces_ambiguity_instead_of_picking_one(tmp_path):
    """Two releases stamping the same file should not happen (each writes into its own
    output dir), so if it does, the answer must make it visible rather than quietly
    choosing -- and must stay deterministic so the cache hash does not flap."""
    tgt = _mk(tmp_path, 'out.h5')
    for rid in ('two_chimney-r0002', 'two_chimney-r0001'):
        al.point_alias(str(tgt), str(tmp_path / al.stamped_alias_name(STEM, rid)))
    plain = str(tmp_path / ('%s.dlc.h5' % STEM))
    al.point_alias(str(tgt), plain)
    assert al.release_for_alias(plain) == 'two_chimney-r0001+two_chimney-r0002'


def test_release_for_alias_ignores_another_sessions_stamp(tmp_path):
    """Sibling sessions share sorted/; only stamps for THIS stem may be consulted."""
    mine, theirs = _mk(tmp_path, 'mine.h5'), _mk(tmp_path, 'theirs.h5')
    al.point_alias(str(theirs),
                   str(tmp_path / al.stamped_alias_name('SLJ017a08_a_NTD', 'two_chimney-r0001')))
    plain = str(tmp_path / ('%s.dlc.h5' % STEM))
    al.point_alias(str(mine), plain)
    assert al.release_for_alias(plain) is None
# [AGENT EDIT END]
