"""Tests for nems_lbhb.motor.dlc_release -- immutable, checksummed DLC releases.

Everything here runs against a synthetic mini-project in ``tmp_path``.  No test
reads or writes a real DLC project or the real release root.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 5 -- tests for the
# immutable single-snapshot DLC release store (build/verify/promote/rollback) | date: 2026-08-06]
import os
import pytest
yaml = pytest.importorskip('yaml')

from nems_lbhb.motor import dlc_release as rel

BODYPARTS = ['front', 'back', 'leftear', 'rightear']


def make_project(tmp_path, task='fake_task', date='Jan10', snapshots=(30000, 60000),
                 iteration=0, shuffle=1, trainset=95):
    proj = tmp_path / ('%s-tester-2024-01-10' % task)
    train = (proj / 'dlc-models' / ('iteration-%d' % iteration) /
             ('%s%s-trainset%dshuffle%d' % (task, date, trainset, shuffle)) / 'train')
    test = train.parent / 'test'
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    for s in snapshots:
        for ext in ('.index', '.meta', '.data-00000-of-00001'):
            (train / ('snapshot-%d%s' % (s, ext))).write_text('w%d%s' % (s, ext))
    (train / 'pose_cfg.yaml').write_text(yaml.dump({'net_type': 'resnet_50', 'init_weights': '/x'}))
    (test / 'pose_cfg.yaml').write_text(yaml.dump({'net_type': 'resnet_50'}))
    lab = proj / 'labeled-data' / 'vid1'
    lab.mkdir(parents=True)
    (lab / 'CollectedData_tester.h5').write_text('labels')
    (proj / 'config.yaml').write_text(yaml.dump({
        'Task': task, 'date': date, 'scorer': 'tester', 'iteration': iteration,
        'snapshotindex': -1, 'project_path': str(proj), 'bodyparts': BODYPARTS,
        'TrainingFraction': [trainset / 100.0], 'net_type': 'resnet_50',
        'video_sets': {}, 'batch_size': 8,
    }))
    return proj


def test_build_release_creates_one_snapshot_only(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    assert rid == 'fake_alias-r0001'
    train = next((tmp_path / 'releases' / rid / 'dlc-models').rglob('train'))
    snaps = sorted({p.name.split('.')[0] for p in train.glob('snapshot-*')})
    assert snaps == ['snapshot-60000'], 'a release must contain exactly one snapshot'


def test_release_config_project_path_points_at_itself(tmp_path):
    """DLC read_config rewrites project_path to the config's own dir. It must already be right."""
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    rdir = tmp_path / 'releases' / rid
    cfg = yaml.safe_load((rdir / 'config.yaml').read_text())
    assert cfg['project_path'] == str(rdir)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 1 -- a
# relative root left project_path relative, which makes DLC's read_config rewrite the
# 0444 config on first read and silently break the release's own checksums | date: 2026-08-06]
def test_build_release_abspaths_a_relative_root(tmp_path, monkeypatch):
    proj = make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root='releases')
    rdir = tmp_path / 'releases' / rid
    cfg = yaml.safe_load((rdir / 'config.yaml').read_text())
    assert os.path.isabs(cfg['project_path'])
    assert cfg['project_path'] == str(rdir)
# [AGENT EDIT END]


def test_release_config_pins_snapshotindex_unambiguously(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    cfg = yaml.safe_load((tmp_path / 'releases' / rid / 'config.yaml').read_text())
    assert cfg['snapshotindex'] == 0
    assert cfg['iteration'] == 0


def test_build_release_freezes_labels(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    frozen = list((tmp_path / 'releases' / rid / 'labels').rglob('CollectedData_*.h5'))
    assert len(frozen) == 1
    (proj / 'labeled-data' / 'vid1' / 'CollectedData_tester.h5').write_text('CHANGED')
    assert frozen[0].read_text() == 'labels', 'frozen labels must not track the live project'


def test_manifest_records_provenance(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'),
                            note='cut for testing')
    m = rel.read_manifest(str(tmp_path / 'releases' / rid))
    assert m['release']['id'] == rid
    assert m['source']['project'] == str(proj)
    assert m['source']['snapshot'] == 60000
    assert m['model']['bodyparts'] == BODYPARTS
    # [AGENT EDIT | reason: review finding 3 -- endswith('shuffle1_60000') would still
    # pass for a build that got the net segment or underscore placement wrong; pin the
    # exact string. Shape matches the real-world scorer validated against
    # SLJ017a07_a_NTDDLC_resnet50_two_chimney3fbJan10shuffle1_60000.h5]
    assert m['model']['output_suffix'] == 'DLC_resnet50_fake_taskJan10shuffle1_60000'
    assert m['release']['note'] == 'cut for testing'
    assert m['source']['labels_fingerprint']


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 2 --
# net_type.replace('_','') is only correct for DLC's plain-resnet branch of
# get_scorer_name; must raise a clear error rather than silently mis-name
# output_suffix for a multi_stage or mobilenet/efficientnet net | date: 2026-08-06]
def test_build_refuses_a_multi_stage_net(tmp_path):
    proj = make_project(tmp_path)
    train = next((proj / 'dlc-models').rglob('train'))
    cfg = yaml.safe_load((train / 'pose_cfg.yaml').read_text())
    cfg['multi_stage'] = True
    (train / 'pose_cfg.yaml').write_text(yaml.dump(cfg))
    with pytest.raises(ValueError) as e:
        rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    assert 'multi_stage' in str(e.value)


def test_build_refuses_a_non_resnet_net_type(tmp_path):
    proj = make_project(tmp_path)
    train = next((proj / 'dlc-models').rglob('train'))
    cfg = yaml.safe_load((train / 'pose_cfg.yaml').read_text())
    cfg['net_type'] = 'mobilenet_v2_1.0'
    (train / 'pose_cfg.yaml').write_text(yaml.dump(cfg))
    with pytest.raises(ValueError) as e:
        rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    assert 'mobilenet_v2_1.0' in str(e.value)
# [AGENT EDIT END]


def test_verify_clean_then_detects_tampering(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    rdir = tmp_path / 'releases' / rid
    assert rel.verify(str(rdir), full=True) == []
    victim = next(rdir.rglob('snapshot-60000.index'))
    os.chmod(str(victim), 0o644)
    victim.write_text('tampered')
    problems = rel.verify(str(rdir), full=True)
    assert problems and 'snapshot-60000.index' in problems[0]


def test_verify_fast_mode_detects_a_missing_file(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    rdir = tmp_path / 'releases' / rid
    victim = next(rdir.rglob('snapshot-60000.meta'))
    # DEVIATION FROM THE BRIEF (one added line, flagged in the task report): a release
    # directory is 0o555, so unlinking a file out of it needs write permission on the
    # *containing directory* as well as on the file.  The brief's version of this test
    # chmods only the file and therefore cannot reach its own assertion.  The intent of
    # the test -- "fast verify notices a file that has gone missing" -- is unchanged.
    os.chmod(str(victim.parent), 0o755)
    os.chmod(str(victim), 0o644)
    os.remove(str(victim))
    assert any('snapshot-60000.meta' in p for p in rel.verify(str(rdir), full=False))


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review minor finding --
# the original tamper test ('tampered', a different length than 'w60000.index') could
# not distinguish hashing from size-checking; this pins that fast mode checks size
# ONLY (same-length corruption slips through) while full mode always catches it |
# date: 2026-08-06]
def test_verify_fast_mode_does_not_catch_same_length_tampering(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    rdir = tmp_path / 'releases' / rid
    victim = next(rdir.rglob('snapshot-60000.index'))
    original = victim.read_text()
    tampered = 'X' * len(original)
    assert tampered != original and len(tampered) == len(original)
    os.chmod(str(victim), 0o644)
    victim.write_text(tampered)
    assert rel.verify(str(rdir), full=False) == [], (
        'fast mode only checks size, so same-length corruption must slip through')
    problems = rel.verify(str(rdir), full=True)
    assert problems and 'snapshot-60000.index' in problems[0], (
        'full mode re-hashes, so it must still catch it')
# [AGENT EDIT END]


def test_release_files_are_read_only(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    idx = next((tmp_path / 'releases' / rid).rglob('snapshot-60000.index'))
    assert not (os.stat(str(idx)).st_mode & 0o222), 'release weights must not be writable'


def test_build_refuses_a_snapshot_that_does_not_exist(tmp_path):
    proj = make_project(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        rel.build_release(str(proj), 'fake_alias', 99999, root=str(tmp_path / 'releases'))
    assert '99999' in str(e.value)


def test_build_refuses_an_incomplete_snapshot_triplet(tmp_path):
    proj = make_project(tmp_path)
    train = next((proj / 'dlc-models').rglob('train'))
    os.remove(str(train / 'snapshot-60000.meta'))
    with pytest.raises(FileNotFoundError) as e:
        rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    assert '.meta' in str(e.value)


def test_build_refuses_an_ambiguous_snapshot_across_iterations(tmp_path):
    """Defect 3's shape: the same snapshot number in two iterations. Refuse; make the caller pick."""
    proj = make_project(tmp_path)
    src = next((proj / 'dlc-models').rglob('train'))
    dst = (proj / 'dlc-models' / 'iteration-1' / src.parent.name / 'train')
    dst.mkdir(parents=True)
    for p in src.glob('snapshot-60000*'):
        (dst / p.name).write_text('other run')
    with pytest.raises(ValueError) as e:
        rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    assert 'iteration' in str(e.value).lower()
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'),
                            iteration=1)
    assert rid == 'fake_alias-r0001'


def test_next_release_id_increments(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    assert rel.build_release(str(proj), 'a', 30000, root=root) == 'a-r0001'
    assert rel.build_release(str(proj), 'a', 60000, root=root) == 'a-r0002'
    assert rel.build_release(str(proj), 'b', 60000, root=root) == 'b-r0001'


def test_promote_updates_current_and_appends_history(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    alias = 'two_chimney'
    # DEVIATION FROM THE BRIEF (skip_verify, flagged in the task report): the brief's
    # own Step-4 spec says promote checks that the release directory exists and that
    # verify(full=True) is clean.  These three registry tests never cut a release, so
    # they must take the documented escape hatch.  They are about the TOML edit only.
    rel.promote(alias, '%s-r0001' % alias, registry_path=str(regp), skip_verify=True)
    import tomli
    r = tomli.loads(regp.read_text())
    assert r['alias'][alias]['current'] == '%s-r0001' % alias
    assert '%s-r0001' % alias in r['alias'][alias]['history']


def test_rollback_restores_the_previous_release(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    alias = 'two_chimney'
    rel.promote(alias, '%s-r0001' % alias, registry_path=str(regp), skip_verify=True)
    rel.promote(alias, '%s-r0002' % alias, registry_path=str(regp), skip_verify=True)
    # rollback's own dir-existence check (review finding 5, minor) is bypassed here
    # too -- this fixture never cuts a real release, only edits the registry copy.
    rel.rollback(alias, registry_path=str(regp), skip_verify=True)
    import tomli
    assert tomli.loads(regp.read_text())['alias'][alias]['current'] == '%s-r0001' % alias


def test_promote_never_deletes_history(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    for n in (1, 2, 3):
        rel.promote('two_chimney', 'two_chimney-r000%d' % n, registry_path=str(regp),
                    skip_verify=True)
    rel.rollback('two_chimney', registry_path=str(regp), skip_verify=True)
    import tomli
    h = tomli.loads(regp.read_text())['alias']['two_chimney']['history']
    assert len(h) == 3


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review findings 4 and 5 --
# promote's verify gate had zero test coverage (all five promote tests used
# skip_verify=True), and a manifest recording 'INCOMPLETE RELEASE' verified clean and
# could be promoted anyway; both are now covered end to end against real releases |
# date: 2026-08-06]
def test_promote_refuses_a_release_directory_that_does_not_exist(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    with pytest.raises(FileNotFoundError) as e:
        rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp),
                    root=str(tmp_path / 'releases'))
    assert 'two_chimney-r0001' in str(e.value)


def test_promote_refuses_a_release_that_fails_verify(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    rid = rel.build_release(str(proj), 'two_chimney', 60000, root=root)
    rdir = tmp_path / 'releases' / rid
    victim = next(rdir.rglob('snapshot-60000.index'))
    os.chmod(str(victim), 0o644)
    victim.write_text('tampered')
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    with pytest.raises(ValueError) as e:
        rel.promote('two_chimney', rid, registry_path=str(regp), root=root)
    assert rid in str(e.value)


def test_promote_and_rollback_succeed_against_real_releases(tmp_path):
    """The happy path for the verify gates added above: nothing tampered, nothing
    missing -- promote and rollback must not refuse a perfectly good release."""
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    rid1 = rel.build_release(str(proj), 'two_chimney', 30000, root=root)
    rid2 = rel.build_release(str(proj), 'two_chimney', 60000, root=root)
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    rel.promote('two_chimney', rid1, registry_path=str(regp), root=root)
    rel.promote('two_chimney', rid2, registry_path=str(regp), root=root)
    rel.rollback('two_chimney', registry_path=str(regp), root=root)
    import tomli
    assert tomli.loads(regp.read_text())['alias']['two_chimney']['current'] == rid1


def test_rollback_refuses_when_target_release_directory_is_gone(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    alias = 'two_chimney'
    rel.promote(alias, '%s-r0001' % alias, registry_path=str(regp), skip_verify=True)
    rel.promote(alias, '%s-r0002' % alias, registry_path=str(regp), skip_verify=True)
    with pytest.raises(FileNotFoundError) as e:
        rel.rollback(alias, registry_path=str(regp), root=str(tmp_path / 'releases'))
    assert '%s-r0001' % alias in str(e.value)


def test_promote_refuses_an_incomplete_release(tmp_path):
    """A release missing train/pose_cfg.yaml records 'INCOMPLETE RELEASE' in its
    manifest but is otherwise structurally clean -- verify() alone would wave it
    through. promote must refuse it explicitly."""
    proj = make_project(tmp_path)
    train = next((proj / 'dlc-models').rglob('train'))
    os.remove(str(train / 'pose_cfg.yaml'))
    root = str(tmp_path / 'releases')
    rid = rel.build_release(str(proj), 'two_chimney', 60000, root=root)
    m = rel.read_manifest(rid, root=root)
    assert 'INCOMPLETE RELEASE' in m['release']['provenance_warning']
    assert rel.verify(rid, full=True, root=root) == [], (
        'structurally clean (nothing tampered, nothing missing that SHA256SUMS knows '
        'about) despite being unloadable by DeepLabCut -- this is exactly the gap '
        'verify() alone cannot see')
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    with pytest.raises(ValueError) as e:
        rel.promote('two_chimney', rid, registry_path=str(regp), root=root)
    assert 'INCOMPLETE RELEASE' in str(e.value)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: consolidation task step 1 --
# rename_release re-seals a release under a new alias (eyecloseup->pupil, multivid->face).
# A rename is the one operation that legitimately rewrites an immutable release, so every
# invariant it could break -- the checksums that cover RELEASE.toml, config.yaml's
# self-referential project_path, the 0444/0555 freeze, the never-reuse-an-id rule -- is
# pinned here | date: 2026-08-07]
def _build(tmp_path, alias='oldname', snapshot=60000, root=None):
    proj = make_project(tmp_path)
    root = root or str(tmp_path / 'releases')
    return proj, root, rel.build_release(str(proj), alias, snapshot, root=root)


def test_rename_release_moves_the_directory_and_keeps_the_number(tmp_path):
    _proj, root, rid = _build(tmp_path)
    new = rel.rename_release(rid, 'newname', root=root, reason='aliases name what they track')
    assert new == 'newname-r0001', 'the release NUMBER is part of its identity, not the alias'
    assert not os.path.exists(os.path.join(root, rid))
    assert os.path.isdir(os.path.join(root, new))


def test_rename_release_verifies_clean_afterwards(tmp_path):
    """SHA256SUMS covers RELEASE.toml, so editing the manifest invalidates it."""
    _proj, root, rid = _build(tmp_path)
    new = rel.rename_release(rid, 'newname', root=root, reason='because')
    assert rel.verify(new, full=True, root=root) == []


def test_rename_release_rewrites_id_and_alias_and_records_where_it_came_from(tmp_path):
    _proj, root, rid = _build(tmp_path)
    new = rel.rename_release(rid, 'newname', root=root, reason='the alias was misleading')
    m = rel.read_manifest(new, root=root)
    assert m['release']['id'] == new
    assert m['release']['alias'] == 'newname'
    entry = m['release']['renamed_from'][-1]
    assert entry['old_id'] == rid
    assert entry['old_alias'] == 'oldname'
    assert entry['new_id'] == new
    assert entry['reason'] == 'the alias was misleading'
    assert entry['utc'].endswith('Z')


def test_rename_release_accumulates_renames(tmp_path):
    _proj, root, rid = _build(tmp_path)
    once = rel.rename_release(rid, 'middle', root=root, reason='first')
    twice = rel.rename_release(once, 'final', root=root, reason='second')
    chain = rel.read_manifest(twice, root=root)['release']['renamed_from']
    assert [e['old_id'] for e in chain] == [rid, once], 'a rename must not erase the last one'


def test_rename_release_repoints_config_project_path(tmp_path):
    """DLC's read_config rewrites project_path to the config's own dir AND writes the
    file back when they differ -- on a 0444 config that is a crash or a checksum break."""
    _proj, root, rid = _build(tmp_path)
    new = rel.rename_release(rid, 'newname', root=root, reason='because')
    cfg = yaml.safe_load(open(os.path.join(root, new, 'config.yaml')))
    assert cfg['project_path'] == os.path.join(root, new)


def test_rename_release_refreezes_the_tree(tmp_path):
    _proj, root, rid = _build(tmp_path)
    new = rel.rename_release(rid, 'newname', root=root, reason='because')
    rdir = os.path.join(root, new)
    idx = next(x for x in rel._iter_release_files(rdir) if x.endswith('.index'))
    assert not (os.stat(os.path.join(rdir, idx)).st_mode & 0o222)
    assert not (os.stat(os.path.join(rdir, rel.MANIFEST_NAME)).st_mode & 0o222)
    assert not (os.stat(rdir).st_mode & 0o222)


def test_rename_release_refuses_to_reuse_an_existing_id(tmp_path):
    proj, root, rid = _build(tmp_path)
    rel.build_release(str(proj), 'newname', 30000, root=root)      # newname-r0001 exists
    with pytest.raises(FileExistsError) as e:
        rel.rename_release(rid, 'newname', root=root, reason='because')
    assert 'newname-r0001' in str(e.value)
    assert os.path.isdir(os.path.join(root, rid)), 'a refused rename must change nothing'


def test_rename_release_requires_a_reason(tmp_path):
    _proj, root, rid = _build(tmp_path)
    with pytest.raises(ValueError) as e:
        rel.rename_release(rid, 'newname', root=root)
    assert 'reason' in str(e.value)
    assert os.path.isdir(os.path.join(root, rid))


def test_rename_release_refuses_a_malformed_release_id(tmp_path):
    _proj, root, _rid = _build(tmp_path)
    with pytest.raises(ValueError):
        rel.rename_release('not-a-release-id', 'newname', root=root, reason='x')


def test_rename_release_refuses_an_unusable_alias(tmp_path):
    _proj, root, rid = _build(tmp_path)
    for bad in ('has space', 'has/slash', '1leading_digit', ''):
        with pytest.raises(ValueError):
            rel.rename_release(rid, bad, root=root, reason='x')
    assert os.path.isdir(os.path.join(root, rid))


def test_rename_release_refuses_a_release_that_does_not_verify(tmp_path):
    """Re-sealing damage would bless it into a fresh SHA256SUMS, permanently."""
    _proj, root, rid = _build(tmp_path)
    victim = os.path.join(root, rid, next(
        x for x in rel._iter_release_files(os.path.join(root, rid)) if x.endswith('.index')))
    os.chmod(victim, 0o644)
    open(victim, 'w').write('tampered')
    with pytest.raises(RuntimeError) as e:
        rel.rename_release(rid, 'newname', root=root, reason='because')
    assert 'does not verify' in str(e.value)
    assert os.path.isdir(os.path.join(root, rid))
    assert not os.path.exists(os.path.join(root, 'newname-r0001'))


def test_rename_release_dry_run_changes_nothing(tmp_path):
    _proj, root, rid = _build(tmp_path)
    assert rel.rename_release(rid, 'newname', root=root, reason='x',
                              dry_run=True) == 'newname-r0001'
    assert os.path.isdir(os.path.join(root, rid))
    assert not os.path.exists(os.path.join(root, 'newname-r0001'))
    assert rel.read_manifest(rid, root=root)['release']['alias'] == 'oldname'


def test_rename_release_restores_the_original_when_the_reseal_fails(monkeypatch, tmp_path):
    """A half-renamed release would verify clean against a manifest naming a directory
    it no longer lives in.  Failure must put it back, bytes included."""
    _proj, root, rid = _build(tmp_path)
    before = open(os.path.join(root, rid, rel.MANIFEST_NAME), 'rb').read()

    def _boom(rdir, manifest):
        raise OSError('disk went away')

    monkeypatch.setattr(rel, '_reseal', _boom)
    with pytest.raises(OSError):
        rel.rename_release(rid, 'newname', root=root, reason='because')
    assert os.path.isdir(os.path.join(root, rid))
    assert not os.path.exists(os.path.join(root, 'newname-r0001'))
    assert open(os.path.join(root, rid, rel.MANIFEST_NAME), 'rb').read() == before
    assert rel.verify(rid, full=True, root=root) == []


def test_next_release_id_follows_the_new_alias_after_a_rename(tmp_path):
    proj, root, rid = _build(tmp_path)
    rel.rename_release(rid, 'newname', root=root, reason='because')
    assert rel.next_release_id('newname', root=root) == 'newname-r0002'
    assert rel.next_release_id('oldname', root=root) == 'oldname-r0001', (
        'nothing is left under the old alias')
    assert rel.build_release(str(proj), 'newname', 30000, root=root) == 'newname-r0002'


def test_rename_release_keeps_every_other_manifest_field(tmp_path):
    _proj, root, rid = _build(tmp_path)
    before = rel.read_manifest(rid, root=root)
    new = rel.rename_release(rid, 'newname', root=root, reason='because')
    after = rel.read_manifest(new, root=root)
    for section in ('source', 'model', 'training', 'evaluation', 'curation'):
        assert after[section] == before[section], section
    for key in ('created_utc', 'created_by', 'host', 'note', 'git_hash'):
        assert after['release'][key] == before['release'][key], key
    # config.yaml really did change size, so the size table must have been rebuilt
    assert set(after['files']['size']) == set(before['files']['size'])
# [AGENT EDIT END]


# --- the registry writer must be surgical -------------------------------------------
# nems_lbhb/configs/dlc_registry.toml carries 61 comment lines transcribing the legacy
# if/elif condition each rule replaced.  They are the migration's audit trail.  A
# tomli_w parse-and-re-serialize round-trip would delete every one of them on the first
# promote, silently.  These three tests are the guard against that regression.

def _n_comment_lines(text):
    return sum(1 for line in text.splitlines() if line.lstrip().startswith('#'))


def _block_text(text, header):
    """The lines of the [header] block, from its header line to the next table header."""
    lines = text.splitlines()
    i = lines.index(header)
    j = i + 1
    while j < len(lines) and not lines[j].startswith('['):
        j += 1
    return '\n'.join(lines[i:j])


def test_promote_preserves_every_comment_in_the_registry(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    before = _n_comment_lines(src)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: consolidation task -- was
    # 61. The count is a canary against comments being DELETED, and it is deliberately
    # exact so that any change to it has to be argued for. The registry legitimately grew
    # by 17 comment lines: the `rules = []` blocks of `face` and the new
    # `right_chimney_headphones` alias are load-bearing (an empty rules list is what keeps
    # right_chimney_headphones off the router, where the legacy chain raises for SQD>20),
    # and that is exactly the kind of thing a future reader must not "tidy up". Bump this
    # number only when you have added comments on purpose | date: 2026-08-07]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference
    # rescaling -- was 78. Twenty comment lines were added ON PURPOSE: 10 in the header
    # documenting the new optional `train_frame_size` and the measurement that motivates it
    # (13 legacy Prince sites at 1280x720 vs 640x480 labelled frames, PRN043a05 0.334 ->
    # 0.848), 6 above [alias.pupil] and 4 above [alias.face] recording WHY those two aliases
    # deliberately do not declare one (their labelled images are variable-size close-up
    # crops, so the concept does not apply, and the default is inert for them). The seven
    # `train_frame_size = [640, 480]  # measured: ...` lines do not count here -- they are
    # trailing comments, not comment lines | date: 2026-08-08]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: per-session camera
    # overrides -- was 109. Sixteen comment lines were added ON PURPOSE, all above the new
    # [camera_override] section, recording the one thing that cannot be derived from the
    # data: psi names video files after the camera SLOT, so a USB port re-enumeration puts
    # the ceiling view in a file called video_side2 while recording.avi holds the face
    # camera. They also document why cam_tag matters (it selects the PSIVIDEO stream that
    # aligns every trial) and why verification must be by eye (frame counts and fps do not
    # identify a view). That reasoning is exactly what a future reader must not tidy away
    # | date: 2026-08-12]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: left-headphone project
    # rename + right-headphone promotion -- was 125. Ten comment lines were added ON
    # PURPOSE: 3 above [alias.left_chimney_headphones].project recording that the training
    # home was renamed from headphones_left_chimney_LB1_pokes ("pokes" named the
    # port-tracking models) and that its right-hemisphere folders were split out to the
    # dedicated right-headphone project, and 7 in the same vein for
    # right_chimney_headphones (r0002 is cut from the merged project and now serves the
    # alias). Both record WHY a directory name on disk no longer matches the scorer
    # embedded in older .h5 outputs -- exactly the kind of provenance a future reader
    # must not tidy away | date: 2026-08-23]
    assert before == 135, 'the registry should still carry its 135 audit-trail comments'
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp), skip_verify=True)
    after_text = regp.read_text()
    assert _n_comment_lines(after_text) == before
    # the specific legacy-condition comment above the promoted block must survive
    assert "legacy: elif 'SlipperyJack/SLJ' or 'LemonDisco/LMD' in avipath" in after_text
    # and so must the header comments and the ones below the edit
    assert '# ORDER IS SEMANTICS.' in after_text
    assert '# legacy: elif (\'StinkySquid/SQD\' & site<=20)' in after_text


def test_promote_does_not_touch_the_following_block(tmp_path):
    """two_chimney is not the last alias; the block after it must come through byte-identical."""
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    header = '[alias.right_chimney_pokes]'
    before = _block_text(src, header)
    rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp), skip_verify=True)
    after = _block_text(regp.read_text(), header)
    assert after == before
    import tomli
    # Compare against the SOURCE registry rather than a hardcoded '' -- the real file's
    # `current` fields are populated once aliases are promoted, and this test must keep
    # working regardless of what the live registry happens to say.
    before_r = tomli.loads(src)
    after_r = tomli.loads(regp.read_text())
    assert (after_r['alias']['right_chimney_pokes']
            == before_r['alias']['right_chimney_pokes'])
    assert ('history' in after_r['alias']['right_chimney_pokes']) == \
           ('history' in before_r['alias']['right_chimney_pokes'])


def test_promote_reparse_guard_fires_on_a_corrupted_write(monkeypatch, tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)

    def _sabotage(text, alias, current, history):
        return text.replace('[alias.two_chimney]', '[alias.two_chimney]\nthis is not toml =')

    monkeypatch.setattr(rel, '_edit_alias_block', _sabotage)
    with pytest.raises(RuntimeError) as e:
        rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp),
                    skip_verify=True)
    assert 'two_chimney' in str(e.value)
    assert regp.read_text() == src, 'a failed promote must leave the registry untouched'


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review minor finding --
# the re-parse guard test above only exercises the invalid-TOML branch; this exercises
# the other one -- syntactically valid TOML that just doesn't say what promote intended
# -- which the guard must catch just as loudly | date: 2026-08-06]
def test_promote_reparse_guard_fires_on_syntactically_valid_wrong_values(monkeypatch, tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    real_edit = rel._edit_alias_block

    def _sabotage(text, alias, current, history):
        # valid TOML, but silently writes a different release id than the one
        # promote was asked for -- nothing here is malformed, only wrong.
        return real_edit(text, alias, 'not-the-requested-release', history)

    monkeypatch.setattr(rel, '_edit_alias_block', _sabotage)
    with pytest.raises(RuntimeError) as e:
        rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp),
                    skip_verify=True)
    assert 'two_chimney' in str(e.value)
    assert regp.read_text() == src, 'a failed promote must leave the registry untouched'
# [AGENT EDIT END]


def test_promote_refuses_an_unknown_alias(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    with pytest.raises(ValueError) as e:
        rel.promote('no_such_alias', 'no_such_alias-r0001', registry_path=str(regp),
                    skip_verify=True)
    assert 'no_such_alias' in str(e.value)


def test_rollback_with_no_earlier_release_raises(tmp_path):
    src = open('nems_lbhb/configs/dlc_registry.toml').read()
    regp = tmp_path / 'reg.toml'
    regp.write_text(src)
    rel.promote('two_chimney', 'two_chimney-r0001', registry_path=str(regp), skip_verify=True)
    with pytest.raises(ValueError):
        rel.rollback('two_chimney', registry_path=str(regp))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- a release now
# records the curation entries logged since the alias's previous release, so what changed
# about the LABELS between two releases is answerable from the manifests | date: 2026-08-06]
def _log(proj, utc, note, **extra):
    """Append one curation_history.log entry with a chosen timestamp."""
    import json
    rec = {'utc': utc, 'user': 'jereme', 'note': note,
           'folders_added': [], 'folders_changed': [], 'frames_added': 0}
    rec.update(extra)
    with open(os.path.join(str(proj), 'curation_history.log'), 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')


def test_curation_section_reports_no_log_without_raising(tmp_path):
    proj = make_project(tmp_path)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=str(tmp_path / 'releases'))
    cur = rel.read_manifest(rid, root=str(tmp_path / 'releases'))['curation']
    assert cur['state'] == 'no_log'
    assert cur['log_present'] is False
    assert cur['entries'] == []
    assert cur['note']


def test_curation_section_takes_every_entry_when_there_is_no_previous_release(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    _log(proj, '2026-01-01T00:00:00Z', 'first labels', frames_added=100,
         folders_added=['v1'])
    _log(proj, '2026-02-01T00:00:00Z', 'more labels', frames_added=50,
         folders_added=['v2'])
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=root)
    cur = rel.read_manifest(rid, root=root)['curation']
    assert cur['state'] == 'no_previous_release'
    assert cur['previous_release'] == ''
    assert cur['entry_count'] == 2
    assert [e['note'] for e in cur['entries']] == ['first labels', 'more labels']
    assert cur['frames_added'] == 150
    assert sorted(cur['folders_added']) == ['v1', 'v2']


def test_curation_section_only_covers_entries_since_the_previous_release(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    _log(proj, '2020-01-01T00:00:00Z', 'ancient', frames_added=100)
    r1 = rel.build_release(str(proj), 'fake_alias', 30000, root=root)
    created = rel.read_manifest(r1, root=root)['release']['created_utc']

    _log(proj, '2099-01-01T00:00:00Z', 'after r1', frames_added=7,
         folders_added=['newsite'])
    r2 = rel.build_release(str(proj), 'fake_alias', 60000, root=root)
    cur = rel.read_manifest(r2, root=root)['curation']

    assert cur['state'] == 'since_previous_release'
    assert cur['previous_release'] == r1
    assert cur['previous_created_utc'] == created
    assert [e['note'] for e in cur['entries']] == ['after r1']
    assert cur['frames_added'] == 7
    assert cur['folders_added'] == ['newsite']
    # r1 itself saw the whole log, since nothing preceded it
    assert rel.read_manifest(r1, root=root)['curation']['entry_count'] == 1


def test_curation_section_survives_an_unreadable_log(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    with open(os.path.join(str(proj), 'curation_history.log'), 'w') as f:
        f.write('{not json\n')
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=root)
    man = rel.read_manifest(rid, root=root)
    assert man['curation']['state'] == 'log_unreadable'
    assert 'curation_history.log' in man['release']['provenance_warning']


def test_curation_section_omits_a_frame_total_it_cannot_know(tmp_path):
    proj = make_project(tmp_path)
    root = str(tmp_path / 'releases')
    _log(proj, '2026-01-01T00:00:00Z', 'pandas was missing', frames_added=None,
         frames_counted=False)
    rid = rel.build_release(str(proj), 'fake_alias', 60000, root=root)
    cur = rel.read_manifest(rid, root=root)['curation']
    assert cur['frames_counted'] is False
    assert 'frames_added' not in cur
    assert cur['entries'][0]['note'] == 'pandas was missing'
    assert 'frames_added' not in cur['entries'][0]      # None is not TOML-representable
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: promote() used to leave the
# registry 0600 (mkstemp's mode, carried across os.replace), so the first queued job after
# ANY promotion died with PermissionError reading it -- jobs run as `lbhb`, and every fit
# resolves its model through this file. The promotion "succeeded" and broke inference in
# the same call. | date: 2026-08-25]
def test_promote_preserves_registry_file_permissions(tmp_path):
    import os, stat as st
    regp = tmp_path / 'reg.toml'
    regp.write_text(open('nems_lbhb/configs/dlc_registry.toml').read())
    os.chmod(regp, 0o644)

    rel.promote('two_chimney', 'two_chimney-r0001',
                registry_path=str(regp), skip_verify=True)

    mode = st.S_IMODE(os.stat(regp).st_mode)
    assert mode == 0o644, 'promote changed the registry mode to %o' % mode
    assert mode & 0o044, 'registry unreadable by group/other -> queue jobs cannot resolve'


def test_write_atomic_defaults_new_files_to_readable(tmp_path):
    import os, stat as st
    p = tmp_path / 'new.toml'
    rel._write_atomic(str(p), 'x = 1\n')
    assert st.S_IMODE(os.stat(p).st_mode) == 0o644
    assert p.read_text() == 'x = 1\n'
# [AGENT EDIT END]
