"""Tests for the model-resolution and alias-writing helpers in dlc_fit_script.

These are the parts of the fit path that decide *which model ran* and *what the
results are called*, and they are the parts that must not need a GPU to check.
`dlc_fit_script` imports DeepLabCut inside its `__main__` block precisely so this
file can import it in psi_nems2, which has no DLC.  `analyze_videos` itself is out
of scope here -- an end-to-end run is a separate, manual check.

Every test drives a synthetic registry rather than
`nems_lbhb/configs/dlc_registry.toml`, so promoting or rolling back a real alias
cannot make them pass or fail.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- unit tests for
# registry/release resolution precedence and the two-alias naming | date: 2026-08-06]
import os

import pytest

from nems_lbhb.motor import dlc_fit_script as fs
from nems_lbhb.motor import dlc_registry


RELEASED_VIDEO = '/auto/data/daq/Animal/REL017/REL017a01_a_NTD.avi'
UNRELEASED_VIDEO = '/auto/data/daq/Animal/UNR017/UNR017a01_a_NTD.avi'
DEMOTED_VIDEO = '/auto/data/daq/Animal/DEM017/DEM017a01_a_NTD.avi'


def registry():
    """A synthetic registry with the three states an alias can be in."""
    aliases = {
        # a promoted release
        'released': {'project': '/proj/released', 'current': 'released-r0002',
                     'history': ['released-r0001', 'released-r0002'],
                     'rules': [{'path_contains': 'REL'}]},
        # never released: nothing to run from but the project
        'unreleased': {'project': '/proj/unreleased', 'current': '',
                       'history': [],
                       'rules': [{'path_contains': 'UNR'}]},
        # releases exist but none is promoted -- the dangerous state
        'demoted': {'project': '/proj/demoted', 'current': '',
                    'history': ['demoted-r0001'],
                    'rules': [{'path_contains': 'DEM'}]},
    }
    return {'_path': '/fake/dlc_registry.toml',
            'schema_version': 1,
            'training_defaults': {'net_type': 'resnet_50', 'augmenter_type': 'imgaug',
                                  'max_snapshots_to_keep': 6, 'saveiters': 10000,
                                  'maxiters': 60000, 'displayiters': 100,
                                  'autotune': False, 'allow_growth': True},
            'aliases': aliases,
            'alias_names': list(aliases)}


# ---------------------------------------------------------------------------
# resolution precedence
# ---------------------------------------------------------------------------

def test_explicit_config_outranks_alias_and_video_rules():
    r = fs.resolve_model(RELEASED_VIDEO, explicit_config='/somewhere/config.yaml',
                         alias='unreleased', registry=registry())
    assert r['config'] == '/somewhere/config.yaml'
    assert r['source'] == 'explicit'
    assert r['release_id'] is None


def test_alias_outranks_video_rules():
    r = fs.resolve_model(RELEASED_VIDEO, alias='unreleased', registry=registry())
    assert r['alias'] == 'unreleased'
    assert r['config'] == os.path.join('/proj/unreleased', 'config.yaml')


def test_video_rules_are_the_fallback():
    r = fs.resolve_model(RELEASED_VIDEO, registry=registry())
    assert r['alias'] == 'released'


def test_a_promoted_alias_resolves_to_its_release():
    r = fs.resolve_model(RELEASED_VIDEO, registry=registry())
    assert r['source'] == 'release'
    assert r['release_id'] == 'released-r0002'
    assert r['config'] == os.path.join(dlc_registry.RELEASE_ROOT,
                                       'released-r0002', 'config.yaml')


def test_an_alias_with_no_release_still_runs_from_its_project():
    r = fs.resolve_model(UNRELEASED_VIDEO, registry=registry())
    assert r['source'] == 'project'
    assert r['release_id'] is None
    assert r['config'] == os.path.join('/proj/unreleased', 'config.yaml')


def test_an_unpromoted_release_refuses_to_fall_back_to_the_project():
    with pytest.raises(RuntimeError) as e:
        fs.resolve_model(DEMOTED_VIDEO, registry=registry())
    msg = str(e.value)
    assert 'demoted-r0001' in msg, 'the refusal must name the release'
    assert 'allow_project=1' in msg, 'the refusal must state the opt-in'


def test_allow_project_is_the_opt_in_out_of_a_release():
    r = fs.resolve_model(RELEASED_VIDEO, allow_project=True, registry=registry())
    assert r['source'] == 'project'
    assert r['release_id'] is None
    assert r['config'] == os.path.join('/proj/released', 'config.yaml')

    # and it unblocks the unpromoted-release case
    r2 = fs.resolve_model(DEMOTED_VIDEO, allow_project=True, registry=registry())
    assert r2['config'] == os.path.join('/proj/demoted', 'config.yaml')


def test_unknown_alias_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        fs.resolve_model(RELEASED_VIDEO, alias='no_such_model', registry=registry())


def test_explicit_config_inside_the_release_root_is_still_a_release(tmp_path):
    root = tmp_path / 'releases'
    rdir = root / 'thing-r0001'
    rdir.mkdir(parents=True)
    (rdir / 'RELEASE.toml').write_text('[release]\nid = "thing-r0001"\n')
    cfg = str(rdir / 'config.yaml')
    assert fs.release_id_of_config(cfg, root=str(root)) == 'thing-r0001'
    # a project config is not
    assert fs.release_id_of_config(str(tmp_path / 'proj' / 'config.yaml'),
                                   root=str(root)) is None
    # nor is a directory under the root with no manifest
    (root / 'junk').mkdir()
    assert fs.release_id_of_config(str(root / 'junk' / 'config.yaml'),
                                   root=str(root)) is None


def test_truthy_refuses_ambiguous_flags():
    assert fs.truthy('1') and fs.truthy('true') and fs.truthy('YES')
    assert not fs.truthy(None) and not fs.truthy('') and not fs.truthy('0')
    with pytest.raises(ValueError):
        fs.truthy('maybe')


# ---------------------------------------------------------------------------
# the two aliases
# ---------------------------------------------------------------------------

STEM = 'SLJ017a07_a_NTD'
SUFFIX = 'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'


def _make_output(tmp_path, suffix=SUFFIX):
    out = tmp_path / (STEM + suffix + '.h5')
    out.write_text('h5')
    return str(out)


# ---------------------------------------------------------------------------
# [AGENT EDIT START | agent: claude | user: wingertj | reason: the recording block gated
# only on release_id, so a "analyze to temp" run with an alias would have written a
# gDLCFit row pointing into scratch | date: 2026-08-14]
# which destinations may be recorded in gDLCFit
# ---------------------------------------------------------------------------

def test_only_a_sorted_directory_is_a_durable_destination():
    assert fs.is_durable_dest('/auto/data/daq/Reishi/REI059/sorted')
    assert fs.is_durable_dest('/auto/data/daq/Reishi/REI059/sorted/')      # trailing slash
    # what the GUI's "analyze to temp" passes as destfolder
    assert not fs.is_durable_dest('/auto/data/daq/Reishi/REI059/tmp/dlc/REI059a03_a_NFB')
    # a 'sorted' further up the path is not the destination
    assert not fs.is_durable_dest('/auto/data/daq/Reishi/REI059/sorted/left_chimney-r0001')


def test_no_destination_is_not_durable():
    for empty in (None, ''):
        assert not fs.is_durable_dest(empty)


def test_temp_and_sorted_are_distinguished_for_the_same_session(tmp_path):
    """The exact pair the GUI produces: same session, two destinations."""
    sorted_dir = tmp_path / 'REI059' / 'sorted'
    temp_dir = tmp_path / 'REI059' / 'tmp' / 'dlc' / 'REI059a03_a_NFB'
    sorted_dir.mkdir(parents=True)
    temp_dir.mkdir(parents=True)
    assert fs.is_durable_dest(str(sorted_dir))
    assert not fs.is_durable_dest(str(temp_dir))


# ---------------------------------------------------------------------------
# publishing a vetted preview instead of re-running the GPU
# ---------------------------------------------------------------------------

RELEASE = 'two_chimney-r0001'


def _make_preview(tmp_path, release_id=RELEASE, stem=STEM, suffix=SUFFIX,
                  stamped=True, companions=True):
    """A temp dir shaped exactly as a release-mode fit leaves it."""
    src = tmp_path / 'tmp' / 'dlc' / stem
    out = src / release_id
    out.mkdir(parents=True)
    target = out / (stem + suffix + '.h5')
    target.write_text('pose')
    if companions:
        (out / (stem + suffix + '_meta.pickle')).write_text('meta')
        (src / ('%s.%s.scale.json' % (stem, release_id))).write_text('{}')
    if stamped:
        fs.write_aliases(str(target), str(src), stem, release_id=release_id)
    return src, target


def test_a_stamped_preview_is_publishable(tmp_path):
    src, target = _make_preview(tmp_path)
    found = fs.publishable_preview(str(src), STEM, RELEASE)
    assert found == os.path.realpath(str(target))


def test_an_unstamped_preview_is_not_publishable(tmp_path):
    """Project-mode output records no network, so it can never become production."""
    src, _target = _make_preview(tmp_path, stamped=False)
    assert fs.publishable_preview(str(src), STEM, RELEASE) is None


def test_a_preview_from_another_release_is_not_publishable(tmp_path):
    """The substitution this whole project exists to prevent."""
    src, _target = _make_preview(tmp_path, release_id='two_chimney-r0002')
    assert fs.publishable_preview(str(src), STEM, RELEASE) is None
    assert fs.publishable_preview(str(src), STEM, 'two_chimney-r0002') is not None


def test_a_cleaned_scratch_dir_is_not_publishable(tmp_path):
    """Scratch is deletable: the alias survives as a dangling link, the pose does not."""
    src, target = _make_preview(tmp_path)
    os.remove(str(target))
    assert fs.publishable_preview(str(src), STEM, RELEASE) is None


def test_publish_copies_the_pose_its_companions_and_writes_both_aliases(tmp_path):
    src, src_target = _make_preview(tmp_path)
    dest = tmp_path / 'sorted'
    dest.mkdir()

    target, written = fs.publish_preview(str(src_target), str(dest), STEM, RELEASE)

    assert os.path.isfile(target)
    assert os.path.dirname(target) == str(dest / RELEASE)     # release-scoped subdir
    assert [os.path.basename(w) for w in written] == [
        '%s.%s.dlc.h5' % (STEM, RELEASE), '%s.dlc.h5' % STEM]
    for w in written:
        assert os.path.realpath(w) == os.path.realpath(target)
    assert os.path.isfile(target[:-len('.h5')] + '_meta.pickle')
    # the sidecar is the only evidence an oversized video was analysed rescaled
    assert os.path.isfile(str(dest / ('%s.%s.scale.json' % (STEM, RELEASE))))


def test_publish_preserves_mtime_so_the_row_key_dates_the_analysis(tmp_path):
    src, src_target = _make_preview(tmp_path)
    dest = tmp_path / 'sorted'
    dest.mkdir()
    old = os.path.getmtime(str(src_target))
    os.utime(str(src_target), (old - 86400, old - 86400))

    target, _written = fs.publish_preview(str(src_target), str(dest), STEM, RELEASE)
    assert abs(os.path.getmtime(target) - (old - 86400)) < 2


def test_publish_refuses_to_clobber_a_real_file_at_an_alias_path(tmp_path):
    src, src_target = _make_preview(tmp_path)
    dest = tmp_path / 'sorted'
    dest.mkdir()
    (dest / ('%s.dlc.h5' % STEM)).write_text('someone else real data')

    with pytest.raises((FileExistsError, RuntimeError)):
        fs.publish_preview(str(src_target), str(dest), STEM, RELEASE)


def test_the_publish_decision_is_a_pure_function_of_the_preview(tmp_path):
    """What the sorted button asks and what the job re-checks are the same question.

    The GUI decides which route to queue; the job re-runs the identical check because
    scratch can be cleaned in between. Both call this, so they cannot disagree.
    """
    src, target = _make_preview(tmp_path)
    assert fs.publishable_preview(str(src), STEM, RELEASE)    # button offers the copy
    os.remove(str(target))                                    # scratch cleaned meanwhile
    assert fs.publishable_preview(str(src), STEM, RELEASE) is None   # job falls back


def test_publish_refuses_to_publish_onto_itself(tmp_path):
    """dest == src would copy a file over itself and then alias it as new."""
    src, src_target = _make_preview(tmp_path)
    with pytest.raises(RuntimeError, match='onto itself'):
        fs.publish_preview(str(src_target), str(src), STEM, RELEASE)


# [AGENT EDIT END]


def test_alias_paths_names_both_links():
    stamped, plain = fs.alias_paths('/dest', STEM, 'two_chimney-r0001')
    assert stamped == '/dest/SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'
    assert plain == '/dest/SLJ017a07_a_NTD.dlc.h5'
    assert fs.alias_paths('/dest', STEM, None)[0] is None


def test_write_aliases_writes_both_and_both_resolve(tmp_path):
    target = _make_output(tmp_path)
    written = fs.write_aliases(target, str(tmp_path), STEM,
                               release_id='two_chimney-r0001')
    assert [os.path.basename(w) for w in written] == [
        'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5', 'SLJ017a07_a_NTD.dlc.h5']
    for w in written:
        assert os.path.islink(w)
        assert os.path.realpath(w) == os.path.realpath(target)
        assert not os.path.isabs(os.readlink(w)), 'aliases must be relative'


def test_write_aliases_without_a_release_writes_only_the_plain_alias(tmp_path):
    target = _make_output(tmp_path)
    written = fs.write_aliases(target, str(tmp_path), STEM, release_id=None)
    assert [os.path.basename(w) for w in written] == ['SLJ017a07_a_NTD.dlc.h5']
    assert not any(n.count('.dlc.h5') and n != 'SLJ017a07_a_NTD.dlc.h5'
                   for n in os.listdir(str(tmp_path)))


def test_a_second_release_does_not_disturb_the_first_stamped_alias(tmp_path):
    first = _make_output(tmp_path)
    fs.write_aliases(first, str(tmp_path), STEM, release_id='two_chimney-r0001')

    second_suffix = 'DLC_resnet50_two_chimney3fbJan10shuffle1_90000'
    second = _make_output(tmp_path, suffix=second_suffix)
    fs.write_aliases(second, str(tmp_path), STEM, release_id='two_chimney-r0002')

    stamped1 = str(tmp_path / ('%s.two_chimney-r0001.dlc.h5' % STEM))
    stamped2 = str(tmp_path / ('%s.two_chimney-r0002.dlc.h5' % STEM))
    plain = str(tmp_path / ('%s.dlc.h5' % STEM))
    assert os.path.realpath(stamped1) == os.path.realpath(first)
    assert os.path.realpath(stamped2) == os.path.realpath(second)
    assert os.path.realpath(plain) == os.path.realpath(second), 'current follows the newest run'


def test_write_aliases_refuses_to_clobber_real_data(tmp_path):
    target = _make_output(tmp_path)
    (tmp_path / ('%s.dlc.h5' % STEM)).write_text('this is a real file, not a link')
    with pytest.raises(FileExistsError):
        fs.write_aliases(target, str(tmp_path), STEM, release_id='two_chimney-r0001')


def test_stamped_collision_between_two_releases_is_refused(tmp_path):
    # two releases that share an output_suffix: r0001 already stamped, r0003 about
    # to overwrite the very file r0001's stamp points at.
    target = _make_output(tmp_path)
    fs.write_aliases(target, str(tmp_path), STEM, release_id='hp-r0001')
    with pytest.raises(RuntimeError) as e:
        fs.assert_no_stamped_collision(str(tmp_path), STEM, 'hp-r0003', SUFFIX)
    assert 'hp-r0001' in str(e.value) and 'hp-r0003' in str(e.value)


def test_no_collision_when_the_suffixes_differ(tmp_path):
    target = _make_output(tmp_path)
    fs.write_aliases(target, str(tmp_path), STEM, release_id='hp-r0001')
    fs.assert_no_stamped_collision(str(tmp_path), STEM, 'hp-r0003',
                                   'DLC_resnet50_otherJan10shuffle1_60000')
    # and re-running the same release is not a collision with itself
    fs.assert_no_stamped_collision(str(tmp_path), STEM, 'hp-r0001', SUFFIX)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 1 -- the
# collision guard used to be called only inside `if release_id is not None:`, so a
# project-mode or explicit-config fit (which writes the SAME scorer-named .h5, since
# the release was cut from that very project) could silently overwrite the file a
# stamped alias already points at. This reproduces that corruption directly against
# assert_no_stamped_collision with release_id=None (what the project/explicit-config
# path now passes), and asserts it is refused. | date: 2026-08-06]
def test_project_mode_write_is_refused_when_it_collides_with_a_stamped_release(tmp_path):
    # a prior release run stamped this session ...
    target = _make_output(tmp_path)
    fs.write_aliases(target, str(tmp_path), STEM, release_id='two_chimney-r0001')
    # ... and now a project-mode / explicit-config run (no release_id of its own)
    # would produce the identical output filename -- exactly what happens when the
    # project is the one the release was cut from.
    with pytest.raises(RuntimeError) as e:
        fs.assert_no_stamped_collision(str(tmp_path), STEM, None, SUFFIX)
    msg = str(e.value)
    assert 'two_chimney-r0001' in msg, 'the refusal must name the release whose stamp is at risk'


def test_project_mode_write_is_fine_when_nothing_is_stamped_yet(tmp_path):
    # no prior stamped alias -- an unstamped project/explicit-config run must not be
    # refused just because it has no release_id of its own.
    _make_output(tmp_path)
    fs.assert_no_stamped_collision(str(tmp_path), STEM, None, SUFFIX)
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# alias-writability preflight (review fix 8)
# ---------------------------------------------------------------------------

# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 8 --
# write_aliases could previously raise FileExistsError on the PLAIN alias after the
# stamped alias was already written and after analyze_videos (GPU work) had already
# run, leaving partial state. assert_aliases_writable checks both alias paths up
# front so this fails cheaply, before analyze_videos. | date: 2026-08-06]
def test_assert_aliases_writable_passes_when_paths_are_free_or_are_links(tmp_path):
    fs.assert_aliases_writable(str(tmp_path), STEM, release_id='two_chimney-r0001')
    target = _make_output(tmp_path)
    fs.write_aliases(target, str(tmp_path), STEM, release_id='two_chimney-r0001')
    # re-running: both paths are now symlinks, not blocking regular files
    fs.assert_aliases_writable(str(tmp_path), STEM, release_id='two_chimney-r0001')


def test_assert_aliases_writable_refuses_a_real_file_at_the_plain_alias(tmp_path):
    (tmp_path / ('%s.dlc.h5' % STEM)).write_text('real data, not a link')
    with pytest.raises(FileExistsError):
        fs.assert_aliases_writable(str(tmp_path), STEM, release_id='two_chimney-r0001')


def test_assert_aliases_writable_refuses_a_real_file_at_the_stamped_alias(tmp_path):
    (tmp_path / ('%s.two_chimney-r0001.dlc.h5' % STEM)).write_text('real data, not a link')
    with pytest.raises(FileExistsError):
        fs.assert_aliases_writable(str(tmp_path), STEM, release_id='two_chimney-r0001')
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# camera override: aliases named after the SESSION, target keeps the cam tag
# ---------------------------------------------------------------------------

# [AGENT EDIT START | agent: claude | user: wingertj | reason: under a registry camera
# override the analysed video is <session>_camC.avi, so the aliases used to land at
# <session>_camC.dlc.h5 -- but baphy_experiment.dlcfile asks for <session>.dlc.h5, so
# the session had no tracking at all as far as any loader was concerned | date: 2026-08-12]
CAM_STEM = STEM + '_camC'          # what DLC analysed, after the override redirect


def _make_camC_output(tmp_path, suffix=SUFFIX):
    out = tmp_path / (CAM_STEM + suffix + '.h5')
    out.write_text('h5')
    return str(out)


def test_camera_override_names_aliases_after_the_session_not_the_camera(tmp_path):
    target = _make_camC_output(tmp_path)
    written = fs.write_aliases(target, str(tmp_path), CAM_STEM,
                               release_id='two_chimney-r0001', alias_stem=STEM)
    # the names every loader asks for ...
    assert [os.path.basename(w) for w in written] == [
        'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5', 'SLJ017a07_a_NTD.dlc.h5']
    for w in written:
        # ... resolving to a target that KEEPS its camC name, which is what
        # baphy_io.load_dlc_trace reads to align against PSIVIDEO_camC
        assert '_camC' in os.path.basename(os.path.realpath(w))
        assert os.path.realpath(w) == os.path.realpath(target)
    assert not os.path.exists(str(tmp_path / (CAM_STEM + '.dlc.h5'))), \
        'nothing should be written under the camera-tagged stem'


def test_alias_stem_defaults_to_the_analysed_stem(tmp_path):
    # a genuine camC analysis -- tracking the side camera on purpose, no override --
    # must keep its camC alias name.  alias_stem=None is that case.
    target = _make_camC_output(tmp_path)
    written = fs.write_aliases(target, str(tmp_path), CAM_STEM,
                               release_id='two_chimney-r0001')
    assert [os.path.basename(w) for w in written] == [
        'SLJ017a07_a_NTD_camC.two_chimney-r0001.dlc.h5', 'SLJ017a07_a_NTD_camC.dlc.h5']


def test_override_preflight_guards_the_paths_that_will_actually_be_written(tmp_path):
    # a real file sitting at the SESSION-named alias must be caught; the check is
    # worthless if it guards <session>_camC.dlc.h5 while write_aliases writes
    # <session>.dlc.h5.
    (tmp_path / ('%s.dlc.h5' % STEM)).write_text('real data, not a link')
    with pytest.raises(FileExistsError):
        fs.assert_aliases_writable(str(tmp_path), CAM_STEM,
                                   release_id='two_chimney-r0001', alias_stem=STEM)


def test_override_collision_guard_scans_session_named_stamps(tmp_path):
    # r0001 stamped this session (under its SESSION name, as an override run does),
    # and an r0003 override run would write the identical camC output filename.
    target = _make_camC_output(tmp_path)
    fs.write_aliases(target, str(tmp_path), CAM_STEM, release_id='hp-r0001',
                     alias_stem=STEM)
    with pytest.raises(RuntimeError) as e:
        fs.assert_no_stamped_collision(str(tmp_path), CAM_STEM, 'hp-r0003', SUFFIX,
                                       alias_stem=STEM)
    assert 'hp-r0001' in str(e.value)
    # re-running the same release is not a collision with itself
    fs.assert_no_stamped_collision(str(tmp_path), CAM_STEM, 'hp-r0001', SUFFIX,
                                   alias_stem=STEM)
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# release verification / transfer-learning refusals (review fix 2)
# ---------------------------------------------------------------------------

# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 2 -- two of
# the highest-risk refusals had no test coverage: verify_resolved_release aborting a
# run over a failed release verification, and release_init_weights refusing a
# net_type mismatch. Both are exercised here by monkeypatching dlc_release rather than
# touching /auto/data/dlc/releases. | date: 2026-08-06]
def test_verify_resolved_release_aborts_on_reported_problems(monkeypatch):
    problems = ['weights.data-00000-of-00001 is missing',
                'pose_cfg.yaml size mismatch (expected 4096, got 12)']
    monkeypatch.setattr(fs.dlc_release, 'verify',
                        lambda release_id, full=False: list(problems))
    with pytest.raises(RuntimeError) as e:
        fs.verify_resolved_release({'source': 'release', 'release_id': 'two_chimney-r0001'})
    msg = str(e.value)
    assert 'two_chimney-r0001' in msg
    for p in problems:
        assert p in msg


def test_verify_resolved_release_is_a_no_op_off_the_release_path():
    # project/explicit mode has nothing to verify against
    assert fs.verify_resolved_release({'source': 'project', 'release_id': None}) == []
    assert fs.verify_resolved_release({'source': 'explicit', 'release_id': None}) == []


def test_release_init_weights_refuses_a_net_type_mismatch(monkeypatch):
    manifest = {'model': {'net_type': 'mobilenet_v2_1.0'},
                'source': {'snapshot': 60000, 'shuffle': 1}}
    monkeypatch.setattr(fs.dlc_release, 'read_manifest', lambda release_id: manifest)
    with pytest.raises(RuntimeError) as e:
        fs.release_init_weights('some-release-r0001', expect_net_type='resnet_50')
    msg = str(e.value)
    assert 'mobilenet_v2_1.0' in msg and 'resnet_50' in msg


def test_release_output_suffix_reads_the_manifest(monkeypatch):
    monkeypatch.setattr(
        fs.dlc_release, 'read_manifest',
        lambda release_id: {'model': {'output_suffix': SUFFIX}})
    assert fs.release_output_suffix('two_chimney-r0001') == SUFFIX


def test_release_output_suffix_refuses_a_manifest_missing_the_suffix(monkeypatch):
    monkeypatch.setattr(fs.dlc_release, 'read_manifest', lambda release_id: {'model': {}})
    with pytest.raises(RuntimeError) as e:
        fs.release_output_suffix('two_chimney-r0001')
    assert 'two_chimney-r0001' in str(e.value)


def test_project_iteration_reads_the_configs_iteration_field(tmp_path):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('iteration: 3\nTask: two_chimney\nother: stuff\n')
    assert fs.project_iteration(str(cfg)) == 3
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# training hyperparameters
# ---------------------------------------------------------------------------

def test_training_hyperparams_come_from_the_registry():
    hp = fs.training_hyperparams(registry())
    assert hp['net_type'] == 'resnet_50' and hp['maxiters'] == 60000


def test_training_hyperparams_refuse_to_default_a_missing_key():
    reg = registry()
    del reg['training_defaults']['net_type']
    with pytest.raises(KeyError) as e:
        fs.training_hyperparams(reg)
    assert 'net_type' in str(e.value)


def test_the_real_registry_supplies_every_required_hyperparameter():
    hp = fs.training_hyperparams()
    assert set(fs._REQUIRED_TRAINING_KEYS) <= set(hp)


def test_trainrun_renders_parseable_toml(tmp_path):
    path = fs.write_trainrun(str(tmp_path), [
        ('run', {'user': 'jereme', 'queueid': 7, 'resume': False,
                 'note': 'a "quoted" path\\here'}),
        ('hyperparameters', {'maxiters': 60000, 'multi_step': [1, 2]}),
    ])
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, 'rb') as f:
        got = tomllib.load(f)
    assert got['run']['user'] == 'jereme' and got['run']['queueid'] == 7
    assert got['run']['resume'] is False
    assert got['run']['note'] == 'a "quoted" path\\here'
    assert got['hyperparameters']['multi_step'] == [1, 2]
# [AGENT EDIT END]


# --- per-release output directory -------------------------------------------------
# A scorer string omits the ITERATION, so two releases of one project can emit
# byte-identical filenames from different weights (two_chimney-r0001 md5 3386f312 vs
# -r0002 md5 55f0fc27, both DLC_resnet50_two_chimney3fbJan10shuffle1_60000). Giving each
# release its own directory makes the collision structurally impossible.

def test_release_output_dir_isolates_releases(tmp_path):
    a = fs.release_output_dir(str(tmp_path), 'two_chimney-r0001')
    b = fs.release_output_dir(str(tmp_path), 'two_chimney-r0002')
    assert a != b
    assert os.path.isdir(a) and os.path.isdir(b)
    assert os.path.basename(a) == 'two_chimney-r0001'


def test_release_output_dir_is_dest_dir_without_a_release(tmp_path):
    """Project mode keeps writing straight into sorted/, so the 552 existing files
    keep their layout."""
    assert fs.release_output_dir(str(tmp_path), None) == str(tmp_path)
    assert fs.release_output_dir(str(tmp_path), '') == str(tmp_path)


def test_release_output_dir_does_not_create_when_asked_not_to(tmp_path):
    p = fs.release_output_dir(str(tmp_path), 'x-r0001', create=False)
    assert not os.path.exists(p)


def test_two_releases_sharing_a_scorer_do_not_collide(tmp_path):
    """The actual defect: identical filenames, different weights, both stamped."""
    suffix = 'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'
    stem = 'SLJ026a02_a_NTD'
    sorted_dir = tmp_path / 'sorted'
    sorted_dir.mkdir()
    targets = {}
    for rid, payload in (('two_chimney-r0001', 'weights-A'), ('two_chimney-r0002', 'weights-B')):
        out = fs.release_output_dir(str(sorted_dir), rid)
        t = os.path.join(out, stem + suffix + '.h5')
        with open(t, 'w') as f:
            f.write(payload)
        fs.write_aliases(t, str(sorted_dir), stem, release_id=rid)
        targets[rid] = t
    # both outputs survive, and each stamped alias still resolves to its OWN release
    for rid, payload in (('two_chimney-r0001', 'weights-A'), ('two_chimney-r0002', 'weights-B')):
        stamped = sorted_dir / ('%s.%s.dlc.h5' % (stem, rid))
        assert open(os.path.realpath(str(stamped))).read() == payload, rid
    # the plain alias points at whichever ran last
    plain = sorted_dir / ('%s.dlc.h5' % stem)
    assert open(os.path.realpath(str(plain))).read() == 'weights-B'


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- `curate` writes
# into the project (merge_datasets rewrites config.yaml and the labeled-data folders), so
# it has to refuse a frozen release for the same reason `training` does; the refusal is a
# module-level function now, which is what makes it checkable without DeepLabCut
# | date: 2026-08-06]
@pytest.mark.parametrize('action', ['curate', 'training'])
def test_a_frozen_release_refuses_both_writing_actions(action):
    resolved = {'alias': 'released', 'release_id': 'released-r0002',
                'config': '/releases/released-r0002/config.yaml',
                'project': '/proj/released', 'source': 'release'}
    with pytest.raises(RuntimeError) as e:
        fs.assert_not_a_release(resolved, action)
    msg = str(e.value)
    assert action in msg and 'released-r0002' in msg
    assert '/proj/released' in msg, 'the refusal must name the writable project'


def test_assert_not_a_release_passes_a_project_or_explicit_config():
    for source in ('project', 'explicit'):
        fs.assert_not_a_release({'source': source, 'alias': 'x', 'release_id': None}, 'curate')


def test_release_refusal_names_the_project_even_without_a_project_key(monkeypatch):
    """resolve_model's 'explicit config path inside RELEASE_ROOT' branch leaves
    project=None; the hint has to come from the registry instead."""
    reg = registry()
    monkeypatch.setattr(dlc_registry, 'load_registry', lambda *a, **k: reg)
    resolved = {'alias': 'released', 'release_id': 'released-r0002',
                'config': '/releases/released-r0002/config.yaml',
                'project': None, 'source': 'release'}
    assert fs.alias_project_hint(resolved) == '/proj/released'
    with pytest.raises(RuntimeError) as e:
        fs.assert_not_a_release(resolved, 'curate')
    assert '/proj/released' in str(e.value)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the session-naming rename ran
# only when no destfolder was given, so "analyse to temp" wrote recordingDLC_<scorer>.h5 --
# a name shared by every session in the lab -- while the same run into sorted/ wrote
# <session>DLC_<scorer>.h5. publishable_preview looks up the session name, so the
# publish-from-temp reuse could never fire on a preview it had just produced.
# | date: 2026-08-17]
def test_a_psi_slot_is_linked_under_the_session_name(tmp_path):
    sess = tmp_path / 'REI117a09_a_NFC'
    sess.mkdir()
    (sess / 'recording.avi').write_bytes(b'x')

    linked = fs.session_named_video(str(sess / 'recording.avi'))

    assert linked == str(tmp_path / 'REI117a09_a_NFC.avi')
    assert os.path.islink(linked)
    assert os.path.realpath(linked) == str(sess / 'recording.avi')


def test_side_slots_keep_their_stream_suffix(tmp_path):
    # the suffix is read back by baphy_io.load_dlc_trace to pick the PSIVIDEO stream that
    # aligns the trace, so it is an interface rather than decoration
    sess = tmp_path / 'REI084a10_a_NFB'
    sess.mkdir()
    for slot, expect in (('video_side.avi', '_camB'),
                         ('video_side2.avi', '_camC'),
                         ('video_pupil.avi', '_pupil')):
        (sess / slot).write_bytes(b'x')
        linked = fs.session_named_video(str(sess / slot))
        assert linked == str(tmp_path / f'REI084a10_a_NFB{expect}.avi')


def test_linking_a_slot_twice_reuses_the_existing_link(tmp_path):
    sess = tmp_path / 'REI117a09_a_NFC'
    sess.mkdir()
    (sess / 'recording.avi').write_bytes(b'x')

    first = fs.session_named_video(str(sess / 'recording.avi'))
    before = os.lstat(first).st_ino
    second = fs.session_named_video(str(sess / 'recording.avi'))

    assert first == second
    assert os.lstat(second).st_ino == before          # not recreated


def test_an_already_session_named_video_passes_through(tmp_path):
    # callers hand in the parent-level <session>.avi link too; it must not be re-wrapped
    vid = tmp_path / 'REI117a09_a_NFC.avi'
    vid.write_bytes(b'x')
    assert fs.session_named_video(str(vid)) == str(vid)


def test_a_missing_non_slot_video_is_refused(tmp_path):
    with pytest.raises(ValueError) as e:
        fs.session_named_video(str(tmp_path / 'nope.avi'))
    assert 'not found' in str(e.value)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: geneval queue action -- the
# script body runs under __main__ so the branch is pinned by source order: it must fire
# AFTER release verification (a candidate release gets checksum-checked before scoring)
# and BEFORE the per-video destination machinery (argv[1] is a directory, not a session
# video, and must never reach session_named_video). | date: 2026-08-18]
def test_geneval_branch_sits_between_verification_and_video_machinery():
    import inspect
    src = inspect.getsource(fs)
    i_verify = src.index('verify_resolved_release(resolved)')
    i_geneval = src.index("if 'geneval' in action_list:")
    i_dest = src.index('if dest_path is None:')
    assert i_verify < i_geneval < i_dest
    block = src[i_geneval:i_dest]
    assert 'evaluate_candidate(' in block
    assert 'sys.exit(0)' in block                 # never falls through to fit machinery
    assert 'update_job_complete' in block         # the queue row still closes
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: evaluate queue action --
# scoring every snapshot requires flipping snapshotindex to 'all' in the LIVE project
# config. If a crash leaves it there, every later inference on that project breaks
# (todo.txt item 8), so the restore must sit in a finally:, not after the call.
# | date: 2026-08-23]
def test_evaluate_restores_snapshotindex_in_a_finally():
    import inspect
    src = inspect.getsource(fs)
    i_eval = src.index("if 'evaluate' in action_list:")
    i_curate = src.index("if 'curate' in action_list:")
    assert i_eval < i_curate
    block = src[i_eval:i_curate]

    assert 'assert_not_a_release(resolved, ' in block   # never evaluates a frozen release
    assert "'all'" in block                             # scores every snapshot
    assert 'evaluate_network(' in block

    # the restore is inside a finally:, and the try opens BEFORE evaluate_network
    i_try = block.index('try:')
    i_call = block.index('evaluate_network(')
    i_finally = block.index('finally:')
    assert i_try < i_call < i_finally
    assert '_orig_index' in block[i_finally:], 'original snapshotindex not restored'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: extract/refine create
# labeled-data folders as the QUEUE account, and a 0755 folder is unwritable by the human
# who labels next -- napari then dies with PermissionError and only root can undo it
# (chmod/rename on a directory needs write access ON that directory). | date: 2026-08-23]
def test_extract_and_refine_open_up_the_folders_they_create():
    import inspect
    src = inspect.getsource(fs)
    for action in ("extract", "refine"):
        i = src.index("if '%s' in action_list:" % action)
        block = src[i:i + 900]
        assert 'open_up_project_paths(' in block, action


def test_open_up_project_paths_is_best_effort_and_world_writable(tmp_path):
    import os
    proj = tmp_path / 'proj'
    d = proj / 'labeled-data' / 'SESSION'
    d.mkdir(parents=True)
    png = d / 'img000.png'
    png.write_bytes(b'x')
    os.chmod(d, 0o755)
    os.chmod(png, 0o644)

    fs.open_up_project_paths(str(proj))

    assert os.stat(d).st_mode & 0o777 == 0o777
    assert os.stat(png).st_mode & 0o777 == 0o666
    # a missing project must not raise -- this runs at the end of a job that succeeded
    fs.open_up_project_paths(str(tmp_path / 'nope'))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: regress session-named predictions, release isolation, metadata and video-worker destination | date: 2026-09-10]
@pytest.mark.parametrize('name,stem', [
    ('recording.avi', 'SQD092b02_a_NFC'),
    ('video_side.avi', 'SQD092b02_a_NFC_camB'),
    ('video_side2.avi', 'SQD092b02_a_NFC_camC'),
    ('video_pupil.avi', 'SQD092b02_a_NFC_pupil'),
    ('custom.avi', 'custom'),
])
def test_analysis_video_stem_is_read_only(tmp_path, name, stem):
    assert fs.analysis_video_stem(tmp_path / 'SQD092b02_a_NFC' / name) == stem
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('release', [None, 'left_chimney-r0001'])
def test_labeled_video_uses_exact_prediction_directory(tmp_path, monkeypatch, release):
    from types import SimpleNamespace
    from pathlib import Path
    folder = tmp_path / release if release else tmp_path
    folder.mkdir(exist_ok=True)
    video = '/videos/SQD092b02_a_NFC.avi'
    prediction = folder / 'SQD092b02_a_NFCDLC_test.h5'
    prediction.touch()
    prediction.with_name(prediction.stem + '_meta.pickle').touch()
    monkeypatch.setattr(fs, 'release_output_suffix', lambda r: 'DLC_test')
    monkeypatch.setattr(fs.dlc_alias, 'dlc_scorer_suffix', lambda c: 'DLC_test')
    calls = []
    dlc = SimpleNamespace(create_labeled_video=lambda *a, **kw: calls.append((a, kw)) or [True])
    fs.create_labeled_videos(dlc, '/config.yaml', [video], str(tmp_path), release)
    assert calls == [(('/config.yaml', [video]),
                      {'videotype': 'avi', 'destfolder': str(folder)})]
    assert fs.labeled_video_prediction(
        tmp_path, '/videos/SQD092b02_a_NFC/recording.avi', 'DLC_test', release) == prediction


def test_labeled_video_rejects_wrong_release_and_missing_metadata(tmp_path):
    folder = tmp_path / 'left-r0001'
    folder.mkdir()
    prediction = folder / 'sessionDLC_test.h5'
    prediction.touch()
    with pytest.raises(FileNotFoundError, match='metadata is missing'):
        fs.labeled_video_prediction(tmp_path, 'session.avi', 'DLC_test', 'left-r0001')
    prediction.with_name(prediction.stem + '_meta.pickle').touch()
    with pytest.raises(FileNotFoundError, match='left-r0002'):
        fs.labeled_video_prediction(tmp_path, 'session.avi', 'DLC_test', 'left-r0002')
    with pytest.raises(FileNotFoundError, match='No prediction'):
        fs.labeled_video_prediction(tmp_path, 'other.avi', 'DLC_test', 'left-r0001')
    with pytest.raises(FileNotFoundError, match='No prediction'):
        fs.labeled_video_prediction(tmp_path, 'session.avi', 'DLC_other', 'left-r0001')
    assert not (tmp_path / 'left-r0002').exists()


def test_labeled_video_surfaces_dlc_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    prediction = tmp_path / 'sessionDLC_test.h5'
    prediction.touch()
    prediction.with_name(prediction.stem + '_meta.pickle').touch()
    monkeypatch.setattr(fs.dlc_alias, 'dlc_scorer_suffix', lambda c: 'DLC_test')
    dlc = SimpleNamespace(create_labeled_video=lambda *a, **kw: [False])
    with pytest.raises(RuntimeError, match='could not create'):
        fs.create_labeled_videos(dlc, '/config.yaml', ['session.avi'], str(tmp_path))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: an explicit inspected publication cannot drift while queued | date: 2026-09-10]
def test_inspected_publication_token_rejects_release_file_and_pointer_changes(tmp_path):
    source = tmp_path / 'source.h5'
    source.write_bytes(b'original')
    token = fs.publication_token(source, 'left-r0001')
    assert len(token) == 64 and token.isalnum()
    fs.check_publication_token(source, 'left-r0001', token)
    with pytest.raises(ValueError, match='changed while queued'):
        fs.check_publication_token(source, 'left-r0002', token)
    with pytest.raises(ValueError, match='changed while queued'):
        fs.check_publication_token(None, 'left-r0001', token)
    source.write_bytes(b'new predictions')
    with pytest.raises(ValueError, match='changed while queued'):
        fs.check_publication_token(source, 'left-r0001', token)
    fs.check_publication_token(source, 'left-r0001', None)  # legacy automatic publish remains compatible
# [AGENT EDIT END]
