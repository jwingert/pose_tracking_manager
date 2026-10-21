# [AGENT EDIT START | agent: claude | user: wingertj | reason: the camera assertion decides
# which PSIVIDEO stream aligns every trial, and it is now writable from the GUI as well as
# the CLI -- both go through this module, so its refusals are the safety net | date: 2026-08-12]
"""Tests for nems_lbhb/motor/dlc_camera_override.py.

Nothing here decodes video (``slot_info``/``sample_frames`` need OpenCV and real files);
what is tested is the part that can silently corrupt data -- path resolution, the derived
``cam_tag``, and every refusal on the write path.
"""
import os

import pytest

from nems_lbhb.motor import dlc_camera_override as co

STEM = 'REI084a10_a_NFB'

REGISTRY_HEAD = '''schema_version = 1

[camera_override]

[training_defaults]
net_type = "resnet_50"
'''


@pytest.fixture
def session(tmp_path):
    """A daq-shaped session directory holding all three psi slots."""
    d = tmp_path / 'Reishi' / 'REI084' / STEM
    d.mkdir(parents=True)
    for slot in co.SLOTS:
        (d / slot).write_bytes(b'not a real video')
    return str(d)


@pytest.fixture
def registry(tmp_path):
    p = tmp_path / 'dlc_registry.toml'
    p.write_text(REGISTRY_HEAD)
    return str(p)


def test_session_dir_accepts_every_shape_a_caller_might_have(session):
    """The GUI holds '<sess>/recording.avi', the fit script holds the parent-level
    '<sess>.avi' symlink, and a CLI user types the directory. All are the same session."""
    assert co.session_dir(session) == session
    assert co.session_dir(os.path.join(session, 'recording.avi')) == session
    assert co.session_dir(os.path.join(session, 'video_side2.avi')) == session
    assert co.session_dir(session + '.avi') == session
    assert co.session_dir(session + '/') == session
    assert co.session_stem(session) == STEM


def test_cam_tag_is_derived_from_the_slot_not_supplied():
    """A tag that disagrees with its slot would mis-align every trial while looking
    deliberate in the file, so callers never get to pass one."""
    assert co.cam_tag_for('recording.avi') is None
    assert co.cam_tag_for('video_side.avi') == 'camB'
    assert co.cam_tag_for('video_side2.avi') == 'camC'


def test_render_block_is_pure_and_carries_the_derived_tag():
    block = co.render_block(STEM, 'video_side2.avi', 'wingertj', note='port swap')
    assert '[camera_override.%s]' % STEM in block
    assert 'cam_tag = "camC"' in block
    assert 'verified_by = "wingertj"' in block
    assert 'port swap' in block


def test_note_quotes_cannot_break_the_toml():
    block = co.render_block(STEM, 'video_side2.avi', 'me', note='he said "it swapped"')
    assert block.count('"') % 2 == 0
    assert 'note = "he said' in block


def test_add_refuses_the_base_slot(session, registry):
    """recording.avi is already the default: an override there asserts nothing, and a
    no-op entry in a file of human assertions is worse than no entry."""
    with pytest.raises(ValueError, match='already the default'):
        co.add(session, 'recording.avi', by='me', registry_path=registry, dry_run=True)


def test_add_refuses_a_file_that_is_not_a_psi_slot(session, registry):
    with pytest.raises(ValueError, match='not a psi camera slot'):
        co.add(session, 'facecam.avi', by='me', registry_path=registry, dry_run=True)


def test_add_refuses_when_the_named_video_is_absent(tmp_path, registry):
    d = tmp_path / 'Reishi' / 'REI084' / STEM
    d.mkdir(parents=True)
    (d / 'recording.avi').write_bytes(b'x')          # side2 deliberately missing
    with pytest.raises(FileNotFoundError):
        co.add(str(d), 'video_side2.avi', by='me', registry_path=registry, dry_run=True)


def test_dry_run_writes_nothing(session, registry):
    before = open(registry).read()
    block, commit = co.add(session, 'video_side2.avi', by='me', registry_path=registry,
                           dry_run=True)
    assert open(registry).read() == before
    assert 'cam_tag = "camC"' in block
    assert commit.startswith('git add ')


def test_add_writes_and_round_trips_through_registry_validation(session, registry):
    from nems_lbhb.motor import dlc_registry
    co.add(session, 'video_side2.avi', by='wingertj', note='verified by eye',
           registry_path=registry)
    dlc_registry.clear_cache()
    entry = dlc_registry.load_registry(registry)['camera_override'][STEM]
    assert entry['video'] == 'video_side2.avi'
    assert entry['cam_tag'] == 'camC'
    assert entry['verified_by'] == 'wingertj'
    # and the anchor survives, so the next entry has somewhere to go
    assert '[training_defaults]' in open(registry).read()


def test_add_refuses_to_overwrite_an_existing_assertion(session, registry):
    co.add(session, 'video_side2.avi', by='first', registry_path=registry)
    with pytest.raises(ValueError, match='already has an entry'):
        co.add(session, 'video_side.avi', by='second', registry_path=registry)


def test_add_refuses_without_the_anchor_rather_than_appending_blindly(session, tmp_path):
    p = tmp_path / 'no_anchor.toml'
    p.write_text('schema_version = 1\n[camera_override]\n')
    with pytest.raises(ValueError, match='anchor'):
        co.add(session, 'video_side2.avi', by='me', registry_path=str(p), dry_run=True)


def test_existing_outputs_finds_this_sessions_tracking_only(session):
    """Those outputs came from the wrong camera, so surfacing them is what stops the
    re-run being forgotten -- but a sibling session's files are not this one's problem."""
    sorted_dir = os.path.join(os.path.dirname(session), 'sorted')
    os.makedirs(sorted_dir)
    for name in ('%s.dlc.h5' % STEM, '%s.left_chimney-r0001.dlc.h5' % STEM,
                 'REI084a09_p_NFB_pupil.dlc.h5'):
        open(os.path.join(sorted_dir, name), 'w').write('x')
    found = [os.path.basename(p) for p in co.existing_outputs(session)]
    assert found == ['%s.dlc.h5' % STEM, '%s.left_chimney-r0001.dlc.h5' % STEM]


def test_existing_outputs_is_empty_when_nothing_was_analysed(session):
    assert co.existing_outputs(session) == []
# [AGENT EDIT END]
