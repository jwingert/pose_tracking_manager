"""Tests for nems_lbhb.motor.dlc_video_scale -- analysis-resolution rescaling.

Everything runs against synthetic videos written into ``tmp_path`` with
``cv2.VideoWriter`` and synthetic DeepLabCut pose files built with pandas.
Nothing under ``/auto/data`` is read and nothing is written outside the tmp dir.

The behaviour being pinned, in one line each:

* 1280x720 (the 13 legacy Prince sites) becomes 640x360, **not** 640x480 --
  squashing 16:9 into 4:3 would distort the geometry the head-fixation frame and
  the arena ppm calibration depend on;
* 640x480 (everything since Feb 2023) is left alone;
* a video smaller than the training frames is never upscaled;
* odd dimensions round to even, because encoders reject odd ones;
* ``rescale_coordinates`` scales x and y exactly and leaves likelihood alone --
  that is what makes a rescaled run indistinguishable from a native one for the
  arena ppm, the head-angle geometry and the HRTF transforms;
* a re-encode that loses frames is refused, because pose is aligned to the neural
  data by row index and a short pose file would misalign silently.

Tests that need cv2 or ffmpeg skip **explicitly**, with the tool named in the
skip reason, so a missing dependency never looks like a pass.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for the new resolution-based inference rescaling | date: 2026-08-08]
import json
import os
import shutil
import subprocess

import pytest

from nems_lbhb.motor import dlc_video_scale as vscale


# libav is chatty on stderr while we deliberately probe re-encodes.
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')

HAVE_FFMPEG = shutil.which(vscale.FFMPEG) is not None

needs_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG,
    reason='ffmpeg not on PATH: the re-encode cannot be tested')


def _cv2():
    return pytest.importorskip(
        'cv2', reason='cv2 not importable: video probing cannot be tested')


# --------------------------------------------------------------------- fixtures

def _write_clip(path, size, n_frames=30, fps=30.0):
    """A short clip at exactly ``size`` = (width, height)."""
    cv2 = _cv2()
    import numpy as np

    rng = np.random.RandomState(0)
    for fourcc_s in ('MJPG', 'XVID', 'mp4v'):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_s),
                                 fps, tuple(size))
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(n_frames):
            frame = rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            cv2.rectangle(frame, (i, 5), (i + 6, 15), (255, 0, 0), -1)
            writer.write(frame)
        writer.release()
        if os.path.getsize(str(path)) > 0:
            return str(path)
    pytest.skip('this OpenCV build cannot write MJPG/XVID/mp4v AVI')


@pytest.fixture
def hd_video(tmp_path):
    """The legacy Prince camera: 1280x720."""
    return _write_clip(tmp_path / 'PRN043a05_a_NTD.avi', (1280, 720))


@pytest.fixture
def sd_video(tmp_path):
    """Everything since Feb 2023: 640x480, the training frame size."""
    return _write_clip(tmp_path / 'PRN075a01_a_NFB.avi', (640, 480))


@pytest.fixture
def scratch(tmp_path):
    """A destination directory that is NOT the source video's own directory."""
    d = tmp_path / 'scratch'
    d.mkdir()
    return str(d)


def _pose_frame(scorer='DLC_resnet50_test', bodyparts=('front', 'back'),
                n_rows=5):
    """A DeepLabCut-shaped pose DataFrame with known x/y/likelihood values."""
    pd = pytest.importorskip('pandas')
    import numpy as np

    cols = pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])
    data = np.zeros((n_rows, len(cols)))
    for j, (_s, _bp, coord) in enumerate(cols):
        if coord == 'x':
            data[:, j] = np.arange(n_rows) + 100.0
        elif coord == 'y':
            data[:, j] = np.arange(n_rows) + 200.0
        else:
            data[:, j] = np.linspace(0.1, 0.9, n_rows)
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def pose_h5(tmp_path):
    df = _pose_frame()
    path = str(tmp_path / 'pose.h5')
    df.to_hdf(path, vscale.DLC_H5_KEY, format='table', mode='w')
    return path


# ------------------------------------------------------------- analysis_scale

def test_hd_video_is_downscaled_to_half_width(hd_video):
    """1280x720 -> 640x360: match the training WIDTH, keep the aspect ratio."""
    info = vscale.analysis_scale(hd_video, train_frame_size=(640, 480))
    assert info['native'] == (1280, 720)
    assert info['analysis_size'] == (640, 360)
    assert info['rescale'] is True
    assert info['scale'] == pytest.approx(0.5)
    assert info['reason']


def test_hd_video_is_not_squashed_into_the_training_aspect(hd_video):
    """640x480 would distort 16:9 content non-uniformly in x and y.

    The head-fixation frame and the arena ppm calibration both assume an
    undistorted view, so the aspect ratio is the thing that must survive, not the
    training frame's height.
    """
    info = vscale.analysis_scale(hd_video, train_frame_size=(640, 480))
    assert info['analysis_size'] != (640, 480)
    aw, ah = info['analysis_size']
    assert aw / float(ah) == pytest.approx(1280 / 720.0, rel=1e-3)


def test_analysis_dimensions_are_even(hd_video):
    aw, ah = vscale.analysis_scale(hd_video)['analysis_size']
    assert aw % 2 == 0 and ah % 2 == 0


def test_training_resolution_video_is_left_alone(sd_video):
    info = vscale.analysis_scale(sd_video, train_frame_size=(640, 480))
    assert info['native'] == (640, 480)
    assert info['analysis_size'] == (640, 480)
    assert info['rescale'] is False
    assert info['scale'] == 1.0


def test_smaller_than_training_is_never_upscaled(tmp_path):
    """Upscaling invents no detail; it only makes the animal too big, which is
    the failure we are fixing, mirrored."""
    small = _write_clip(tmp_path / 'small.avi', (320, 240))
    info = vscale.analysis_scale(small, train_frame_size=(640, 480))
    assert info['rescale'] is False
    assert info['scale'] == 1.0
    assert info['analysis_size'] == (320, 240)


def test_odd_dimensions_round_to_even(tmp_path):
    """Encoders reject odd dimensions, so rounding has to land on even."""
    # 642x362 at scale 640/642 gives 640 x 360.87 -> 640x360 (even).
    odd = _write_clip(tmp_path / 'odd.avi', (642, 362))
    info = vscale.analysis_scale(odd, train_frame_size=(640, 480))
    aw, ah = info['analysis_size']
    assert aw % 2 == 0 and ah % 2 == 0
    assert (aw, ah) == (640, 360)


def test_scale_is_the_realised_x_scale_not_the_requested_one(tmp_path):
    """Rounding to even changes the transform actually applied; `scale` must
    describe what happened, or rescale_coordinates inverts the wrong thing."""
    vid = _write_clip(tmp_path / 'oddish.avi', (642, 362))
    info = vscale.analysis_scale(vid, train_frame_size=(640, 480))
    assert info['scale'] == pytest.approx(info['analysis_size'][0] / 642.0)


def test_unprobeable_video_raises_rather_than_defaulting(tmp_path):
    """An unknown native size must never become 'no rescale needed' -- that
    silently analyses at the wrong scale and the output looks fine."""
    _cv2()
    junk = tmp_path / 'junk.avi'
    junk.write_bytes(b'not a video' * 50)
    with pytest.raises(RuntimeError):
        vscale.analysis_scale(str(junk))


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vscale.analysis_scale(str(tmp_path / 'nope.avi'))


def test_bad_train_frame_size_raises(sd_video):
    with pytest.raises(ValueError):
        vscale.analysis_scale(sd_video, train_frame_size=(0, 480))


# -------------------------------------------------------------- rescaled_copy

@needs_ffmpeg
def test_rescaled_copy_has_the_requested_size_and_same_basename(hd_video, scratch):
    """Same basename: DeepLabCut names its output <stem><scorer>.h5, so the
    rescaled run must produce exactly the filename a native run would."""
    info = vscale.analysis_scale(hd_video)
    out = vscale.rescaled_copy(hd_video, info['analysis_size'], scratch)
    assert os.path.basename(out) == os.path.basename(hd_video)
    assert os.path.dirname(out) == scratch
    assert vscale.probe_size(out) == (640, 360)


@needs_ffmpeg
def test_rescaled_copy_preserves_frame_count(hd_video, scratch):
    out = vscale.rescaled_copy(hd_video, (640, 360), scratch)
    assert vscale.frame_count(out) == vscale.frame_count(hd_video)


@needs_ffmpeg
def test_rescaled_copy_leaves_the_source_untouched(hd_video, scratch):
    """This module is read-only with respect to /auto/data/daq."""
    before = (os.path.getsize(hd_video), os.path.getmtime(hd_video))
    vscale.rescaled_copy(hd_video, (640, 360), scratch)
    assert (os.path.getsize(hd_video), os.path.getmtime(hd_video)) == before
    assert vscale.probe_size(hd_video) == (1280, 720)


@needs_ffmpeg
def test_rescaled_copy_refuses_to_overwrite_its_own_source(hd_video):
    with pytest.raises(ValueError) as e:
        vscale.rescaled_copy(hd_video, (640, 360),
                             os.path.dirname(hd_video))
    assert 'source video' in str(e.value)


def test_rescaled_copy_rejects_odd_analysis_size(hd_video, scratch):
    with pytest.raises(ValueError):
        vscale.rescaled_copy(hd_video, (641, 360), scratch)


def test_rescaled_copy_without_ffmpeg_raises(hd_video, scratch, monkeypatch):
    """No ffmpeg must be an error, not a quiet fall back to native analysis --
    the whole failure this module fixes is invisible in the output."""
    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: None)
    with pytest.raises(RuntimeError) as e:
        vscale.rescaled_copy(hd_video, (640, 360), scratch)
    assert 'ffmpeg' in str(e.value)


def test_truncated_reencode_is_caught(hd_video, scratch, monkeypatch):
    """The frame-count check is what stops a short re-encode.

    A pose file with fewer rows than the recording misaligns against the neural
    data by row index, silently and totally.  Here ffmpeg is replaced by a fake
    that writes a *valid but shorter* clip, which is the shape of the failure.
    """
    _cv2()
    n_src = vscale.frame_count(hd_video)
    assert n_src > 4

    def fake_run(cmd, **kwargs):
        dest = cmd[-1]
        _write_clip(dest, (640, 360), n_frames=max(1, n_src // 3))

        class Done(object):
            returncode = 0
            stdout = ''
            stderr = ''
        return Done()

    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: '/usr/bin/' + n)
    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError) as e:
        vscale.rescaled_copy(hd_video, (640, 360), scratch)
    assert 'frames' in str(e.value)
    # and the short file is not left lying around to be analyzed
    assert not os.path.exists(os.path.join(scratch,
                                           os.path.basename(hd_video)))


def test_rescaled_copy_reports_a_failed_ffmpeg(hd_video, scratch, monkeypatch):

    class Failed(object):
        returncode = 1
        stdout = ''
        stderr = 'Invalid argument\n'

    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: '/usr/bin/' + n)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: Failed())
    with pytest.raises(RuntimeError) as e:
        vscale.rescaled_copy(hd_video, (640, 360), scratch)
    assert 'exited 1' in str(e.value)


# --------------------------------------------------------- rescale_coordinates

def test_rescale_coordinates_scales_xy_exactly(pose_h5):
    """Exact for a known factor: 0.5 -> everything doubles."""
    pd = pytest.importorskip('pandas')
    before = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)

    n = vscale.rescale_coordinates(pose_h5, 0.5)
    assert n == len(before.index)

    after = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    coords = after.columns.get_level_values('coords')
    for col, kind in zip(after.columns, coords):
        if kind in ('x', 'y'):
            assert list(after[col]) == pytest.approx(
                [v * 2.0 for v in before[col]])


def test_rescale_coordinates_leaves_likelihood_untouched(pose_h5):
    """Likelihood is dimensionless; scaling it would corrupt every QC threshold."""
    pd = pytest.importorskip('pandas')
    before = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    vscale.rescale_coordinates(pose_h5, 0.5)
    after = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)

    coords = after.columns.get_level_values('coords')
    lik = [c for c, k in zip(after.columns, coords) if k == 'likelihood']
    assert lik, 'fixture must contain likelihood columns'
    for col in lik:
        assert list(after[col]) == list(before[col])


def test_rescale_coordinates_preserves_shape_and_columns(pose_h5):
    pd = pytest.importorskip('pandas')
    before = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    vscale.rescale_coordinates(pose_h5, 0.5)
    after = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    assert after.shape == before.shape
    assert list(after.columns) == list(before.columns)
    assert after.columns.names == before.columns.names


def test_rescale_coordinates_round_trips_to_native(pose_h5):
    """0.5 then 2.0 returns the original numbers -- the transform is exact."""
    pd = pytest.importorskip('pandas')
    before = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    vscale.rescale_coordinates(pose_h5, 0.5)
    vscale.rescale_coordinates(pose_h5, 2.0)
    after = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    for col in before.columns:
        assert list(after[col]) == pytest.approx(list(before[col]))


def test_rescale_coordinates_scale_one_is_a_noop(pose_h5):
    pd = pytest.importorskip('pandas')
    before = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    n = vscale.rescale_coordinates(pose_h5, 1.0)
    after = pd.read_hdf(pose_h5, vscale.DLC_H5_KEY)
    assert n == len(before.index)
    for col in before.columns:
        assert list(after[col]) == list(before[col])


def test_rescale_coordinates_rejects_nonpositive_scale(pose_h5):
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            vscale.rescale_coordinates(pose_h5, bad)


def test_rescale_coordinates_rejects_a_non_dlc_frame(tmp_path):
    """Blindly scaling an unrecognised table could corrupt real data."""
    pd = pytest.importorskip('pandas')
    path = str(tmp_path / 'plain.h5')
    pd.DataFrame({'a': [1.0, 2.0]}).to_hdf(path, vscale.DLC_H5_KEY,
                                           format='table', mode='w')
    with pytest.raises(ValueError) as e:
        vscale.rescale_coordinates(path, 0.5)
    assert 'coords' in str(e.value)


# ------------------------------------------------------------------- provenance

def test_sidecar_records_what_happened(tmp_path):
    info = {'native': (1280, 720), 'analysis_size': (640, 360), 'scale': 0.5,
            'rescale': True, 'reason': 'native is larger than training frames'}
    path = vscale.write_scale_sidecar(str(tmp_path), 'PRN043a05_a_NTD', info,
                                      release_id='right_chimney-r0001')
    assert os.path.basename(path) == \
        'PRN043a05_a_NTD.right_chimney-r0001.scale.json'
    with open(path) as fh:
        payload = json.load(fh)
    assert payload['native_size'] == [1280, 720]
    assert payload['analysis_size'] == [640, 360]
    assert payload['scale'] == 0.5
    assert payload['rescaled'] is True
    assert payload['reason']
    assert 'native pixels' in payload['coordinates']


def test_sidecar_name_without_a_release():
    assert vscale.scale_sidecar_name('X01_a_NTD') == 'X01_a_NTD.scale.json'


def test_sidecar_is_rewritable(tmp_path):
    """Re-analysing a session must not fail on an existing sidecar."""
    info = {'native': (640, 480), 'analysis_size': (640, 480), 'scale': 1.0,
            'rescale': False, 'reason': 'already at training resolution'}
    a = vscale.write_scale_sidecar(str(tmp_path), 'stem', info)
    b = vscale.write_scale_sidecar(str(tmp_path), 'stem', info)
    assert a == b
    assert json.load(open(b))['rescaled'] is False


# ------------------------------------------------------- registry integration

def test_registry_exposes_train_frame_size():
    from nems_lbhb.motor import dlc_registry
    r = dlc_registry.load_registry()
    for alias in r['alias_names']:
        size = dlc_registry.resolve(alias, r)['train_frame_size']
        assert len(size) == 2 and all(int(v) > 0 for v in size)


def test_registry_train_frame_size_defaults_when_absent(tmp_path):
    from nems_lbhb.motor import dlc_registry
    cfg = tmp_path / 'reg.toml'
    cfg.write_text('schema_version = 1\n'
                   '[alias.a]\nproject = "/tmp/a"\ncurrent = ""\nrules = []\n'
                   '[alias.b]\nproject = "/tmp/b"\ncurrent = ""\n'
                   'train_frame_size = [1280, 720]\nrules = []\n')
    r = dlc_registry.load_registry(str(cfg))
    assert dlc_registry.resolve('a', r)['train_frame_size'] == \
        dlc_registry.DEFAULT_TRAIN_FRAME_SIZE
    assert dlc_registry.resolve('b', r)['train_frame_size'] == (1280, 720)


def test_registry_rejects_a_malformed_train_frame_size(tmp_path):
    """A typo here silently picks the analysis resolution for every video the
    alias routes, and the resulting pose files look perfectly well-formed."""
    from nems_lbhb.motor import dlc_registry
    cfg = tmp_path / 'bad.toml'
    cfg.write_text('schema_version = 1\n'
                   '[alias.a]\nproject = "/tmp/a"\ncurrent = ""\n'
                   'train_frame_size = [640]\nrules = []\n')
    with pytest.raises(ValueError) as e:
        dlc_registry.load_registry(str(cfg))
    assert 'train_frame_size' in str(e.value)


def test_fit_script_reads_the_size_from_the_alias():
    from nems_lbhb.motor import dlc_fit_script as fs
    from nems_lbhb.motor import dlc_registry

    aliases = {'big': {'project': '/proj/big', 'current': '',
                       'train_frame_size': [1280, 720], 'rules': []},
               'plain': {'project': '/proj/plain', 'current': '', 'rules': []}}
    reg = {'_path': '/fake.toml', 'aliases': aliases,
           'alias_names': list(aliases)}
    assert fs.resolved_train_frame_size({'alias': 'big'}, reg) == (1280, 720)
    assert fs.resolved_train_frame_size({'alias': 'plain'}, reg) == \
        dlc_registry.DEFAULT_TRAIN_FRAME_SIZE
    # an explicit config path belongs to no alias -> the measured lab default
    assert fs.resolved_train_frame_size({'alias': None}, reg) == \
        dlc_registry.DEFAULT_TRAIN_FRAME_SIZE
# [AGENT EDIT END]
