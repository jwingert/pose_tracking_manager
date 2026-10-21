"""Tests for nems_lbhb.motor.dlc_video_qc and scripts/dlc_video_audit.py.

Everything runs against synthetic videos written into ``tmp_path`` with
``cv2.VideoWriter``.  Nothing under ``/auto/data`` is read, and nothing is ever
written outside the tmp dir.

The damaged fixture is a byte-truncated copy of a valid clip, which reproduces
the real failure mode faithfully: the container header still advertises the full
frame count, ffmpeg recovers part of the stream, and OpenCV -- the decoder
DeepLabCut uses -- gives up well before the end.  That is exactly the shape of
``/auto/data/daq/LemonDisco/LMD008/LMD008a02_a_NFB/recording.avi`` (header
77,385, ffprobe 52,190, OpenCV 2,461).

Tests that need cv2 or ffmpeg skip **explicitly**, with the tool named in the
skip reason, so a missing dependency never looks like a pass.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

from nems_lbhb.motor import dlc_video_qc as vqc


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'dlc_video_audit.py')

# libav writes thousands of 'ac-tex damaged' lines to stderr while we
# deliberately decode broken files; they drown the pytest output.
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')

HAVE_FFPROBE = shutil.which('ffprobe') is not None
HAVE_FFMPEG = shutil.which('ffmpeg') is not None

needs_ffprobe = pytest.mark.skipif(
    not HAVE_FFPROBE, reason='ffprobe not on PATH: full_check cannot be tested')
needs_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason='ffmpeg not on PATH: error-line counting cannot be '
                            'tested')


def _cv2():
    return pytest.importorskip(
        'cv2', reason='cv2 not importable: the OpenCV probe cannot be tested')


def _load_audit_module():
    spec = importlib.util.spec_from_file_location('dlc_video_audit', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- fixtures

def _write_clip(path, n_frames=240, size=(64, 48), fps=30.0):
    """A short, deliberately compressible-but-not-trivial clip.

    Tries a few fourccs because which ones a given OpenCV build can *write* is
    not guaranteed; the test skips rather than fails if none work.
    """
    cv2 = _cv2()
    import numpy as np

    rng = np.random.RandomState(0)
    for fourcc_s in ('MJPG', 'XVID', 'mp4v'):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_s),
                                 fps, size)
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(n_frames):
            frame = rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            x = i % (size[0] - 12)
            cv2.rectangle(frame, (x, 10), (x + 8, 20), (255, 0, 0), -1)
            writer.write(frame)
        writer.release()
        if os.path.getsize(str(path)) > 0:
            return fourcc_s
    pytest.skip('this OpenCV build cannot write MJPG/XVID/mp4v AVI')


@pytest.fixture
def good_video(tmp_path):
    path = tmp_path / 'good.avi'
    _write_clip(path)
    return str(path)


@pytest.fixture
def truncated_video(tmp_path, good_video):
    """A valid clip with its tail chopped off -- header intact, stream is not."""
    path = tmp_path / 'truncated.avi'
    with open(good_video, 'rb') as fh:
        data = fh.read()
    with open(str(path), 'wb') as fh:
        fh.write(data[:int(len(data) * 0.4)])
    return str(path)


# ------------------------------------------------------------------ quick tier

def test_good_video_passes_quick(good_video):
    r = vqc.quick_check(good_video, n_seeks=6, sequential_frames=50)
    assert r['checked'] is True
    assert r['ok'] is True, r['reason']
    assert r['reason'] == ''
    assert r['header_frames'] == 240
    assert r['seek_attempts'] == 6
    assert r['seek_failures'] == 0
    assert r['sequential_ok'] is True
    assert (r['width'], r['height']) == (64, 48)
    # the escalation budget is not spent on a healthy file
    assert r['deep_scan_frames'] is None


def test_truncated_video_fails_quick(truncated_video):
    r = vqc.quick_check(truncated_video, n_seeks=6, sequential_frames=50)
    assert r['checked'] is True, 'a truncated file must still be *checked*'
    assert r['ok'] is False
    # header still lies about the length -- that is what makes this invisible
    assert r['header_frames'] == 240
    assert r['seek_failures'] > 0


def test_truncated_video_reason_is_useful(truncated_video):
    """The reason has to say how much of the video OpenCV can actually reach.

    'corrupt' is not actionable; '89 of the 240 frames the header claims' tells
    you what fraction of a re-analysis would be garbage.
    """
    r = vqc.quick_check(truncated_video, n_seeks=6, sequential_frames=50)
    reason = r['reason']
    assert reason, 'a failing check must explain itself'
    assert 'OpenCV' in reason
    assert r['deep_scan_frames'] is not None, 'must escalate to locate the stop'
    assert 0 < r['deep_scan_frames'] < r['header_frames']
    assert str(r['deep_scan_frames']) in reason
    assert str(r['header_frames']) in reason


def test_quick_check_is_the_opencv_verdict(truncated_video):
    """Sequential reach below the header count is the DeepLabCut-breaking mode.

    ffprobe is more forgiving than OpenCV, so ``ok`` must key off the OpenCV
    evidence rather than the decodable-frame ratio.
    """
    r = vqc.quick_check(truncated_video, n_seeks=6, sequential_frames=50)
    frac = r['deep_scan_frames'] / float(r['header_frames'])
    assert frac < vqc.SEQUENTIAL_REACH_FRACTION
    assert r['ok'] is False


def test_empty_file_is_reported_not_raised(tmp_path):
    path = tmp_path / 'empty.avi'
    path.write_bytes(b'')
    r = vqc.quick_check(str(path))
    assert r['ok'] is False
    assert r['checked'] is True
    assert 'empty' in r['reason']


def test_garbage_bytes_do_not_raise(tmp_path):
    path = tmp_path / 'garbage.avi'
    path.write_bytes(b'this is not a video' * 100)
    r = vqc.quick_check(str(path))       # must not raise
    assert r['ok'] is False
    assert r['reason']


def test_tiny_file_is_not_called_empty(tmp_path):
    """A sub-MB file rounds to 0.0 MB -- it is broken, but it is not *empty*.

    Regression: the emptiness test used to compare the rounded ``file_mb``, so
    every file under 50 kB was reported as '0 bytes' regardless of its contents.
    """
    path = tmp_path / 'tiny.avi'
    path.write_bytes(b'x' * 100)
    r = vqc.quick_check(str(path))
    assert r['ok'] is False
    assert 'empty' not in r['reason']


def test_quick_check_without_cv2_reports_not_checked(monkeypatch, good_video):
    """No cv2 must mean 'we did not look', never 'the file is fine' or a crash."""
    monkeypatch.setitem(sys.modules, 'cv2', None)   # `import cv2` -> ImportError
    r = vqc.quick_check(good_video)
    assert r['checked'] is False
    assert r['ok'] is False
    assert 'cv2' in r['reason']


# ------------------------------------------------------------------- full tier

@needs_ffprobe
@needs_ffmpeg
def test_good_video_passes_full(good_video):
    r = vqc.full_check(good_video)
    assert r['checked'] is True
    assert r['ok'] is True, r['reason']
    assert r['header_frames'] == 240
    assert r['decodable_frames'] == 240
    assert r['ratio'] == 1.0
    assert r['error_lines'] == 0


@needs_ffprobe
@needs_ffmpeg
def test_truncated_video_fails_full(truncated_video):
    r = vqc.full_check(truncated_video)
    assert r['checked'] is True
    assert r['ok'] is False
    assert r['decodable_frames'] < r['header_frames']
    assert 0.0 < r['ratio'] < 1.0
    assert 'decodable' in r['reason']


@needs_ffprobe
def test_full_check_without_ffmpeg_still_reports_frame_counts(monkeypatch,
                                                              good_video):
    real_which = shutil.which
    monkeypatch.setattr(shutil, 'which',
                        lambda n, *a, **k: None if n == vqc.FFMPEG
                        else real_which(n, *a, **k))
    r = vqc.full_check(good_video)
    assert r['checked'] is True
    assert r['header_frames'] == 240
    assert r['decodable_frames'] == 240
    assert r['error_lines'] is None
    assert 'ffmpeg' in r['reason']
    # a tool we could not run is not evidence the file is sound
    assert r['ok'] is False


def test_full_check_without_ffprobe_degrades_gracefully(monkeypatch, tmp_path):
    """No ffprobe: report clearly, do not crash, do not claim a verdict."""
    path = tmp_path / 'anything.avi'
    path.write_bytes(b'x' * 1024)
    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: None)
    r = vqc.full_check(str(path))       # must not raise
    assert r['checked'] is False
    assert r['ok'] is False
    assert 'ffprobe' in r['reason']
    assert 'not found' in r['reason']


def test_full_check_timeout_degrades_gracefully(monkeypatch, tmp_path):
    path = tmp_path / 'anything.avi'
    path.write_bytes(b'x' * 1024)
    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: '/usr/bin/' + n)

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get('timeout', 1))

    monkeypatch.setattr(subprocess, 'run', boom)
    r = vqc.full_check(str(path), timeout=1.0)
    assert r['ok'] is False
    assert r['checked'] is False
    assert 'timed out' in r['reason']


def test_full_check_reports_nonzero_ffprobe_exit(monkeypatch, tmp_path):
    path = tmp_path / 'anything.avi'
    path.write_bytes(b'x' * 1024)
    monkeypatch.setattr(shutil, 'which', lambda n, *a, **k: '/usr/bin/' + n)

    class Fake(object):
        returncode = 1
        stdout = ''
        stderr = 'moov atom not found\n'

    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: Fake())
    r = vqc.full_check(str(path))
    assert r['ok'] is False
    assert r['checked'] is True
    assert 'exited 1' in r['reason']
    assert 'moov atom not found' in r['reason']


# ----------------------------------------------------------------- missing file

@pytest.mark.parametrize('fn', [vqc.quick_check, vqc.full_check])
def test_missing_file_raises(tmp_path, fn):
    with pytest.raises(FileNotFoundError):
        fn(str(tmp_path / 'does_not_exist.avi'))


def test_check_wrapper_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        vqc.check(str(tmp_path / 'nope.avi'))


def test_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vqc.quick_check(str(tmp_path))


# --------------------------------------------------------------------- wrapper

def test_check_dispatches(good_video):
    assert vqc.check(good_video)['checker'] == 'quick'


@needs_ffprobe
def test_check_full_dispatches(good_video):
    assert vqc.check(good_video, full=True)['checker'] == 'full'


# --------------------------------------------------------------------- helpers

def test_count_error_lines_expands_repeats():
    stderr = ('[mpeg4] ac-tex damaged\n'
              '    Last message repeated 18 times\n'
              '\n'
              '[mpeg4] header damaged\n')
    assert vqc._count_error_lines(stderr) == 20


def test_count_error_lines_empty():
    assert vqc._count_error_lines('') == 0
    assert vqc._count_error_lines(None) == 0


def test_video_for_dlc_h5_symlink_layout(tmp_path):
    pen = tmp_path / 'PRN075'
    (pen / 'sorted').mkdir(parents=True)
    h5 = pen / 'sorted' / 'PRN075a01_a_NFB.dlc.h5'
    h5.write_bytes(b'')
    (pen / 'PRN075a01_a_NFB.avi').write_bytes(b'x')
    assert vqc.video_for_dlc_h5(str(h5)) == str(pen / 'PRN075a01_a_NFB.avi')


def test_video_for_dlc_h5_recording_layout(tmp_path):
    pen = tmp_path / 'PRN075'
    (pen / 'sorted').mkdir(parents=True)
    (pen / 'PRN075a01_a_NFB').mkdir()
    h5 = pen / 'sorted' / 'PRN075a01_a_NFB.dlc.h5'
    h5.write_bytes(b'')
    (pen / 'PRN075a01_a_NFB' / 'recording.avi').write_bytes(b'x')
    assert vqc.video_for_dlc_h5(str(h5)) == str(
        pen / 'PRN075a01_a_NFB' / 'recording.avi')


def test_video_for_dlc_h5_missing_returns_none(tmp_path):
    pen = tmp_path / 'PRN075'
    (pen / 'sorted').mkdir(parents=True)
    h5 = pen / 'sorted' / 'PRN075a01_a_NFB.dlc.h5'
    h5.write_bytes(b'')
    assert vqc.video_for_dlc_h5(str(h5)) is None


# ---------------------------------------------------------------- audit script

def test_audit_session_and_animal():
    mod = _load_audit_module()
    animal, session = mod.session_of(
        '/auto/data/daq/LemonDisco/LMD008/LMD008a02_a_NFB/recording.avi')
    assert animal == 'LemonDisco'
    assert session == 'LMD008a02_a_NFB'


def test_audit_resolves_h5_to_avi(tmp_path):
    mod = _load_audit_module()
    pen = tmp_path / 'PRN075'
    (pen / 'sorted').mkdir(parents=True)
    (pen / 'sorted' / 'a.dlc.h5').write_bytes(b'')
    (pen / 'sorted' / 'b.dlc.h5').write_bytes(b'')
    (pen / 'a.avi').write_bytes(b'x')
    paths, unresolved = mod.resolve_videos(str(tmp_path / '*' / 'sorted' /
                                               '*.dlc.h5'))
    assert paths == [str(pen / 'a.avi')]
    assert unresolved == [str(pen / 'sorted' / 'b.dlc.h5')]


def test_audit_row_marks_damage(truncated_video, good_video):
    mod = _load_audit_module()
    bad = mod.audit_one(truncated_video)
    good = mod.audit_one(good_video)
    assert bad['ok'] is False and bad['checked'] is True
    assert bad['reason'].startswith('quick:')
    assert good['ok'] is True
    assert set(mod.CSV_COLUMNS) >= {'path', 'session', 'animal',
                                    'header_frames', 'decodable_frames',
                                    'ratio', 'seek_failures', 'sequential_ok',
                                    'ok', 'reason'}


def test_audit_missing_file_is_a_row_not_a_crash(tmp_path):
    """The library raises on a missing file; the sweep must survive one."""
    mod = _load_audit_module()
    row = mod.audit_one(str(tmp_path / 'gone.avi'))
    assert row['ok'] is False
    assert row['checked'] is False
    assert 'gone.avi' in row['reason']


@needs_ffprobe
@needs_ffmpeg
def test_audit_only_suspect_escalates_only_failures(truncated_video, good_video):
    mod = _load_audit_module()
    bad = mod.audit_one(truncated_video, only_suspect=True)
    good = mod.audit_one(good_video, only_suspect=True)
    assert bad['tier'] == 'quick+full'
    assert bad['decodable_frames'] not in ('', None)
    assert good['tier'] == 'quick', 'a passing video must not pay for ffprobe'
    assert good['decodable_frames'] == ''


def test_audit_writes_csv(tmp_path, good_video):
    import csv as _csv
    mod = _load_audit_module()
    out = tmp_path / 'report.csv'
    mod.write_csv([mod.audit_one(good_video)], str(out))
    with open(str(out)) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]['ok'] == 'True'
    assert rows[0]['path'] == good_video
