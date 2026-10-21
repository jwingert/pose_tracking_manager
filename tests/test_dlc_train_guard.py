"""Tests for nems_lbhb.motor.dlc_train_guard.

The first six tests are the contract from the Task 7 brief, verbatim.  The rest
cover the paths the brief's six leave open: the polling `timeout`, a corrupt
lockfile, releasing someone else's lock, the shuffle filter on the fresh-iteration
guard, and slugging in `run_modelprefix`.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- tests for the
# training lock and the fresh-iteration guard | date: 2026-08-06]
import errno
import json
import os
import time

import pytest

from nems_lbhb.motor import dlc_train_guard as g


# ---------------------------------------------------------------------------
# the brief's six
# ---------------------------------------------------------------------------

def test_lock_is_exclusive(tmp_path):
    p = g.acquire_lock(str(tmp_path), {'user': 'a', 'run_id': '1'})
    assert os.path.exists(p)
    with pytest.raises(RuntimeError) as e:
        g.acquire_lock(str(tmp_path), {'user': 'b', 'run_id': '2'})
    assert 'a' in str(e.value), 'the refusal must name who holds the lock'
    g.release_lock(p)
    p2 = g.acquire_lock(str(tmp_path), {'user': 'b', 'run_id': '2'})
    assert os.path.exists(p2)


def test_lock_holder_reports_metadata(tmp_path):
    g.acquire_lock(str(tmp_path), {'user': 'jereme', 'run_id': 'abc', 'queueid': 7})
    h = g.lock_holder(str(tmp_path))
    assert h['user'] == 'jereme' and h['queueid'] == 7 and h['pid'] == os.getpid()


def test_lock_holder_is_none_when_unlocked(tmp_path):
    assert g.lock_holder(str(tmp_path)) is None


def test_fresh_iteration_guard_refuses_a_populated_train_dir(tmp_path):
    train = tmp_path / 'dlc-models' / 'iteration-1' / 'fake-trainset95shuffle1' / 'train'
    train.mkdir(parents=True)
    (train / 'snapshot-60000.index').write_text('w')
    with pytest.raises(RuntimeError) as e:
        g.assert_fresh_iteration(str(tmp_path), 1)
    assert 'snapshot-60000' in str(e.value)
    g.assert_fresh_iteration(str(tmp_path), 1, resume=True)      # explicit opt-in is allowed


def test_fresh_iteration_guard_passes_on_an_empty_iteration(tmp_path):
    (tmp_path / 'dlc-models' / 'iteration-2').mkdir(parents=True)
    g.assert_fresh_iteration(str(tmp_path), 2)


def test_run_modelprefix_is_unique_per_run():
    a = g.run_modelprefix('jereme', 'aaa')
    b = g.run_modelprefix('jereme', 'bbb')
    assert a != b and a.startswith('runs/') and 'jereme' in a


# ---------------------------------------------------------------------------
# beyond the brief
# ---------------------------------------------------------------------------

def test_lock_records_pid_host_and_time_that_the_caller_cannot_spoof(tmp_path):
    # a caller passing pid/host/started_utc must not be able to lie about them:
    # a stale-lock diagnosis depends on those three being the truth.
    g.acquire_lock(str(tmp_path), {'user': 'x', 'pid': 999999,
                                   'host': 'nowhere', 'started_utc': 'never'})
    h = g.lock_holder(str(tmp_path))
    assert h['pid'] == os.getpid()
    assert h['host'] != 'nowhere'
    assert h['started_utc'].endswith('Z')


def test_lock_refusal_names_host_pid_and_start_time(tmp_path):
    g.acquire_lock(str(tmp_path), {'user': 'a', 'run_id': '1', 'queueid': 42})
    with pytest.raises(RuntimeError) as e:
        g.acquire_lock(str(tmp_path), {'user': 'b'})
    msg = str(e.value)
    for expected in (str(os.getpid()), '42', g.lock_path(str(tmp_path))):
        assert expected in msg


def test_lock_timeout_gives_up_and_still_names_the_holder(tmp_path):
    g.acquire_lock(str(tmp_path), {'user': 'a'})
    t0 = time.time()
    with pytest.raises(RuntimeError) as e:
        g.acquire_lock(str(tmp_path), {'user': 'b'}, timeout=1.0)
    assert 1.0 <= time.time() - t0 < 10.0
    assert 'a' in str(e.value)


def test_corrupt_lockfile_raises_rather_than_being_ignored(tmp_path):
    with open(g.lock_path(str(tmp_path)), 'w') as f:
        f.write('{not json')
    with pytest.raises(RuntimeError) as e:
        g.lock_holder(str(tmp_path))
    assert 'unreadable' in str(e.value)
    # and acquiring must not silently steamroll it
    with pytest.raises(RuntimeError):
        g.acquire_lock(str(tmp_path), {'user': 'b'})


def test_release_lock_refuses_another_process_lock_unless_forced(tmp_path):
    p = g.acquire_lock(str(tmp_path), {'user': 'a'})
    with open(p) as f:
        rec = json.load(f)
    rec['pid'] = os.getpid() + 1000000
    with open(p, 'w') as f:
        json.dump(rec, f)

    with pytest.raises(RuntimeError):
        g.release_lock(p)
    assert os.path.exists(p)

    assert g.release_lock(p, force=True) is True
    assert not os.path.exists(p)
    assert g.release_lock(p) is False       # tolerates absence


def test_acquire_lock_refuses_a_project_that_is_not_a_directory(tmp_path):
    missing = str(tmp_path / 'nope')
    with pytest.raises(ValueError) as e:
        g.acquire_lock(missing, {'user': 'a'})
    assert missing in str(e.value)


def test_fresh_iteration_guard_is_scoped_to_the_requested_shuffle(tmp_path):
    train = tmp_path / 'dlc-models' / 'iteration-1' / 'fake-trainset95shuffle2' / 'train'
    train.mkdir(parents=True)
    (train / 'snapshot-60000.index').write_text('w')
    # shuffle 2 is populated; a shuffle-1 run does not overwrite it
    g.assert_fresh_iteration(str(tmp_path), 1, shuffle=1)
    with pytest.raises(RuntimeError):
        g.assert_fresh_iteration(str(tmp_path), 1, shuffle=2)


def test_fresh_iteration_guard_lists_every_snapshot_at_risk(tmp_path):
    train = tmp_path / 'dlc-models' / 'iteration-1' / 'fake-trainset95shuffle1' / 'train'
    train.mkdir(parents=True)
    for n in (10000, 60000):
        (train / ('snapshot-%d.index' % n)).write_text('w')
    with pytest.raises(RuntimeError) as e:
        g.assert_fresh_iteration(str(tmp_path), 1)
    msg = str(e.value)
    assert 'snapshot-10000' in msg and 'snapshot-60000' in msg
    assert 'resume=1' in msg, 'the refusal must state the opt-in'


def test_fresh_iteration_guard_passes_on_a_project_with_no_dlc_models(tmp_path):
    g.assert_fresh_iteration(str(tmp_path), 0)


def test_run_modelprefix_slugs_unsafe_characters():
    p = g.run_modelprefix('lab/user', 'run id/2')
    assert p.startswith('runs/')
    assert p.count('/') == 1, 'the run component must not introduce path separators'
    assert 'lab-user' in p and 'run-id-2' in p
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 7 -- the
# "lock vanished between the O_EXCL failure and the read" branch used to `continue`
# unconditionally, skipping both the deadline check and the sleep. If a lockfile kept
# disappearing and reappearing at exactly the wrong moment, acquire_lock could spin
# without bound. This drives that exact condition (os.open always EEXIST on the lock
# path, _read_lock always reporting no lock there) and asserts the retry is bounded
# and fails loudly instead of hanging. | date: 2026-08-06]
def test_acquire_lock_bounds_the_vanished_lock_retry(tmp_path, monkeypatch):
    target = g.lock_path(str(tmp_path))
    real_open = os.open

    def flaky_open(path, flags, mode=0o777, *a, **kw):
        if path == target:
            raise OSError(errno.EEXIST, 'File exists')
        return real_open(path, flags, mode, *a, **kw)

    monkeypatch.setattr(g.os, 'open', flaky_open)
    monkeypatch.setattr(g, '_read_lock', lambda path: None)

    t0 = time.time()
    with pytest.raises(RuntimeError) as e:
        g.acquire_lock(str(tmp_path), {'user': 'a'}, timeout=0)
    assert time.time() - t0 < 5.0, 'the vanished-lock retry must be bounded, not spin forever'
    assert 'kept appearing and disappearing' in str(e.value)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- curation:
# label inventory, the merge guard against DLC's silent skip, the recorded delta and
# the staleness check that keeps `training` from running on unmerged labels
# | date: 2026-08-06]
# ---------------------------------------------------------------------------
# the brief's five
# ---------------------------------------------------------------------------

def test_curation_delta_counts_added_frames(tmp_path):
    before = {'v1': {'frames': 10, 'sha256': 'a'}}
    after = {'v1': {'frames': 12, 'sha256': 'b'}, 'v2': {'frames': 5, 'sha256': 'c'}}
    d = g.curation_delta(before, after)
    assert d['folders_added'] == ['v2']
    assert d['folders_changed'] == ['v1']
    assert d['frames_before'] == 10 and d['frames_after'] == 17 and d['frames_added'] == 7


def test_merge_guard_refuses_when_iteration_did_not_advance(tmp_path):
    """DLC's merge_datasets flags a bad folder and silently leaves iteration alone."""
    import yaml
    (tmp_path / 'config.yaml').write_text(yaml.dump({'iteration': 1}))
    with pytest.raises(RuntimeError) as e:
        g.assert_merge_succeeded(str(tmp_path), iteration_before=1)
    assert 'iteration' in str(e.value)


def test_merge_guard_passes_when_iteration_advanced(tmp_path):
    import yaml
    (tmp_path / 'config.yaml').write_text(yaml.dump({'iteration': 2}))
    g.assert_merge_succeeded(str(tmp_path), iteration_before=1)


def test_merge_guard_names_the_flagged_folder(tmp_path):
    """When a labeled-data folder is missing its CollectedData file, say which one."""
    import yaml
    (tmp_path / 'config.yaml').write_text(yaml.dump({'iteration': 1}))
    ok = tmp_path / 'labeled-data' / 'good'
    bad = tmp_path / 'labeled-data' / 'broken'
    ok.mkdir(parents=True)
    bad.mkdir(parents=True)
    (ok / 'CollectedData_x.h5').write_text('l')
    with pytest.raises(RuntimeError) as e:
        g.assert_merge_succeeded(str(tmp_path), iteration_before=1)
    assert 'broken' in str(e.value)


def test_append_curation_log_is_append_only(tmp_path):
    p = g.append_curation_log(str(tmp_path), {'frames_added': 7, 'folders_added': ['v2'],
                                              'folders_changed': [], 'frames_before': 10,
                                              'frames_after': 17}, 'jereme', 'first')
    g.append_curation_log(str(tmp_path), {'frames_added': 0, 'folders_added': [],
                                          'folders_changed': [], 'frames_before': 17,
                                          'frames_after': 17}, 'jereme', 'second')
    lines = [l for l in open(p).read().splitlines() if l.strip()]
    assert len(lines) == 2 and 'first' in lines[0] and 'second' in lines[1]


# ---------------------------------------------------------------------------
# beyond the brief: the inventory
# ---------------------------------------------------------------------------

def _labeled(tmp_path, folder, text='labels', name='CollectedData_svd.h5'):
    d = tmp_path / 'labeled-data' / folder
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(text)
    return f


def _stub_pandas(monkeypatch, rows_by_path=None, default_rows=3):
    """Put a minimal `pandas` in sys.modules so label_inventory can count rows."""
    import sys
    import types
    mod = types.ModuleType('pandas')

    def read_hdf(path, *a, **kw):
        n = (rows_by_path or {}).get(os.path.basename(str(path)), default_rows)
        return list(range(n))

    mod.read_hdf = read_hdf
    monkeypatch.setitem(sys.modules, 'pandas', mod)
    return mod


def test_label_inventory_ignores_folders_with_no_collected_data(tmp_path, monkeypatch):
    """A bare folder name over-reports: right_chimney_PRNfb has an EMPTY
    labeled-data/SQD032a07_a_NTD_recording/, and it contributes no frames."""
    _stub_pandas(monkeypatch)
    _labeled(tmp_path, 'has_labels')
    (tmp_path / 'labeled-data' / 'SQD032a07_a_NTD_recording').mkdir(parents=True)
    inv = g.label_inventory(str(tmp_path))
    assert list(inv) == ['has_labels']
    assert inv['has_labels']['frames'] == 3
    assert inv['has_labels']['frames_source'] == 'pandas'
    assert inv['has_labels']['sha256']


def test_label_inventory_reports_unavailable_rather_than_guessing_a_count(tmp_path,
                                                                         monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'pandas', None)      # `import pandas` -> ImportError
    _labeled(tmp_path, 'v1')
    inv = g.label_inventory(str(tmp_path))
    assert inv['v1']['frames'] is None
    assert inv['v1']['frames_source'] == 'unavailable'


def test_label_inventory_sha_tracks_content_not_mtime(tmp_path, monkeypatch):
    _stub_pandas(monkeypatch)
    _labeled(tmp_path, 'v1', text='one')
    a = g.label_inventory(str(tmp_path))
    _labeled(tmp_path, 'v1', text='two')
    b = g.label_inventory(str(tmp_path))
    assert a['v1']['sha256'] != b['v1']['sha256']
    assert g.curation_delta(a, b)['folders_changed'] == ['v1']


def test_label_inventory_folds_in_duplicate_collected_data_files(tmp_path, monkeypatch):
    """Five real folders hold 'CollectedData_svd [1].h5' next to the real one."""
    _stub_pandas(monkeypatch, default_rows=2)
    _labeled(tmp_path, 'v1', text='real')
    _labeled(tmp_path, 'v1', text='dup', name='CollectedData_svd [1].h5')
    inv = g.label_inventory(str(tmp_path))
    assert inv['v1']['file_count'] == 2
    assert inv['v1']['frames'] == 4
    assert len(inv['v1']['files']) == 2


def test_curation_delta_refuses_to_invent_a_frame_count(tmp_path):
    before = {'v1': {'frames': None, 'sha256': 'a'}}
    after = {'v1': {'frames': None, 'sha256': 'a'}, 'v2': {'frames': 5, 'sha256': 'c'}}
    d = g.curation_delta(before, after)
    assert d['frames_counted'] is False
    assert d['frames_before'] is None and d['frames_after'] is None
    assert d['frames_added'] is None
    assert d['folders_added'] == ['v2']


def test_curation_delta_reports_a_removed_folder(tmp_path):
    d = g.curation_delta({'v1': {'frames': 3, 'sha256': 'a'}}, {})
    assert d['folders_removed'] == ['v1']
    assert d['frames_after'] == 0


def test_merge_guard_passes_when_every_folder_has_labels(tmp_path):
    import yaml
    (tmp_path / 'config.yaml').write_text(yaml.dump({'iteration': 1}))
    _labeled(tmp_path, 'v1')
    with pytest.raises(RuntimeError) as e:
        g.assert_merge_succeeded(str(tmp_path), iteration_before=1)
    msg = str(e.value)
    assert 'every labeled-data folder' in msg, msg      # nothing to blame -> say so


def test_merge_guard_uses_dlcs_own_flagging_rule(tmp_path):
    """DLC flags a folder holding neither MachineLabelsRefine.h5 nor
    CollectedData_<scorer>.h5 -- the scorer is exact, and refined folders pass."""
    import yaml
    (tmp_path / 'config.yaml').write_text(yaml.dump({'iteration': 1, 'scorer': 'svd'}))
    _labeled(tmp_path, 'wrong_scorer', name='CollectedData_someone_else.h5')
    _labeled(tmp_path, 'right_scorer', name='CollectedData_svd.h5')
    _labeled(tmp_path, 'refined', name='MachineLabelsRefine.h5')
    (tmp_path / 'labeled-data' / 'right_scorer_labeled').mkdir()   # DLC skips these
    assert g.unlabelled_folders(str(tmp_path), scorer='svd') == ['wrong_scorer']
    with pytest.raises(RuntimeError) as e:
        g.assert_merge_succeeded(str(tmp_path), iteration_before=1)
    assert 'wrong_scorer' in str(e.value)


def test_append_curation_log_records_who_when_and_the_label_mtime(tmp_path, monkeypatch):
    _stub_pandas(monkeypatch)
    _labeled(tmp_path, 'v1')
    p = g.append_curation_log(str(tmp_path), g.curation_delta({}, g.label_inventory(str(tmp_path))),
                              'jereme', 'merged v1')
    rec = json.loads(open(p).read().splitlines()[-1])
    assert rec['user'] == 'jereme' and rec['note'] == 'merged v1'
    assert rec['utc'].endswith('Z')
    assert rec['labels_newest_mtime'] > 0
    assert rec['folders_added'] == ['v1']


# ---------------------------------------------------------------------------
# beyond the brief: the staleness check that gates `training`
# ---------------------------------------------------------------------------

def test_training_refuses_when_there_is_no_curation_log(tmp_path, monkeypatch):
    _stub_pandas(monkeypatch)
    _labeled(tmp_path, 'v1')
    with pytest.raises(RuntimeError) as e:
        g.assert_curation_current(str(tmp_path))
    msg = str(e.value)
    assert 'curate' in msg
    assert g.CURATION_LOG_NAME in msg


def test_training_refuses_when_labels_are_newer_than_the_last_curation(tmp_path,
                                                                       monkeypatch):
    _stub_pandas(monkeypatch)
    f = _labeled(tmp_path, 'v1')
    g.append_curation_log(str(tmp_path), g.curation_delta({}, g.label_inventory(str(tmp_path))),
                          'jereme', 'merged v1')
    g.assert_curation_current(str(tmp_path))              # current: passes

    # someone labels another session afterwards and goes straight to Train
    later = os.path.getmtime(str(f)) + 3600
    f2 = _labeled(tmp_path, 'v2')
    os.utime(str(f2), (later, later))
    with pytest.raises(RuntimeError) as e:
        g.assert_curation_current(str(tmp_path))
    assert 'curate' in str(e.value)
    assert 'v2' in str(e.value), 'name the labels that are newer than the last curation'


def test_curation_is_current_again_after_a_second_curate(tmp_path, monkeypatch):
    _stub_pandas(monkeypatch)
    f = _labeled(tmp_path, 'v1')
    g.append_curation_log(str(tmp_path), g.curation_delta({}, {}), 'jereme', 'first')
    later = os.path.getmtime(str(f)) + 3600
    os.utime(str(f), (later, later))
    with pytest.raises(RuntimeError):
        g.assert_curation_current(str(tmp_path))
    g.append_curation_log(str(tmp_path), g.curation_delta({}, g.label_inventory(str(tmp_path))),
                          'jereme', 'second')
    g.assert_curation_current(str(tmp_path))


def test_assert_curation_current_passes_a_project_with_no_labels_at_all(tmp_path):
    # nothing to merge, nothing to be stale about
    g.assert_curation_current(str(tmp_path))


def test_read_curation_log_skips_blank_lines_and_raises_on_corruption(tmp_path):
    path = os.path.join(str(tmp_path), g.CURATION_LOG_NAME)
    with open(path, 'w') as f:
        f.write(json.dumps({'utc': '2026-08-06T00:00:00Z', 'user': 'a', 'note': 'x'}) + '\n')
        f.write('\n')
    assert len(g.read_curation_log(str(tmp_path))) == 1
    with open(path, 'a') as f:
        f.write('{not json\n')
    with pytest.raises(RuntimeError) as e:
        g.read_curation_log(str(tmp_path))
    assert g.CURATION_LOG_NAME in str(e.value)
# [AGENT EDIT END]


# --- fail-open paths closed after review -----------------------------------------

def _mk_project_with_label(tmp_path, mtime=None):
    proj = tmp_path / 'proj'
    lab = proj / 'labeled-data' / 'V1'
    lab.mkdir(parents=True)
    f = lab / 'CollectedData_x.h5'
    f.write_text('labels')
    if mtime is not None:
        os.utime(str(f), (mtime, mtime))
    return proj, f


def test_curation_log_stamps_the_mtime_the_caller_saw(tmp_path):
    """The DLC labeling GUI does not take the curate lock, so a label written DURING
    the run was never merged. Stamping a post-merge stat would record it as curated."""
    proj, f = _mk_project_with_label(tmp_path, mtime=1000.0)
    g.append_curation_log(str(proj), {'frames_added': 0}, 'u', 'n',
                          labels_newest_mtime=1000.0)
    # a label lands after the merge started
    os.utime(str(f), (5000.0, 5000.0))
    with pytest.raises(RuntimeError) as e:
        g.assert_curation_current(str(proj))
    assert 'newer than the last recorded curation' in str(e.value)


def test_curation_log_without_the_caller_mtime_still_works(tmp_path):
    proj, f = _mk_project_with_label(tmp_path, mtime=1000.0)
    g.append_curation_log(str(proj), {'frames_added': 0}, 'u', 'n')
    g.assert_curation_current(str(proj))


def test_entry_missing_labels_newest_mtime_refuses_rather_than_softening(tmp_path):
    """A hand-edited or foreign entry has no exact reference. Refusing is fail-safe;
    the old wall-clock+tolerance fallback could wave a stale label through."""
    import json
    proj, f = _mk_project_with_label(tmp_path, mtime=1000.0)
    with open(g.curation_log_path(str(proj)), 'a') as fh:
        fh.write(json.dumps({'utc': '2026-08-06T12:00:00', 'user': 'u', 'note': ''}) + '\n')
    with pytest.raises(RuntimeError) as e:
        g.assert_curation_current(str(proj))
    assert 'labels_newest_mtime' in str(e.value)
    assert 'curate' in str(e.value)


# --- refinement data is training data --------------------------------------------
# DLC's merge_datasets accepts a folder holding EITHER CollectedData_<scorer>.h5 or
# MachineLabelsRefine.h5, so refined outlier frames count. Watching only the former
# left the staleness gate blind to a freshly refined folder.

def test_label_files_counts_refinement_data(tmp_path):
    proj = tmp_path / 'proj'
    (proj / 'labeled-data' / 'refined').mkdir(parents=True)
    (proj / 'labeled-data' / 'refined' / 'MachineLabelsRefine.h5').write_text('r')
    (proj / 'labeled-data' / 'manual').mkdir(parents=True)
    (proj / 'labeled-data' / 'manual' / 'CollectedData_x.h5').write_text('m')
    (proj / 'labeled-data' / 'empty').mkdir(parents=True)
    found = {os.path.basename(os.path.dirname(f)) for f in g.label_files(str(proj))}
    assert found == {'refined', 'manual'}


def test_refined_folder_appears_in_the_inventory(tmp_path):
    proj = tmp_path / 'proj'
    (proj / 'labeled-data' / 'refined').mkdir(parents=True)
    (proj / 'labeled-data' / 'refined' / 'MachineLabelsRefine.h5').write_text('r')
    assert 'refined' in g.label_inventory(str(proj))


def test_staleness_gate_fires_on_newly_refined_data(tmp_path):
    """The fail-open this closes: refine outliers, then train without curating."""
    proj = tmp_path / 'proj'
    lab = proj / 'labeled-data' / 'V1'
    lab.mkdir(parents=True)
    manual = lab / 'CollectedData_x.h5'
    manual.write_text('m')
    os.utime(str(manual), (1000.0, 1000.0))
    g.append_curation_log(str(proj), {'frames_added': 0}, 'u', 'n',
                          labels_newest_mtime=1000.0)
    g.assert_curation_current(str(proj))          # clean before refinement
    refined = proj / 'labeled-data' / 'V2'
    refined.mkdir(parents=True)
    rf = refined / 'MachineLabelsRefine.h5'
    rf.write_text('r')
    os.utime(str(rf), (9000.0, 9000.0))
    with pytest.raises(RuntimeError) as e:
        g.assert_curation_current(str(proj))
    assert 'MachineLabelsRefine.h5' in str(e.value)
