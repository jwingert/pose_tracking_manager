# [AGENT EDIT START | agent: claude | user: wingertj | reason: per-alias generalization
# sets -- the held-out property is only worth anything if the store enforces it, so the
# leakage refusal, the append-only manual manifest, and compatibility with the layout
# scripts/dlc_generalization_setup.py already built all get pinned here | date: 2026-08-18]
import csv
import os

import pytest
import yaml

from nems_lbhb.motor import dlc_generalization as gen


def _video(tmp_path, animal, session, name='recording.avi'):
    d = tmp_path / 'daq' / animal / session[:6] / session
    d.mkdir(parents=True, exist_ok=True)
    v = d / name
    v.write_bytes(b'x')
    return str(v)


def _project(tmp_path, video_sets=(), labeled=()):
    p = tmp_path / 'proj'
    (p / 'labeled-data').mkdir(parents=True, exist_ok=True)
    for name in labeled:
        (p / 'labeled-data' / name).mkdir()
    cfg = p / 'config.yaml'
    with open(cfg, 'w') as f:
        yaml.safe_dump({'video_sets': {v: {'crop': '0'} for v in video_sets},
                        'bodyparts': ['front', 'back']}, f)
    return str(cfg)


def test_add_follows_the_setup_scripts_layout(tmp_path):
    root = str(tmp_path / 'gen')
    v = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    link = gen.add_video('left_chimney', v, by='wingertj', root=root)

    # videos/<Animal>_<session>.avi, exactly as dlc_generalization_setup writes
    assert link.endswith('left_chimney/videos/Reishi_REI900a01_a_NFB.avi')
    assert os.path.islink(link)
    assert os.path.realpath(link) == os.path.realpath(v)

    entries = gen.list_videos('left_chimney', root=root)
    assert [e['session'] for e in entries] == ['REI900a01_a_NFB']
    assert entries[0]['animal'] == 'Reishi'
    assert entries[0]['exists'] and entries[0]['added_by'] == 'wingertj'


def test_setup_script_manifest_csv_is_read_and_never_written(tmp_path):
    root = str(tmp_path / 'gen')
    d = tmp_path / 'gen' / 'left_chimney'
    (d / 'videos').mkdir(parents=True)
    daq = _video(tmp_path, 'Prince', 'PRN024a01_a_NTD', name='PRN024a01_a_NTD.avi')
    os.symlink(daq, d / 'videos' / 'Prince_PRN024a01_a_NTD.avi')
    with open(d / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video_stem', 'animal', 'eval_mode',
                                          'avi_path'])
        w.writeheader()
        w.writerow({'video_stem': 'PRN024a01_a_NTD', 'animal': 'Prince',
                    'eval_mode': 'cross_animal', 'avi_path': daq})
    before = open(d / 'manifest.csv').read()

    entries = gen.list_videos('left_chimney', root=root)
    assert entries[0]['added_by'] == 'dlc_generalization_setup'
    assert entries[0]['note'] == 'cross_animal'

    # a GUI add must land in manifest.json -- the CSV is regenerated wholesale by the
    # setup script, so a manual row there would be clobbered on its next run
    v2 = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    gen.add_video('left_chimney', v2, by='w', root=root)
    assert open(d / 'manifest.csv').read() == before
    assert os.path.exists(d / 'manifest.json')
    sessions = {e['session'] for e in gen.list_videos('left_chimney', root=root)}
    assert sessions == {'PRN024a01_a_NTD', 'REI900a01_a_NFB'}


def test_duplicate_session_is_refused(tmp_path):
    root = str(tmp_path / 'gen')
    v = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    gen.add_video('left_chimney', v, by='w', root=root)
    with pytest.raises(ValueError) as e:
        gen.add_video('left_chimney', v, by='w', root=root)
    assert 'already in' in str(e.value)


def test_a_training_session_cannot_enter_the_set(tmp_path):
    # add_video_to_dataset renames copies to <session>_<basename>.avi, so the training
    # name CONTAINS the stem -- the check must be substring, not equality
    root = str(tmp_path / 'gen')
    cfg = _project(tmp_path,
                   video_sets=['/proj/videos/REI900a01_a_NFB_recording.avi'])
    v = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    with pytest.raises(ValueError) as e:
        gen.add_video('left_chimney', v, by='w', project_config=cfg, root=root)
    assert 'training project' in str(e.value)


def test_labeled_data_outside_video_sets_still_counts_as_training(tmp_path):
    # one add_new_videos call away from entering video_sets -- treated as leaked
    cfg = _project(tmp_path, labeled=['SQD901b02_a_NFB_recording'])
    assert gen.in_training('SQD901b02_a_NFB', cfg)
    assert not gen.in_training('REI900a01_a_NFB', cfg)


def test_missing_target_is_reported_not_hidden(tmp_path):
    root = str(tmp_path / 'gen')
    v = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    gen.add_video('left_chimney', v, by='w', root=root)
    os.remove(v)
    entries = gen.list_videos('left_chimney', root=root)
    assert len(entries) == 1 and entries[0]['exists'] is False


def test_recipe_labels_in_the_companion_never_the_training_project(tmp_path):
    root = str(tmp_path / 'gen')
    v = _video(tmp_path, 'Reishi', 'REI900a01_a_NFB')
    gen.add_video('left_chimney', v, by='w', root=root)
    cfg = _project(tmp_path)
    recipe = gen.labeling_recipe('left_chimney', cfg, root=root)
    assert 'create_new_project' in recipe
    assert 'label_frames' in recipe
    assert str(tmp_path / 'gen') in recipe          # companion lives in the set dir
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: a project name is not a
# model identity -- the same candidate evaluated at two snapshots landed in
# results_sampled.csv indistinguishable. Every row now records the snapshot its config
# resolved to, and schema growth rewrites the file instead of silently misaligning new
# columns against the old header. | date: 2026-08-18]
def test_resolved_snapshot_follows_the_configs_index(tmp_path):
    p = tmp_path / 'proj'
    train = p / 'dlc-models' / 'iteration-2' / 'modelJan1-trainset95shuffle1' / 'train'
    train.mkdir(parents=True)
    for n in (75000, 100000, 125000, 150000, 160000):
        (train / f'snapshot-{n}.index').write_bytes(b'')
    cfg = p / 'config.yaml'
    with open(cfg, 'w') as f:
        yaml.safe_dump({'iteration': 2, 'snapshotindex': 3}, f)
    assert gen.resolved_snapshot(str(cfg)) == 150000     # index 3, numerically sorted

    with open(cfg, 'w') as f:
        yaml.safe_dump({'iteration': 2, 'snapshotindex': -1}, f)
    assert gen.resolved_snapshot(str(cfg)) == 160000     # -1 = last


def test_append_rows_rewrites_when_the_schema_grows(tmp_path):
    path = str(tmp_path / 'results.csv')
    gen._append_rows(path, [{'candidate': 'a', 'ratio_mean': '0.9'}])
    # a later writer adds a column the first rows predate
    gen._append_rows(path, [{'candidate': 'b', 'ratio_mean': '0.8',
                             'snapshot': '150000'}])
    rows = gen.sampled_results.__wrapped__ if hasattr(gen.sampled_results, '__wrapped__') \
        else None
    import csv
    got = list(csv.DictReader(open(path, newline='')))
    assert [r['candidate'] for r in got] == ['a', 'b']
    assert got[0]['snapshot'] == ''                      # old row: explicit empty, not shifted
    assert got[1]['snapshot'] == '150000'
    import os, stat
    assert os.stat(path).st_mode & 0o666 == 0o666        # both accounts can rewrite
# [AGENT EDIT END]


def test_h5_selection_demands_the_snapshots_own_suffix(tmp_path):
    # a 150000 evaluation silently re-scored the 160000 run's predictions because
    # sorted()[-1] is lexicographic and '_160000.h5' sorts last -- attribution must come
    # from the resolved identity, and a missing match must raise, never guess
    d = str(tmp_path)
    for snap in (150000, 160000):
        (tmp_path / f'Prince_PRN024DLC_resnet50_mergedshuffle1_{snap}.h5').write_bytes(b'x')

    got = gen._h5_for_snapshot(d, 'Prince_PRN024', 150000)
    assert got.endswith('_150000.h5')

    with pytest.raises(RuntimeError):
        gen._h5_for_snapshot(d, 'Prince_PRN024', 999999)     # no such output
    with pytest.raises(RuntimeError):
        gen._h5_for_snapshot(d, 'Prince_PRN024', None)       # ambiguous, unresolvable
