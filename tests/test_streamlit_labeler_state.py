"""Offline regression checks for routing freshness and conditional GUI execution."""
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: test metadata changes, manual overrides, cache invalidation, and hidden-section state | date: 2026-09-09]
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from nems_lbhb.motor.streamlit_vid_labeler.labeler_state import apply_routing


LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_metadata_correction_updates_auto_selection_but_preserves_manual_choice():
    state = {}
    context = ("SQD092b02", "geometry")
    apply_routing(state, context, {"alias": "right_chimney"})
    apply_routing(state, context, {"alias": "left_chimney"})
    assert state["routing_alias"] == "left_chimney"
    state.update(routing_alias="two_chimney", routing_manual=True)
    apply_routing(state, context, {"alias": "right_chimney"})
    assert state["routing_alias"] == "two_chimney"
    apply_routing(state, ("another_video", "geometry"), {"alias": "left_chimney"})
    assert state["routing_alias"] == "left_chimney"
    assert not state["routing_manual"]


def test_failed_refresh_never_arms_an_unverified_model():
    state = {}
    context = ("video", "geometry")
    failed = {"alias": "right_chimney", "refresh_error": "DB down"}
    apply_routing(state, context, failed)
    assert "routing_alias" not in state
    assert state["routing_declined"]
    apply_routing(state, context, {"alias": "left_chimney"})
    apply_routing(state, context, failed)
    assert state["routing_alias"] == "left_chimney"
    apply_routing(state, context, {"alias": None})
    assert "routing_alias" not in state


@pytest.fixture
def core(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    module = importlib.import_module("labeler_core")
    monkeypatch.setattr(module, "PREFS_FILE", tmp_path / "prefs.json")
    return module


def test_tab_selector_preserves_selection_and_formatted_option_values(core):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
import labeler_core as core
core.inject_css()
section = core.tab_selector("Section", ["Model", "Update model"], key="app_section")
step = core.tab_selector("Step", range(3), key="update_section",
                         format_func=lambda i: f"Step {i + 1}")
st.radio("Routing method", ["Geometry", "Manual"], key="ordinary_radio")
st.write(f"Selected: {section}, {step}")
''').run()
    assert not app.exception
    assert app.radio(key="app_section").value == "Model"
    assert app.radio(key="update_section").options == ["Step 1", "Step 2", "Step 3"]
    app.radio(key="app_section").set_value("Update model").run()
    app.radio(key="update_section").set_value(2).run()
    app.radio(key="ordinary_radio").set_value("Manual").run()
    assert not app.exception
    assert app.radio(key="app_section").value == "Update model"
    assert app.radio(key="update_section").value == 2


def test_routing_refresh_expiry_registry_change_failure_and_recovery(core, monkeypatch):
    from nems_lbhb.motor import dlc_routing
    state = State()
    monkeypatch.setattr(core.st, "session_state", state)
    clock, registry, calls = [100.0], [1], []
    monkeypatch.setattr(core.time, "time", lambda: clock[0])
    monkeypatch.setattr(core, "_registry_mtime", lambda: registry[0])
    result = {"alias": "right_chimney", "observed": {"db_errors": []}}

    def suggest(*args, **kwargs):
        calls.append(args)
        return dict(result)

    monkeypatch.setattr(dlc_routing, "suggest", suggest)
    assert core.routing_suggestion("video")["alias"] == "right_chimney"
    result["alias"] = "left_chimney"
    assert core.routing_suggestion("video")["alias"] == "right_chimney"
    assert len(calls) == 1
    clock[0] += 31
    assert core.routing_suggestion("video")["alias"] == "left_chimney"
    registry[0] += 1
    core.routing_suggestion("video")
    assert len(calls) == 3
    result["observed"] = {"db_errors": ["ear lookup failed"]}
    core.refresh_routing()
    failed = core.routing_suggestion("video")
    assert failed["alias"] == "left_chimney"
    assert failed["refresh_error"]
    assert failed["checked_at"] == 131
    result["observed"] = {"db_errors": []}
    core.refresh_routing()
    assert not core.routing_suggestion("video")["refresh_error"]
    state.clear()  # a new browser session must not inherit another session's cache
    core.routing_suggestion("video")
    assert len(calls) == 6


def test_overlay_reuses_data_and_detects_symlink_retarget(core, monkeypatch, tmp_path):
    monkeypatch.setattr(core.st, "session_state", State())
    first, second, pointer = (tmp_path / name for name in ("first.h5", "second.h5", "current.h5"))
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    pointer.symlink_to(first)
    reads = []

    def read(stamp):
        reads.append(stamp)
        return pd.DataFrame({"x": [len(reads), 20]})

    monkeypatch.setattr(core, "_prediction_data", read)
    assert core.prediction_frame(pointer, 0)["x"] == 1
    assert core.prediction_frame(pointer, 1)["x"] == 20
    assert len(reads) == 1
    pointer.unlink()
    pointer.symlink_to(second)
    assert core.prediction_frame(pointer, 0)["x"] == 2
    second.write_bytes(b"updated prediction")
    assert core.prediction_frame(pointer, 0)["x"] == 3


def test_model_discovery_is_shared_until_explicit_refresh(core, monkeypatch):
    scans = []
    core.discover_dlc_models.clear()

    def glob(*args, **kwargs):
        scans.append(args)
        return ["/models/one/config.yaml"]

    monkeypatch.setattr(core.glob, "glob", glob)
    assert core.discover_dlc_models() == {"one": "/models/one/config.yaml"}
    assert core.discover_dlc_models() == {"one": "/models/one/config.yaml"}
    assert len(scans) == 1
    core.refresh_files()
    core.discover_dlc_models()
    assert len(scans) == 2
    core.discover_dlc_models.clear()


def test_geometry_edits_mark_unsaved_and_block_reload(core, monkeypatch):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import labeler_core as core
from tabs import arena_geometry
core.init_session_state()
core.retain_widget_values()
st.session_state.setdefault("geometry_test_seeded", False)
if not st.session_state.geometry_test_seeded:
    st.session_state.points = {"A": (0, 0), "B": (10, 0)}
    st.session_state.distances = {"A_B": 0.0}
    st.session_state.geometry_test_seeded = True
arena_geometry.render()
''').run(timeout=20)
    assert not app.exception
    app.number_input(key="dist_A_B").set_value(1.0).run()
    assert not app.exception
    assert app.session_state["arena_config_dirty"]
    state = State(arena_config_dirty=True, video_path="video")
    monkeypatch.setattr(core.st, "session_state", state)
    calls = []
    monkeypatch.setattr(core, "load_sorted_config_for_session", calls.append)
    core.reload_arena_config()
    assert not calls
    state.arena_config_dirty = False
    core.reload_arena_config()
    assert calls == ["video"]


def test_widget_values_survive_hiding_their_section(core):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import labeler_core as core
core.retain_widget_values()
section = st.radio("Section", ["Model", "Geometry"], key="app_section")
if section == "Model":
    st.selectbox("Model", ["left", "right"], key="routing_alias")
else:
    st.text_input("Shift", key="fixation_shift_x")
''').run(timeout=20)
    app.selectbox(key="routing_alias").select("right").run()
    app.radio(key="app_section").set_value("Geometry").run()
    app.text_input(key="fixation_shift_x").input("0.75").run()
    app.radio(key="app_section").set_value("Model").run()
    assert not app.exception
    assert app.selectbox(key="routing_alias").value == "right"
    assert app.session_state["fixation_shift_x"] == "0.75"


def test_arena_reload_replaces_retained_geometry_values(core, monkeypatch):
    from io import BytesIO
    state = State(points={}, distances={}, speakers={}, current_frame=None,
                  dist_A_B=99.0, R_dist=99.0, R_angle=99.0, geometry_edit_fingerprint="old")
    monkeypatch.setattr(core.st, "session_state", state)
    config = BytesIO(b'[distances]\nA_B = 2.0\n[speakers.R]\ndistance = 3.0\nangle = 30.0\n')
    assert core.import_config(config)
    assert state.dist_A_B == 2.0
    assert state.R_dist == 3.0
    assert state.R_angle == 30.0
    assert "geometry_edit_fingerprint" not in state
    assert not state.arena_config_dirty
    assert not core.import_config(BytesIO(b'invalid toml'))


def test_entry_executes_only_selected_section(core, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from tabs import model, update_model, database, new_model, arena_markers, arena_geometry
    monkeypatch.setattr(core, "discover_dlc_models", lambda: {})
    monkeypatch.setattr(core, "code_banner", lambda: None)
    calls = []
    names = ["Model", "Update model", "Database", "New DLC model", "Arena markers", "Arena geometry"]
    modules = [model, update_model, database, new_model, arena_markers, arena_geometry]
    for name, module in zip(names, modules):
        monkeypatch.setattr(module, "render", lambda name=name: calls.append(name))
    app = AppTest.from_file(str(LABELER / "behavior_config_labeler_tabs.py")).run(timeout=20)
    assert not app.exception
    assert calls == ["Model"]
    for name in names[1:]:
        calls.clear()
        app.radio(key="app_section").set_value(name).run()
        assert not app.exception
        assert calls == [name]


def test_fit_status_can_skip_pointer_reads_without_changing_other_columns(monkeypatch):
    from nems_lbhb.motor import dlc_fit_record as fr
    row = {c: None for c in fr.COLUMNS}
    row.update(id=1, video_stem="SQD092b02_a_NFC", stream="", qc="{}",
               attribution="unknown", target_mtime_utc="2026-09-09", siteid="SQD092b")
    received = []

    def fits_for(**filters):
        received.append(filters)
        return pd.DataFrame([row])

    monkeypatch.setattr(fr, "fits_for", fits_for)
    monkeypatch.setattr(fr, "is_stale", lambda row: True)

    def no_pointer_reads(df):
        raise AssertionError("Database-only view accessed filesystem pointers")

    monkeypatch.setattr(fr, "_is_current_series", no_pointer_reads)
    result = fr.fit_status(siteid="SQD092b", include_current=False)
    assert received == [{"siteid": "SQD092b"}]
    assert result["is_current"].isna().all()
    assert result["stale"].all()


def test_real_model_picker_refresh_and_section_round_trip(core, monkeypatch, tmp_path):
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from nems_lbhb.motor import dlc_routing

    session = tmp_path / "SQD092" / "SQD092b02_a_NFC"
    session.mkdir(parents=True)
    video = session / "recording.avi"
    video.write_bytes(b"stub")
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text("Task: test\nscorer: test\nproject_path: " + str(project)
                      + "\nbodyparts: []\nvideo_sets: {}\niteration: 0\n")
    monkeypatch.setattr(core, "discover_dlc_models", lambda: {"project": str(config)})
    monkeypatch.setattr(core, "code_banner", lambda: None)
    monkeypatch.setattr(core, "remember_pref", lambda *args: None)
    monkeypatch.setattr(core, "video_frame_count", lambda path: 2)
    monkeypatch.setattr(core, "read_frame", lambda *args: Image.new("RGB", (64, 48)))
    monkeypatch.setattr(core, "h5_options", lambda *args, **kwargs: {})
    monkeypatch.setattr(core, "fit_status_cached", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(core, "resolve_alias", lambda alias: {
        "alias": alias, "config": str(config), "project": str(project), "source": "project"})
    suggestion = ["right_chimney"]

    def suggest(*args, **kwargs):
        return {"method": "geometry", "alias": suggestion[0], "observed": {"db_errors": []},
                "candidates": [{"alias": suggestion[0], "release": None, "why": [], "unknown": []}]}

    monkeypatch.setattr(dlc_routing, "suggest", suggest)
    app = AppTest.from_file(str(LABELER / "behavior_config_labeler_tabs.py"))
    app.query_params["video"] = str(video)
    app.run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="routing_alias").value == "right_chimney"
    app.selectbox(key="routing_alias").select("two_chimney").run()
    assert not app.exception
    suggestion[0] = "left_chimney"
    app.button(key="routing_refresh").click().run()
    assert not app.exception
    assert app.selectbox(key="routing_alias").value == "two_chimney"
    app.button(key="routing_follow").click().run()
    assert not app.exception
    assert app.selectbox(key="routing_alias").value == "left_chimney"
    app.radio(key="app_section").set_value("New DLC model").run()
    assert not app.exception
    app.radio(key="app_section").set_value("Model").run()
    assert not app.exception
    assert app.selectbox(key="routing_alias").value == "left_chimney"
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: verify browser-independent spread and session/release-aware labeled-video preflight | date: 2026-09-10]
def test_spread_uses_bounded_svg_without_mutating_qc(core):
    from types import SimpleNamespace
    import numpy as np
    points = np.linspace(-4, 4, 12000).reshape(-1, 2)
    original = points.copy()
    result = SimpleNamespace(table=pd.DataFrame({'role': ['rigid']}, index=['ear']),
                             scatter_proc={'ear': points}, reference=None,
                             mean_shape_proc=None, mean_shape=None)
    figure = core._hfqc_figure(result)
    assert all(trace.type != 'scattergl' for trace in figure.data)
    assert len(figure.data[0].x) == 1000
    assert np.max(np.abs(figure.data[0].x)) <= 3
    np.testing.assert_array_equal(points, original)
    assert 'QC statistics unchanged' in core._hfqc_frame_caption(result)


@pytest.mark.parametrize('camera', ['recording.avi', 'video_side2.avi'])
def test_video_button_finds_session_named_release_prediction(core, monkeypatch, tmp_path, camera):
    from nems_lbhb.motor import dlc_fit_script as fs
    video = tmp_path / 'SQD092b02_a_NFC' / 'recording.avi'
    video.parent.mkdir()
    video.touch()
    actual = video.with_name(camera)
    actual.touch()
    folder = core.session_temp_dir(video) / 'left_chimney-r0001'
    folder.mkdir(parents=True)
    prediction = folder / (fs.analysis_video_stem(actual) + 'DLC_test.h5')
    prediction.touch()
    prediction.with_name(prediction.stem + '_meta.pickle').touch()
    monkeypatch.setattr(core.st, 'session_state', State(video_path=str(video)))
    monkeypatch.setattr(core, 'production_alias', lambda: 'left_chimney')
    monkeypatch.setattr(core, 'resolve_alias', lambda a: {
        'release_id': 'left_chimney-r0001', 'config': '/release/config.yaml'})
    monkeypatch.setattr(core, 'ceiling_video', lambda v: (str(actual), None))
    monkeypatch.setattr(fs, 'release_output_suffix', lambda r: 'DLC_test')
    calls, warnings = [], []
    monkeypatch.setattr(core, 'queue_dlc_job', lambda *a, **kw: calls.append((a, kw)) or 'test job')
    monkeypatch.setattr(core.st, 'warning', warnings.append)
    core.video_to_temp()
    assert not warnings
    assert calls == [((video, None, str(folder.parent), 'video'),
                      {'extra_note': 'temp_labeled_video', 'alias': 'left_chimney'})]
    prediction.unlink()
    core.video_to_temp()
    assert len(calls) == 1
    assert 'No prediction' in warnings[-1]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: verify writable montage caching, browser-playable encoding, and actionable failures | date: 2026-09-10]
@pytest.fixture
def montage_source(core, monkeypatch, tmp_path):
    import imageio
    import numpy as np
    source_dir = tmp_path / 'release'
    source_dir.mkdir()
    video = source_dir / 'session_labeled.mp4'
    with imageio.get_writer(str(video), fps=10, codec='libx264') as writer:
        for i in range(30):
            writer.append_data(np.full((32, 32, 3), i * 7, dtype=np.uint8))
    cache = tmp_path / 'preview_cache'
    cache.mkdir()
    monkeypatch.setattr(core, '_montage_cache_dir', lambda: cache)
    source_dir.chmod(0o555)
    try:
        yield video, cache
    finally:
        source_dir.chmod(0o755)


def test_montage_encodes_outside_readonly_release(core, montage_source):
    video, cache = montage_source
    output, starts, error = core.extract_video_montage(video, n_clips=3, clip_duration=0.5)
    assert error is None
    assert output.parent == cache and output.suffix == '.mp4'
    assert len(starts) == 3
    assert list(video.parent.iterdir()) == [video]
    cap = core.cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened()
        assert int(cap.get(core.cv2.CAP_PROP_FRAME_COUNT)) == 15
    finally:
        cap.release()


def test_montage_cache_tracks_parameters_and_source_revision(core, montage_source):
    video, _ = montage_source
    original = core.montage_cache_path(video, 3, 0.5)
    assert core.montage_cache_path(video, 4, 0.5) != original
    assert core.montage_cache_path(video, 3, 1.0) != original
    video.touch()
    assert core.montage_cache_path(video, 3, 0.5) != original


def test_montage_reports_encoder_error_and_removes_partial_output(core, monkeypatch, montage_source):
    import imageio
    video, cache = montage_source

    def fail(*args, **kwargs):
        raise PermissionError('test output directory denied')

    monkeypatch.setattr(imageio, 'get_writer', fail)
    output, starts, error = core.extract_video_montage(video, n_clips=3, clip_duration=0.5)
    assert output is None and starts is None
    assert 'PermissionError' in error and 'directory denied' in error
    assert not list(cache.iterdir())
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: regress independent fit selection, provenance, synchronized visuals and explicit publication | date: 2026-09-10]
@pytest.fixture
def inspection(core):
    module = importlib.import_module('fit_inspection')
    module.inventory.clear()
    module._score.clear()
    return module


@pytest.fixture
def fit_files(tmp_path):
    session = 'SQD092b02_a_NFC'
    video = tmp_path / session / 'recording.avi'
    video.parent.mkdir()
    video.touch()
    releases = tmp_path / 'releases'
    fits = []
    for origin, rid in [('temp', 'left-r0001'), ('sorted', 'right-r0001')]:
        folder = (tmp_path / 'tmp' / 'dlc' / session if origin == 'temp' else tmp_path / 'sorted')
        (folder / rid).mkdir(parents=True)
        prediction = folder / rid / (session + 'DLC_test.h5')
        prediction.touch()
        (folder / (session + '.' + rid + '.dlc.h5')).symlink_to(prediction)
        (folder / (session + '.dlc.h5')).symlink_to(prediction)
        release = releases / rid
        release.mkdir(parents=True)
        (release / 'RELEASE.toml').write_text(
            f'[release]\nid="{rid}"\nalias="{rid.split("-")[0]}"\n'
            '[model]\noutput_suffix="DLC_test"\n')
        (release / 'config.yaml').write_text('bodyparts: []\n')
        fits.append(prediction)
    return video, releases, fits


def test_inventory_sees_all_models_origins_and_deduplicates_links(inspection, fit_files):
    video, releases, paths = fit_files
    # Another session in the shared sorted directory must not leak into this one.
    (paths[1].parent / 'SQD092b03_a_NFCDLC_test.h5').touch()
    rows = inspection.inventory(str(video), str(releases))
    assert len(rows) == 2
    assert {f['release_id'] for f in rows} == {'left-r0001', 'right-r0001'}
    assert rows[0]['published'] and rows[0]['origins'] == ['sorted']
    assert rows[1]['origins'] == ['temp'] and not rows[1]['published']
    assert all(f['video'] == str(video) for f in rows)
    assert inspection.inventory(str(video.parent) + '.avi', str(releases)) == rows


def test_inventory_does_not_guess_unstamped_model_or_accept_wrong_manifest(inspection, fit_files):
    video, releases, paths = fit_files
    (paths[0].parent.parent / 'recordingDLC_legacy.h5').touch()
    (releases / 'left-r0001' / 'RELEASE.toml').write_text(
        '[release]\nid="left-r0001"\nalias="left"\n[model]\noutput_suffix="DLC_other"\n')
    rows = inspection.inventory(str(video), str(releases))
    unknown = [f for f in rows if not f['release_id']]
    assert len(unknown) == 2
    assert all(not f['config'] and not f['alias'] for f in unknown)
    assert any('Unverified release' in f['warning'] for f in unknown)


def test_inspection_stays_selected_when_route_or_inventory_changes(inspection):
    fits = [{'id': 'a'}, {'id': 'b'}]
    state = State(routing_alias='right')
    inspection.select_defaults(fits, state, 'session')
    state.update(inspect_a='b', inspect_b='a', routing_alias='left')
    inspection.select_defaults(fits + [{'id': 'c'}], state, 'session')
    assert state.inspect_a == 'b' and state.inspect_b == 'a'
    assert state.routing_alias == 'left'
    inspection.select_defaults([{'id': 'a'}], state, 'session')
    assert state.inspect_a == 'a'
    inspection.select_defaults(fits, state, 'other')
    assert state.inspect_a == 'a' and state.inspect_b is None


def test_inspection_views_share_selections_and_optional_overrides(inspection, core, monkeypatch):
    from streamlit.testing.v1 import AppTest
    fits = [dict(id=name, path='/'+name+'.h5', label=name, video='/source.avi', camera='recording.avi',
                 origins=['temp'], roots={}, release_id=None, alias=None, config='',
                 warning='', published=False, provenance='unknown') for name in ('a', 'b', 'c')]
    monkeypatch.setattr(inspection, 'inventory', lambda *args: fits)
    monkeypatch.setattr(core, 'resolve_alias', lambda *args: None)
    calls = []
    for name, method in [('Frame', 'render_frames'), ('Tracking spread', 'render_spread'),
                         ('Labelled video', 'render_montage')]:
        monkeypatch.setattr(inspection, method, lambda fits, cutoff, name=name: calls.append((name, [f['id'] for f in fits])))
    app = AppTest.from_string('''
import streamlit as st
import labeler_core as core
import fit_inspection as inspection
core.init_session_state()
core.inject_css()
inspection.render('/session/recording.avi')
''').run()
    assert not app.exception
    app.selectbox(key='inspect_a').set_value('b').run()
    app.selectbox(key='inspect_b').set_value('c').run()
    for view in inspection.VIEW_KEYS:
        calls.clear()
        app.radio(key='model_view').set_value(view).run()
        assert not app.exception
        assert calls == [(view, ['b', 'c'])]
    app.checkbox(key='inspect_unlinked').check().run()
    app.selectbox(key='inspect_video_a').set_value('a').run()
    assert calls[-1] == ('Labelled video', ['a', 'c'])
    app.selectbox(key='inspect_video_b').set_value(None).run()
    assert calls[-1] == ('Labelled video', ['a'])
    app.radio(key='model_view').set_value('Frame').run()
    assert calls[-1] == ('Frame', ['b', 'c'])
    app.radio(key='model_view').set_value('Labelled video').run()
    assert calls[-1] == ('Labelled video', ['a'])
    app.checkbox(key='inspect_unlinked').uncheck().run()
    assert app.selectbox(key='inspect_a').value == 'b'


def test_score_uses_fit_reference_and_invalidates_on_cutoff(inspection, core, monkeypatch, tmp_path):
    prediction = tmp_path / 'prediction.h5'
    prediction.touch()
    monkeypatch.setattr(core.st, 'session_state', State(selected_dlc_model_config='/wrong/route.yaml'))
    calls = []
    monkeypatch.setattr(inspection, '_score', lambda key: calls.append(key) or 'result')
    fit = {'id': str(prediction), 'path': str(prediction), 'config': ''}
    inspection.compute_scores([fit], .2)
    assert calls[0][1] is None  # unknown provenance never borrows the routed model
    assert inspection.existing_score(fit, .2) == ('result', None)
    assert inspection.existing_score(fit, .5) == (None, 'not computed')
    prediction.write_bytes(b'changed')
    assert inspection.existing_score(fit, .2) == (None, 'not computed')


def test_publish_inspected_fit_is_exact_and_never_falls_back_to_inference(inspection, core, monkeypatch, fit_files):
    video, releases, paths = fit_files
    fit = next(f for f in inspection.inventory(str(video), str(releases)) if 'temp' in f['origins'])
    monkeypatch.setattr(core, 'resolve_alias', lambda alias: {'release_id': 'left-r0001'})
    monkeypatch.setattr(core, 'ceiling_video', lambda v: (str(video), None))
    calls = []
    monkeypatch.setattr(core, 'queue_dlc_job', lambda *a, **kw: calls.append((a, kw)))
    inspection.publish_fit(fit, str(video))
    assert calls[0][0][3] == 'publish'
    assert calls[0][1]['fallback'] is False
    assert calls[0][1]['alias'] == 'left'
    assert len(calls[0][1]['extra_args']['publish_token']) == 64
    monkeypatch.setattr(core, 'resolve_alias', lambda alias: {'release_id': 'left-r0002'})
    with pytest.raises(ValueError, match='currently deployed'):
        inspection.publish_fit(fit, str(video))
    assert len(calls) == 1


def test_composite_montage_uses_same_source_frames_for_both_fits(inspection, core, monkeypatch, montage_source, tmp_path):
    import numpy as np
    video, cache = montage_source
    columns = pd.MultiIndex.from_product([['scorer'], ['nose'], ['x', 'y', 'likelihood']])
    table = pd.DataFrame(np.tile([10, 10, 1], (30, 1)), columns=columns)
    fits = []
    for name in ('a', 'b'):
        path = tmp_path / (name+'.h5')
        path.touch()
        fits.append({'id': name, 'path': str(path), 'video': str(video), 'camera': 'recording.avi'})
    monkeypatch.setattr(inspection, 'prediction_table', lambda fit: table)
    output = inspection.make_montage(fits, .2, 3, .5)
    assert output.parent == cache
    cap = core.cv2.VideoCapture(str(output))
    try:
        assert int(cap.get(core.cv2.CAP_PROP_FRAME_COUNT)) == 15
        assert int(cap.get(core.cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(cap.get(core.cv2.CAP_PROP_FRAME_HEIGHT)) == 32
    finally:
        cap.release()
    assert inspection.montage_path(fits[::-1], .2, 3, .5) != output
    fits[1]['video'] = '/different/camera.avi'
    with pytest.raises(ValueError, match='same known camera'):
        inspection.make_montage(fits, .2, 3, .5)


def test_inspected_population_uses_selected_alias_not_route(inspection, core, monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    # A and B now share one figure pair (2026-09-16); the population toggle moved to the
    # settings popover in render(), so the view reads it from state rather than owning it
    monkeypatch.setattr(core, 'hfqc_compare_figure', lambda pairs, **k: core.go.Figure())
    monkeypatch.setattr(core, 'hfqc_compare_bars', lambda pairs: core.go.Figure())
    monkeypatch.setattr(core, 'hfqc_summary_chips', lambda r: 'spread 0.20 · tracked 99%')
    monkeypatch.setattr(core, '_hfqc_style_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(core, '_hfqc_frame_caption', lambda r: 'test')
    result = SimpleNamespace(warnings=[], has_reference=True, table=pd.DataFrame())
    monkeypatch.setattr(inspection, 'existing_score', lambda *a: (result, None))
    calls = []
    monkeypatch.setattr(core, '_hfqc_population_section', lambda *a, **kw: calls.append(kw))
    app = AppTest.from_string('''
import fit_inspection as i
i.render_spread([{'id': 'a', 'alias': 'left'}, {'id': 'b', 'alias': 'right'}], .2)
''').run()
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the database comparison is one plot in the spread view's strip; only picking it runs the query | date: 2026-09-25]
    assert not app.exception
    assert not calls
    app.radio(key='inspect_spread_plot').set_value('database').run()
    assert not app.exception
    assert [c['alias'] for c in calls] == ['left', 'right']
    assert len({c['key_prefix'] for c in calls}) == 2
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the session finder makes
# hopping between sessions the normal way to drive the app, so the state reset that used
# to happen once per typed path now has to be right every time | date: 2026-09-15]
def test_open_session_retargets_the_page_and_drops_the_previous_arena_state(core, monkeypatch):
    state = State(video_path="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  video_path_input_field="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  current_frame="decoded image", current_frame_key=("stamp", 7),
                  frame_number_input=7, points={"A": (1, 2)}, distances={"A_B": 1.0},
                  fixation_point=(3, 4), last_imported_config="old.toml",
                  auto_config_name="recording-PRN001a01_a_NTD-dlc_arena_config.toml",
                  auto_config_loaded_for="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  arena_config_dirty=False, video_path_error="Not found: typo")
    monkeypatch.setattr(core.st, "session_state", state)

    assert core.open_session("/daq/PRN002/PRN002a03_a_NFC/recording.avi")

    assert state["video_path"] == "/daq/PRN002/PRN002a03_a_NFC/recording.avi"
    assert state["current_frame"] is None and state["current_frame_key"] is None
    assert state["video_path_error"] == ""
    # the strip's two widgets are left alone and asked for instead -- writing them here
    # is what raised StreamlitAPIException on every finder click
    assert state["pending_strip_sync"] is True
    assert state["video_path_input_field"] == "/daq/PRN001/PRN001a01_a_NTD/recording.avi"
    assert state["frame_number_input"] == 7
    # a session the finder surfaced as needing a config has none to load, so anything
    # left over would be drawn on -- and saved against -- the wrong video
    assert state["points"] == {} and state["distances"] == {}
    assert state["fixation_point"] is None
    assert state.get("last_imported_config") is None
    assert state.get("auto_config_name") is None
    assert state.get("auto_config_loaded_for") is None


def test_open_session_refuses_to_discard_unsaved_arena_annotations(core, monkeypatch):
    state = State(video_path="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  points={"A": (1, 2)}, arena_config_dirty=True)
    monkeypatch.setattr(core.st, "session_state", state)

    assert not core.open_session("/daq/PRN002/PRN002a03_a_NFC/recording.avi")

    assert state["video_path"] == "/daq/PRN001/PRN001a01_a_NTD/recording.avi"
    assert state["points"] == {"A": (1, 2)}
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: open_session wrote two keys
# that belong to widgets in the always-visible session strip, and a tab body runs AFTER the
# strip has instantiated them -- clicking a finder row raised StreamlitAPIException
# | date: 2026-09-16]
STRIP = '''
import streamlit as st
import labeler_core as core

core.init_session_state()
{adopt}
st.session_state.setdefault("video_path_input_field", st.session_state.get("video_path", ""))
st.text_input("Video path", key="video_path_input_field")
st.number_input("Frame", min_value=0, max_value=100, step=1, key="frame_number_input")

# a tab body, which the real app renders below the strip
if st.session_state.get("do_open"):
    st.session_state.do_open = False
    core.open_session("/daq/PRN002/PRN002a03_a_NFC/recording.avi")
'''


def test_opening_a_session_never_writes_a_live_session_strip_widget(core):
    """The crash a finder click produced: the strip owns those keys by the time we run.

    Streamlit refuses any write to a widget's key after that widget has been instantiated
    in the current run, and the strip is always visible -- it is built before whichever
    tab body is showing.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(STRIP.format(adopt=""))
    at.run(timeout=30)
    at.session_state["do_open"] = True
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["video_path"] == "/daq/PRN002/PRN002a03_a_NFC/recording.avi"


def test_the_strip_adopts_the_opened_session_on_the_following_run(core):
    """Deferring the write must not lose it: the box has to catch up, or the path shown
    disagrees with the video the rest of the page is on."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(STRIP.format(adopt="core.adopt_pending_session()"))
    at.run(timeout=30)
    # park on a late frame first: landing on a different video while the box still names
    # frame 60 asks for a frame the new recording may not have
    at.number_input(key="frame_number_input").set_value(60).run(timeout=30)
    assert at.number_input(key="frame_number_input").value == 60

    at.session_state["do_open"] = True
    at.run(timeout=30)          # the run that calls open_session
    at.run(timeout=30)          # the rerun its caller triggers

    assert not at.exception
    assert at.text_input(key="video_path_input_field").value == \
        "/daq/PRN002/PRN002a03_a_NFC/recording.avi"
    assert at.number_input(key="frame_number_input").value == 0


def test_the_real_session_strip_adopts_a_pending_retarget(core):
    """Against the actual app file, not a replica of it.

    The two tests above build their own strip, so they would both keep passing if
    behavior_config_labeler_tabs.py stopped calling adopt_pending_session().
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(LABELER / "behavior_config_labeler_tabs.py"))
    at.run(timeout=120)
    at.session_state["video_path"] = "/daq/PRN002/PRN002a03_a_NFC/recording.avi"
    at.session_state["pending_strip_sync"] = True
    at.run(timeout=120)

    assert not at.exception
    assert at.text_input(key="video_path_input_field").value == \
        "/daq/PRN002/PRN002a03_a_NFC/recording.avi"
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: config_name_input is a
# RETAINED widget, so the save target survived a session switch -- saving markers placed
# for the new session overwrote the previous session's config | date: 2026-09-16]
def test_opening_a_session_does_not_carry_the_previous_save_target_over(core, monkeypatch):
    """The Save box must re-seed from the new video, not keep pointing at the old file.

    ``config_name_input`` is in retain_widget_values, so it outlives the switch, and
    Streamlit prefers an existing session-state value over a widget's ``value=``. Left
    alone it names the PREVIOUS session's config -- and Save writes where the box says,
    so the markers you placed for this session land on that one.
    """
    old = "/daq/PRN001/sorted/recording-PRN001a01_a_NTD-dlc_arena_config.toml"
    state = State(video_path="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  config_name_input=old, arena_config_dirty=False, points={})
    monkeypatch.setattr(core.st, "session_state", state)

    assert core.open_session("/daq/PRN002/PRN002a03_a_NFC/recording.avi")

    assert "config_name_input" not in state
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the GUI pinned every job to a
# machine from a hardcoded list, which it cannot keep current as workers come and go
# | date: 2026-09-16]
def test_the_gui_lets_the_queue_choose_the_machine(core, monkeypatch):
    """No ``require_machine``: the queue master knows which workers exist, the GUI does not.

    The pin was a workaround for one bad node (capybara, 2026-08-18). A hardcoded pool
    goes stale in both directions -- it kept sending work to three machines while agouti
    sat idle, and it would have kept excluding a node long after it was repaired.
    """
    captured = {}

    def fake_add(args, note, **kwargs):
        captured.update(kwargs)
        return [(4242, "queued")]

    monkeypatch.setattr(core.db, "add_job_to_queue", fake_add)
    monkeypatch.setattr(core, "current_dlc_user", lambda: "wingertj")

    message = core.queue_dlc_job("/daq/A/A1/A1a01_a_NTD/recording.avi", alias="two_chimney")

    assert "4242" in message                       # the queue call really happened
    assert captured["linux_user"] == "lbhb"
    assert captured.get("require_machine", "") == ""


def test_no_hardcoded_machine_list_survives_in_the_gui(core):
    assert not hasattr(core, "DEV_MACHINE_POOL")
    assert not hasattr(core, "queue_require_machine")
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame box and the arena-config status/reload live in Arena markers, not the session strip | date: 2026-09-25]
def test_frame_and_arena_config_controls_live_only_in_arena_markers(core, monkeypatch, tmp_path):
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from nems_lbhb.motor import dlc_routing

    session = tmp_path / "SQD092" / "SQD092b02_a_NFC"
    session.mkdir(parents=True)
    video = session / "recording.avi"
    video.write_bytes(b"stub")
    (tmp_path / "SQD092" / "sorted").mkdir()
    (tmp_path / "SQD092" / "sorted" /
     "recording-SQD092b02_a_NFC-dlc_arena_config.toml").write_text(
        f'[metadata]\nvideo_path = "{video}"\nframe_number = 1\n'
        'frame_width = 64\nframe_height = 48\n[points]\nA = [1, 2]\n')
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text("Task: test\nscorer: test\nproject_path: " + str(project)
                      + "\nbodyparts: []\nvideo_sets: {}\niteration: 0\n")
    monkeypatch.setattr(core, "discover_dlc_models", lambda: {"project": str(config)})
    monkeypatch.setattr(core, "code_banner", lambda: None)
    monkeypatch.setattr(core, "remember_pref", lambda *args: None)
    monkeypatch.setattr(core, "video_frame_count", lambda path: 2)
    reads = []
    monkeypatch.setattr(core, "read_frame",
                        lambda *args: reads.append(args) or Image.new("RGB", (64, 48)))
    monkeypatch.setattr(core, "h5_options", lambda *args, **kwargs: {})
    monkeypatch.setattr(core, "fit_status_cached", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(core, "resolve_alias", lambda alias: {
        "alias": alias, "config": str(config), "project": str(project), "source": "project"})
    monkeypatch.setattr(dlc_routing, "suggest", lambda *a, **k: {
        "method": "geometry", "alias": "right_chimney", "observed": {"db_errors": []},
        "candidates": [{"alias": "right_chimney", "release": None, "why": [], "unknown": []}]})

    app = AppTest.from_file(str(LABELER / "behavior_config_labeler_tabs.py"))
    app.query_params["video"] = str(video)
    app.run(timeout=30)
    assert not app.exception
    assert not [w for w in app.number_input if w.key == "frame_number_input"]
    assert not [b for b in app.button if b.key == "arena_reload"]
    assert not app.success                           # no "imported successfully" banner

    app.radio(key="app_section").set_value("Arena markers").run()
    assert not app.exception
    assert app.number_input(key="frame_number_input").value == 1   # from the config
    assert app.button(key="arena_reload")
    assert any("config · recording-SQD092b02" in c.value for c in app.caption)
    assert reads                                     # the tab decoded its own frame
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the logo is built from baked DLC poses at runtime, no image asset | date: 2026-09-25]
def test_the_logo_is_a_rainbow_skeleton_of_three_poses(core):
    import logo
    svg = logo.logo_svg()
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<g opacity=") == 3                        # three frames
    assert svg.count("<line ") == 3 * len(logo.EDGES)           # head + spine per frame
    assert svg.count("<circle ") == 3 * len(logo.PARTS)
    assert logo.COLOURS["front"] == "#8000ff" and logo.COLOURS["rear"] == "#ff0000"
    assert all(0 <= v <= 100 for pose in logo.POSES for xy in pose.values() for v in xy)
# [AGENT EDIT END]
