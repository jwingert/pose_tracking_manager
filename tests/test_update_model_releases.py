"""Update tab: which release a training project serves, and what data it was fed."""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for naming releases in the Update tab's project picker and diffing a project's labels against its release | date: 2026-09-25]

import importlib
from pathlib import Path

import pandas as pd
import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"


@pytest.fixture
def core(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    pytest.importorskip("tables")
    monkeypatch.syspath_prepend(str(LABELER))
    module = importlib.import_module("labeler_core")
    monkeypatch.setattr(module, "PREFS_FILE", tmp_path / "prefs.json")
    module._releases_by_project.clear()
    return module


@pytest.fixture
def update_model(core):
    return importlib.import_module("tabs.update_model")


def write_labels(root, subdir, folder, rows):
    path = Path(root) / subdir / folder / "CollectedData_x.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [float(v) for v in rows]}).to_hdf(path, key="df_with_missing")
    return path


def test_a_project_untouched_since_its_release_reports_identical_labels(core, tmp_path):
    for folder, rows in (("A", [1, 2]), ("B", [3])):
        write_labels(tmp_path / "rel", "labels", folder, rows)
        write_labels(tmp_path / "proj", "labeled-data", folder, rows)

    diff = core.release_label_diff(tmp_path / "rel", tmp_path / "proj")
    assert diff["same"] == ["A", "B"]
    assert not (diff["changed"] or diff["added"] or diff["removed"])
    assert diff["release_frames"] == diff["project_frames"] == 3


def test_label_drift_is_reported_per_video(core, tmp_path):
    write_labels(tmp_path / "rel", "labels", "same", [1])
    write_labels(tmp_path / "proj", "labeled-data", "same", [1])
    write_labels(tmp_path / "rel", "labels", "edited", [1, 2])
    write_labels(tmp_path / "proj", "labeled-data", "edited", [1, 5])
    write_labels(tmp_path / "rel", "labels", "dropped", [1])
    write_labels(tmp_path / "proj", "labeled-data", "new", [1, 2, 3])

    diff = core.release_label_diff(tmp_path / "rel", tmp_path / "proj")
    assert (diff["same"], diff["changed"], diff["added"], diff["removed"]) == (
        ["same"], ["edited"], ["new"], ["dropped"])
    assert (diff["release_frames"], diff["project_frames"]) == (4, 6)


def test_the_summary_says_whether_retraining_uses_the_releases_data(update_model):
    base = dict(release_id="left-r0002", frames=92, folders=["A", "B"],
                iteration=3, snapshot=50000, project_frames=92,
                same=["A", "B"], changed=[], added=[], removed=[])
    line, detail = update_model.training_data_summary(base)
    assert line.startswith("`left-r0002` trained on 92 frames · 2 videos")
    assert "project labels = release" in line
    assert "Trained on: A, B" in detail

    drifted = dict(base, project_frames=120, added=["C"], changed=["A"])
    line, detail = update_model.training_data_summary(drifted)
    assert "+1 new · 1 relabelled videos · 120 frames now" in line
    assert "New since release: C" in detail and "Relabelled since release: A" in detail


def test_releases_are_mapped_to_the_project_they_were_cut_from(core, monkeypatch, tmp_path):
    from nems_lbhb.motor import dlc_release
    registry = {"alias_names": ["left", "two"]}
    history = {"left": ("left-r0002", ["left-r0001", "left-r0002"]),
               "two": ("two-r0001", ["two-r0001"])}
    unpromoted = {"left": ["left-r0003", "left-r0001"], "two": []}
    sources = {"left-r0001": tmp_path / "old_proj", "left-r0002": tmp_path / "proj",
               "left-r0003": tmp_path / "proj", "two-r0001": tmp_path / "two_proj"}
    monkeypatch.setattr(core.dlc_registry, "load_registry", lambda *a, **k: registry)
    monkeypatch.setattr(core, "alias_release_info",
                        lambda a: (history[a][0], history[a][1], None))
    monkeypatch.setattr(core, "unpromoted_releases", lambda a: unpromoted[a])
    monkeypatch.setattr(dlc_release, "read_manifest",
                        lambda rid: {"source": {"project": str(sources[rid])}})

    got = core.releases_by_project(registry_mtime=1.0)
    assert got[str(tmp_path / "proj")] == [("left", "left-r0002", "current"),
                                          ("left", "left-r0003", "candidate")]
    assert got[str(tmp_path / "old_proj")] == [("left", "left-r0001", "previous")]
    assert got[str(tmp_path / "two_proj")] == [("two", "two-r0001", "current")]
# [AGENT EDIT END]
