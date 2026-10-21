"""What the session finder must report about each free-moving session on disk.

The filesystem half is real (a tmp_path tree shaped like /auto/data/daq); only the two
database reads are injected, because the interesting logic is entirely in what the
`sorted/` listing and gDLCFit say about a session, not in SQL.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for the new per-session DLC status scan | date: 2026-09-15]

import pandas as pd
import pytest

from nems_lbhb.motor import dlc_session_status


RAW_COLUMNS = ["rawid", "siteid", "runclass", "behavior", "parmfile", "resppath",
               "pendate"]


def raw_rows(resppath, *specs):
    """gDataRaw-shaped rows: each spec is (rawid, parmfile_stem, runclass)."""
    return pd.DataFrame(
        [(rawid, stem[:7], runclass, "active", stem + ".m", str(resppath), "2026-01-01")
         for rawid, stem, runclass in specs],
        columns=RAW_COLUMNS)


@pytest.fixture
def pen(tmp_path):
    """One penetration directory with a sorted/ subdirectory."""
    root = tmp_path / "Prince" / "PRN001"
    (root / "sorted").mkdir(parents=True)
    return root


def session(pen, stem, video="recording.avi", dlc=False, config=False):
    if video:
        (pen / stem).mkdir(exist_ok=True)
        (pen / stem / video).write_bytes(b"")
    if dlc:
        (pen / "sorted" / (stem + ".dlc.h5")).write_bytes(b"")
    if config:
        (pen / "sorted" / ("recording-" + stem + "-dlc_arena_config.toml")).write_text("")


def inventory(pen, raw, fits=None, **kwargs):
    fits = pd.DataFrame(columns=["rawid", "routing_alias", "release_id",
                                 "attribution", "stale"]) if fits is None else fits
    return dlc_session_status.session_status(
        animal="Prince", query=lambda sql, params=None: raw, fits=fits, **kwargs)


def test_reports_what_each_session_is_missing(pen):
    session(pen, "PRN001a01_a_NTD", dlc=True, config=True)
    session(pen, "PRN001a02_a_NTD")
    session(pen, "PRN001a03_a_NTD", dlc=True)
    session(pen, "PRN001a04_a_NTD", config=True)
    raw = raw_rows(pen, (1, "PRN001a01_a_NTD", "NTD"), (2, "PRN001a02_a_NTD", "NTD"),
                   (3, "PRN001a03_a_NTD", "NTD"), (4, "PRN001a04_a_NTD", "NTD"))

    needs = inventory(pen, raw).set_index("rawid")["needs"].to_dict()

    assert needs == {1: "", 2: "dlc+config", 3: "config", 4: "dlc"}


def test_session_with_no_directory_at_all_has_nothing_to_do(pen):
    raw = raw_rows(pen, (1, "PRN001a05_p_FTC", "FTC"))

    row = inventory(pen, raw).iloc[0]

    assert row["needs"] == "no video"
    assert row["video"] is None


def test_session_directory_without_a_camera_file_has_nothing_to_do(pen):
    (pen / "PRN001a06_p_NFB").mkdir()
    (pen / "PRN001a06_p_NFB" / "trial_log.csv").write_text("")
    raw = raw_rows(pen, (1, "PRN001a06_p_NFB", "NFB"))

    assert inventory(pen, raw).iloc[0]["needs"] == "no video"


def test_ceiling_view_on_a_side_camera_slot_is_still_a_video(pen):
    session(pen, "PRN001a07_a_NTD", video="video_side2.avi", dlc=True, config=True)
    raw = raw_rows(pen, (1, "PRN001a07_a_NTD", "NTD"))

    row = inventory(pen, raw).iloc[0]

    assert row["slot"] == "video_side2.avi"
    assert row["video"].endswith("PRN001a07_a_NTD/video_side2.avi")
    assert row["needs"] == ""


def test_legacy_arena_config_filename_still_counts_as_a_config(pen):
    session(pen, "PRN001a08_a_NTD", dlc=True)
    (pen / "sorted" / "PRN001a08_a_NTD_dlc_arena_config.toml").write_text("")
    raw = raw_rows(pen, (1, "PRN001a08_a_NTD", "NTD"))

    assert inventory(pen, raw).iloc[0]["needs"] == ""


def test_a_stale_fit_is_not_a_work_item(pen):
    # staleness means the alias now points at a newer release; re-running every such
    # session is a bulk decision, not something to mix into "what is missing here"
    session(pen, "PRN001a09_a_NTD", dlc=True, config=True)
    raw = raw_rows(pen, (1, "PRN001a09_a_NTD", "NTD"))
    fits = pd.DataFrame([(1, "right_chimney", "right_chimney-r0002", "stamped", True)],
                        columns=["rawid", "routing_alias", "release_id",
                                 "attribution", "stale"])

    row = inventory(pen, raw, fits=fits).iloc[0]

    assert row["needs"] == ""
    assert row["release_id"] == "right_chimney-r0002"


def test_scope_is_passed_as_a_bind_parameter_not_inlined_sql(pen):
    seen = {}

    def spy(sql, params=None):
        seen["sql"], seen["params"] = sql, params
        return raw_rows(pen)

    dlc_session_status.session_status(animal="Prince", query=spy, fits=pd.DataFrame())

    assert seen["params"] == ("Prince",)
    assert "m.animal = %s" in seen["sql"]
    assert "Prince" not in seen["sql"]
# [AGENT EDIT END]
