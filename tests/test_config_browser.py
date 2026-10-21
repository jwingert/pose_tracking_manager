"""The server-side config browser: what it lists, where it starts, what a click does.

This exists because the browser's own file dialog cannot reach ``/auto`` at all. Every
browser on the labeler host is a snap, and neither AppArmor profile grants ``/auto/**`` --
only ``@{HOME}/[^s.]**`` and the desktop portal's proxy directory. The Streamlit *server*
is unconfined, so a listing it produces sidesteps the sandbox entirely.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: snap-confined browsers cannot
# read /auto, so uploading a config from the network share is impossible from the file
# dialog; list and import server-side instead | date: 2026-09-16]

import importlib
from pathlib import Path

import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"

USERS_ROOT = "/auto/users"


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def core(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    module = importlib.import_module("labeler_core")
    monkeypatch.setattr(module, "PREFS_FILE", tmp_path / "prefs.json")
    return module


@pytest.fixture
def markers(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    return importlib.import_module("tabs.arena_markers")


@pytest.fixture
def site(tmp_path):
    """A site directory: a sorted/ holding two configs, a sub-folder, and noise."""
    sorted_dir = tmp_path / "SQD008" / "sorted"
    sorted_dir.mkdir(parents=True)
    for name in ("recording-SQD008a05_a_NTD-dlc_arena_config.toml",
                 "recording-SQD008a03_a_NTD-dlc_arena_config.toml"):
        (sorted_dir / name).write_text("")
    (sorted_dir / "SQD008a05_a_NTD.dlc.h5").write_text("")        # not a config
    (sorted_dir / ".hidden-dlc_arena_config.toml").write_text("")  # not offered
    (sorted_dir / "zz_nested").mkdir()
    (sorted_dir / "aa_nested").mkdir()
    (tmp_path / "SQD008" / "SQD008a05_a_NTD").mkdir()
    return sorted_dir


# ── the listing ───────────────────────────────────────────────────────────────

def test_directories_come_first_then_files_each_alphabetically(core, site):
    _resolved, entries = core.browse_dir(site)
    assert list(entries) == [
        "..", "aa_nested/", "zz_nested/",
        "recording-SQD008a03_a_NTD-dlc_arena_config.toml",
        "recording-SQD008a05_a_NTD-dlc_arena_config.toml",
    ]


def test_a_trailing_slash_says_which_entries_navigate(core, site):
    # picking a directory moves you; picking a file imports it. One glance has to say
    # which is about to happen, because the two are not undoable in the same way
    _resolved, entries = core.browse_dir(site)
    assert all(entries[label].is_dir() for label in entries if label.endswith("/"))


def test_only_files_with_the_requested_suffix_are_offered(core, site):
    _resolved, entries = core.browse_dir(site)
    assert not any(name.endswith(".h5") for name in entries)


def test_hidden_entries_are_skipped(core, site):
    _resolved, entries = core.browse_dir(site)
    assert not any(name.startswith(".") and name != ".." for name in entries)


def test_the_parent_is_always_reachable(core, site):
    _resolved, entries = core.browse_dir(site)
    assert entries[".."] == site.parent


def test_the_filesystem_root_offers_no_parent(core):
    # '/' is its own parent, so an unguarded '..' would be a dead entry that looks live
    resolved, entries = core.browse_dir("/")
    assert resolved == Path("/")
    assert ".." not in entries


def test_dot_dot_in_the_typed_path_is_resolved_away(core, site):
    resolved, _entries = core.browse_dir(str(site / "zz_nested" / ".."))
    assert resolved == site


def test_a_tilde_is_expanded(core):
    resolved, _entries = core.browse_dir("~")
    assert resolved == Path.home().resolve()


def test_a_missing_folder_raises_rather_than_looking_empty(core, tmp_path):
    # an empty listing and an unreachable path are different answers, and the second one
    # is the one a typo produces -- silently showing "no configs here" would hide it
    with pytest.raises(OSError):
        core.browse_dir(tmp_path / "nope")


def test_a_file_path_is_not_a_folder(core, site):
    with pytest.raises(OSError):
        core.browse_dir(site / "SQD008a05_a_NTD.dlc.h5")


# ── where it starts ───────────────────────────────────────────────────────────

def test_the_browser_starts_in_this_sessions_sorted_directory(markers, site):
    video = site.parent / "SQD008a05_a_NTD" / "recording.avi"
    assert markers.browse_default(str(video)) == str(site)


def test_without_a_session_it_starts_somewhere_that_exists(markers):
    assert markers.browse_default("") == USERS_ROOT


def test_switching_sessions_repoints_the_browser(core, monkeypatch):
    # the folder is a retained widget value, so it would otherwise keep showing the
    # previous site's sorted/ -- the one place you are least likely to want
    state = State(video_path="/daq/PRN001/PRN001a01_a_NTD/recording.avi",
                  am_browse_dir="/daq/PRN001/sorted", arena_config_dirty=False, points={})
    monkeypatch.setattr(core.st, "session_state", state)

    assert core.open_session("/daq/PRN002/PRN002a03_a_NFC/recording.avi")

    assert "am_browse_dir" not in state


# ── what a click does ─────────────────────────────────────────────────────────

def test_choosing_a_directory_moves_into_it(markers, core, monkeypatch, site):
    state = State(am_browse_dir=str(site), am_browse_pick="aa_nested/")
    monkeypatch.setattr(core.st, "session_state", state)

    markers.browse_navigate()

    assert state["am_browse_dir"] == str(site / "aa_nested")
    assert not state["am_browse_pick"]        # the old label means nothing here


def test_choosing_a_file_stays_put_so_it_can_be_imported(markers, core, monkeypatch, site):
    name = "recording-SQD008a05_a_NTD-dlc_arena_config.toml"
    state = State(am_browse_dir=str(site), am_browse_pick=name)
    monkeypatch.setattr(core.st, "session_state", state)

    markers.browse_navigate()

    assert state["am_browse_dir"] == str(site)
    assert state["am_browse_pick"] == name


def test_the_selected_file_is_what_import_would_read(markers, core, monkeypatch, site):
    name = "recording-SQD008a05_a_NTD-dlc_arena_config.toml"
    state = State(am_browse_dir=str(site), am_browse_pick=name)
    monkeypatch.setattr(core.st, "session_state", state)

    assert markers.browse_selection() == site / name


def test_nothing_is_selected_when_the_pick_is_a_directory_or_absent(markers, core,
                                                                    monkeypatch, site):
    state = State(am_browse_dir=str(site), am_browse_pick="aa_nested/")
    monkeypatch.setattr(core.st, "session_state", state)
    assert markers.browse_selection() is None

    state["am_browse_pick"] = None
    assert markers.browse_selection() is None


def test_a_vanished_selection_does_not_crash_the_picker(markers, core, monkeypatch, site):
    # the folder is listed fresh on every run, and a config can be moved or renamed
    # between the run that listed it and the run that reads the selection
    state = State(am_browse_dir=str(site), am_browse_pick="gone-dlc_arena_config.toml")
    monkeypatch.setattr(core.st, "session_state", state)

    assert markers.browse_selection() is None
    markers.browse_navigate()                 # must not raise


def test_an_unreachable_folder_does_not_crash_navigation(markers, core, monkeypatch,
                                                          tmp_path):
    state = State(am_browse_dir=str(tmp_path / "nope"), am_browse_pick="anything/")
    monkeypatch.setattr(core.st, "session_state", state)

    markers.browse_navigate()                 # must not raise

    assert state["am_browse_dir"] == str(tmp_path / "nope")


# ── rendered ──────────────────────────────────────────────────────────────────

SCRIPT = """
import labeler_core as core
from tabs import arena_markers
core.init_session_state()
arena_markers._save_import({video!r})
"""


def test_the_expander_offers_a_path_box_a_listing_and_the_jumps(core, markers, site):
    from streamlit.testing.v1 import AppTest

    video = str(site.parent / "SQD008a05_a_NTD" / "recording.avi")
    at = AppTest.from_string(SCRIPT.format(video=video)).run(timeout=30)
    assert not at.exception

    assert at.text_input(key="am_browse_dir").value == str(site)
    options = at.selectbox(key="am_browse_pick").options
    assert "recording-SQD008a05_a_NTD-dlc_arena_config.toml" in options
    labels = [b.label for b in at.button]
    assert any(USERS_ROOT in label for label in labels)
    assert any("sorted" in label.lower() for label in labels)


def test_typing_a_folder_relists_without_touching_the_browser(core, markers, site):
    from streamlit.testing.v1 import AppTest

    video = str(site.parent / "SQD008a05_a_NTD" / "recording.avi")
    at = AppTest.from_string(SCRIPT.format(video=video)).run(timeout=30)
    at.text_input(key="am_browse_dir").set_value(str(site / "aa_nested")).run(timeout=30)
    assert not at.exception

    assert at.selectbox(key="am_browse_pick").options == [".."]


def test_an_unreachable_typed_folder_says_so(core, markers, site):
    from streamlit.testing.v1 import AppTest

    video = str(site.parent / "SQD008a05_a_NTD" / "recording.avi")
    at = AppTest.from_string(SCRIPT.format(video=video)).run(timeout=30)
    at.text_input(key="am_browse_dir").set_value("/no/such/place").run(timeout=30)
    assert not at.exception

    assert any("/no/such/place" in w.value for w in at.warning)
# [AGENT EDIT END]
