"""Action buttons report their outcome through their own colour, hover and a dialog."""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for core.action_button / core.say | date: 2026-09-28]

from pathlib import Path

import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"

SCRIPT = """
import sys
sys.path.insert(0, {labeler!r})
import streamlit as st
import labeler_core as core

def ok():
    core.say("success", "job queued · 123")

def caveat():
    core.say("warning", "queued from a project · no provenance")

def returns_error():
    return "Error launching job: no database"

def boom():
    raise RuntimeError("refused: wrong camera")

core.action_button("Run", "b_" + st.session_state.get("which", "ok"),
                   dict(ok=ok, caveat=caveat, returns_error=returns_error,
                        boom=boom)[st.session_state.get("which", "ok")])
"""


@pytest.fixture
def app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string(SCRIPT.format(labeler=str(LABELER)))


def styles(at):
    return " ".join(m.value for m in at.markdown if "<style>" in m.value)


def click(at, which):
    at.session_state["which"] = which
    at.run()
    at.button(key=f"b_{which}").click().run()
    return at


def test_success_turns_the_button_green_with_the_message_in_its_hover(app):
    at = click(app, "ok")
    assert not at.exception and not at.success and not at.error
    assert "#2e7d32" in styles(at) and "st-key-b_ok" in styles(at)
    assert "job queued · 123" in at.button(key="b_ok").help


def test_a_caveat_is_amber(app):
    at = click(app, "caveat")
    assert "#b26a00" in styles(at) and not at.warning


@pytest.mark.parametrize("which", ["returns_error", "boom"])
def test_a_failure_is_red_and_opens_a_dialog_once(app, which):
    at = click(app, which)
    assert "#c62828" in styles(at)
    assert at.error, "the dialog body shows the error"
    at.run()
    assert not at.error, "the dialog is not re-opened on the next rerun"
    assert "#c62828" in styles(at), "the colour stays"


def test_the_outcome_is_forgotten_when_the_session_changes(app):
    at = click(app, "ok")
    at.session_state["video_path"] = "/somewhere/else/recording.avi"
    at.run()
    assert "#2e7d32" not in styles(at)


def test_say_writes_normally_outside_an_action(app):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(f"""
import sys
sys.path.insert(0, {str(LABELER)!r})
import labeler_core as core
core.say("warning", "plain warning")
""").run()
    assert [w.value for w in at.warning] == ["plain warning"]


def test_key_class_matches_streamlits_frontend(app):
    import importlib, sys
    sys.path.insert(0, str(LABELER))
    core = importlib.import_module("labeler_core")
    assert core.key_class("camera_go::SQD041a05 x") == "st-key-camera_go--SQD041a05-x"
# [AGENT EDIT END]
