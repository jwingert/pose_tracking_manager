"""Tab 4 — New DLC model.

Deliberately isolated: every widget here writes to a ``new_*`` session key, so the
form can never be pre-filled from — or write back into — the config.yaml of the
model the session is routed to (todo 8).
"""
from pathlib import Path

import streamlit as st

# [AGENT EDIT START | agent: Codex | user: wingertj | reason: constructing the app must not import the DLC runtime | date: 2026-09-09]
# DeepLabCut is loaded only when Create project is pressed.
# [AGENT EDIT END]

import labeler_core as core

DEFAULTS = {
    "new_project_name": "",
    "new_experimenter": "",
    "new_working_dir": "/auto/data/dlc",
    "new_bodyparts": "headpost\nfront\nleftear\nnose",
    "new_videos": "",
    "new_copy_videos": True,
}


def render():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (value if key != "new_experimenter"
                                     else core.current_dlc_user())

    left, right = st.columns(2)

    with left:
        st.markdown("##### Project")
        st.text_input("Project name", key="new_project_name",
                      placeholder="e.g. headtrack_v4")
        st.text_input("Experimenter", key="new_experimenter")
        st.text_input("Working directory", key="new_working_dir")
        st.text_area("Bodyparts (one per line)", key="new_bodyparts", height=150)

    with right:
        st.markdown("##### Seed videos")
        st.text_area("Video paths (one per line)", key="new_videos", height=110,
                     placeholder="/auto/data/daq/.../rawvid.avi")
        add_col, copy_col = st.columns([1.2, 1])
        with add_col:
            current_video = st.session_state.get("video_path", "")
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: writing the text
            # area's key after it was drawn raised StreamlitAPIException; a click callback
            # runs before the widgets are built | date: 2026-09-28]
            st.button("Add current session's video", disabled=not current_video,
                      use_container_width=True, key="nm_add_current",
                      on_click=_add_current_video, args=(current_video,))
            # [AGENT EDIT END]
        with copy_col:
            st.checkbox("copy videos into project", key="new_copy_videos")

        st.markdown("##### Create")
        name = st.session_state.new_project_name.strip()
        experimenter = st.session_state.new_experimenter.strip()
        working_dir = st.session_state.new_working_dir.strip()
        videos = [v.strip() for v in st.session_state.new_videos.splitlines() if v.strip()]
        bodyparts = [b.strip() for b in st.session_state.new_bodyparts.splitlines()
                     if b.strip()]

        problems = []
        if not name:
            problems.append("a project name")
        if not experimenter:
            problems.append("an experimenter")
        if not working_dir:
            problems.append("a working directory")
        if not videos:
            problems.append("at least one seed video")
        missing_videos = [v for v in videos if not Path(v).exists()]

        if name and experimenter and working_dir:
            st.code(f"{working_dir}/{name}-{experimenter}-<date>/config.yaml", language=None)
        if problems:
            st.caption("Needs " + ", ".join(problems) + ".")
        if missing_videos:
            st.caption(f":orange[⚠ {len(missing_videos)} video path(s) missing]",
                       help="These paths do not exist:\n\n" +
                            "\n\n".join(missing_videos))

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
        core.action_button("Create project", "nm_create", _create_project, type="primary",
                           use_container_width=True,
                           disabled=bool(problems) or bool(missing_videos))

        created = st.session_state.get("new_project_created")
        if created:
            st.caption("next · label frames (Update model → curate), then register an alias")
            st.code(created, language=None)
        # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: callbacks for the create
# form; read the form from session state at click time | date: 2026-09-28]
def _add_current_video(current_video):
    existing = [v.strip() for v in (st.session_state.get("new_videos") or "").splitlines()
                if v.strip()]
    if current_video and current_video not in existing:
        existing.append(current_video)
    st.session_state.new_videos = "\n".join(existing)


def _create_project():
    """The Create button's job: a DLC project from the form as it stands."""
    import deeplabcut                                         # heavyweight: only on click
    state = st.session_state
    videos = [v.strip() for v in (state.get("new_videos") or "").splitlines() if v.strip()]
    bodyparts = [b.strip() for b in (state.get("new_bodyparts") or "").splitlines()
                 if b.strip()]
    config_path = deeplabcut.create_new_project(
        state.new_project_name.strip(), state.new_experimenter.strip(), videos,
        working_directory=state.new_working_dir.strip(),
        copy_videos=bool(state.get("new_copy_videos")))
    core.say("success", f"project created · {config_path}")
    if bodyparts:
        try:
            core.write_config_values(config_path, bodyparts=bodyparts)
            core.say("success", f"bodyparts written · {', '.join(bodyparts)}")
        except Exception as e:                                # noqa: BLE001
            core.say("warning", f"Project created, but bodyparts were not written: {e}")
    state.new_project_created = str(config_path)
    core.refresh_files()
# [AGENT EDIT END]
