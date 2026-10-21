# [AGENT EDIT START | agent: claude | user: wingertj | reason: the GUI is where a camera
# mix-up is actually noticed -- the user is looking at the frames -- so the assertion should
# be makeable there, through the same validated write path as the CLI | date: 2026-08-12]
"""Camera-slot panel: which psi slot holds the ceiling view for this session.

psi names video files after the camera SLOT, not the content, so a USB port
re-enumeration silently swaps them.  The old pipeline routed on camera NAME, which is why
a swap used to be undetectable; the fix is a human assertion, and this is where a human
can make it while looking at the evidence.

All rules live in ``nems_lbhb.motor.dlc_camera_override``.  This module only renders.
"""
import os

import streamlit as st

from nems_lbhb.motor import dlc_camera_override as co


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the panel hid behind a
# collapsed expander, so which camera the page is showing -- the one fact that invalidates
# every marker and overlay on screen when it is wrong -- took a click to discover. Status
# is now always visible; the slot comparison, which costs a decode of every slot, stays
# behind a toggle. | date: 2026-08-17]
def render(video_path, user='unknown', expanded=False, key='cam'):
    """Draw the panel for the session containing ``video_path``.

    Safe to call with a missing/blank path -- it renders nothing rather than raising, so
    it can sit unconditionally in the app body.
    """
    if not video_path or not os.path.exists(str(video_path)):
        return
    try:
        sess = co.session_dir(video_path)
        stem = co.session_stem(video_path)
    except Exception as e:                                    # noqa: BLE001
        st.warning(f"camera panel unavailable for {video_path}: {e}")
        return

    why = ("psi names these files after the camera SLOT, not what the camera saw, so a "
           "re-plugged port can put the ceiling view in `video_side*`. Frame counts and "
           "fps do not identify a view -- compare the pictures.")
    existing = co.current(video_path)
    if existing:
        note = (existing.get('note') or '').strip()
        st.caption(f"Ceiling view: **{existing['video']}** · asserted by "
                   f"{existing.get('verified_by')} {existing.get('verified_date')}",
                   help="\n\n".join(x for x in (note, why) if x))
    else:
        st.caption("Ceiling view: **recording.avi** (default)", help=why)

    # the comparison is the expensive part (a seek+decode in every slot), and it is only
    # wanted when the frames on screen look like the wrong camera
    compare = st.checkbox("Compare camera slots", key=f"{key}_compare::{sess}",
                          value=bool(expanded),
                          help="Shows every psi slot side by side at matched wall-clock "
                               "times, and lets you assert which one holds the ceiling "
                               "view. Use it when the frame above looks wrong.")
    if not compare:
        return                    # the caption above already carries the assertion

    if existing:
        _render_existing(existing)

    try:
        info = co.slot_info(sess)
    except Exception as e:                                    # noqa: BLE001
        st.error(f"could not read the video slots: {e}")
        return
    if not info:
        st.info("no psi video slots (recording.avi / video_side.avi / video_side2.avi) here.")
        return

    _render_slot_table(info)
    _render_thumbnails(sess, info)

    if existing:
        st.caption("Already recorded. Change it by editing the registry by hand.")
        return

    _render_assert_form(sess, stem, info, user)
# [AGENT EDIT END]


def _render_existing(entry):
    st.success(
        f"**Ceiling view: `{entry['video']}`** (cam_tag `{entry.get('cam_tag')}`) -- "
        f"asserted by {entry.get('verified_by')} on {entry.get('verified_date')}."
    )
    note = (entry.get('note') or '').strip()
    if note:
        st.caption(note)


def _render_slot_table(info):
    rows = []
    for slot, m in info.items():
        rows.append({
            'file': slot,
            'frames': m['n_frames'],
            'fps': round(m['fps']),
            'duration (s)': round(m['duration']),
            'size': f"{m['width']}x{m['height']}",
            'cam_tag': m['cam_tag'] or '(base)',
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_thumbnails(sess, info):
    """One row per slot, sampled at the SAME wall-clock times so views are comparable."""
    key = f"camera_frames::{sess}"
    if key not in st.session_state:
        with st.spinner("grabbing frames from each slot..."):
            try:
                st.session_state[key] = co.sample_frames(sess, info=info)
            except Exception as e:                            # noqa: BLE001
                st.error(f"could not sample frames: {e}")
                st.session_state[key] = {}
    frames = st.session_state[key]
    for slot, tiles in frames.items():
        if not tiles:
            continue
        st.markdown(f"**{slot}** &nbsp; `cam_tag {info[slot]['cam_tag'] or '(base)'}`",
                    unsafe_allow_html=True)
        cols = st.columns(len(tiles))
        for col, (t, img) in zip(cols, tiles):
            # contrast-stretched upstream: a 100 fps slot is nearly unreadable raw, and
            # that is exactly the kind of slot a mix-up involves
            col.image(img, caption=f"t = {t:.0f}s", use_container_width=True)


def _render_assert_form(sess, stem, info, user):
    st.markdown("---")
    st.markdown("**Assert which slot holds the ceiling (top-down) view**")

    options = list(info.keys())
    default = options.index('recording.avi') if 'recording.avi' in options else 0
    choice = st.radio("Ceiling view is in:", options, index=default,
                      key=f"camera_choice::{sess}", horizontal=True)

    if co.cam_tag_for(choice) is None:
        st.caption("`recording.avi` is the default -- nothing to record.")
        return

    st.warning(f"Asserts the top-down view for **{stem}** is in `{choice}` "
               f"(cam_tag `{co.cam_tag_for(choice)}`).",
               icon="⚠️")
    st.caption("check the frames first",
               help="The tag decides which PSIVIDEO frame counts align every trial. An "
                    "assertion made without looking mis-aligns the whole session while "
                    "looking deliberate in the file.")

    note = st.text_input("Note (what you saw, how you verified)",
                         key=f"camera_note::{sess}",
                         placeholder="port swap; ceiling view confirmed by eye at 4 time points")

    block, commit = co.add(sess, choice, by=user, note=note, dry_run=True)
    with st.expander("What will be written to the registry", expanded=False):
        st.code(block, language='toml')

    existing_h5 = co.existing_outputs(sess)
    if existing_h5:
        st.error(
            f"{len(existing_h5)} existing `.dlc.h5` for this session came from the WRONG "
            f"camera. Quarantine and re-run them, or they will keep being loaded:\n\n"
            + "\n".join(f"- `{os.path.basename(p)}`" for p in existing_h5))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button,
    # failures in a dialog. The commit command used to be drawn and then wiped by an
    # immediate st.rerun(); it is kept in state and stays under the button | date: 2026-09-28]
    import labeler_core as core
    core.action_button(f"Record override: {choice}", f"camera_go::{sess}", _record_override,
                       args=(co, sess, stem, choice, user), type="primary")
    commit_cmd = st.session_state.get(f"camera_commit::{sess}")
    if commit_cmd:
        st.caption("not committed · run from the repo")
        st.code(commit_cmd, language='bash')


def _record_override(co, sess, stem, choice, user):
    # the note is read at click time: callback args are frozen at the previous run
    note = st.session_state.get(f"camera_note::{sess}", "")
    _block, commit = co.add(sess, choice, by=user, note=note)   # raises when refused
    st.session_state.pop(f"camera_frames::{sess}", None)
    st.session_state[f"camera_commit::{sess}"] = commit
    return f"recorded · {stem} → {choice}"
    # [AGENT EDIT END]
# [AGENT EDIT END]
