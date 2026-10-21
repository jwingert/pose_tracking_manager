"""Tab 6 — Arena geometry: scale, axis, fixation and speakers.

Reads the markers placed in Arena markers; changes here feed the same config the
markers tab saves.
"""
import streamlit as st

import labeler_core as core


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: isolate geometry controls and protect unsaved edits from config reload | date: 2026-09-09]
@st.fragment
def render():
# [AGENT EDIT END]
    if not st.session_state.points:
        st.info("Place at least two markers in Arena markers first — everything here is "
                "measured between them.")
        return

    left, right = st.columns(2)
    with left:
        _scale()
        _axis_and_fixation()
    with right:
        _speakers()
        _geometry_check()

    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: mark derived geometry dirty when any editor input changes | date: 2026-09-09]
    import json
    keys = ("distances", "xaxis_points", "fixation_point_select", "fixation_shift_x",
            "fixation_shift_y", "speaker_orientation", "speakers")
    fingerprint = json.dumps({k: st.session_state.get(k) for k in keys}, sort_keys=True)
    previous = st.session_state.get("geometry_edit_fingerprint")
    if previous is not None and previous != fingerprint:
        core.mark_arena_dirty()
    st.session_state.geometry_edit_fingerprint = fingerprint
    if st.session_state.get("arena_config_dirty"):
        st.caption(":orange[Unsaved geometry — save in Arena markers]")
    # [AGENT EDIT END]


# ── scale ──────────────────────────────────────────────────────────────────────
def _scale():
    st.markdown("##### Scale — pair distances")
    st.caption("Type the real-world distance for the pairs you measured; leave the rest "
               "at 0 and they are skipped.")

    core.recompute_pixels_per_meter()
    ppm_pairs = st.session_state.get("pixels_per_meter", {})

    for pair in sorted(st.session_state.distances.keys()):
        label_col, input_col, ppm_col = st.columns([1.5, 0.8, 0.7])
        with label_col:
            st.markdown(f"<div style='padding-top:6px;font-size:13px'>"
                        f"{pair.replace('_', ' ↔ ', 1)}</div>", unsafe_allow_html=True)
        with input_col:
            st.session_state.setdefault(f"dist_{pair}",
                                        float(st.session_state.distances[pair]))
            st.number_input(f"{pair} (m)", min_value=0.0, step=0.01,
                            key=f"dist_{pair}", label_visibility="collapsed")
            st.session_state.distances[pair] = float(st.session_state[f"dist_{pair}"])
        with ppm_col:
            value = ppm_pairs.get(pair)
            st.markdown(f"<div style='padding-top:6px;font-size:12px;opacity:.6'>"
                        f"{value:.0f} px/m</div>" if value else
                        "<div style='padding-top:6px;font-size:12px;opacity:.35'>—</div>",
                        unsafe_allow_html=True)

    ppm_pairs, avg = core.recompute_pixels_per_meter()
    if avg:
        values = list(ppm_pairs.values())
        spread = (max(values) - min(values)) / avg * 100 if len(values) > 1 else 0.0
        st.success(f"average {avg:.0f} px/m · spread {spread:.1f} % "
                   f"over {len(values)} measured pair(s)")
    else:
        st.info("No measured pairs yet — enter at least one distance above.")


# ── axis + fixation ────────────────────────────────────────────────────────────
def _axis_and_fixation():
    st.markdown("##### Axis & fixation")
    marker_keys = list(st.session_state.points.keys())

    x1_col, x2_col = st.columns(2)
    with x1_col:
        x1 = st.selectbox("x1 (left end of axis)", marker_keys, index=0,
                          key="xaxis1_point_select")
    with x2_col:
        default_x2 = 1 if len(marker_keys) > 1 else 0
        x2 = st.selectbox("x2 (right end of axis)", marker_keys, index=default_x2,
                          key="xaxis2_point_select")
    st.session_state.xaxis_points = (x1, x2)
    if x1 == x2:
        st.warning("x1 and x2 are the same marker — the rotation is undefined until they "
                   "differ.")

    base_col, sx_col, sy_col = st.columns([1.4, 0.8, 0.8])
    with base_col:
        default_fixation = st.session_state.get("fixation_point_select", marker_keys[0])
        st.selectbox("Fixation base marker", marker_keys,
                     index=marker_keys.index(default_fixation)
                     if default_fixation in marker_keys else 0,
                     key="fixation_point_select")
    with sx_col:
        st.session_state.setdefault("fixation_shift_x", "0.0")
        st.text_input("Shift x (m)", key="fixation_shift_x")
    with sy_col:
        st.session_state.setdefault("fixation_shift_y", "0.0")
        st.text_input("Shift y (m)", key="fixation_shift_y")

    fixation_point = core.compute_fixation_point_from_state()
    if fixation_point is None:
        st.error("Enter valid numbers for the shifts (metres).")
    else:
        st.session_state.fixation_point = fixation_point
        fx, fy = fixation_point
        st.caption(f"Fixation lands at x={fx:.1f} px, y={fy:.1f} px — drawn in red on the "
                   f"Arena markers frame. The shift is applied in the rotated basis.")


# ── speakers ───────────────────────────────────────────────────────────────────
def _speakers():
    st.markdown("##### Speakers")
    st.selectbox("Positioned relative to the new x-axis at the",
                 ["Top", "Right", "Bottom", "Left"],
                 index=["Top", "Right", "Bottom", "Left"].index(
                     st.session_state.get("speaker_orientation", "Top")),
                 key="speaker_orientation")

    if not st.session_state.speakers:
        st.info("No speakers defined yet.")
    for name in list(st.session_state.speakers.keys()):
        name_col, dist_col, ang_col, del_col = st.columns([0.8, 1, 1, 0.7])
        with name_col:
            st.markdown(f"<div style='padding-top:6px'><strong>{name}</strong></div>",
                        unsafe_allow_html=True)
        with dist_col:
            st.session_state.setdefault(f"{name}_dist",
                                        float(st.session_state.speakers[name]["distance"]))
            st.number_input(f"{name} distance (m)", min_value=0.0, step=0.1,
                            key=f"{name}_dist", label_visibility="collapsed")
            st.session_state.speakers[name]["distance"] = float(st.session_state[f"{name}_dist"])
        with ang_col:
            st.session_state.setdefault(f"{name}_angle",
                                        float(st.session_state.speakers[name]["angle"]))
            st.number_input(f"{name} angle (deg)", min_value=-180.0, max_value=180.0,
                            step=1.0, key=f"{name}_angle", label_visibility="collapsed")
            st.session_state.speakers[name]["angle"] = float(st.session_state[f"{name}_angle"])
        with del_col:
            if st.button("remove", key=f"del_speaker_{name}"):
                del st.session_state.speakers[name]
                # [AGENT EDIT START | agent: Codex | user: wingertj | reason: preserve unsaved speaker changes | date: 2026-09-09]
                core.mark_arena_dirty()
                # [AGENT EDIT END]
                st.rerun()
    st.caption("distance (m) · angle (°) from the axis side above")

    with st.form("add_speaker_form", clear_on_submit=True):
        new_speaker_name = st.text_input("New speaker name", placeholder="e.g. C")
        if st.form_submit_button("Add speaker") and new_speaker_name:
            if new_speaker_name in st.session_state.speakers:
                st.warning("Speaker already exists.")
            else:
                st.session_state.speakers[new_speaker_name] = {"distance": 1.0,
                                                               "angle": 0.0}
                # [AGENT EDIT START | agent: Codex | user: wingertj | reason: preserve unsaved speaker changes | date: 2026-09-09]
                core.mark_arena_dirty()
                # [AGENT EDIT END]
                st.rerun()


def _geometry_check():
    st.markdown("##### Geometry check")
    st.plotly_chart(core.speaker_layout_figure(), use_container_width=True,
                    config={"displayModeBar": False}, key="ag_speaker_layout")
    st.caption("Speakers in metres from the fixation point, in the rotated frame — the "
               "same geometry that gets written into the arena config.")
