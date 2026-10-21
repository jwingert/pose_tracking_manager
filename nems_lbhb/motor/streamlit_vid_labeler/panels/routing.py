# [AGENT EDIT START | agent: claude | user: wingertj | reason: expose the routing METHOD, so
# geometry is the default suggestion but the legacy site-based router stays reachable for
# re-running a historical file with the model that produced it | date: 2026-08-13]
"""Routing panel: which model fits this recording, and why.

Shows a suggestion, never a decision. The method selector (geometry / legacy) sets which
router produced it; every alias remains selectable regardless, because the constraint that
keeps arbitrary routing safe is provenance -- every release-based fit is stamped and
recorded in ``gDLCFit`` -- not a restrictive router.

All logic is in ``nems_lbhb.motor.dlc_routing``; this module only renders.
"""
import contextlib

import streamlit as st

from nems_lbhb.motor import dlc_registry
from nems_lbhb.motor import dlc_routing

METHODS = {
    'geometry': 'Chimney side, audio, probes, camera stream. No animal or site rules.',
    'legacy': 'The migrated site rules — reproduces what a historical file was analysed with.',
}

# Shown instead of an arbitrary default when the router has no suggestion.  Without it the
# selectbox fell back to index 0 -- alphabetically 'pupil', an eye close-up -- and Streamlit
# wrote that into routing_alias, arming the Run buttons with a model nothing had chosen.
NO_MODEL = '— none suggested —'


@contextlib.contextmanager
def _wrapper(expanded):
    """Own expander when given a bool; a bare container when ``expanded is None``.

    The tabbed layout already gives this panel its own titled section, and Streamlit
    refuses an expander inside an expander -- which the "other candidate(s)" expander
    in _render_suggestion would otherwise trip.
    """
    if expanded is None:
        with st.container():
            yield
    else:
        with st.expander("Model routing", expanded=expanded):
            yield


def render(video_path, expanded=True, key='routing', result=None, extra=None, info=None,
           settings=None):
    """Draw the panel and return the alias the user has settled on, or ``None``.

    The return value is a suggestion the caller may use as a default -- it does not
    commit anything.  ``expanded=None`` skips this panel's own expander, for callers
    that already have it inside one.  ``result`` lets a caller that has already run
    :func:`dlc_routing.suggest` (to seed the alias before this panel renders) pass the
    answer in rather than paying for a second lookup.  ``extra`` is a zero-argument
    callable rendered in the third column of the picker row, so a caller can sit its own
    model-selection control beside this one instead of stacking it underneath.

    ``info`` switches to the compact layout: no evidence line, no suggestion banner --
    the suggestion is marked in the picker instead, and ``info(result)`` renders at the
    end of the picker row, where the caller puts everything that explains the choice.
    The method radio moves into a ⚙ popover at the very end of that row, followed by
    ``settings()`` for the caller's own rarely-touched controls.
    """
    if not video_path:
        return None

    with _wrapper(expanded):
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the method radio and
        # four st.metric tiles filled two tall rows to show four short facts, most of them
        # em dashes. One row, one line. | date: 2026-08-17]
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: compact mode moves the method radio into the row's ⚙ popover | date: 2026-09-25]
        if info is not None:
            method = st.session_state.get(f'{key}_method', 'geometry')
            obs_col = None
        else:
            method_col, obs_col = st.columns([1, 2.2])
            with method_col:
                method = _method_radio(key)
        # [AGENT EDIT END]

        if result is None:
            try:
                result = dlc_routing.suggest(video_path, method=method)
            except Exception as e:                            # noqa: BLE001
                st.error(f"routing failed: {e}")              # suggest() should never raise
                return None

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: compact mode -- evidence, suggestion reasons and the resolved release move into one info popover at the end of the picker row; the release is stated once, in the header | date: 2026-09-25]
        if info is not None:
            if not (result.get('candidates') or []):
                st.caption(":orange[⚠ no suggestion]",
                           help=result.get('note') or 'no suggestion for this recording.')
            return _render_override(result, key, extra, info=info, settings=settings)
        # [AGENT EDIT END]
        with obs_col:
            if method == 'geometry':
                _render_observation(result.get('observed') or {})

        _render_suggestion(result)
        return _render_override(result, key, extra)
        # [AGENT EDIT END]


def _render_observation(obs):
    """What the recording IS -- four facts, on one line.

    The rule that turned each fact into a suggestion goes in the hover, not on the page:
    it matters when a suggestion looks wrong, and is noise the rest of the time.  An
    unobserved signal shows as an em dash rather than a sentence explaining its absence.
    """
    bits, why = [], []
    for field, label, ev_key in (('chimney', 'chimney', 'chimney'),
                                 ('audio', 'audio', 'audio'),
                                 ('probes', 'probes', 'chimney_count'),
                                 ('stream', 'stream', 'stream')):
        value = obs.get(field)
        bits.append(f"{label} **{value if value not in (None, '') else '—'}**")
        reason = (obs.get('evidence') or {}).get(ev_key)
        if reason:
            why.append(f"{label}: {reason}")
    st.markdown(" · ".join(bits), help="\n\n".join(why) or None)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the same facts as markdown, for the compact layout's info popover | date: 2026-09-25]
def observation_markdown(obs):
    """The evidence line and the rule behind each fact, as popover markdown."""
    lines = []
    for field, label, ev_key in (('chimney', 'chimney', 'chimney'),
                                 ('audio', 'audio', 'audio'),
                                 ('probes', 'probes', 'chimney_count'),
                                 ('stream', 'stream', 'stream')):
        value = obs.get(field)
        reason = (obs.get('evidence') or {}).get(ev_key)
        lines.append(f"- {label} **{value if value not in (None, '') else '—'}**"
                     + (f" — {reason}" if reason else ""))
    return "\n".join(lines)


def suggestion_markdown(result):
    """Why the top alias was suggested, and what else matched, as popover markdown."""
    candidates = result.get('candidates') or []
    if not candidates:
        return result.get('note') or 'no suggestion for this recording.'
    top = candidates[0]
    lines = [f"**`{top['alias']}`** → release `{top.get('release') or '(none)'}`"]
    # a reason that is just an observed fact is already under Observed
    seen = set(((result.get('observed') or {}).get('evidence') or {}).values())
    lines += [f"- {line}" for line in (top.get('why') or []) + (top.get('unknown') or [])
              if line not in seen]
    if len(candidates) > 1:
        lines.append("\n**Also matched**")
        lines += [f"- `{c['alias']}` → `{c.get('release') or '(none)'}`"
                  for c in candidates[1:]]
    return "\n".join(lines)
# [AGENT EDIT END]


def _render_suggestion(result):
    """The headline, with the runners-up folded into its hover.

    The alternates were a whole expander for information that is only consulted when the
    suggestion looks wrong -- and every alias is selectable in the picker regardless, so
    the list was never a control, only an explanation.
    """
    candidates = result.get('candidates') or []
    if not candidates:
        st.warning(result.get('note') or 'no suggestion for this recording.')
        return

    top = candidates[0]
    why = [str(line) for line in (top.get('why') or []) + (top.get('unknown') or [])]
    if len(candidates) > 1:
        why.append("---")
        why.append("**Also matched:**")
        for c in candidates[1:]:
            detail = "; ".join(str(line) for line in
                               (c.get('why') or []) + (c.get('unknown') or []))
            why.append(f"`{c['alias']}` → release `{c.get('release') or '(none)'}`"
                       + (f" — {detail}" if detail else ""))

    st.success(f"**Suggested: `{top['alias']}`**"
               + (f"  →  release `{top['release']}`" if top.get('release') else
                  '  (no current release)'),
               icon="🧭")
    if why:
        st.caption(f"why this one" + (f" · {len(candidates) - 1} other match(es)"
                                      if len(candidates) > 1 else ""),
                   help="\n\n".join(why))


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the alias picker, the other
# candidates and the caller's project override were three stacked full-width rows holding
# one control each; they are the same decision at three levels of escalation, so they
# belong on one row. | date: 2026-08-17]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: one definition of the method radio, used inline or inside the ⚙ popover | date: 2026-09-25]
def _method_radio(key):
    return st.radio("Routing method", list(METHODS),
                    format_func=lambda m: m.capitalize(),
                    key=f'{key}_method', horizontal=True,
                    label_visibility='collapsed',
                    help="Geometry: " + METHODS['geometry'] + "\n\nLegacy: "
                         + METHODS['legacy'])
# [AGENT EDIT END]


def _render_override(result, key, extra=None, info=None, settings=None):
    """Every alias stays selectable -- the suggestion only sets the default."""
    try:
        names = list(dlc_registry.load_registry()['alias_names'])
    except Exception as e:                                    # noqa: BLE001
        st.caption(f"could not list aliases: {e}")
        return result.get('alias')

    suggested = result.get('alias')
    # nothing routed: offer an explicit non-choice rather than defaulting to whichever
    # alias sorts first, so a model is never armed without someone selecting it
    options = names if suggested in names else [NO_MODEL] + names
    state_key = f'{key}_alias'
    # a stored value that is not in the list makes the selectbox raise; drop it and
    # fall back to the suggestion (registries change, session state outlives them)
    if st.session_state.get(state_key) not in options:
        st.session_state.pop(state_key, None)
    index = options.index(suggested) if suggested in options else 0

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: compact mode is one
    # line -- "Model to use", the picker, the project override and one ⚙ holding the
    # routing method, the refresh and everything ⓘ used to explain | date: 2026-09-25]
    if info is not None:
        label_col, pick_col, extra_col, gear_col = st.columns(
            [0.62, 1.2, 1.2, 0.22], gap="small", vertical_alignment="center")
        with label_col:
            # sized like the page's ##### section titles ("Inspect existing fits")
            st.markdown("<p style='font-size:1.25rem;font-weight:600;margin:0;"
                        "white-space:nowrap;line-height:1.2;position:relative;top:-8px'>"
                        "Model to use</p>",
                        unsafe_allow_html=True)
        with gear_col:
            with st.popover("⚙", use_container_width=True):
                st.caption("routing")
                _method_radio(key)
                if settings is not None:
                    settings()
                st.divider()
                info(result)
    else:
        pick_col, extra_col = st.columns([1, 1.15])
    # [AGENT EDIT END]
    with pick_col:
        # sync_routing_for_video seeds this key via the Session State API before the
        # page renders (that is what makes the suggestion follow the video); passing
        # index= on a run where the key already exists makes Streamlit warn that the
        # default is being ignored -- it is, by design, so only offer it when the key
        # is genuinely unset
        default = {} if state_key in st.session_state else {"index": index}
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: record explicit model overrides independently of automatic suggestions | date: 2026-09-09]
        import labeler_core as core
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: compact mode marks the suggestion in the list instead of a banner above it | date: 2026-09-25]
        if info is not None:
            default["format_func"] = (lambda name: f"{name} · suggested"
                                      if name == suggested else name)
        # [AGENT EDIT END]
        chosen = st.selectbox("Model to use", options, key=state_key,
                              on_change=core.select_routing_manually,
                              label_visibility="collapsed" if info is not None else "visible",
                              **default)
        # [AGENT EDIT END]
        if chosen == NO_MODEL:
            st.caption(":orange[no model selected]",
                       help="This router had no suggestion for this recording. Pick an "
                            "alias, switch routing method, or use the project override.")
            chosen = None
        elif suggested and chosen != suggested:
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: "Use suggestion" moved here from the routing status rows | date: 2026-09-25]
            # stacked, not columns: this row is already nested inside the caller's
            # columns, and Streamlit allows one level of nesting
            st.caption(f":orange[⚠ overrides `{suggested}`]")
            if st.button("use suggestion", key=f"{key}_follow",
                         on_click=core.follow_routing):
                st.rerun()
            # [AGENT EDIT END]

    if extra is not None:
        with extra_col:
            extra()
    return chosen
# [AGENT EDIT END]
# [AGENT EDIT END]
