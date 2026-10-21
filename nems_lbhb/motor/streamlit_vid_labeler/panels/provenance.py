# [AGENT EDIT START | agent: claude | user: wingertj | reason: nothing in the GUI could
# answer "which model produced this session's tracking, and is it current?" -- the question
# the whole lifecycle project exists to make answerable | date: 2026-08-12]
"""Provenance panels: what is deployed, and what produced this session.

Two read-only views over data that already exists:

* the registry's alias -> release pointers (``dlc_registry`` + ``dlc_release``);
* the ``gDLCFit`` row for the session currently selected (``dlc_fit_record.fit_status``).

There was a third -- a per-alias tally of recorded tracking whose alias has since moved
on. It was removed on purpose. A fit being older than its alias's current release says
nothing about whether that fit is any good, which is the only thing a labelling session
can act on; it outnumbered the real gaps several to one and read as a backlog.

Read-only on purpose. Promotion edits a git-tracked registry and re-analysis costs GPU
time; neither should be a click in a page whose main job is labelling. This tells you what
is true and what the command would be.
"""
import contextlib
import os

import streamlit as st

from nems_lbhb.motor import dlc_registry

# [AGENT EDIT START | agent: claude | user: wingertj | reason: same columns as the Database
# tab's browse table, so one fit cannot read differently in two places | date: 2026-09-16]
# The Database tab's columns minus the site, which is already in the heading above this
# table. Spelled out rather than imported from labeler_core because this panel is also
# loaded by package path, where `labeler_core` is not a top-level module;
# tests/test_database_tab.py pins the two lists to each other instead.
SESSION_COLUMNS = ["video_stem", "stream", "routing_alias", "release_id",
                   "reproducible", "head_spread", "head_conf", "target_mtime_utc"]
# [AGENT EDIT END]


@contextlib.contextmanager
def _wrapper(title, expanded):
    """Own expander when given a bool; a bare container when ``expanded is None``.

    A caller that has already given this panel a titled section of its own (the
    Database tab) does not want a second collapsible box around it.
    """
    if expanded is None:
        with st.container():
            yield
    else:
        with st.expander(title, expanded=expanded):
            yield


def render_deployed(expanded=False):
    """Alias -> current release -> snapshot, straight from the registry."""
    with _wrapper("Deployed models (registry)", expanded):
        try:
            reg = dlc_registry.load_registry()
        except Exception as e:                                # noqa: BLE001
            st.error(f"could not load the registry: {e}")
            return
        # load_registry moves the TOML's [alias.*] tables to 'aliases' and keeps the
        # file order in 'alias_names', so the panel lists them as the registry reads.
        aliases = reg.get('aliases') or {}
        order = reg.get('alias_names') or sorted(aliases)
        if not aliases:
            st.info("no aliases in the registry.")
            return

        rows = []
        for name in order:
            block = aliases.get(name) or {}
            current = block.get('current')
            rows.append({
                'alias': name,
                'current release': current or '(none)',
                'releases': len(block.get('history') or []),
                'qc': block.get('qc', ''),
                'bodyparts': len(block.get('bodyparts') or []),
                'description': (block.get('description') or '')[:60],
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

        names = [r['alias'] for r in rows]
        pick = st.selectbox("Release detail", names, key="prov_alias_detail")
        _render_release_detail(aliases.get(pick, {}).get('current'))


def _render_release_detail(release_id):
    if not release_id:
        st.caption("this alias has no current release.")
        return
    try:
        from nems_lbhb.motor import dlc_release
        manifest = dlc_release.read_manifest(release_id)
    except Exception as e:                                    # noqa: BLE001
        st.warning(f"could not read {release_id}'s manifest: {e}")
        return

    rel = manifest.get('release', {}) or {}
    src = manifest.get('source', {}) or {}
    ev = manifest.get('evaluation', {}) or {}
    cols = st.columns(4)
    cols[0].metric("snapshot", src.get('snapshot', '?'))
    cols[1].metric("iteration", src.get('iteration', '?'))
    cols[2].metric("test error (px)", _fmt(ev.get('test_error_px')))
    cols[3].metric("labelled frames", src.get('label_frames', '?'))
    st.caption(f"built by {rel.get('created_by', '?')} on {rel.get('created_utc', '?')} "
               f"from `{src.get('project', '?')}`")
    if rel.get('note'):
        st.caption(str(rel['note']))

    warn = rel.get('provenance_warning')
    if warn:
        # e.g. left_chimney-r0001, whose original weights were destroyed by an in-place
        # retrain; the release is honest about being a reconstruction.
        st.warning(f"**provenance warning:** {warn}")


def render_session(video_path, expanded=False):
    """The gDLCFit row(s) for the session containing ``video_path``."""
    if not video_path:
        return
    stem = _session_stem(video_path)
    with _wrapper(f"Tracking provenance for {stem}", expanded):
        try:
            # [AGENT EDIT START | agent: Codex | user: wingertj | reason: scope session queries before loading and skip unused pointer checks | date: 2026-09-09]
            import re
            import labeler_core as core
            match = re.match(r'^([A-Za-z]{3}\d+[a-z])', stem)
            if not match:
                st.caption("Cannot identify the site for this video.")
                return
            df = core.fit_status_cached(core._registry_mtime(), siteid=match.group(1))
            # [AGENT EDIT END]
        except Exception as e:                                # noqa: BLE001
            st.warning(f"gDLCFit unavailable: {e}")
            return
        if df is None or not len(df):
            st.info("no rows in gDLCFit yet.")
            return

        mine = df[df['video_stem'].fillna('').str.startswith(stem)]
        if not len(mine):
            st.caption(f"nothing recorded for {stem}",
                       help="If a .dlc.h5 exists on disk, the sweeper "
                            "(scripts/dlc_sync_fits.py) has not seen it yet.")
            return

        mine = core.prepare_fits(mine)
        st.dataframe(mine[[c for c in SESSION_COLUMNS if c in mine.columns]],
                     hide_index=True, use_container_width=True)


def _session_stem(video_path):
    """'<session>' for any of the shapes a caller might hold."""
    p = str(video_path).rstrip('/')
    base = os.path.basename(p)
    if base.startswith(('recording', 'video_')):
        return os.path.basename(os.path.dirname(p))
    return os.path.splitext(base)[0]


def _fmt(v):
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "?"
# [AGENT EDIT END]
