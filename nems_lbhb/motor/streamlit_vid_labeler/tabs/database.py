"""Tab 4 — Database: what is deployed, and every fit on record.

Two questions, neither about the session in front of you: which model each alias
currently deploys, and which recorded fits need attention.

"Needs attention" is deliberately only two things -- a fit nobody can re-run, and a fit
whose head markers scatter.  Everything else gDLCFit stores (how the release id was
recovered, whether the alias has moved on since, where a QC flag threshold happens to
sit) is archaeology: real, occasionally vital, and answerable with a SQL query by
whoever needs it.  Shown in a browse table it is nine columns of detail nobody acts on,
sitting between the operator and the two that they do.

Read-only on purpose. Promotion edits a git-tracked registry and re-analysis costs GPU
time; neither should be a click in a page whose main job is labelling.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: minimal Database tab -- drop
# staleness, release stamping and the QC flag columns; keep provenance plus the two
# head-marker statistics that identify what needs attention | date: 2026-09-16]
import streamlit as st

import labeler_core as core
from tabs import load_panel

# What a fit row means is defined once, in labeler_core, because the Model tab shows the
# same columns for a single session.
COLUMNS = core.FIT_COLUMNS

HELP = {
    "reproducible": "Whether the release that produced this tracking still exists and "
                    "could be run again. False covers both a fit no release could be "
                    "pinned to at all and one whose weights no longer reproduce it.",
    "head_spread": "qc rms_spread_mean: how far the rigid head markers scatter once the "
                   "head is fixed, in head-lengths, averaged over those markers. "
                   "Reference-free -- it is measured against the video's own means, so "
                   "it needs no training labels. Bigger is worse. Over the 428 scored "
                   "rows on 2026-09-16 the median was 0.23 and the worst tenth sat "
                   "above 0.35.",
    "head_conf": "qc p_median: the median DLC likelihood across the rigid head markers. "
                 "Saturates among healthy fits -- nine rows in ten were above 0.90 on "
                 "2026-09-16 -- so a low value is meaningful but a high one proves "
                 "little on its own.",
    "no qc": "Rows that predate QC entirely have no spread and no confidence at all, "
             "and the count beside this caption says how many are in view. An active "
             "threshold drops them rather than sweeping them in: an unmeasured session "
             "is not a measured-bad one.",
}


def apply_filters(table, text, alias, spread_min, conf_max, only_broken):
    """The prepared frame narrowed by the controls above it.

    A row whose QC was never computed has neither statistic, and an ACTIVE threshold
    drops it -- comparing against NaN does that for free, which is why no ``fillna``
    appears here. A quarter of the corpus predates QC; letting those rows answer "yes"
    to every hunt for bad tracking would park an unactionable block at the top of it.
    """
    view = table
    if text:
        needle = str(text)
        mask = view["video_stem"].fillna("").astype(str).str.contains(
            needle, case=False, regex=False)
        mask |= view["siteid"].fillna("").astype(str).str.contains(
            needle, case=False, regex=False)
        view = view[mask]
    if alias and alias != "(any)":
        view = view[view["routing_alias"] == alias]
    if spread_min > 0:
        view = view[view["head_spread"] >= spread_min]
    if conf_max < 1:
        view = view[view["head_conf"] <= conf_max]
    if only_broken:
        view = view[~view["reproducible"].astype(bool)]
    return view


@st.fragment(run_every=30)
def render():
    core.retain_widget_values("db_filter_alias db_filter_text db_filter_spread "
                              "db_filter_conf db_filter_broken prov_alias_detail".split())
    st.button("Refresh database", key="database_refresh",
              on_click=core.fit_status_cached.clear)

    st.markdown("##### Deployed models")
    panel = load_panel("provenance")
    if panel is None:
        st.warning("provenance panel unavailable")
    else:
        try:
            panel.render_deployed(expanded=None)
        except Exception as e:                                # noqa: BLE001
            st.warning(f"registry unavailable: {e}")

    st.divider()
    _fit_browser()


def _fit_browser():
    """Every gDLCFit row, filterable. The Model tab shows just this session's."""
    st.markdown("##### Recorded fits (gDLCFit)")
    try:
        df = core.fit_status_cached(core._registry_mtime())
    except Exception as e:                                    # noqa: BLE001
        st.warning(f"gDLCFit unavailable: {e}")
        return
    if df is None or not len(df):
        st.info("no rows in gDLCFit yet.")
        return

    table = core.prepare_fits(df)
    aliases = ["(any)"] + sorted(str(a) for a in table["routing_alias"].dropna().unique())

    f1, f2, f3, f4, f5 = st.columns([1.7, 1.1, 1, 1, 1.2])
    with f1:
        text = st.text_input("Site or session", key="db_filter_text",
                             placeholder="e.g. REI059")
    with f2:
        alias = st.selectbox("Alias", aliases, key="db_filter_alias")
    with f3:
        spread_min = st.number_input("spread ≥", min_value=0.0, max_value=5.0,
                                     value=0.0, step=0.05, format="%.2f",
                                     key="db_filter_spread", help=HELP["head_spread"])
    with f4:
        conf_max = st.number_input("conf ≤", min_value=0.0, max_value=1.0,
                                   value=1.0, step=0.01, format="%.2f",
                                   key="db_filter_conf", help=HELP["head_conf"])
    with f5:
        st.write("")
        only_broken = st.checkbox("not reproducible", key="db_filter_broken",
                                  help=HELP["reproducible"])

    view = apply_filters(table, text, alias, spread_min, conf_max, only_broken)
    st.dataframe(view[COLUMNS], hide_index=True, use_container_width=True, height=420,
                 column_config={
                     "reproducible": st.column_config.CheckboxColumn(
                         "reproducible", help=HELP["reproducible"]),
                     "head_spread": st.column_config.NumberColumn(
                         "head_spread", format="%.3f", help=HELP["head_spread"]),
                     "head_conf": st.column_config.NumberColumn(
                         "head_conf", format="%.3f", help=HELP["head_conf"]),
                 })
    st.caption(f"{len(view)} row(s) · {int(view['head_spread'].isna().sum())} without QC",
               help=HELP["no qc"])
# [AGENT EDIT END]
