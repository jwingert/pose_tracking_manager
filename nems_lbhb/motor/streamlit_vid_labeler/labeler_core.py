"""Shared helpers for the tabbed DLC Arena Labeler.

Everything here was lifted from ``behavior_config_labeler_plotly.py`` so the two
layouts can live side by side: the original single-page script is untouched and
still runnable, and this module holds the logic the tab modules call.  Nothing in
here renders the page layout -- see ``behavior_config_labeler_tabs.py`` and
``tabs/``.
"""
import base64
import copy
import glob
import json
import math
import os
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: keep disposable browser previews outside worker-owned release directories | date: 2026-09-10]
import hashlib
import tempfile
# [AGENT EDIT END]

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tomli
import tomli_w
import yaml
from PIL import Image
from plotly.subplots import make_subplots

# [AGENT EDIT START | agent: Codex | user: wingertj | reason: defer DLC imports until an action needs them | date: 2026-09-09]
import time
from labeler_state import apply_routing


class _LazyDLC:
    def __getattr__(self, name):
        import deeplabcut
        return getattr(deeplabcut, name)


deeplabcut = _LazyDLC()
# [AGENT EDIT END]
from nems_lbhb import db

# Optional pieces of the lifecycle stack.  The single-page app imported these
# unguarded; a missing one took the whole page down, which is a bad trade for a
# panel that degrades fine.
try:
    from nems_lbhb.motor import dlc_headfix_qc as hfqc
except Exception:                                             # noqa: BLE001
    hfqc = None

# [AGENT EDIT START | agent: claude | user: wingertj | reason: reading a NEW module
# constant off hfqc turned a version skew into a hard "Spread QC failed: no attribute
# DEFAULT_TRACK_P_CUT". Two ways that happens and neither is exotic: Streamlit reloads the
# app script but NOT imported package modules, so an app running across an edit keeps the
# old module in sys.modules; and the deployed checkout at /auto/users/lbhb/Code/nems_db
# predates these constants entirely. The rest of the panel works fine without them, so
# resolve once with a fallback rather than letting one missing constant take QC down.
# | date: 2026-08-17]
# Values mirror dlc_headfix_qc's own defaults, which are baphy_io's `dlc_threshold`; see
# the LOADER_P_CUT comment there. Both gates are the loader's -- they are separate names
# for one number so a caller CAN probe a stricter tracking bar, not because they differ.
SPREAD_P_CUT = getattr(hfqc, 'DEFAULT_P_CUT', 0.2) if hfqc else 0.2
TRACK_P_CUT = getattr(hfqc, 'DEFAULT_TRACK_P_CUT', 0.2) if hfqc else 0.2
# [AGENT EDIT END]
try:
    from nems_lbhb.motor import dlc_registry
except Exception:                                             # noqa: BLE001
    dlc_registry = None
try:
    from nems_lbhb.motor import dlc_generalization
except Exception:                                             # noqa: BLE001
    dlc_generalization = None


PREFS_FILE = Path.home() / ".dlc_labeler_prefs.json"

DLC_CONFIG_GLOB = "/auto/data/dlc/**/config.yaml"

INITIAL_STATE = {
    "video_path": "",
    "current_frame": None,
    "points": {},
    "selected_point": None,
    "fixation_point": None,
    "distances": {},
    "speakers": {"R": {"distance": 1.0, "angle": 30.0},
                 "L": {"distance": 1.0, "angle": -30.0}},
    "speaker_orientation": "Top",
    "ppm_avg": 1.0,
    "cdict": None,
    "plotly_key_version": 0,
    "last_imported_config": None,
    "dlc_bodyparts": [],
    "dlc_videos": [],
    "selected_dlc_model_config": "",
}


class _PlotlyFigureWithConfig:
    """Carries a plotly config through streamlit_plotly_events."""

    def __init__(self, fig, config):
        self._fig = fig
        self._config = config

    def to_json(self):
        payload = json.loads(self._fig.to_json())
        payload["config"] = self._config
        return json.dumps(payload)


# ── preferences ────────────────────────────────────────────────────────────────
def load_user_prefs():
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                return json.load(f)
        except Exception:                                     # noqa: BLE001
            return {}
    return {}


def save_user_prefs(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:                                    # noqa: BLE001
        st.warning(f"Could not save preferences: {e}")


def remember_pref(key, value):
    prefs = load_user_prefs()
    if prefs.get(key) != value:
        prefs[key] = value
        save_user_prefs(prefs)


def current_dlc_user():
    """Who is driving this session, for job attribution."""
    name = st.session_state.get("dlc_run_user") or load_user_prefs().get("dlc_run_user")
    if name:
        return str(name).strip()
    import getpass
    try:
        return getpass.getuser()
    except Exception:                                         # noqa: BLE001
        return "unknown"


# ── session bootstrap / chrome ─────────────────────────────────────────────────
def init_session_state(query_params=None):
    seeded = dict(INITIAL_STATE)
    if query_params is not None:
        seeded["video_path"] = query_params.get("video", "") or ""
    for k, v in seeded.items():
        if k not in st.session_state:
            st.session_state[k] = v


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: restore tab-like navigation without executing hidden sections | date: 2026-09-09]
def inject_css():
    css = """
        <style>
        .plotly-plot { background-color: transparent !important; }
        .modebar, .modebar-container,
        .plotly .modebar, .plotly .modebar-container, .plotly .modebar-btn {
            display: none !important;
        }
        .block-container { padding-top: 2.2rem; padding-left: 1.4rem; padding-right: 1.4rem; }
        /* [AGENT EDIT START | agent: claude | user: wingertj | reason: Streamlit's fixed
           60 px top bar is painted in the page colour; with the tightened padding above it
           covered the top of the header logo. Transparent, and click-through except for its
           own Deploy/menu buttons | date: 2026-09-25] */
        [data-testid="stHeader"] { background: transparent; pointer-events: none; }
        [data-testid="stElementContainer"]:has(.outcome-style) { display: none; }
        /* white on an icy accent is unreadable: primary buttons get dark text. Outcome
           colours (action_button) set their own white text and win with !important */
        button[kind="primary"], button[data-testid="stBaseButton-primary"] {
            color: THEME_ACCENT_TEXT;
            /* pinned: Streamlit's built-in Light/Dark themes (chosen in Settings, saved
               per browser) bring their own RED primary and ignore config.toml */
            background-color: THEME_ACCENT; border-color: THEME_ACCENT;
        }
        button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
            filter: brightness(1.08);
        }
        button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {
            color: THEME_ACCENT_TEXT; font-weight: 600;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"],
        [data-testid="stHeader"] [data-testid="stDecoration"] { pointer-events: auto; }
        /* [AGENT EDIT END] */
        /* Only navigation radios get the tab treatment. Keep native inputs for
           keyboard access; hide the decorative circle, not the input or label. */
        [class*="st-key-labeler_tabs_"] [role="radiogroup"] {
            gap: .25rem;
            border-bottom: 1px solid rgba(128, 128, 128, .4);
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"] {
            margin: 0 !important;
            padding: .65rem 1rem;
            border-radius: .4rem .4rem 0 0;
            border-bottom: 3px solid transparent;
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"] > div:first-of-type {
            display: none;
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"] > div:last-of-type {
            padding-left: 0;
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"]:hover {
            background: rgba(128, 128, 128, .1);
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"]:has(input:checked) {
            background: THEME_ACCENT_TINT;
            border-bottom-color: THEME_ACCENT;
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"]:has(input:checked) p {
            font-weight: 600;
        }
        [class*="st-key-labeler_tabs_"] label[data-baseweb="radio"]:has(input:focus-visible) {
            outline: 2px solid THEME_ACCENT;
            outline-offset: -2px;
        }
        /* [AGENT EDIT START | agent: claude | user: wingertj | reason: the video path is an
           editable session badge -- chip() colours and type, still a text box | date: 2026-09-25] */
        .st-key-session_badges > div[data-testid="stVerticalBlock"],
        .st-key-session_badges[data-testid="stVerticalBlock"] {
            flex-direction: row; flex-wrap: nowrap; justify-content: flex-end;
            align-items: center; gap: 6px;
        }
        .st-key-session_badges [data-testid="stElementContainer"],
        .st-key-session_badges [data-testid="stMarkdown"],
        .st-key-session_badges [data-testid="stMarkdownContainer"] { width: auto !important; }
        .st-key-session_badges [data-testid="stElementContainer"]:not(:has([data-testid="stTextInput"])) {
            flex: 0 0 auto;             /* the tags never wrap or shrink */
        }
        .st-key-session_badges [data-testid="stElementContainer"]:has([data-testid="stTextInput"]) {
            flex: 0 1 440px; min-width: 160px;
            transform: translateY(8px);   /* level with chip(), whose markdown block sits lower */
        }
        .st-key-session_badges [data-testid="stTextInput"] { width: 100%; }
        .st-key-session_badges [data-testid="stMarkdownContainer"] p { margin: 0; }
        .st-key-session_badges [data-baseweb="input"] {
            background: THEME_BADGE; border: none; border-radius: 2px; min-height: 0;
            height: 25px;
        }
        .st-key-session_badges [data-baseweb="input"]::before {
            content: "video"; color: THEME_BADGE_TEXT; opacity: .65; font-size: 12px;
            padding-left: 10px; align-self: center; white-space: nowrap;
        }
        .st-key-session_badges [data-baseweb="input"] > div { background: transparent; }
        .st-key-session_badges input {
            color: THEME_BADGE_TEXT; -webkit-text-fill-color: THEME_BADGE_TEXT; font-size: 12px;
            font-weight: 700; padding: 0 10px 0 6px; height: 25px; line-height: 25px;
            text-overflow: ellipsis;
        }
        .st-key-session_badges [data-baseweb="input"]:focus-within {
            outline: 1px solid THEME_ACCENT;
        }
        /* [AGENT EDIT END] */
        </style>
    """
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: theme colours come from THEME, not literals | date: 2026-09-28]
    for name in ("accent_tint", "accent_text", "accent", "badge_text", "badge"):  # longest first
        css = css.replace("THEME_" + name.upper(), THEME[name])
    st.markdown(css, unsafe_allow_html=True)
    # [AGENT EDIT END]


def tab_selector(label, options, *, key, format_func=str):
    """Tab-styled, keyboard-accessible navigation with lazy panel execution.

    Streamlit 1.40 runs every st.tabs body. A keyed radio preserves selection
    and lets callers execute only the active body; the container scopes its CSS.
    """
    with st.container(key=f"labeler_tabs_{key}"):
        return st.radio(label, options, key=key, format_func=format_func,
                        horizontal=True, label_visibility="collapsed")
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: warnings rendered as
# full-width boxes carrying whole paragraphs (release provenance notes, QC caveats). A
# note is now a short tag on one line; the full text is never dropped, it moves into the
# hover popup -- "if it must be said, say it in under ten words" | date: 2026-09-25]
_NOTE_RULES = (
    (re.compile(r"training reference built from only (\d+) labeled frames"),
     lambda m: f"weak reference · {m.group(1)} frames"),
    (re.compile(r"only (\d+) valid frames"), lambda m: f"{m.group(1)} valid frames · noisy"),
    (re.compile(r"within-video QC only"), lambda m: "within-video QC only"),
    (re.compile(r"^Unverified release"), lambda m: "unverified release"),
    (re.compile(r"^Scaled imported points"), lambda m: "points rescaled to this frame"),
    (re.compile(r"missing saved frame size"), lambda m: "no saved frame size"),
    (re.compile(r"^Config video path differs"), lambda m: "config from another video"),
    (re.compile(r"^Config video path not found"), lambda m: "config's video missing"),
    (re.compile(r"^Failed to load configuration"), lambda m: "config failed to load"),
    (re.compile(r"^Could not read frame (\d+)"), lambda m: f"frame {m.group(1)} unreadable"),
    # release notes that lead with a capitalised verdict: DAMAGED PROVENANCE, AMBIGUOUS,
    # MISLEADING SOURCE PROJECT NAME, DEFECT 9 ... -- the verdict IS the short form
    # (a 3-letter acronym like "DLC test error ..." is not a verdict)
    (re.compile(r"^([A-Z]{4,}(?:[ -](?:[A-Z]{3,}|\d+)\b){0,4})"),
     lambda m: m.group(1).capitalize()),
)


def short_note(text, fallback=None):
    """``(label, alarming)`` -- a few-word tag for a warning, and whether it is one.

    Known QC caveats and capitalised release verdicts are alarming; anything else is a
    plain note labelled ``fallback`` (or its first five words).
    """
    text = str(text or "").strip()
    for pattern, label in _NOTE_RULES:
        m = pattern.search(text)
        if m:
            return label(m), True
    if fallback:
        return fallback, False
    words = text.split()
    return " ".join(words[:5]) + (" …" if len(words) > 5 else ""), False


def notes_caption(texts, prefix="", fallback=None):
    """One caption line of tags for ``texts``; every full text rides in the hover."""
    texts = [t for t in texts if t and str(t).strip()]
    if not texts:
        return
    tags = []
    for t in texts:
        label, alarming = short_note(t, fallback)
        tag = f":orange[⚠ {label}]" if alarming else label
        if tag not in tags:                  # two texts can reduce to the same tag
            tags.append(tag)
    st.caption((f"{prefix} · " if prefix else "") + " · ".join(tags),
               help="\n\n".join(str(t).strip() for t in texts))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: button clicks reported
# their outcome by writing st.success/st.error text into the layout. An action button
# now reports through its own colour -- green done, amber done with a caveat, red failed
# -- with the messages in its hover, and a failure opens a dialog instead of a box on
# the page | date: 2026-09-28]
_OUTCOMES = "_action_outcomes"
_RANK = {"success": 0, "info": 0, "warning": 1, "error": 2}
OUTCOME_STYLE = {"ok": "#2e7d32", "warn": "#b26a00", "error": "#c62828"}


def say(kind, text):
    """``st.success/info/warning/error(text)`` -- or, inside an action button's job,
    a message for that button's outcome instead of text in the layout."""
    log = st.session_state.get("_action_log")
    if log is None:
        getattr(st, kind)(text)
    else:
        log.append((kind, str(text)))


def key_class(key):
    """The CSS class Streamlit 1.40 gives a keyed element (its frontend ``wr()``)."""
    return "st-key-" + re.sub(r"[^a-zA-Z0-9_-]", "-", str(key).strip())


def _run_action(key, fn, args, kwargs):
    st.session_state["_action_log"] = []
    try:
        result = fn(*args, **(kwargs or {}))
        # job functions that return their message rather than saying it
        logged = [t for _k, t in st.session_state["_action_log"]]
        if isinstance(result, str) and result.strip() and result not in logged:
            failed = result.lstrip().lower().startswith("error")
            st.session_state["_action_log"].append(("error" if failed else "success",
                                                    result))
        elif result is False:
            st.session_state["_action_log"].append(("error", "failed"))
    except Exception as exc:                                  # noqa: BLE001
        st.session_state["_action_log"].append(("error", f"{type(exc).__name__}: {exc}"))
    log = st.session_state.pop("_action_log")
    state = ("ok", "warn", "error")[max((_RANK.get(k, 0) for k, _ in log), default=0)]
    st.session_state.setdefault(_OUTCOMES, {})[key] = dict(
        state=state, messages=log, dialog=state == "error",
        scope=st.session_state.get("video_path", ""))


def action_outcome(key):
    """The last outcome of ``key``'s job for the session on screen, or None."""
    out = (st.session_state.get(_OUTCOMES) or {}).get(key)
    if out and out["scope"] != st.session_state.get("video_path", ""):
        st.session_state[_OUTCOMES].pop(key, None)
        return None
    return out


def _outcome_dialog(label, messages):
    def body():
        for kind, text in messages:
            getattr(st, "error" if kind == "error" else
                    "warning" if kind == "warning" else "info")(text)
    st.dialog(f"{label} · failed")(body)()


def action_button(label, key, fn, args=(), kwargs=None, help=None, **button_kw):
    """``st.button`` that runs ``fn`` and shows how it went on the button itself.

    ``fn`` reports with :func:`say` (or returns a message string, or raises). The
    button turns green/amber/red, its hover lists the messages, and a failure opens a
    dialog once. The colour lasts until the next click or a change of session.
    """
    out = action_outcome(key)
    if out:
        colour = OUTCOME_STYLE[out["state"]]
        # the style element's own slot is hidden (inject_css) so the button does not
        # shift down a row; a <style> applies even inside a display:none container
        st.markdown(f"<div class='outcome-style'><style>.{key_class(key)} button "
                    f"{{background:{colour} !important;border-color:{colour} !important;"
                    f"color:#fff !important}} .{key_class(key)} button p "
                    f"{{color:#fff !important}}</style></div>", unsafe_allow_html=True)
        detail = "\n\n".join(text for _kind, text in out["messages"])
        help = "\n\n".join(t for t in (detail, help) if t) or None
    clicked = st.button(label, key=key, on_click=_run_action,
                        args=(key, fn, tuple(args), kwargs), help=help, **button_kw)
    if out and out.get("dialog"):
        out["dialog"] = False
        _outcome_dialog(label, out["messages"])
    return clicked
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: one accent colour and a
# dark default (.streamlit/config.toml); these mirror it for colours set in CSS/HTML.
# Green/amber/red stay reserved for status | date: 2026-09-28]
THEME = {
    "accent": "#7DD3FC",                    # icy blue -- the 10%
    "accent_tint": "rgba(125, 211, 252, .16)",
    "accent_text": "#0B1119",               # text ON the accent: dark, for contrast
    "panel": "#18222F",                     # the 30%
    # translucent + inherited text: readable whichever theme the viewer picked in
    # Settings (a saved browser choice overrides the dark default)
    "badge": "rgba(128, 146, 170, .20)",
    "badge_text": "inherit",
}
# [AGENT EDIT END]


def chip(label, value, tone="neutral", title=None):
    """A small session-identity badge for the top strip."""
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: light badges were bright patches on the dark theme; dark tints, accent for identity, status tones only for status | date: 2026-09-28]
    colors = {
        "neutral": (THEME["badge"], "inherit"),
        "accent": ("rgba(125, 211, 252, .28)", "inherit"),
        "warn": ("rgba(178, 106, 0, .30)", "inherit"),
        "bad": ("rgba(198, 40, 40, .30)", "inherit"),
    }
    # [AGENT EDIT END]
    bg, fg = colors.get(tone, colors["neutral"])
    hover = f" title='{title}'" if title else ""
    return (f"<span{hover} style='display:inline-block;padding:3px 10px;margin-right:6px;"
            f"background:{bg};color:{fg};font-size:12px;border-radius:2px;"
            f"white-space:nowrap'><span style='opacity:.65'>{label}</span> "
            f"<strong>{value}</strong></span>")


# ── DLC model discovery / aliases ──────────────────────────────────────────────
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: share expensive NFS discovery across reruns and panels | date: 2026-09-09]
@st.cache_data(ttl=300, max_entries=1, show_spinner=False)
def discover_dlc_models():
    configs = glob.glob(DLC_CONFIG_GLOB, recursive=True)
    names = [Path(p).parent.name for p in configs]
    return dict(sorted(zip(names, configs)))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: cache file contents by identity while keeping widget updates outside cached functions | date: 2026-09-09]
def file_stamp(path):
    """Resolve symlinks as well as detecting replacement/edits at the same path."""
    real = os.path.realpath(str(path))
    stat = os.stat(real)
    return real, stat.st_mtime_ns, stat.st_size


@st.cache_data(max_entries=64, show_spinner=False)
def _read_yaml(stamp):
    with open(stamp[0]) as f:
        return yaml.safe_load(f)


def read_yaml(path):
    return _read_yaml(file_stamp(path))


@st.cache_data(max_entries=4, show_spinner=False)
def _prediction_data(stamp):
    return pd.read_hdf(stamp[0])


def prediction_frame(path, frame_number):
    # Keep the selected table in this session too: st.cache_data otherwise copies
    # the full dataframe back to the caller even on a cache hit for every frame.
    stamp = file_stamp(path)
    cached = st.session_state.get("prediction_data")
    if cached is None or cached[0] != stamp:
        cached = (stamp, _prediction_data(stamp))
        st.session_state.prediction_data = cached
    return cached[1].iloc[int(frame_number)]


@st.cache_data(ttl=30, max_entries=64, show_spinner=False)
def fit_status_cached(registry_mtime, siteid=None):
    from nems_lbhb.motor import dlc_fit_record
    return dlc_fit_record.fit_status(siteid=siteid, include_current=False)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: one definition of what a
# gDLCFit row MEANS, shared by the Database tab's browse table and the Model tab's
# per-session table -- two column sets would let the same fit read differently in two
# places | date: 2026-09-16]
FIT_COLUMNS = ["siteid", "video_stem", "stream", "routing_alias", "release_id",
               "reproducible", "head_spread", "head_conf", "target_mtime_utc"]


def reproducible_flag(table):
    """True where a fit names a release that exists and could be run again.

    Two stored columns folded into the one question an operator acts on. gDLCFit's
    ``reproducible`` is NULL for a row no release could be pinned to at all, and a blank
    cell in a triage table reads as "nothing wrong here" -- the opposite of what it
    means -- so unknown is reported as False.

    Deliberately NOT done in ``dlc_fit_record``: ``is_stale`` keys off the raw
    ``reproducible == 0`` and the sweeper writes the raw column, so widening the meaning
    at the source would change behaviour in code the GUI does not own.
    """
    if "release_id" not in table.columns or "reproducible" not in table.columns:
        return pd.Series(False, index=table.index, dtype=bool)
    flag = pd.to_numeric(table["reproducible"], errors="coerce").fillna(0) == 1
    return table["release_id"].notna() & flag


def prepare_fits(table):
    """A ``fit_status`` frame with the GUI's derived columns added.

    The two QC statistics are renamed here, once, so a threshold and the column it is
    compared against can never come from different places. ``head_spread`` is the qc
    document's ``rms_spread_mean`` (rigid head-marker scatter in head-lengths, measured
    against the video's own means) and ``head_conf`` its ``p_median`` (median DLC
    likelihood over those markers).
    """
    out = table.copy()
    out["reproducible"] = reproducible_flag(table)
    for dest, src in (("head_spread", "rms_spread_mean"), ("head_conf", "p_median")):
        out[dest] = (pd.to_numeric(out[src], errors="coerce") if src in out.columns
                     else pd.Series(float("nan"), index=out.index))
    for name in FIT_COLUMNS:
        if name not in out.columns:
            out[name] = None
    return out
# [AGENT EDIT END]


@st.cache_resource(show_spinner=False)
def _routing_engine():
    """Use bounded MySQL waits for interactive metadata reads only."""
    from sqlalchemy import create_engine
    engine = db.Engine()
    if engine.dialect.name != "mysql":
        return engine
    return create_engine(engine.url, pool_pre_ping=True, pool_recycle=1600,
                         pool_timeout=3,
                         connect_args={"connect_timeout": 3, "read_timeout": 5,
                                       "write_timeout": 5})


class _RoutingDB:
    @staticmethod
    def pd_query(sql, params=None):
        with _routing_engine().connect() as connection:
            return db._exec_driver_sql_to_df(connection, sql, params)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the session finder needs a
# whole-animal view of what is on disk, cached so switching filters does not re-scan NFS
# | date: 2026-09-15]
@st.cache_data(ttl=300, show_spinner=False)
def recorded_animals():
    """Animals with recordings, most recently recorded first.

    Recency ordering rather than alphabetical: the finder is used on whatever is being
    collected now, and the selectbox is type-ahead searchable for everything else.
    """
    df = _RoutingDB.pd_query(
        "SELECT animal, MAX(pendate) AS lastdate FROM gPenetration "
        "WHERE animal IS NOT NULL AND animal <> '' "
        "GROUP BY animal ORDER BY lastdate DESC")
    return [a for a in df["animal"].tolist() if a]


@st.cache_data(ttl=600, show_spinner="scanning sorted/ · up to a minute cold")
def session_status(animal):
    """Every recording for one animal, with what sorted/ and gDLCFit say about it.

    Scoped per animal so runclass filtering happens in the frame instead of going back
    to disk.  The TTL is long and the Rescan button explicit because the cost is the NFS
    attribute cache, not the row count: a first look at an animal measured 34-64 s cold
    and 0.22-0.31 s warm, and a short TTL would hand that cold minute back at random.
    """
    from nems_lbhb.motor import dlc_session_status
    return dlc_session_status.session_status(animal=animal, query=_RoutingDB.pd_query,
                                           fits=fit_status_cached(_registry_mtime()))


def refresh_session_status():
    fit_status_cached.clear()
    session_status.clear()
    recorded_animals.clear()
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Rescan must also re-read routing and queue state for the finder's bulk-queue section | date: 2026-09-22]
    routed_session.clear()
    queue_states.clear()
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: queue the routed model for
# many missing-dlc sessions at once from the session finder, instead of opening each one
# and pressing "Production → sorted" on the Model tab | date: 2026-09-22]
FINDER_QUEUE_NOTE = "analyze_to_sorted"
QUEUE_STATE_NAMES = {0: "queued", -1: "running", 1: "done", 2: "dead"}


@st.cache_data(ttl=600, max_entries=4096, show_spinner=False)
def routed_session(video_path, registry_mtime):
    """The geometry router's answer for one video, flattened for a table row.

    Always the geometry method -- the Model tab's radio is a per-session choice and the
    legacy router exists to reproduce historical fits, not to pick new ones.
    ``registry_mtime`` is only a cache key, so a promote or rollback re-routes.
    """
    try:
        from nems_lbhb.motor import dlc_routing
        result = dlc_routing.suggest(str(video_path), method="geometry", db=_RoutingDB())
    except Exception as exc:                                  # noqa: BLE001
        return {"alias": None, "release": None, "unknown": "", "error": str(exc)}
    errors = (result.get("observed") or {}).get("db_errors") or []
    top = (result.get("candidates") or [{}])[0]
    return {"alias": result.get("alias"), "release": top.get("release"),
            "unknown": "; ".join(top.get("unknown") or []),
            "error": "; ".join(errors) or (None if result.get("alias")
                                           else result.get("note"))}


def finder_job_notes(video_path, alias, release_id, dest="sorted"):
    """Every tQueue note a ``dest`` job for this session and release can carry.

    temp: the Model tab's "Preview → temp" job, the same note either place queues it.
    sorted: the fit, and the preview copy :func:`queue_routed_fit` (or the Model tab)
    queues instead when a vetted preview exists -- either one already queued is the
    same work.
    """
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: finder can bulk-queue previews to temp, not only production to sorted | date: 2026-09-22]
    if dest == "temp":
        return (_job_note(video_path, None, "fit", "_temp_analysis", alias, release_id),)
    # [AGENT EDIT END]
    return (_job_note(video_path, None, "fit", "_" + FINDER_QUEUE_NOTE, alias, release_id),
            _job_note(video_path, None, "publish", "_publish_temp", alias, release_id))


@st.cache_data(ttl=30, show_spinner=False)
def queue_states(notes):
    """``{note: 'queued'|'running'|'done'|'dead'}`` for the notes already in tQueue."""
    notes = [n for n in notes if n]
    if not notes:
        return {}
    marks = ", ".join(["%s"] * len(notes))
    df = _RoutingDB.pd_query(
        f"SELECT note, complete FROM tQueue WHERE note IN ({marks})", tuple(notes))
    return {r.note: QUEUE_STATE_NAMES.get(int(r.complete), str(r.complete))
            for r in df.itertuples(index=False)}


def queue_routed_fit(video_path, alias, release_id, dest="sorted"):
    """Queue one session under ``alias``, without touching page state.

    ``dest="temp"`` is the Model tab's "Preview → temp": a stamped release fit into the
    session's tmp/dlc folder, for vetting.  ``dest="sorted"`` is "Production → sorted":
    the same fit into sorted/, or a copy of a vetted preview of that same release when
    one is already in temp.  Returns the queue's message.
    """
    if not (video_path and os.path.exists(str(video_path))):
        return f"Error: no video at {video_path}"
    if not (alias and release_id):
        return f"Error: {alias or 'no alias'} has no release to run"
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: finder can bulk-queue previews to temp, not only production to sorted | date: 2026-09-22]
    if dest == "temp":
        return queue_dlc_job(video_path, None, str(session_temp_dir(video_path)), "fit",
                             extra_note="temp_analysis", alias=alias)
    # [AGENT EDIT END]
    preview = None
    try:
        from nems_lbhb.motor import dlc_fit_script as fs
        preview = fs.publishable_preview(str(session_temp_dir(video_path)),
                                         Path(video_path).parents[0].name, release_id)
    except Exception:                                         # noqa: BLE001
        pass
    if preview:
        return queue_dlc_job(video_path, None, None, "publish", extra_note="publish_temp",
                             alias=alias, src=str(session_temp_dir(video_path)),
                             fallback=True)
    return queue_dlc_job(video_path, None, None, "fit", extra_note=FINDER_QUEUE_NOTE,
                         alias=alias)
# [AGENT EDIT END]


def refresh_files():
    discover_dlc_models.clear()
    _h5_options_cached.clear()
    _project_summary.clear()
    qc_population.clear()
    fit_status_cached.clear()
    _labeled_frame_counts.clear()
    session_status.clear()
    recorded_animals.clear()


def mark_arena_dirty():
    st.session_state.arena_config_dirty = True


def retain_widget_values(only=None):
    """Detach value widgets from cleanup when their section is hidden.

    Button/upload/component event state is deliberately excluded. Durable edit
    values must survive conditional rendering and fragment reruns.
    """
    keys = """routing_alias routing_method use_raw_model dlc_model_select dlc_notes
        dlc_run_user cm_overlay_on cm_overlay_h5 am_overlay_on
        am_overlay_h5 selected_point point_remove_select config_name_input
        fixation_point_select fixation_shift_x fixation_shift_y speaker_orientation
        xaxis1_point_select xaxis2_point_select frame_number_input
        mu_project_select mu_curate_note mu_gen_add_path mu_gen_candidate mu_gen_metric
        mu_gen_nframes mu_iteration mu_overlay_iters mu_release_alias mu_release_note
        mu_release_snapshot mu_snapshot training_mode training_resume
        transfer_learning_release transfer_learning_snapshot hfqc_h5_select hfqc_p_cut
        am_browse_dir am_browse_pick
        labeled_video_select play_mode n_clips clip_duration db_filter_alias
        db_filter_text db_filter_spread db_filter_conf db_filter_broken prov_alias_detail
        new_bodyparts new_copy_videos new_experimenter new_project_name new_videos
        new_working_dir app_section update_section model_view""".split()
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: preserve inspection values across hidden top-level sections without retaining action buttons | date: 2026-09-10]
    keys += """inspect_a inspect_b inspect_unlinked inspect_cut inspect_frame
        inspect_clips inspect_seconds inspect_spread_plot inspect_frame_a inspect_frame_b
        inspect_spread_a inspect_spread_b inspect_video_a inspect_video_b""".split()
    # [AGENT EDIT END]
    keys += [k for k in st.session_state if k.startswith(
        ("dist_", "cam_compare::", "camera_choice::", "camera_note::"))]
    keys += [f"{name}_{suffix}" for name in st.session_state.get("speakers", {})
             for suffix in ("dist", "angle")]
    for key in (keys if only is None else only):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
# [AGENT EDIT END]


def alias_names():
    if dlc_registry is None:
        return []
    try:
        return list(dlc_registry.load_registry()["alias_names"])
    except Exception:                                         # noqa: BLE001
        return []


def resolve_alias(alias):
    """Full registry resolution for an alias, or None.

    Returns what ``dlc_registry.resolve`` returns — ``alias, release_id, config,
    project, snapshot, bodyparts, qc, source, train_frame_size`` — so callers can tell
    a frozen **release** (what production inference runs) from a live training
    **project**.  Falls back to a directory-name match, marked ``source='guess'``, when
    the registry is unavailable.
    """
    if not alias:
        return None
    if dlc_registry is not None:
        try:
            return dict(dlc_registry.resolve(alias))
        except Exception:                                     # noqa: BLE001
            pass
    config = _fallback_config_for_alias(alias)
    if not config:
        return None
    return {"alias": alias, "release_id": None, "config": config,
            "project": str(Path(config).parent), "snapshot": None, "bodyparts": [],
            "qc": None, "source": "guess", "train_frame_size": None}


def _fallback_config_for_alias(alias):
    models = discover_dlc_models()
    if alias in models:
        return models[alias]
    hits = [p for name, p in models.items() if name.split("-", 1)[0] == alias]
    if len(hits) == 1:
        return hits[0]
    hits = [p for name, p in models.items() if name.startswith(alias)]
    return hits[0] if len(hits) == 1 else None


def _coerce_config_path(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("config_path", "config", "config_yaml", "path"):
            got = _coerce_config_path(value.get(key))
            if got:
                return got
        project = value.get("project_path") or value.get("project")
        if project:
            guess = Path(project) / "config.yaml"
            return str(guess) if guess.exists() else None
        return None
    text = str(value)
    if text.endswith("config.yaml"):
        return text if os.path.exists(text) else None
    guess = Path(text) / "config.yaml"
    return str(guess) if guess.exists() else None


def resolve_alias_config(alias, models=None):
    """alias -> config.yaml path (thin wrapper over :func:`resolve_alias`)."""
    resolved = resolve_alias(alias)
    return resolved.get("config") if resolved else None


def active_model():
    """The resolution the page is currently working from, as a dict (never None).

    ``via`` is ``'alias'`` or ``'raw'``; ``source`` is ``'release'``, ``'project'``,
    ``'guess'`` or ``'raw'``.  ``release_id`` is what stamps a fit — a run without one
    produces tracking that nothing on disk attributes to a network.
    """
    return st.session_state.get("active_model") or {
        "alias": None, "release_id": None, "config": "", "project": None,
        "source": "raw", "via": "raw", "snapshot": None, "train_frame_size": None}


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the update loop was reading
# and writing selected_dlc_model_config, which for a released alias is the FROZEN release
# -- read-only, one snapshot, no evaluation-results and no curation log. Every step of the
# loop belongs to the mutable project instead | date: 2026-08-14]
def routed_project_config():
    """``config.yaml`` of the mutable project behind the ACTIVE model (no override)."""
    project = active_model().get("project")
    if project:
        candidate = Path(project) / "config.yaml"
        if candidate.exists():
            return str(candidate)
    return st.session_state.get("selected_dlc_model_config", "")


def project_config_path():
    """``config.yaml`` of the MUTABLE project all update work targets.

    ``selected_dlc_model_config`` is what inference runs from; for a released alias that
    is the release, which is ``r--r--r--`` and carries exactly one snapshot.  Adding
    frames, curating, evaluating and training all belong to the training project the
    release was cut from — ``dlc_registry.resolve`` returns both.

    [AGENT EDIT | agent: claude | user: wingertj | reason: the working project was ONLY
    reachable through the Model tab's routing/raw-override, so updating a project the
    router does not point at (a successor like merged_left_chimney) forced a detour
    through the Advanced picker on a different tab. The Update tab now pins the working
    project directly ('update_project_config'); every project act -- add frames, curate,
    train, evaluate, release -- reads it from here, so they cannot disagree about which
    tree they touch. | date: 2026-08-18]
    """
    override = st.session_state.get("update_project_config")
    if override and os.path.exists(str(override)):
        return str(override)
    return routed_project_config()


def mutable_projects():
    """``{name: config path}`` of TRAINING projects only — the release store excluded.

    Everything the Update tab does (labelling, curation, training, cutting a release)
    is a project act; a frozen release is read-only and holds one snapshot, so offering
    it here would only manufacture confusing failures.
    """
    from nems_lbhb.motor import dlc_attribution
    release_root = os.path.abspath(dlc_attribution.RELEASE_ROOT)
    out = {}
    for name, cfg in discover_dlc_models().items():
        if os.path.abspath(str(cfg)).startswith(release_root + os.sep):
            continue
        out[name] = cfg
    return out
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the Update tab listed
# training projects by directory name only, so nothing said which one a release was cut
# from or what data it was trained on. A release's RELEASE.toml records both -- its
# source project and a frozen copy of the exact label files -- so the tab can name the
# release beside its project and say how the project's labels differ now | date: 2026-09-25]
def releases_by_project(registry_mtime=None):
    """``{project_dir: [(alias, release_id, status), ...]}`` for registry aliases.

    ``status`` is ``'current'`` (what the alias serves), ``'previous'`` (served it
    once) or ``'candidate'`` (cut, never promoted). Archive releases are not registry
    aliases and never appear. Ordered current, candidate, previous.
    """
    return _releases_by_project(registry_mtime if registry_mtime is not None
                                else _registry_mtime())


@st.cache_data(ttl=300, show_spinner=False)
def _releases_by_project(registry_mtime):
    out = {}
    if dlc_registry is None:
        return out
    try:
        from nems_lbhb.motor import dlc_release
        names = dlc_registry.load_registry()["alias_names"]
    except Exception:                                         # noqa: BLE001
        return out
    order = {"current": 0, "candidate": 1, "previous": 2}
    for alias in names:
        current, history, _next = alias_release_info(alias)
        for rid in [r for r in [current] if r] + unpromoted_releases(alias):
            try:
                project = dlc_release.read_manifest(rid).get("source", {}).get("project")
            except Exception:                                 # noqa: BLE001
                continue
            status = ("current" if rid == current else
                      "previous" if rid in history else "candidate")
            if project:
                out.setdefault(os.path.abspath(project), []).append((alias, rid, status))
    for rows in out.values():
        rows.sort(key=lambda r: order[r[2]])
    return out


def label_files(root, subdir):
    """``{(folder, filename): path}`` for every ``<root>/<subdir>/*/CollectedData_*.h5``."""
    found = {}
    for path in glob.glob(os.path.join(str(root), subdir, "*", "CollectedData_*.h5")):
        found[(os.path.basename(os.path.dirname(path)), os.path.basename(path))] = path
    return found


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _label_rows(path):
    try:
        return len(pd.read_hdf(path))
    except Exception:                                         # noqa: BLE001
        return None


def release_label_diff(release_dir, project_dir):
    """How the project's labels today differ from the ones a release was trained on.

    Compared by content, per label file, the same way ``dlc_release`` fingerprints them
    -- so ``unchanged`` is exactly "retraining this project uses the release's data".
    Returns ``{release_frames, project_frames, same, changed, added, removed}``; the
    last four are sorted video (label folder) names.
    """
    frozen = label_files(release_dir, "labels")
    live = label_files(project_dir, "labeled-data")
    same, changed = set(), set()
    for key in set(frozen) & set(live):
        (same if _file_sha(frozen[key]) == _file_sha(live[key]) else changed).add(key[0])
    folders = lambda keys: {k[0] for k in keys}
    added = folders(set(live) - set(frozen)) - same - changed
    removed = folders(set(frozen) - set(live)) - same - changed

    def total(files):
        counts = [_label_rows(p) for p in files.values()]
        return None if any(c is None for c in counts) else sum(counts)

    return {"release_frames": total(frozen), "project_frames": total(live),
            "same": sorted(same - changed), "changed": sorted(changed),
            "added": sorted(added), "removed": sorted(removed)}


def _label_stamp(root, subdir):
    return tuple(sorted((k, file_stamp(p)) for k, p in label_files(root, subdir).items()))


@st.cache_data(ttl=600, max_entries=64, show_spinner=False)
def _release_label_diff_cached(release_dir, project_dir, stamp):
    return release_label_diff(release_dir, project_dir)


def release_training_data(release_id, project_dir):
    """What ``release_id`` was trained on, and how ``project_dir`` differs today.

    ``None`` when the release cannot be read. Cached on the project's label-file stamps,
    so saving a frame in the labeler is seen on the next run.
    """
    try:
        from nems_lbhb.motor import dlc_release
        manifest = dlc_release.read_manifest(release_id)
        release_dir = dlc_release._release_dir(release_id)
    except Exception:                                         # noqa: BLE001
        return None
    source = manifest.get("source", {})
    diff = _release_label_diff_cached(release_dir, str(project_dir),
                                      _label_stamp(project_dir, "labeled-data"))
    return dict(diff, release_id=release_id,
                folders=list(source.get("label_folders") or []),
                frames=source.get("label_frames") or diff.get("release_frames"),
                iteration=source.get("iteration"), snapshot=source.get("snapshot"))
# [AGENT EDIT END]


def production_alias():
    """The alias to pass as ``alias=`` for inference, or None when running raw.

    Returns None when the user has explicitly overridden to a project config: an
    explicit config path implies ``allow_project`` in ``dlc_fit_script.resolve_model``
    and produces an UNSTAMPED fit, which is a deliberate choice, not a default.
    """
    model = active_model()
    if model.get("via") == "alias" and model.get("alias"):
        return model["alias"]
    return None


def load_dlc_config_fields(config_path):
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse parsed configuration on model and frame changes | date: 2026-09-09]
    config = read_yaml(config_path)
    # [AGENT EDIT END]

    st.session_state.dlc_project_name = config.get("Task", "")
    st.session_state.dlc_experimenter = config.get("scorer", "")
    st.session_state.dlc_working_dir = config.get("project_path", "")
    st.session_state.dlc_bodyparts = config.get("bodyparts", [])
    st.session_state.dlc_videos = list(config.get("video_sets", {}).keys())
    st.session_state.dlc_notes = config.get("notes", "")

    project_path = Path(config.get("project_path", ""))
    iterations = sorted([p.name for p in (project_path / "dlc-models").glob("iteration-*")
                        if p.is_dir()])
    current_iter = config.get("iteration", 0)
    guess = f"iteration-{current_iter}"
    st.session_state.selected_iter = guess if guess in iterations else (
        iterations[-1] if iterations else guess)
    return config


def model_iterations(config_path):
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse unchanged YAML without caching widget state | date: 2026-09-09]
    cfg = read_yaml(config_path)
    # [AGENT EDIT END]
    project_path = Path(cfg.get("project_path", ""))
    iterations = sorted([p.name for p in (project_path / "dlc-models").glob("iteration-*")
                        if p.is_dir()],
                        key=lambda n: int(n.split("-")[-1]))
    return cfg, project_path, iterations


def iteration_snapshots(project_path, iteration):
    snapshot_dir = next(iter((Path(project_path) / "dlc-models" / iteration).glob("*/train")), None)
    if not snapshot_dir or not snapshot_dir.exists():
        return []
    return sorted(int(p.stem.split("-")[1]) for p in snapshot_dir.glob("snapshot-*.meta"))


def write_config_values(config_path, **values):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.update(values)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg


def set_snapshotindex(config_path, value):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    old_index = config.get("snapshotindex", "all")
    config["snapshotindex"] = value
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return old_index


def save_notes_callback():
    config_path = project_config_path()                       # never the frozen release
    if not config_path:
        return
    write_config_values(config_path, notes=st.session_state.get("dlc_notes", ""))
    st.success("Notes saved!")


def is_dlc_model_trained(config_path, shuffle=1):
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse parsed YAML while keeping action-time snapshot checks fresh | date: 2026-09-09]
    cfg = read_yaml(config_path)
    # [AGENT EDIT END]
    training_folder = Path(cfg["project_path"]) / "dlc-models"
    if not training_folder.exists():
        return False
    return len(list(training_folder.rglob("snapshot-*.meta"))) > 0


def outlier_frames_exist(config_path, video_path):
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse parsed project configuration | date: 2026-09-09]
    cfg = read_yaml(config_path)
    # [AGENT EDIT END]
    labeled_data_dir = (Path(cfg["project_path"]) / "labeled-data"
                        / Path(video_path).parents[0].name)
    if not labeled_data_dir.exists():
        return False
    frame_files = list(labeled_data_dir.glob("*.png")) + list(labeled_data_dir.glob("*.bmp"))
    return len(frame_files) > 0


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: cache advisory label counts rather than scanning the training tree during model selection | date: 2026-09-09]
def labeled_frame_counts(config_path):
    return _labeled_frame_counts(file_stamp(config_path))


@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
def _labeled_frame_counts(stamp):
    config_path = stamp[0]
# [AGENT EDIT END]
    """(videos, frames) currently in the project's labeled-data tree."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["project_path"]) / "labeled-data"
        folders = [p for p in root.glob("*") if p.is_dir()]
        frames = sum(len(list(p.glob("*.png"))) + len(list(p.glob("*.bmp")))
                     for p in folders)
        return len(folders), frames
    except Exception:                                         # noqa: BLE001
        return 0, 0


# ── frames / figures ───────────────────────────────────────────────────────────
def read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: avoid reopening unchanged videos on every widget interaction | date: 2026-09-09]
@st.cache_data(max_entries=64, show_spinner=False)
def _video_frame_count(stamp):
    cap = cv2.VideoCapture(stamp[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def video_frame_count(video_path):
    return _video_frame_count(file_stamp(video_path))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: dlc_fit_script honours the
# per-session camera override but the GUI did not, so every pixel-facing tool (frame
# slider, DLC overlay, arena marker placement) silently used a DIFFERENT video than the
# one DLC analysed. REI084a10_a_NFB: recording.avi is the face camera (168101 frames @
# 100fps), the ceiling view is video_side2.avi (50429 @ 30fps), and the .dlc.h5 has 50429
# rows. | date: 2026-08-17]
def ceiling_video(video_path):
    """``(path, override)`` -- the file that actually holds the top-down view.

    DERIVED, never stored.  ``video_path`` stays the session's identity: it names the
    arena config (:func:`default_config_name` builds the filename from ``vp.stem``), the
    temp directory, and the argument handed to the queue -- where ``dlc_fit_script`` does
    its own registry lookup, so the redirect must be decided in exactly one place rather
    than baked into what the GUI passes along.

    ``override`` is the registry block when one applies, else ``None``.  A missing target
    returns the original path with the override attached, so the caller can say the
    registry and the disk disagree instead of silently reading the wrong camera.
    """
    if not video_path:
        return "", None
    try:
        over = dlc_registry.camera_override(str(video_path))
    except Exception:                                         # noqa: BLE001
        return str(video_path), None                          # registry unreadable: no redirect
    if not over:
        return str(video_path), None

    vp = Path(video_path)
    sess = vp.parent if vp.name.startswith(("recording", "video_")) else vp.with_suffix("")
    target = sess / over["video"]
    if not target.exists():
        return str(video_path), dict(over, missing=str(target))
    return str(target), over


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame number box moved
# out of the session strip into the tab that shows the frame; the camera-override notice
# comes with it, as a tag | date: 2026-09-25]
def render_frame_picker(video_path):
    """The Frame box for ``video_path``, then the decoded frame (or None).

    Reads through :func:`ceiling_video`, so an overridden session shows the camera DLC
    analysed -- the reason the override notice lives beside the box.
    """
    if not (video_path and os.path.exists(video_path)):
        st.number_input("Frame", min_value=0, value=0, disabled=True,
                        key="frame_number_disabled")
        return None
    frame_video, over = ceiling_video(video_path)
    total = video_frame_count(frame_video)
    last = max(total - 1, 0)
    st.session_state["frame_number_input"] = min(
        int(st.session_state.get("frame_number_input", 0) or 0), last)
    st.number_input("Frame", min_value=0, max_value=last, step=1,
                    key="frame_number_input",
                    help=f"{total} frames in {os.path.basename(frame_video)}")
    if over and over.get("missing"):
        st.caption(":orange[⚠ override camera missing]",
                   help=f"Registry says this session's ceiling view is `{over['video']}`, "
                        f"but it is not on disk ({over['missing']}). Showing "
                        f"`{os.path.basename(video_path)}` -- NOT what a fit would analyse.")
    elif over:
        st.caption(f"camera · {over['video']}",
                   help=f"Camera override: frames, overlay and markers come from "
                        f"`{over['video']}` (cam_tag `{over.get('cam_tag')}`), not "
                        f"`{os.path.basename(video_path)}`.")
    frame = ensure_current_frame(video_path, st.session_state.get("frame_number_input", 0))
    if frame is None:
        notes_caption([f"Could not read frame {st.session_state.get('frame_number_input', 0)}"
                       f" from {os.path.basename(frame_video)}."])
    return frame
# [AGENT EDIT END]


def ensure_current_frame(video_path, frame_number):
    """Read the frame only when the (video, index) it should show has changed.

    Streamlit reruns the whole script on every widget touch, so an unconditional read
    would re-decode on each keystroke.  A seek+read costs 12-125 ms on these files, which
    is fine once per actual change and wasteful several times a second.
    """
    resolved, _over = ceiling_video(video_path)
    if not resolved or not os.path.exists(resolved):
        return None
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: invalidate the decoded frame when a video is replaced at the same path | date: 2026-09-09]
    key = (file_stamp(resolved), int(frame_number))
    # [AGENT EDIT END]
    if (st.session_state.get("current_frame") is not None
            and st.session_state.get("current_frame_key") == key):
        return st.session_state.current_frame
    frame = read_frame(resolved, frame_number)
    st.session_state.current_frame = frame
    st.session_state.current_frame_key = key if frame is not None else None
    return frame
# [AGENT EDIT END]


def get_cmap_color(cmap_name="viridis"):
    names = list(st.session_state.points.keys())
    if not names:
        st.session_state.cdict = {}
        return
    crange = np.linspace(0, 1, len(names))
    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:                                         # noqa: BLE001
        cmap = plt.cm.viridis
    cdict = {name: cmap(crange[ni]) for ni, name in enumerate(names)}
    st.session_state.cdict = {
        n: f"rgb({int(v[0] * 255)}, {int(v[1] * 255)}, {int(v[2] * 255)})"
        for n, v in cdict.items()}


def compute_fixation_point_from_state():
    """Fixation point from current state, without UI side-effects."""
    if not st.session_state.points:
        return None
    fix_key = st.session_state.get("fixation_point_select", None)
    if fix_key not in st.session_state.points:
        return None
    try:
        shift_x_m = float(st.session_state.get("fixation_shift_x", "0.0"))
        shift_y_m = float(st.session_state.get("fixation_shift_y", "0.0"))
    except ValueError:
        return None

    base_x, base_y = st.session_state.points[fix_key]
    ppm = st.session_state.ppm_avg
    fx, fy = base_x, base_y

    xaxis_points = st.session_state.get("xaxis_points", None)
    if xaxis_points and all(p in st.session_state.points for p in xaxis_points):
        x1 = np.array(st.session_state.points[xaxis_points[0]])
        x2 = np.array(st.session_state.points[xaxis_points[1]])
        delta = x2 - x1
        if np.hypot(delta[0], delta[1]) > 0:
            theta = np.arctan2(delta[1], delta[0])
            cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            base_rot = (np.array([base_x, base_y]) - x1) @ R.T
            base_rot = base_rot + np.array([shift_x_m, shift_y_m]) * ppm
            R_inv = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
            fx, fy = (base_rot @ R_inv.T) + x1
    else:
        fx = base_x + shift_x_m * ppm
        fy = base_y + shift_y_m * ppm
    return float(fx), float(fy)


def create_plotly_figure(image):
    """Frame + annotation markers, sized to the image."""
    try:
        pil_img = image
        if hasattr(pil_img, "convert"):
            pil_img = pil_img.convert("RGB")
        img_array = np.asarray(pil_img)
    except Exception:                                         # noqa: BLE001
        img_array = np.array(image)

    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    fig = go.Figure(go.Image(z=img_array, colormodel="rgb", hoverinfo="none"))

    cdict = st.session_state.get("cdict", None)
    if not cdict and st.session_state.points:
        get_cmap_color()
        cdict = st.session_state.get("cdict", {})
    cdict = cdict or {}

    for name, (x, y) in st.session_state.points.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=12, color=cdict.get(name, "rgb(255,255,255)"),
                        line=dict(width=2, color="white")),
            text=[name], textposition="top center",
            textfont=dict(color="white"), showlegend=False))

    fig.update_layout(
        width=image.width, height=image.height,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest", hoverdistance=5,
        xaxis=dict(visible=False, range=[0, image.width], fixedrange=True,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[image.height, 0], fixedrange=True))
    return fig


def add_fixation_trace(fig):
    if st.session_state.get("fixation_point") is not None:
        fx, fy = st.session_state.fixation_point
        fig.add_trace(go.Scatter(
            x=[fx], y=[fy], mode="markers",
            marker=dict(color="red", size=15, symbol="x"),
            name="Fixation Point", showlegend=False))
    return fig


def add_dlc_overlay(fig, h5_path, frame_number):
    """Draw one frame of a DLC prediction file onto an existing figure."""
    try:
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: draw a single frame from reusable prediction data | date: 2026-09-09]
        keypoints = prediction_frame(h5_path, frame_number)
        scorer = keypoints.index.get_level_values(0)[0]
        names_bodyparts = list(keypoints[scorer].index.get_level_values(0).unique())
        # [AGENT EDIT END]
        for bp in names_bodyparts:
            fig.add_trace(go.Scatter(
                x=[keypoints[scorer][bp]["x"]], y=[keypoints[scorer][bp]["y"]],
                mode="markers+text", marker=dict(color="cyan", size=10),
                text=[bp], name=f"DLC: {bp}", showlegend=False))
    except Exception as e:                                    # noqa: BLE001
        st.error(f"Failed to load DLC predictions: {e}")
    return fig


def img_to_base64(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ── arena config save / import ─────────────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: TOML has no null, so ONE
# None anywhere in the config aborts the whole dump with "Object of type <class
# 'NoneType'> is not TOML serializable" -- and the config carries several optional
# fields that are None in the ordinary case (camera_override on any session WITHOUT an
# override, fixation_base_marker and xaxis_pair before they are chosen, frame_* with no
# frame loaded). That broke .toml saving for normal sessions; .yaml/.json were fine
# because both can represent null. A missing key and an explicit null read the same here
# -- every consumer uses .get(key, default) -- so absent is the honest TOML spelling of
# "not set". | date: 2026-08-26]
def _toml_drop_none(obj):
    """Recursively drop None values so a config with unset options is TOML-writable."""
    if isinstance(obj, dict):
        return {k: _toml_drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_toml_drop_none(v) for v in obj if v is not None]
    return obj
# [AGENT EDIT END]


def save_config(config_name, video_path):
    """Save the arena configuration plus a figure snapshot."""
    if not config_name:
        say("error", "Please enter a configuration name")
        return

    config_path = Path(config_name)
    plot_path = config_path.with_name(
        config_path.stem.replace("dlc_arena_config", "dlc_arena_plot") + ".png")

    def _query_year_from_celldb(penname):
        if not penname:
            say("error", "Celldb query skipped: missing penname.")
            return None
        sql = (" select gDataRaw.*,gPenetration.pendate from gDataRaw "
               "INNER JOIN gCellMaster on gDataRaw.masterid = gCellMaster.id "
               "INNER JOIN gPenetration on gPenetration.id = gCellMaster.penid "
               f"where parmfile like '{penname}' order by pendate desc")
        try:
            pend = db.pd_query(sql)
        except Exception as e:                                # noqa: BLE001
            say("error", f"Celldb query failed for penname {penname}: {e}")
            return None
        if pend is None or pend.empty:
            say("error", f"Celldb query returned no records for penname {penname}.")
            return None
        pendate = pend.pendate.iloc[0]
        try:
            return str(int(pendate.year))
        except Exception:                                     # noqa: BLE001
            try:
                return str(int(pd.to_datetime(pendate).year))
            except Exception:                                 # noqa: BLE001
                say("error", f"Failed to parse pendate for penname {penname}: {pendate}")
                return None

    def _build_web_arena_jpg_path(video_path_value, config_path_value=None):
        vp_local = Path(video_path_value)
        runid = vp_local.parent.name if vp_local.parent and vp_local.parent.name else vp_local.stem
        animal_id = None
        if config_path_value:
            cp_local = Path(config_path_value)
            if len(cp_local.parents) > 2 and cp_local.parents[0].name == "sorted":
                animal_id = cp_local.parents[2].name
        if not animal_id and len(vp_local.parents) > 1 and vp_local.parents[1].name:
            animal_id = vp_local.parents[1].name
        if not animal_id and runid:
            m = re.match(r"([A-Za-z]+\d+)", runid)
            if m:
                animal_id = m.group(1)
        year = _query_year_from_celldb(runid)
        if not (animal_id and year and runid):
            return None
        return (Path("/auto/data/web/analysis") / animal_id.lower() / year
                / f"{runid}.arena_config.jpg")

    def _infer_dlc_model_from_sorted(video_path_value):
        vp_local = Path(video_path_value)
        runid = vp_local.parent.name if vp_local.parent and vp_local.parent.name else vp_local.stem
        h5_path = vp_local.parents[1] / "sorted" / f"{runid}.dlc.h5"
        if not h5_path.exists():
            return None, None
        try:
            target = h5_path.resolve()
        except Exception:                                     # noqa: BLE001
            target = h5_path
        target_stem = target.stem
        model_part = (target_stem.split("_DLC_", 1)[1] if "_DLC_" in target_stem
                      else target_stem)
        model_part = re.sub(r"(?:_)?shuffle\d+_\d+$", "", model_part)
        config_guess = Path("/auto/data/dlc") / model_part / "config.yaml"
        return model_part, (str(config_guess) if config_guess.exists() else None)

    frame = st.session_state.get("current_frame", None)
    if frame is not None:
        fig = add_fixation_trace(create_plotly_figure(frame))
        try:
            fig.write_image(str(plot_path), format="png",
                            width=frame.width, height=frame.height)
            say("success", f"Saved plot image to: {plot_path}")
        except Exception as e:                                # noqa: BLE001
            say("error", f"Could not save plot image: {e}")
        try:
            if "sorted" in config_path.parts:
                web_jpg_path = _build_web_arena_jpg_path(video_path, config_path)
                if web_jpg_path is None:
                    say("error", "Could not determine web JPG path from celldb. "
                             "Skipping web JPG save.")
                else:
                    web_jpg_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.write_image(str(web_jpg_path), format="jpg",
                                    width=frame.width, height=frame.height)
                    say("success", f"Saved web JPG to: {web_jpg_path}")
        except Exception as e:                                # noqa: BLE001
            say("warning", f"Could not save web JPG: {e}")
    else:
        say("warning", "No frame loaded — cannot save plot image.")

    dlc_model_name = st.session_state.get("dlc_model_select", "")
    dlc_model_config = st.session_state.get("selected_dlc_model_config", "")
    inferred_model_name, inferred_config_path = _infer_dlc_model_from_sorted(video_path)
    if inferred_model_name:
        dlc_model_name = inferred_model_name
        if inferred_config_path:
            dlc_model_config = inferred_config_path

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: markers are clicked on
    # the CEILING view, which is not always the typed path. Recording only the typed path
    # made an overridden session's config indistinguishable from one drawn on the wrong
    # camera. video_path keeps its meaning (session identity) for existing readers.
    # | date: 2026-08-17]
    _frame_video, _cam_over = ceiling_video(st.session_state.get("video_path", ""))
    config = {
        "metadata": {
            "video_path": st.session_state.get("video_path", None),
            "frame_video_path": _frame_video or None,
            "camera_override": (_cam_over or {}).get("video"),
            # [AGENT EDIT END]
            "frame_number": st.session_state.get("frame_number_input", None),
            "frame_width": getattr(frame, "width", None),
            "frame_height": getattr(frame, "height", None),
        },
        "points": st.session_state.points,
        "distances": st.session_state.distances,
        "ppm_pairs": st.session_state.get("pixels_per_meter", {}),
        "average_ppm": {"average_ppm": st.session_state.ppm_avg},
        "fixation_base_marker": st.session_state.get("fixation_point_select", None),
        "fixation_shift": {
            "x_shift_m": st.session_state.get("fixation_shift_x", "0.0"),
            "y_shift_m": st.session_state.get("fixation_shift_y", "0.0"),
        },
        "fixation_point": {"fixation_point": st.session_state.fixation_point},
        "xaxis_pair": {"xaxis_pair": st.session_state.get("xaxis_points", None)},
        "speaker_orientation": st.session_state.speaker_orientation,
        "speakers": st.session_state.speakers,
        "arena_plot_path": str(plot_path),
        "dlc_model": {"name": dlc_model_name, "config_path": dlc_model_config},
    }

    name = str(config_name)
    if name.endswith(".toml"):
        with open(name, "wb") as f:
            tomli_w.dump(_toml_drop_none(config), f)
    elif name.endswith(".yaml"):
        with open(name, "w") as f:
            yaml.dump(config, f)
    elif name.endswith(".json"):
        with open(name, "w") as f:
            json.dump(config, f, indent=2)
    else:
        say("error", "Unsupported format. Use .toml, .yaml, or .json")
        return

    st.session_state.arena_config_dirty = False
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: a successful save establishes the arena-config version now on disk | date: 2026-09-09]
    if name.endswith(".toml"):
        st.session_state.auto_config_path = name
        st.session_state.auto_config_name = Path(name).name
        st.session_state.auto_config_stamp = file_stamp(name)
    # [AGENT EDIT END]
    say("success", f"Configuration saved as {name}")


# [AGENT EDIT START | agent: claude | user: wingertj | reason: import messages rendered
# wherever the import ran -- for the auto-load that was the top of every tab. They are
# kept as notes for the Arena markers tab to show as tags | date: 2026-09-25]
def _arena_note(text):
    st.session_state.setdefault("arena_config_notes", []).append(str(text))
# [AGENT EDIT END]


def import_config(config_file):
    """Import a previously saved TOML arena config and apply it."""
    st.session_state.arena_config_notes = []
    try:
        config = tomli.load(config_file)

        if "metadata" in config:
            config_video = config["metadata"].get("video_path", None)
            frame_number = config["metadata"].get("frame_number", 0)
            st.session_state.imported_config_video = config_video
            if config_video and os.path.exists(config_video):
                current_video = st.session_state.get("video_path", None)
                if current_video and str(Path(current_video)) == str(Path(config_video)):
                    try:
                        st.session_state.frame_number_input = int(frame_number)
                    except (TypeError, ValueError):
                        st.session_state.frame_number_input = 0
                    # through ceiling_video, so an overridden session shows the camera
                    # DLC analysed rather than flashing the typed one for one rerun
                    ensure_current_frame(config_video,
                                         st.session_state.frame_number_input)
                else:
                    _arena_note(f"Config video path differs from current: {config_video}")
            elif config_video:
                _arena_note(f"Config video path not found: {config_video}")

        if "points" in config:
            st.session_state.points = {k: tuple(v) for k, v in config["points"].items()}
            get_cmap_color()
            if st.session_state.points:
                first_key = next(iter(st.session_state.points))
                if st.session_state.get("selected_point") not in st.session_state.points:
                    st.session_state.selected_point = first_key
                if st.session_state.get("point_remove_select") not in st.session_state.points:
                    st.session_state.point_remove_select = first_key

        if "distances" in config:
            st.session_state.distances = config["distances"]
        if "ppm_pairs" in config:
            st.session_state.pixels_per_meter = config["ppm_pairs"]
        if "average_ppm" in config and "average_ppm" in config["average_ppm"]:
            st.session_state.ppm_avg = config["average_ppm"]["average_ppm"]

        if "fixation_base_marker" in config:
            fix_key = config["fixation_base_marker"]
            if not st.session_state.points or fix_key in st.session_state.points:
                st.session_state.fixation_point_select = fix_key
        if "fixation_shift" in config:
            st.session_state.fixation_shift_x = str(config["fixation_shift"].get("x_shift_m", "0.0"))
            st.session_state.fixation_shift_y = str(config["fixation_shift"].get("y_shift_m", "0.0"))
        if "fixation_point" in config and "fixation_point" in config["fixation_point"]:
            st.session_state.fixation_point = tuple(config["fixation_point"]["fixation_point"])

        if "xaxis_pair" in config and "xaxis_pair" in config["xaxis_pair"]:
            xaxis_points = tuple(config["xaxis_pair"]["xaxis_pair"] or ())
            st.session_state.xaxis_points = xaxis_points
            if len(xaxis_points) == 2 and all(p in st.session_state.points for p in xaxis_points):
                st.session_state.xaxis1_point_select = xaxis_points[0]
                st.session_state.xaxis2_point_select = xaxis_points[1]

        if "speakers" in config:
            st.session_state.speakers = config["speakers"]
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: retained geometry widgets must adopt the values explicitly reloaded from disk | date: 2026-09-09]
        for pair, value in st.session_state.get("distances", {}).items():
            st.session_state[f"dist_{pair}"] = float(value)
        for speaker, values in st.session_state.get("speakers", {}).items():
            st.session_state[f"{speaker}_dist"] = float(values["distance"])
            st.session_state[f"{speaker}_angle"] = float(values["angle"])
        st.session_state.pop("geometry_edit_fingerprint", None)
        # [AGENT EDIT END]
        if "speaker_orientation" in config:
            st.session_state.speaker_orientation = config["speaker_orientation"]
        if "dlc_model" in config:
            st.session_state.imported_dlc_model = config["dlc_model"].get("name", "")

        current_frame = st.session_state.get("current_frame", None)
        if current_frame is not None and "metadata" in config:
            saved_w = config["metadata"].get("frame_width", None)
            saved_h = config["metadata"].get("frame_height", None)
            if saved_w and saved_h:
                scale_x = current_frame.width / float(saved_w)
                scale_y = current_frame.height / float(saved_h)
                if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
                    st.session_state.points = {
                        k: (float(v[0]) * scale_x, float(v[1]) * scale_y)
                        for k, v in st.session_state.points.items()}
                    if st.session_state.fixation_point is not None:
                        fx, fy = st.session_state.fixation_point
                        st.session_state.fixation_point = (float(fx) * scale_x,
                                                           float(fy) * scale_y)
                    if abs(scale_x - scale_y) <= 1e-3:
                        if "ppm_pairs" in config:
                            st.session_state.pixels_per_meter = {
                                k: float(v) * scale_x for k, v in config["ppm_pairs"].items()}
                        if "average_ppm" in config and "average_ppm" in config["average_ppm"]:
                            st.session_state.ppm_avg = (
                                float(config["average_ppm"]["average_ppm"]) * scale_x)
                        _arena_note(f"Scaled imported points/fixation and ppm by "
                                   f"x={scale_x:.4f}, y={scale_y:.4f} to match current frame.")
                    else:
                        _arena_note(f"Scaled imported points/fixation by x={scale_x:.4f}, "
                                   f"y={scale_y:.4f}. PPM not scaled (aspect ratio changed).")
            else:
                _arena_note("Imported config is missing saved frame size; points may "
                           "not align to current frame.")

        current_video = st.session_state.get("video_path", None)
        if current_video:
            st.session_state.auto_config_loaded_for = str(Path(current_video))
        st.session_state.plotly_key_version = st.session_state.get("plotly_key_version", 0) + 1
        st.session_state.arena_config_dirty = False
        # success is the loaded config's name in the Arena markers tab, not a banner
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: do not record a failed import as a successful refresh | date: 2026-09-09]
        return True
        # [AGENT EDIT END]
    except Exception as e:                                    # noqa: BLE001
        _arena_note(f"Failed to load configuration: {e}")
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: report import failure to the reload caller | date: 2026-09-09]
        return False
        # [AGENT EDIT END]


def import_config_path(config_path):
    try:
        with open(config_path, "rb") as f:
            # [AGENT EDIT START | agent: Codex | user: wingertj | reason: propagate parse/validation failures | date: 2026-09-09]
            return import_config(f)
            # [AGENT EDIT END]
    except Exception as e:                                    # noqa: BLE001
        say("error", f"Failed to load configuration from {config_path}: {e}")
        return False


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the browser's own file
# dialog cannot reach /auto -- every browser on this host is a snap and neither AppArmor
# profile grants /auto/**, only @{HOME}/[^s.]** and the portal proxy dir. The Streamlit
# process is unconfined, so a listing it produces is the only route to the share
# | date: 2026-09-16]
def browse_dir(folder, suffix=".toml"):
    """``(resolved_folder, {label: Path})`` for a server-side file picker.

    Directories are labelled with a trailing ``/`` so one glance says whether choosing an
    entry navigates or imports -- the two are not undoable in the same way.  ``..`` comes
    first, and is omitted at the filesystem root, where it would be a dead entry that
    looks live.  Hidden entries are skipped; a dotted path can still be reached by typing
    it.

    Raises ``OSError`` when the folder cannot be listed. That is deliberate: an empty
    listing and an unreachable path are different answers, and the second is what a typo
    produces, so a caller must be able to say so rather than show "nothing here".

    Note for ``/auto``: it is an autofs indirect mount, so listing it shows only the keys
    currently mounted. Reaching ``/auto/users`` by typing or by a jump button triggers the
    automount; browsing down from ``/`` may not find it.
    """
    resolved = Path(folder).expanduser().resolve()
    entries = {}
    if resolved.parent != resolved:
        entries[".."] = resolved.parent
    dirs, files = [], []
    for child in resolved.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            dirs.append(child)
        elif not suffix or child.name.endswith(suffix):
            files.append(child)
    for child in sorted(dirs, key=lambda c: c.name.lower()):
        entries[child.name + "/"] = child
    for child in sorted(files, key=lambda c: c.name.lower()):
        entries[child.name] = child
    return resolved, entries
# [AGENT EDIT END]


def default_config_name(video_path):
    vp = Path(video_path)
    out_path = vp.parents[1] / "sorted"
    return str(out_path / f"{vp.stem}-{vp.parent.stem}-dlc_arena_config.toml")


def load_sorted_config_for_session(video_path):
    vp = Path(video_path)
    session_name = vp.parents[0].name
    sorted_path = vp.parents[1] / "sorted"
    if not sorted_path.exists():
        return None
    expected_name = f"{vp.stem}-{vp.parent.stem}-dlc_arena_config.toml"
    exact_match = sorted_path / expected_name
    if exact_match.exists():
        latest_config = exact_match
    else:
        candidates = list(sorted_path.glob(f"{vp.stem}*dlc_arena_config.toml"))
        if not candidates:
            candidates = list(sorted_path.glob(f"{session_name}*dlc_arena_config.toml"))
        if not candidates:
            return None

        def score(p):
            common = os.path.commonprefix([p.name, expected_name])
            return (len(common), p.stat().st_mtime)
        latest_config = max(candidates, key=score)
    if import_config_path(latest_config):
        st.session_state.auto_config_loaded_for = str(vp)
        st.session_state.auto_config_name = latest_config.name
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: track the version of the loaded arena config | date: 2026-09-09]
        st.session_state.auto_config_path = str(latest_config)
        st.session_state.auto_config_stamp = file_stamp(latest_config)
        st.session_state.pop("geometry_edit_fingerprint", None)
        # [AGENT EDIT END]
        return latest_config
    return None


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the session finder makes
# switching sessions the normal way to drive the app, so "point the page somewhere else"
# needs one definition; typing a path only ever reset the decoded frame, which was enough
# when switching was rare | date: 2026-09-15]
_ARENA_STATE_KEYS = ("points", "selected_point", "fixation_point", "distances",
                     "last_imported_config")
_AUTO_CONFIG_KEYS = ("auto_config_loaded_for", "auto_config_name", "auto_config_path",
                     "auto_config_stamp")

# Widget keys belonging to a tab that CANNOT be rendering when a session is opened (the
# finder is), so they are safe to drop here -- unlike the session strip's, which are live.
# `config_name_input` is retained across reruns and Streamlit prefers an existing
# session-state value over a widget's `value=`, so leaving it would keep the Save box
# pointing at the previous session's config file.
_STALE_TARGET_KEYS = ("config_name_input", "am_browse_dir", "am_browse_pick")


def adopt_pending_session(state=None):
    """Point the session strip's widgets at ``video_path``.  Call BEFORE building them.

    The other half of :func:`open_session`, which cannot write these keys itself. Also
    does the first-run seeding the strip used to do inline, so there is one place that
    decides what the path box starts out holding.
    """
    state = st.session_state if state is None else state
    if state.pop("pending_strip_sync", False):
        state["video_path_input_field"] = state.get("video_path", "")
        state["frame_number_input"] = 0
    else:
        state.setdefault("video_path_input_field", state.get("video_path", ""))


def open_session(video_path):
    """Retarget the whole page at ``video_path``.  False if that would lose edits.

    Refuses while the arena config is dirty, for the reason the "Reload arena config"
    button is disabled then: those annotations exist only in session state, and nothing
    would be left of them afterwards.

    The arena keys are RESET rather than left for the next config load to overwrite.
    The sessions this gets used on are mostly the ones with no config to load -- that is
    what puts them in the finder -- so markers carried over from the previous video would
    be drawn on this one, and saved against it.
    """
    if st.session_state.get("arena_config_dirty"):
        st.session_state.open_session_error = (
            "unsaved arena config · save or discard before switching")
        return False
    st.session_state.video_path = str(video_path)
    st.session_state.video_path_error = ""
    st.session_state.current_frame = None
    st.session_state.current_frame_key = None
    # the path box and the frame box belong to the session strip, which is ALWAYS
    # visible and therefore already instantiated by the time a tab body runs; Streamlit
    # refuses a write to a live widget's key. Ask instead, and let the strip adopt it on
    # the next run -- which the caller triggers with st.rerun(). Every other key reset
    # below is either plain state or belongs to a tab that is not the one calling this.
    st.session_state.pending_strip_sync = True
    for key in _ARENA_STATE_KEYS:
        # deepcopy: INITIAL_STATE holds mutable defaults, and handing out the same dict
        # would make the next marker click edit the template
        st.session_state[key] = copy.deepcopy(INITIAL_STATE.get(key))
    for key in _AUTO_CONFIG_KEYS + _STALE_TARGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("geometry_edit_fingerprint", None)
    st.session_state.pop("open_session_error", None)
    # retire the click component's key so a pending click on the old frame cannot land
    st.session_state.plotly_key_version = st.session_state.get("plotly_key_version", 0) + 1
    return True
# [AGENT EDIT END]


# ── job queueing ───────────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: explicit config reload must preserve unsaved edits | date: 2026-09-09]
def reload_arena_config():
    if not st.session_state.get("arena_config_dirty"):
        load_sorted_config_for_session(st.session_state.get("video_path", ""))


def arena_config_changed():
    path = st.session_state.get("auto_config_path")
    if not path:
        return False
    try:
        return file_stamp(path) != st.session_state.get("auto_config_stamp")
    except OSError:
        return True
# [AGENT EDIT END]


def _job_note(videopath, configpath, action, add_on, alias=None, release_id=None):
    video_name = f"{Path(videopath).parents[0].name}/{Path(videopath).name}"
    if alias:
        model_part = f"alias_{alias}-release_{release_id or 'project'}"
    else:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the note named the
        # config FILE ('config.yaml' -- identical for every DLC project), and
        # add_job_to_queue matches existing rows BY NOTE, so queueing the same action for
        # a second model hijacked the first model's queue row and log. The parent
        # directory is the project/release name, which is the model's actual identity.
        # | date: 2026-08-18]
        model_name = Path(configpath).parent.name if configpath else "registry"
        try:
            with open(configpath, "r") as f:
                cfg = yaml.safe_load(f)
            model_part = (f"config_{model_name}-iter_{cfg['iteration']}"
                          f"-snapshotindex_{cfg['snapshotindex']}")
        except Exception:                                     # noqa: BLE001
            model_part = f"config_{model_name}"
        # [AGENT EDIT END]
    return f"DLC Job:{model_part}-video_{video_name}-action_{action}{add_on}"


def queue_dlc_job(videopath, configpath=None, destpath=None, action="fit", extra_note="",
                  init_weights=None, from_release=None, alias=None, allow_project=False,
                  resume=False, note=None, user=None, src=None, fallback=False,
                  extra_args=None):
    """Queue a DLC job through the standard wrapper script.

    ``dlc_fit_script`` resolves the model by precedence: an explicit config path in
    argv[3] wins, then ``alias=<name>``, then the registry rules for the video.  The
    PRODUCTION path for inference is ``alias=`` with an EMPTY config, because that is
    what resolves to a frozen release, verifies it against its own SHA256SUMS and
    stamps the output ``<stem>.<release_id>.dlc.h5``.  An explicit config path implies
    ``allow_project`` and produces an unstamped fit — useful for trying a model, not
    for anything that has to stay attributable.

    Pass ``alias=`` for inference (fit/video) and ``configpath=`` for project-side work
    (training, curate, extract, refine), which by definition runs in the mutable project.
    """
    try:
        release_id = None
        if alias and not configpath:
            resolved = resolve_alias(alias) or {}
            release_id = resolved.get("release_id")

        # argv is positional up to dest, so the config slot must be present-but-empty
        # when the model comes from an alias
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: db.add_job_to_queue
        # joins args with plain spaces into one shell string, so a bare "" placeholder
        # collapses and every later positional argument shifts left by one -- the
        # destfolder arrived in argv[3], where dlc_fit_script expects the config path,
        # and the job analysed "into" its own destination. Quote it so the shell passes a
        # real empty argument | date: 2026-08-17]
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: a relative
        # videopath ("videos/<session>.avi") is relative to the PROJECT, but the queue
        # daemon runs jobs from an arbitrary cwd -- 7 extract jobs died in
        # session_named_video's existence check. Absolutize the argv against the config's
        # project dir; _job_note below still receives the short form, so job notes (the
        # dedupe key) are unchanged | date: 2026-08-19]
        argv_video = str(videopath)
        if configpath and not os.path.isabs(argv_video):
            candidate = os.path.join(os.path.dirname(str(configpath)), argv_video)
            if os.path.exists(candidate):
                argv_video = candidate
        args = [argv_video, action, str(configpath) if configpath else "NOCONFIG"]
        # [AGENT EDIT END]
        if destpath is not None:
            args.append(str(destpath))
        # [AGENT EDIT END]

        if alias:
            args.append(f"alias={alias}")
        if allow_project:
            args.append("allow_project=1")
        if init_weights:
            args.append(f"init_weights={init_weights}")
        if from_release:
            args.append(f"from_release={from_release}")
        if resume:
            args.append("resume=1")
        if src:
            args.append(f"src={src}")
        if fallback:
            args.append("fallback=1")
        for k, v in (extra_args or {}).items():               # action-specific key=value
            args.append(f"{k}={v}")

        if user is None:
            user = current_dlc_user()
        from nems_lbhb.motor import dlc_train_guard
        slug = dlc_train_guard._slug(user)
        args.append(f"user={slug}")
        if note:
            # free text recorded with the curation entry; keep it argv-safe, since
            # parse_key_value_args splits on '='
            args.append(f"note={str(note).replace('=', '-')}")

        is_training = "train" in action
        force_rerun = not is_training
        add_on = "" if is_training else ("_" + extra_note if extra_note else "")

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: GPU_job=1 was
        # hardcoded for EVERY action -- deliberate for publish (its fallback can become a
        # full fit) but a silent tax on the rest: frame extraction is k-means on
        # downsampled frames, pure CPU, and each such job held a GPU slot doing no GPU
        # work. A job needs the GPU pool only when some action in the list can run the
        # network: fit/training/geneval always, publish because of fallback=1.
        # | date: 2026-08-18]
        _GPU_ACTIONS = {"fit", "training", "geneval", "publish", "evaluate"}
        gpu_job = 1 if any(a.strip() in _GPU_ACTIONS
                           for a in str(action).split(",")) else 0
        # [AGENT EDIT END]
        job_note = _job_note(videopath, configpath, action, add_on, alias, release_id)
        result = db.add_job_to_queue(
            args, job_note, force_rerun=force_rerun, GPU_job=gpu_job,
            linux_user=queue_linux_user(),
            user=user, executable_path=wrapper_path(), script_path=script_path(),
            priority=1)

        result_list = list(result)
        if result_list:
            queueids, messages = zip(*result_list)
            return f"Job queued (ID: {queueids[0]}): {messages[0].strip()}"
        return f"Job submitted: {job_note}"
    except Exception as e:                                    # noqa: BLE001
        return f"Error launching job: {e}"


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the deployed checkout is
# from 2025-12-18 and contains none of dlc_registry/dlc_release/dlc_alias/dlc_train_guard/
# dlc_fit_record, so every queued job ran pre-lifecycle code while the GUI described a
# system that was not deployed | date: 2026-08-17]
DEV_CODE_ROOT = "/auto/users/wingertj/nems_db"
DEPLOYED_CODE_ROOT = "/auto/users/lbhb/Code/nems_db"

# Flip to False once DEPLOYED_CODE_ROOT is updated; nothing else has to change.
USE_DEV_CODE = True

# Account the queue runs jobs as. Production is `lbhb`: uniform service-account
# ownership is what actually makes release directories immutable, since `chmod a-w` is
# only advisory on NFS. In dev mode the jobs run as the developer instead, so output is
# owned by whoever is iterating and does not need lbhb to clean up. Anything that writes
# into /auto/data/dlc/releases must go back to `lbhb` before it is trusted as immutable.
# Stays `lbhb` even in dev mode. Tried `wingertj` on 2026-08-17: the job was marked dead
# (complete=2) with NO job log at all, i.e. the daemon never launched it -- it can only
# switch to accounts it is configured for, and historically only nems/davids/svd/lbhb
# appear. `lbhb` has real successes, so the account is fine; the recent DLC failures under
# it were the stale-code bug (job 2649791: the Dec-2025 script has no key=value parsing,
# so it read `alias=left_chimney` as the destination directory), which the dev wrapper
# fixes. Keeping lbhb also preserves the service-account ownership that release
# immutability depends on.
DEV_QUEUE_USER = "lbhb"
PRODUCTION_QUEUE_USER = "lbhb"

# No machine pin. The GUI passes no `require_machine`, so the queue master places every
# job: it is the thing that knows which workers exist, are healthy, and have a free GPU
# slot, and a list living here can only go stale in both directions. The previous pool
# ("hyena", "rhino", "manatee", picked at random) was a workaround for capybara eating
# unpinned jobs on 2026-08-18; by 2026-09-16 capybara had completed 336 of 346 unpinned
# DLC jobs, while the pin was still excluding agouti -- the healthiest node in the pool,
# 321 of 322. If a node goes bad again, take it out of the queue-master rotation rather
# than teaching this file about hostnames.


def queue_linux_user():
    return DEV_QUEUE_USER if USE_DEV_CODE else PRODUCTION_QUEUE_USER


def code_root():
    """Checkout that queued jobs execute."""
    return DEV_CODE_ROOT if USE_DEV_CODE else DEPLOYED_CODE_ROOT


def wrapper_path():
    """Shell wrapper the queue invokes.

    The dev wrapper additionally prepends ``code_root()`` to PYTHONPATH: pointing only
    the SCRIPT at the live tree is not enough, because ``import nems_lbhb`` otherwise
    resolves through dlc5's easy-install.pth to the deployed checkout, and the job would
    run new entry-point code against old library code.
    """
    if USE_DEV_CODE:
        return os.path.join(DEV_CODE_ROOT, "nems_lbhb/motor/dlc_fit_wrapper_dev")
    return os.path.join(DEPLOYED_CODE_ROOT, "nems_lbhb/motor/dlc_fit_wrapper")


def script_path():
    return os.path.join(code_root(), "nems_lbhb/motor/dlc_fit_script.py")


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: avoid spawning git on every interaction | date: 2026-09-09]
@st.cache_data(ttl=60, show_spinner=False)
def code_banner():
# [AGENT EDIT END]
    """One line for the GUI header: which code the queue will run."""
    if not USE_DEV_CODE:
        return None
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "-C", DEV_CODE_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                         # noqa: BLE001
        sha = "unknown"
    return f"{DEV_CODE_ROOT} @ {sha} · as {queue_linux_user()}"
# [AGENT EDIT END]


def session_temp_dir(video_path):
    vp = Path(video_path)
    return vp.parents[1] / f"tmp/dlc/{vp.parents[0].name}"


def _explicit_config(run_from_project, alias):
    """The config to pass a job, or ``None`` to let the alias resolve its release.

    ``run_from_project`` means the MUTABLE project -- passing the release config here
    made the checkbox a no-op that only dropped the alias, producing an unstamped fit
    from the very release it claimed to bypass.
    """
    if run_from_project:
        return project_config_path()
    if alias:
        return None                                           # production: alias resolves it
    return st.session_state.get("selected_dlc_model_config", "")


def analyze_to_temp(run_from_project=False):
    """Queue a fit into this session's tmp/dlc folder.

    Production route: ``alias=`` and no config, so the job runs a verified release and
    stamps its output.  ``run_from_project`` forces the mutable project config instead
    (unstamped, for trying a model out).
    """
    video_path = Path(st.session_state.get("video_path", ""))
    temp_path = str(session_temp_dir(video_path))
    alias = None if run_from_project else production_alias()
    config_path = _explicit_config(run_from_project, alias)

    if not str(video_path) or not os.path.exists(video_path):
        say("error", "Invalid or missing video path.")
        return
    if not alias and (not config_path or not os.path.exists(config_path)):
        say("error", "No alias and no valid DLC config path.")
        return
    message = queue_dlc_job(video_path, config_path, temp_path, "fit",
                            extra_note="temp_analysis", alias=alias)
    say("error" if str(message).startswith("Error") else "success",
        f"DLC job queued → {temp_path}: {message}")
    if not alias:
        st.caption("No provenance (explicit config implies allow_project) — the "
                   "output carries no release id.")


# [AGENT EDIT START | agent: claude | user: wingertj | reason: "analyze to sorted" after
# vetting a preview re-ran the GPU to recompute bytes already sitting in tmp/dlc
# | date: 2026-08-14]
def publishable_preview(video_path):
    """The vetted preview that could be published instead of re-run, or ``None``.

    Only a preview stamped with the release the alias resolves to *right now* counts --
    an unstamped or older-release preview is refused by the job for the same reason it
    is not offered here.
    """
    if not video_path:
        return None
    model = active_model()
    release_id = model.get("release_id")
    if not release_id or model.get("via") != "alias":
        return None
    try:
        from nems_lbhb.motor import dlc_fit_script as fs
        stem = Path(video_path).parents[0].name
        return fs.publishable_preview(str(session_temp_dir(video_path)), stem, release_id)
    except Exception:                                         # noqa: BLE001
        return None


# [AGENT EDIT END]


def analyze_to_sorted(run_from_project=False):
    """Get this session into ``sorted/``, by the cheapest route that is correct.

    One action, two implementations. When a preview of the *currently deployed* release
    is already sitting in temp, the bytes a fresh fit would compute are on disk already,
    so the job copies them instead of running inference. Otherwise it analyzes.

    The job is queued with ``fallback=1``: this check runs now, the job runs later, and
    scratch can be cleaned in between. Rather than fail, the job then does the full fit.
    """
    video_path = st.session_state.get("video_path", "")
    alias = None if run_from_project else production_alias()
    config_path = _explicit_config(run_from_project, alias)

    if not video_path or not os.path.exists(video_path):
        say("error", "Invalid or missing video path.")
        return
    if not alias and (not config_path or not os.path.exists(config_path)):
        say("error", "No alias and no valid DLC config path.")
        return
    if not alias:
        say("warning", "Writing to sorted/ from an explicit project config: the result "
                   "carries no release id, so gDLCFit cannot attribute it to a release.")

    preview = None if run_from_project else publishable_preview(video_path)
    if preview:
        message = queue_dlc_job(video_path, config_path, None, "publish",
                                extra_note="publish_temp", alias=alias,
                                src=str(session_temp_dir(video_path)), fallback=True)
        say("error" if str(message).startswith("Error") else "success",
            f"Copying the vetted preview (no inference): {message}")
        return

    message = queue_dlc_job(video_path, config_path, None, "fit",
                            extra_note="analyze_to_sorted", alias=alias)
    say("error" if str(message).startswith("Error") else "success",
        f"DLC job queued: {message}")


def video_to_temp(run_from_project=False):
    video_path = Path(st.session_state.get("video_path", ""))
    temp_path = session_temp_dir(video_path)
    alias = None if run_from_project else production_alias()
    config_path = _explicit_config(run_from_project, alias)

    if not str(video_path) or not os.path.exists(video_path):
        say("error", "Invalid or missing video path.")
        return
    if not alias and (not config_path or not os.path.exists(config_path)):
        say("error", "No alias and no valid DLC config path.")
        return
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: match session/camera and selected release rather than the raw recording filename | date: 2026-09-10]
    try:
        from nems_lbhb.motor import dlc_fit_script as fs
        resolved = (resolve_alias(alias) or {}) if alias else {}
        release_id = resolved.get("release_id")
        model_config = config_path or resolved.get("config")
        if not model_config:
            raise ValueError("Could not resolve the selected model's config.")
        actual_video, override = ceiling_video(video_path)
        if override and override.get("missing"):
            raise FileNotFoundError(f"Camera override video is missing: {override['missing']}")
        suffix = (fs.release_output_suffix(release_id) if release_id
                  else fs.dlc_alias.dlc_scorer_suffix(model_config))
        prediction = fs.labeled_video_prediction(temp_path, actual_video, suffix, release_id)
    except (OSError, ValueError, RuntimeError) as exc:
        say("warning", str(exc))
        return
    st.caption(f"Using prediction: {prediction}")
    # [AGENT EDIT END]
    message = queue_dlc_job(video_path, config_path, str(temp_path), "video",
                            extra_note="temp_labeled_video", alias=alias)
    say("error" if str(message).startswith("Error") else "success",
        f"DLC job queued: {message}")


def curate_command(note=""):
    """The shell call that accepts the labels — shown, never queued.

    ``curate`` merges the labelled folders into the training set and appends what
    changed to ``curation_history.log``; ``training`` refuses to start while the labels
    on disk are newer than the last curate.
    """
    video_path = st.session_state.get("video_path", "") or "<video.avi>"
    config_path = project_config_path() or "<config.yaml>"    # curation is a project act
    parts = ["python", script_path(), str(video_path), "curate", str(config_path),
             f"user={current_dlc_user()}"]
    if note:
        parts.append(f'note="{str(note).replace(chr(34), "")}"')
    return " ".join(parts)


def add_video_to_dataset():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = project_config_path()                       # add_new_videos writes here
    if not str(video_path) or not config_path:
        raise RuntimeError("Missing video or config path.")
    if not st.session_state.get("dlc_videos"):
        raise RuntimeError("Model config lists no videos; cannot infer the dataset video folder.")

    model_video_base = Path(st.session_state.dlc_videos[0]).parent
    new_name = str(video_path.parents[0].name) + "_" + str(video_path.name)
    new_video = str(model_video_base / new_name)

    if os.path.exists(new_video):
        say("info", f"Video already exists at {new_video}, skipping copy")
    else:
        st.write(f"Copying video to {new_video}...")
        shutil.copy(video_path, new_video)

    try:
        deeplabcut.add_new_videos(config=config_path, videos=[new_video], copy_videos=False)
        if is_dlc_model_trained(config_path):
            message = queue_dlc_job(new_video, config_path,
                                    str(Path(new_video).parents[0]), "fit,refine")
            say("error" if str(message).startswith("Error") else "success",
        f"DLC job queued: {message}")
            return "Video added; analysis + outlier extraction queued."
        message = queue_dlc_job(new_video, config_path,
                                str(Path(new_video).parents[0]), "extract")
        say("error" if str(message).startswith("Error") else "success",
        f"DLC job queued: {message}")
        return "Video added and frames queued for extraction (model not trained yet)."
    except Exception as e:                                    # noqa: BLE001
        raise RuntimeError(f"Failed to add video: {e}")


def labeling_command(config_path):
    """The exact call to paste into a DLC shell, and which one it is."""
    if is_dlc_model_trained(config_path):
        return "refine", (f"import deeplabcut\n"
                          f"deeplabcut.refine_labels('{config_path}')")
    return "label", (f"import deeplabcut\n"
                     f"deeplabcut.label_frames('{config_path}')")


def train_dlc_model():
    """Queue the 'training' action against the project (never a release).

    ``training`` refuses to start when the labels on disk are newer than the last
    ``curate``, so step 2 has to have been accepted first.  Transfer learning takes
    either a snapshot path (``init_weights``) or a frozen release (``from_release``,
    which is resolved from the release and checked against the configured net_type).
    """
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = project_config_path()                       # training is never a release
    if not config_path:
        raise RuntimeError("Missing DLC config path.")  # [AGENT EDIT | agent: claude | user: wingertj | reason: failures raise so action_button shows them red, not a green sentence | date: 2026-09-28]
    try:
        init_weights = None
        from_release = None
        mode = st.session_state.get("training_mode", "Train from scratch")
        if mode == "Transfer learning from snapshot":
            selected_snapshot = st.session_state.get("transfer_learning_snapshot", None)
            if selected_snapshot:
                init_weights = selected_snapshot.replace(".meta", "")
        elif mode == "Transfer learning from release":
            from_release = st.session_state.get("transfer_learning_release", None)

        message = queue_dlc_job(video_path, config_path, "training", "training",
                                init_weights=init_weights, from_release=from_release,
                                resume=bool(st.session_state.get("training_resume")),
                                user=current_dlc_user())
        if str(message).startswith("Error"):
            raise RuntimeError(message)
        say("info", message)
        if from_release:
            return f"Training queued, initialised from release {from_release}."
        if init_weights:
            return f"Training queued with transfer learning from {Path(init_weights).name}."
        return "Training queued from scratch."
    except Exception as e:                                    # noqa: BLE001
        raise RuntimeError(f"Training failed: {e}") from e


# ── releases ───────────────────────────────────────────────────────────────────
def alias_release_info(alias):
    """``(current_release, history, next_id)`` for an alias, best effort."""
    if not alias or dlc_registry is None:
        return None, [], None
    try:
        block = dlc_registry.load_registry()["aliases"].get(alias) or {}
    except Exception:                                         # noqa: BLE001
        return None, [], None
    current = block.get("current") or None
    history = list(block.get("history") or [])
    next_id = None
    try:
        from nems_lbhb.motor import dlc_release
        next_id = dlc_release.next_release_id(alias)
    except Exception:                                         # noqa: BLE001
        pass
    return current, history, next_id


def queue_generalization_eval(alias, candidate_config, n_frames=None):
    """Queue the sampled generalization evaluation as a GPU job.

    The unit of work is the ALIAS's whole set, so argv[1] carries the set's videos
    directory rather than one session -- ``geneval`` never reads it, but the slot is
    positional.  The candidate goes in the config slot, so a release candidate is
    checksum-verified by ``resolve_model`` before a single frame is scored.
    """
    if dlc_generalization is None:
        raise RuntimeError("dlc_generalization unavailable.")
    if n_frames is None:
        n_frames = dlc_generalization.DEFAULT_SAMPLE_FRAMES
    videos_dir = dlc_generalization.videos_dir(alias)
    return queue_dlc_job(videos_dir, configpath=candidate_config, action="geneval",
                         user=current_dlc_user(),
                         extra_args={"gen_alias": alias, "n_frames": int(n_frames)})


def unpromoted_releases(alias):
    """Release ids cut for ``alias`` but not currently serving it, newest first.

    These are the CANDIDATES: built, immutable, testable from the Model tab's Advanced
    picker, and not yet what the alias runs.  Read from the release store rather than
    the registry history, which only learns about a release at promotion.
    """
    if not alias:
        return []
    try:
        from nems_lbhb.motor import dlc_release
        numbers = dlc_release._release_numbers(alias)
    except Exception:                                         # noqa: BLE001
        return []
    current, _history, _next = alias_release_info(alias)
    return [rid for _n, rid in sorted(numbers, reverse=True) if rid != current]


def build_release_dry_run(project, alias, snapshot, iteration=None, note=""):
    """Dry-run a release build: reports what would be frozen, writes nothing."""
    from nems_lbhb.motor import dlc_release
    return dlc_release.build_release(project, alias, snapshot, iteration=iteration,
                                     note=note, dry_run=True)


def release_commands(project, alias, snapshot, note=""):
    """The build / verify / promote / rollback calls, as copy-paste python.

    Not buttons: promotion edits the git-tracked registry TOML, and the provenance
    panels in this app are read-only for the same reason — what production runs should
    not change from a page whose main job is labelling.
    """
    return (
        "from nems_lbhb.motor import dlc_release\n"
        f"rid = dlc_release.build_release({project!r}, {alias!r}, {snapshot!r},"
        f" note={note!r})\n"
        "dlc_release.verify(rid, full=True)\n"
        f"print(dlc_release.promote({alias!r}, rid))      # edits dlc_registry.toml\n"
        f"# dlc_release.rollback({alias!r})               # if it has to go back")


# ── routing suggestion ─────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: the routing suggestion only
# ever applied to the first video -- 'routing_alias' is a widget key, so once it existed
# Streamlit ignored the index= computed from the suggestion and silently kept the previous
# session's alias | date: 2026-08-14]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: the key was (video, method)
# only, so the suggestion outlived a PROMOTION: after left_chimney_headphones moved to
# r0004 the banner still read r0003 while "Model to use" -- which resolves through
# load_registry, itself keyed on the file's mtime -- correctly said r0004. A suggestion
# names a release, so the registry's mtime belongs in the key. | date: 2026-08-26]
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: session-local routing refresh preserves last successful DB evidence | date: 2026-09-09]
ROUTING_REFRESH_SECONDS = 30


def refresh_routing():
    st.session_state.pop("routing_checked_at", None)


def select_routing_manually():
    st.session_state.routing_manual = True


def follow_routing():
    st.session_state.routing_manual = False
    refresh_routing()
# [AGENT EDIT END]


def _registry_mtime():
    """Cache-busting stamp: changes on every promote/rollback. 0.0 when unreadable."""
    from nems_lbhb.motor import dlc_registry
    try:
        return os.path.getmtime(dlc_registry.DEFAULT_REGISTRY_PATH)
    except OSError:
        return 0.0
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: bound database staleness without sharing old observations with new sessions | date: 2026-09-09]
def routing_suggestion(video_path, method="geometry"):
    """Fresh on session entry, then at most one DB read per 30 seconds/session.

    A registry edit also invalidates the result. Failures retain previous evidence
    for this context and are displayed explicitly; failed observations never route.
    """
    if not video_path:
        return None
    context = (str(video_path), method, _registry_mtime())
    now = time.time()
    previous = st.session_state.get("routing_result")
    same_context = st.session_state.get("routing_result_context") == context
    checked = st.session_state.get("routing_checked_at", 0)
    if same_context and previous is not None and now - checked < ROUTING_REFRESH_SECONDS:
        return previous
    try:
        from nems_lbhb.motor import dlc_routing
        result = dlc_routing.suggest(str(video_path), method=method, db=_RoutingDB())
        errors = (result.get("observed") or {}).get("db_errors", [])
        if errors:
            raise RuntimeError("; ".join(errors))
        result["checked_at"] = now
        result["refresh_error"] = None
    except Exception as exc:
        result = dict(previous) if same_context and previous else {
            "method": method, "alias": None, "candidates": [], "observed": {},
            "checked_at": None}
        result["refresh_error"] = str(exc)
    st.session_state.routing_result = result
    st.session_state.routing_result_context = context
    st.session_state.routing_checked_at = now
    return result
# [AGENT EDIT END]


def routing_method():
    """Which router the panel's radio is set to (geometry until touched)."""
    return st.session_state.get("routing_method", "geometry")


def sync_routing_for_video(video_path):
    """Apply fresh routing evidence before constructing the model picker.

    Writing the *widget key* before the selectbox is created is what makes the
    suggestion take effect on the same run: Streamlit prefers a value already in
    session_state over the widget's ``index=``.  That same precedence is why this has
    to update automatic selections when metadata changes. Explicit manual choices
    survive refreshes for the same video/method.
    """
    if not video_path:
        return None
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: apply refreshed suggestions even when the video has not changed | date: 2026-09-09]
    stamp = (str(video_path), routing_method())
    result = routing_suggestion(video_path, stamp[1])
    return apply_routing(st.session_state, stamp, result)
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: poll only the visible model page and propagate changes through model resolution | date: 2026-09-09]
@st.fragment(run_every=30)
def routing_status():
    before = st.session_state.get("routing_result")
    result = routing_suggestion(st.session_state.get("video_path", ""), routing_method())
    # A new check time alone only redraws this small status fragment. Propagate
    # changed evidence/selection through the full model-resolution path.
    revision = lambda value: {k: v for k, v in (value or {}).items() if k != "checked_at"}
    if revision(result) != revision(before):
        st.rerun()
    if not result:
        return
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: "Routing checked" + a
    # full-width Refresh button + Use suggestion took three rows; now one ↻ on the heading
    # row with the check time in its hover. "Use suggestion" sits beside the picker's
    # "overrides" tag, the only time it applies | date: 2026-09-25]
    checked = result.get("checked_at")
    when = time.strftime("%H:%M:%S", time.localtime(checked)) if checked else "—"
    if st.button("↻ refresh", key="routing_refresh", on_click=refresh_routing,
                 help="Re-checks on its own every 30 s."):
        st.rerun()
    st.caption(f"checked {when}")
    if result.get("refresh_error"):
        st.caption(":orange[⚠ refresh failed]",
                   help="Routing refresh failed; keeping the last successful selection. "
                        + result["refresh_error"])
    # [AGENT EDIT END]
# [AGENT EDIT END]


# ── update-model step state ────────────────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: numbered headings alone do
# not say where you are in the loop; each step reports its own state so a refusal further
# down (training vs an uncurated label set) is visible before you click | date: 2026-08-14]
def _curation_state(project):
    """``(state, detail)`` for the curate step, without parsing a guard traceback."""
    try:
        from nems_lbhb.motor import dlc_train_guard as guard
    except Exception:                                         # noqa: BLE001
        return "na", "guard unavailable"
    try:
        files = guard.label_files(project)
        if not files:
            return "na", "no labels in this project yet"
        entries = guard.read_curation_log(project)
        if not entries:
            return "warn", "never curated — training will refuse"
        reference = entries[-1].get("labels_newest_mtime")
        if reference is None:
            return "warn", "last log entry has no mtime reference — curate again"
        newer = [p for p in files if os.path.getmtime(p) > float(reference)]
        if newer:
            return "warn", f"{len(newer)} label file(s) newer than the last curate"
        last = entries[-1]
        return "ok", f"curated {str(last.get('utc', '?'))[:16]} by {last.get('user', '?')}"
    except Exception as e:                                    # noqa: BLE001
        return "na", str(e)[:120]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: make project summary inputs explicit for caching | date: 2026-09-09]
def update_steps(video_path, config_path=None, model=None):
# [AGENT EDIT END]
    """State of each step in the update loop, as ``[{n, title, state, detail}]``.

    ``state`` is ``ok`` (done), ``todo`` (next action), ``warn`` (will be refused),
    ``blocked`` (an earlier step gates it) or ``na`` (not applicable / unreadable).

    Reads the PROJECT, not the release: labelled frames, evaluation results and the
    curation log only exist in the mutable tree.
    """
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: summaries must describe the pinned project rather than a different routed project | date: 2026-09-09]
    model = active_model() if model is None else model
    config_path = config_path or project_config_path()
    project = str(Path(config_path).parent) if config_path else ""
    # [AGENT EDIT END]
    trained = False
    try:
        trained = is_dlc_model_trained(config_path)
    except Exception:                                         # noqa: BLE001
        pass

    if not video_path:
        frames = ("na", "no video selected")
    else:
        try:
            here = outlier_frames_exist(config_path, video_path)
        except Exception:                                     # noqa: BLE001
            here = False
        session = Path(video_path).parents[0].name
        frames = (("ok", f"{session} is in labeled-data")
                  if here else ("todo", f"{session} not in labeled-data yet"))

    curate = _curation_state(project) if project else ("na", "no project")

    try:
        n_eval = len(all_eval_files(config_path) or {})
    except Exception:                                         # noqa: BLE001
        n_eval = 0
    evaluate = (("ok", f"{n_eval} iteration(s) evaluated") if n_eval else
                ("todo", "no evaluation results yet") if trained else
                ("na", "nothing trained to evaluate"))

    if curate[0] == "warn":
        retrain = ("blocked", "curate first — training refuses on uncurated labels")
    else:
        holder = None
        try:
            from nems_lbhb.motor import dlc_train_guard as guard
            holder = guard.lock_holder(project) if project else None
        except Exception:                                     # noqa: BLE001
            holder = None
        retrain = (("warn", f"training lock held: {holder}") if holder else
                   ("todo", "ready to queue"))

    alias = model.get("alias")
    if not alias:
        release = ("na", "no alias — a release belongs to a registry alias")
    else:
        current, _history, next_id = alias_release_info(alias)
        release = ("todo", f"{alias}: current {current or '(none)'} → next {next_id or '?'}")

    # [AGENT EDIT | agent: claude | user: wingertj | reason: evaluation moved to rung 1 --
    # the loop STARTS by measuring where the model stands (labelled-frame error side by
    # side with the generalization set), which is what decides whether frames are worth
    # adding at all. It is also where you return after training to pick the snapshot.
    # | date: 2026-08-18]
    return [
        {"n": 1, "title": "Evaluate", "state": evaluate[0], "detail": evaluate[1]},
        {"n": 2, "title": "Add frames", "state": frames[0], "detail": frames[1]},
        {"n": 3, "title": "Curate labels", "state": curate[0], "detail": curate[1]},
        {"n": 4, "title": "Retrain", "state": retrain[0], "detail": retrain[1]},
        {"n": 5, "title": "Release", "state": release[0], "detail": release[1]},
    ]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse advisory project diagnostics while keeping action-time guards uncached | date: 2026-09-09]
@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
def _project_summary(video_path, stamp, alias, registry_mtime):
    return update_steps(video_path, stamp[0], model={"alias": alias})


def project_summary(video_path, config_path):
    return _project_summary(video_path, file_stamp(config_path),
                            active_model().get("alias"), _registry_mtime())
# [AGENT EDIT END]


def scale_sidecars(video_path):
    """``[(path, dict)]`` of ``*.scale.json`` sidecars for this session.

    A rescaled fit is deliberately indistinguishable from a native one — coordinates
    are converted back to native pixels — so the sidecar is the only record that the
    video was re-encoded before inference.  Worth surfacing where someone is reading
    those coordinates.
    """
    if not video_path:
        return []
    vp = Path(video_path)
    if len(vp.parents) < 2:
        return []
    session = vp.parents[0].name
    out = []
    for folder in (session_temp_dir(vp), vp.parents[1] / "sorted"):
        if not folder.exists():
            continue
        for p in sorted(set(folder.glob("*.scale.json"))
                        | set(folder.glob("*/*.scale.json"))):
            if folder.name == "sorted" and session not in p.name:
                continue
            try:
                with open(p, "r") as f:
                    out.append((p, json.load(f)))
            except Exception:                                 # noqa: BLE001
                continue
    return out


def evaluate_all_snapshots(config_path):
    original_index = set_snapshotindex(config_path, "all")
    deeplabcut.evaluate_network(config_path, plotting=False)
    set_snapshotindex(config_path, original_index)


def get_eval_file(config_path, iteration=None):
    cfg = yaml.safe_load(open(config_path))
    proj_path = Path(cfg["project_path"])
    iteration = iteration or f"iteration-{cfg['iteration']}"
    results_file = (proj_path / "evaluation-results" / iteration
                    / "CombinedEvaluation-results.csv")
    return str(results_file) if results_file.exists() else None


def all_eval_files(config_path):
    """Every iteration's combined evaluation CSV, oldest first."""
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: avoid repeated YAML parsing for evaluation browsing | date: 2026-09-09]
    cfg = read_yaml(config_path)
    # [AGENT EDIT END]
    root = Path(cfg["project_path"]) / "evaluation-results"
    out = {}
    if root.exists():
        for d in sorted(root.glob("iteration-*"),
                        key=lambda p: int(p.name.split("-")[-1])):
            f = d / "CombinedEvaluation-results.csv"
            if f.exists():
                out[d.name] = str(f)
    return out


def snapshot_error_figure(eval_files):
    """Test error vs training iterations, one trace per DLC iteration."""
    fig = go.Figure()
    for iteration, path in eval_files.items():
        try:
            df = pd.read_csv(path)
        except Exception:                                     # noqa: BLE001
            continue
        col = ("Test error with p-cutoff" if "Test error with p-cutoff" in df.columns
               else "Test error(px)")
        if col not in df.columns or "Training iterations:" not in df.columns:
            continue
        series = df.groupby("Training iterations:")[col].mean()
        fig.add_trace(go.Scatter(x=list(series.index), y=list(series.values),
                                 mode="lines+markers", name=iteration))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="training iterations",
                      yaxis_title="mean test error (px)",
                      legend=dict(font=dict(size=10)))
    return fig


# ── temp .h5 / labeled video listings ──────────────────────────────────────────
def list_dlc_h5_files(folder_path, match_substr=None):
    """``.h5`` files in a folder AND one level down.

    A release-based fit analyzes into ``<dest>/<release_id>/`` — each release gets its
    own output directory, because a scorer string does not identify a release — so a
    flat glob of the temp folder misses every stamped result.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    files = sorted(set(folder_path.glob("*.h5")) | set(folder_path.glob("*/*.h5")))
    if match_substr:
        files = [p for p in files if match_substr in p.name]
    return files


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the pickers listed
# ``p.name`` and rebuilt the path as ``folder / name``, which silently dropped the
# ``<release_id>/`` component that list_dlc_h5_files goes out of its way to find -- so
# choosing a release-stamped file raised "does not exist". The basename is also AMBIGUOUS:
# two releases of one alias emit the same scorer filename, so r0001 and r0002 were
# indistinguishable in the list. | date: 2026-08-17]
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: share prediction discovery across frame and marker views | date: 2026-09-09]
def h5_options(folders, match_substr=None):
    if isinstance(folders, (str, Path)):
        folders = {'': folders}
    entries = tuple((origin, str(value[0]), value[1]) if isinstance(value, tuple)
                    else (origin, str(value), match_substr)
                    for origin, value in folders.items() if value)
    return _h5_options_cached(entries)


@st.cache_data(ttl=15, max_entries=64, show_spinner=False)
def _h5_options_cached(entries):
    folders = {origin: (path, match) for origin, path, match in entries}
    return _h5_options_uncached(folders)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: retain the existing discovery implementation behind the bounded cache | date: 2026-09-09]
def _h5_options_uncached(folders, match_substr=None):
# [AGENT EDIT END]
    """``{label: Path}`` for a prediction-file picker, DEDUPED by real file.

    ``folders`` maps an origin name to a directory, e.g.
    ``{'temp': <tmp/dlc/session>, 'sorted': <sorted>}``; a bare path is treated as
    ``{'': path}``.

    Deduping is the point.  The aliases beside an analysis are SYMLINKS to it, so one
    release fit contributes three listings of one file -- ``<stem>.dlc.h5``,
    ``<stem>.<release>.dlc.h5`` and the target itself.  Offering all three implies a
    choice that does not exist: picking a different one redraws identical markers, which
    looks like a broken control rather than three names for one thing.

    Labels say where the file is and which release made it, because that is the
    distinction a user is actually choosing between.
    """
    if isinstance(folders, (str, Path)):
        folders = {'': folders}

    by_real, options = {}, {}
    for origin, folder in folders.items():
        # a value may carry its own filter: tmp/dlc/<session>/ is already session-scoped,
        # but sorted/ holds every session in the penetration and must be narrowed
        own_filter = match_substr
        if isinstance(folder, tuple):
            folder, own_filter = folder
        if not folder:
            continue
        folder = Path(folder)
        for p in list_dlc_h5_files(folder, match_substr=own_filter):
            real = Path(os.path.realpath(p))
            if real in by_real:
                continue                                       # an alias for one already listed
            by_real[real] = (origin, folder, p)

    for real, (origin, folder, found) in by_real.items():
        # a release fit analyses into <dest>/<release_id>/, so the parent directory names
        # the release; anything flat predates release stamping
        parent = real.parent.name
        release = parent if Path(folder).name != parent else "no release"
        label = " · ".join(x for x in (origin, release) if x)
        if label in options:                                   # two fits, same release
            label = f"{label} · {real.stem[:28]}"
        options[label] = real
    return dict(sorted(options.items()))


def session_h5_folders(video_path):
    """``{'temp': ..., 'sorted': ...}`` for a session, for :func:`h5_options`."""
    if not video_path:
        return {}
    vp = Path(video_path)
    if len(vp.parents) < 2:
        return {}
    return {"temp": session_temp_dir(vp),
            "sorted": (vp.parents[1] / "sorted", vp.parents[0].name)}


def h5_pick(options, state_key):
    """The ``Path`` for the picker at ``state_key``, dropping a stale stored label.

    Session state outlives the directory it described -- a preview gets cleaned, a new
    release appears -- and a stored value absent from the options makes the selectbox
    raise rather than fall back.
    """
    if st.session_state.get(state_key) not in options:
        st.session_state.pop(state_key, None)
    chosen = st.session_state.get(state_key)
    return options.get(chosen) if chosen else None
# [AGENT EDIT END]


def get_dlc_highlight_tokens(config_path):
    if not config_path or not os.path.exists(config_path):
        return None, None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:                                         # noqa: BLE001
        return None, None

    model_name = Path(config_path).parent.name
    highlight_base = model_name.split("-", 1)[0] if model_name else None
    shuffle = cfg.get("shuffle", None)
    iter_num = cfg.get("iteration", None)
    snapshot_idx = cfg.get("snapshotindex", None)
    snapshot_val = None

    try:
        iter_num = int(iter_num)
    except Exception:                                         # noqa: BLE001
        iter_num = None

    if iter_num is not None:
        snapshot_indices = iteration_snapshots(cfg.get("project_path", ""),
                                               f"iteration-{iter_num}")
        if snapshot_indices:
            if not isinstance(snapshot_idx, int):
                snapshot_idx = len(snapshot_indices) - 1
            snapshot_idx = max(0, min(snapshot_idx, len(snapshot_indices) - 1))
            snapshot_val = snapshot_indices[snapshot_idx]

    tokens = []
    if shuffle is not None:
        tokens.append(f"shuffle{shuffle}")
    if snapshot_val is not None:
        tokens.append(f"_{snapshot_val}")
    return highlight_base, (tokens or None)


def render_h5_list(label, folder_path, match_substr=None, highlight_substr=None,
                   highlight_tokens=None):
    st.caption(f"{label}: {folder_path}")
    h5_files = list_dlc_h5_files(folder_path, match_substr=match_substr)
    if not h5_files:
        st.write("No .h5 files found.")
        return
    lines = []
    highlight_key = highlight_substr.lower() if highlight_substr else None
    highlight_tokens_l = [t.lower() for t in highlight_tokens] if highlight_tokens else []
    for p in h5_files:
        name_lower = p.name.lower()
        token_match = (all(t in name_lower for t in highlight_tokens_l)
                       if highlight_tokens_l else True)
        if highlight_key and highlight_key in name_lower and token_match:
            lines.append(f"- <span style='color:#2c455d; font-weight:600;'>{p.name}</span> "
                         f"<span style='color:#5980a6'>← current model</span>")
        else:
            lines.append(f"- {p.name}")
    st.markdown("<div style='white-space: pre-wrap; word-break: break-word; "
                f"font-size:12px'>{chr(10).join(lines)}</div>", unsafe_allow_html=True)


def get_available_labeled_videos():
    video_path = Path(st.session_state.get("video_path", ""))
    if not str(video_path) or not video_path.exists():
        return []
    temp_path = session_temp_dir(video_path)
    if not temp_path.exists():
        return []
    labeled_videos = sorted(set(temp_path.glob("*_labeled.mp4"))
                            | set(temp_path.glob("*/*_labeled.mp4")))
    labeled_videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return labeled_videos


def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
    except Exception:                                         # noqa: BLE001
        pass
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:                                         # noqa: BLE001
        pass
    return None


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: private writable montage cache keyed by source revision and clip settings | date: 2026-09-10]
@st.cache_resource(show_spinner=False)
def _montage_cache_dir():
    return Path(tempfile.mkdtemp(prefix="dlc-labeler-montages-"))


def montage_cache_path(video_path, n_clips=5, clip_duration=1.0, output_suffix="_montage"):
    identity = (file_stamp(video_path), int(n_clips), float(clip_duration), output_suffix)
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    return _montage_cache_dir() / f"{digest}.mp4"
# [AGENT EDIT END]


def extract_video_montage(video_path, n_clips=5, clip_duration=1.0,
                          output_suffix="_montage"):
    """Evenly-spaced clips concatenated into one short montage."""
    video_path = Path(video_path)
    if not video_path.exists():
        return None, None, f"Video path does not exist: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, f"Could not open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None, None, f"Invalid video properties: fps={fps}, frames={frame_count}"

    total_duration = frame_count / fps
    if total_duration < n_clips * clip_duration:
        cap.release()
        return None, None, (f"Video too short: {total_duration:.1f}s < "
                            f"{n_clips * clip_duration:.1f}s needed")

    usable_duration = total_duration - clip_duration
    start_times = [usable_duration * i / (n_clips - 1) if n_clips > 1 else 0
                   for i in range(n_clips)]
    frames_per_clip = int(clip_duration * fps)

    all_frames = []
    try:
        for start_time in start_times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
            for _ in range(frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not all_frames:
        return None, None, "No frames were extracted"

    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: encode atomically in writable cache and report real errors instead of blaming codecs | date: 2026-09-10]
    scratch_path = None
    try:
        import imageio
        output_path = montage_cache_path(video_path, n_clips, clip_duration, output_suffix)
        fd, scratch_name = tempfile.mkstemp(dir=output_path.parent, suffix=".mp4")
        os.close(fd)
        scratch_path = Path(scratch_name)
        # H.264/yuv420p is browser-playable; an XVID AVI fallback is not.
        with imageio.get_writer(str(scratch_path), format="FFMPEG", fps=fps,
                                codec="libx264", pixelformat="yuv420p", quality=7,
                                output_params=["-movflags", "+faststart"]) as writer:
            for frame in all_frames:
                writer.append_data(frame)
        check = cv2.VideoCapture(str(scratch_path))
        try:
            valid, _ = check.read()
        finally:
            check.release()
        if not valid:
            raise RuntimeError("H.264 encoder produced no decodable frames; check FFmpeg output in the server log")
        os.replace(scratch_path, output_path)
        return output_path, start_times, None
    except Exception as exc:                                  # noqa: BLE001
        return None, None, f"Montage creation failed ({type(exc).__name__}): {exc}"
    finally:
        if scratch_path is not None and scratch_path.exists():
            scratch_path.unlink()
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: the visible viewer owns polling, so hidden nested fragments cannot keep refreshing | date: 2026-09-09]
# Polling is owned by tabs.model._viewer.
# [AGENT EDIT END]
def render_video_preview_section():
    """Labeled-video picker + montage generator, refreshed every 60 s."""
    labeled_videos = get_available_labeled_videos()
    if not labeled_videos:
        st.info("No labeled videos in temp. Queue 'Labeled video → temp' first.")
        return

    video_names = [v.name for v in labeled_videos]
    selected_idx = st.selectbox("Labeled video", range(len(video_names)),
                               format_func=lambda i: video_names[i],
                               key="labeled_video_select")
    if selected_idx is None:
        return
    selected_video = labeled_videos[selected_idx]

    mode_col, clip_col, dur_col = st.columns([1.1, 0.8, 0.9])
    with mode_col:
        play_mode = st.radio("Mode", ["Montage", "Full"], horizontal=True, key="play_mode")
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: default montage to 10 clips x 2 s | date: 2026-09-23]
    with clip_col:
        n_clips = st.number_input("Clips", value=10, min_value=3, max_value=10, step=1,
                                  key="n_clips")
    with dur_col:
        clip_duration = st.number_input("Sec/clip", value=2.0, min_value=0.5,
                                        max_value=2.0, step=0.5, key="clip_duration")
    # [AGENT EDIT END]

    if play_mode == "Montage":
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: reuse only a complete cached montage matching the source revision and settings | date: 2026-09-10]
        try:
            montage_path = montage_cache_path(selected_video, n_clips, clip_duration)
        except OSError as exc:
            st.error(f"Cannot access montage cache: {exc}")
            return
        if st.session_state.get("montage_source") != str(montage_path):
            st.session_state.pop("montage_path", None)
            st.session_state.pop("montage_start_times", None)
        st.session_state.montage_source = str(montage_path)
        # [AGENT EDIT END]

        if st.button("Generate montage", key="gen_montage"):
            with st.spinner(f"Creating {int(n_clips)}-clip montage..."):
                montage_result, start_times, err_msg = extract_video_montage(
                    selected_video, n_clips=int(n_clips),
                    clip_duration=float(clip_duration))
                if montage_result is not None:
                    montage_path = montage_result
                    st.session_state["montage_start_times"] = start_times
                    st.session_state["montage_path"] = str(montage_path)
                else:
                    st.error(f"Failed: {err_msg}")

        if "montage_path" in st.session_state:
            montage_path = Path(st.session_state["montage_path"])
        if montage_path and montage_path.exists():
            st.video(str(montage_path))
            start_times = st.session_state.get("montage_start_times", None)
            if start_times:
                st.caption("From: " + ", ".join(f"{t:.0f}s" for t in start_times))
        else:
            st.info("Click 'Generate montage'")
    else:
        file_size_mb = selected_video.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            st.warning(f"Video is {file_size_mb:.0f} MB")
        st.video(str(selected_video))


# ── head-aligned spread QC ─────────────────────────────────────────────────────
_HFQC_FLAG_COLOR = {"ok": "#2e7d32", "warn": "#ef6c00", "fail": "#c62828", "": "#9e9e9e"}


@st.cache_data(show_spinner=False)
def _hfqc_reference(config_path, config_mtime, labels_fingerprint):
    return hfqc.build_training_reference(config_path)


@st.cache_data(show_spinner=False)
def _hfqc_score(h5_path, h5_mtime, config_path, config_mtime, labels_fingerprint, p_cut):
    """``(result, track_cut)`` -- the cut tracking was ACTUALLY measured at.

    Both gates default to baphy_io's ``dlc_threshold``, so this normally equals ``p_cut``.
    It is returned rather than assumed because an older ``dlc_headfix_qc`` has no
    ``track_p_cut`` parameter: the fallback below still produces a correct result, and the
    population filter keys on the value that was really used rather than one this module
    merely intended.
    """
    ref = _hfqc_reference(config_path, config_mtime, labels_fingerprint)
    try:
        return hfqc.score_video(h5_path, ref, p_cut=p_cut,
                                track_p_cut=TRACK_P_CUT), TRACK_P_CUT
    except TypeError:
        return hfqc.score_video(h5_path, ref, p_cut=p_cut), p_cut


def _hfqc_candidate_h5(video_path, config_path):
    video_path = Path(video_path)
    if len(video_path.parents) < 2:
        return []
    session = video_path.parents[0].name
    temp_dir = session_temp_dir(video_path)
    sorted_dir = video_path.parents[1] / "sorted"

    candidates = list_dlc_h5_files(temp_dir)
    if sorted_dir.exists():
        candidates += [p for p in sorted_dir.glob("*.h5") if session in p.name]
        candidates += [p for p in sorted_dir.glob("*/*.h5") if session in p.name]

    highlight_substr, highlight_tokens = get_dlc_highlight_tokens(config_path)
    matched = hfqc.match_h5_files(candidates, highlight_substr, highlight_tokens)
    return sorted(matched, key=lambda p: p.stat().st_mtime, reverse=True)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: a spread number in isolation
# is uninterpretable -- "ratio 0.75" is only good or bad relative to what this model does
# everywhere else. gDLCFit already stores the same two headline metrics for every recorded
# fit, so the comparison is a query rather than a recomputation. | date: 2026-08-17]
def release_qc_summary(release_id):
    """``{n, ratio_median, track_median}`` of recorded fits for ONE release, or None.

    The candidate-vs-current question in the release step needs the current release's
    own numbers, not the whole alias's -- an alias's population mixes every release it
    has ever served.
    """
    if not release_id:
        return None
    pop = qc_population(None)
    if pop.empty or "release_id" not in pop.columns:
        return None
    sub = pop[pop.release_id == release_id]
    if sub.empty:
        return None
    return {"n": int(len(sub)),
            "ratio_median": (float(sub.ratio_mean.dropna().median())
                             if sub.ratio_mean.notna().any() else None),
            "track_median": (float(sub.track_mean.dropna().median())
                             if sub.track_mean.notna().any() else None)}


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: match population cache freshness to its visible refresh interval | date: 2026-09-09]
@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
# [AGENT EDIT END]
def qc_population(alias=None):
    """Recorded QC for every fit in ``gDLCFit``, as a tidy DataFrame.

    ``alias`` restricts to one routing alias, which is the comparison that means
    something: spread is measured against a model's OWN training labels, so a ratio from
    ``two_chimney`` and one from ``left_chimney`` are not the same quantity.

    Empty DataFrame on any failure -- this is context, and must never take the QC view
    down with it.
    """
    sql = ("SELECT siteid, animal, routing_alias, release_id, qc FROM gDLCFit "
           "WHERE qc IS NOT NULL AND qc NOT IN ('null', '{}')")
    params = ()
    if alias:
        sql += " AND routing_alias = %s"
        params = (alias,)
    try:
        raw = db.pd_query(sql, params)
    except Exception:                                         # noqa: BLE001
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        try:
            qc = json.loads(r.qc) if isinstance(r.qc, str) else (r.qc or {})
        except Exception:                                     # noqa: BLE001
            continue
        if qc.get("ratio_worst") is None and qc.get("track_worst") is None:
            continue
        rows.append({"siteid": r.siteid, "animal": r.animal,
                     "alias": r.routing_alias, "release_id": r.release_id,
                     # QC v5 primary headline: reference-free self-referential spread
                     "rms_spread_mean": qc.get("rms_spread_mean"),
                     "rms_spread_worst": qc.get("rms_spread_worst"),
                     "rms_spread_proc_mean": qc.get("rms_spread_proc_mean"),
                     "ratio_mean": qc.get("ratio_mean"),
                     "track_mean": qc.get("track_mean"),
                     "ratio_worst": qc.get("ratio_worst"),
                     "track_worst": qc.get("track_worst"),
                     "offset_mean": qc.get("offset_mean"),
                     "p_median": qc.get("p_median"), "p_cut": qc.get("p_cut"),
                     "track_p_cut": qc.get("track_p_cut"),
                     "qc_version": qc.get("qc_module_version"),
                     "frames": qc.get("frames")})
    return pd.DataFrame(rows)


def qc_comparable(pop, p_cut, track_p_cut=None, metric="ratio_mean"):
    """The subset of ``pop`` measured the same way as a fresh result.

    A spread scored at p_cut 0.2 and one scored at 0.9 are different measurements of
    different frames -- REI084a10 reads 0.86x at 0.9 and 2.75x at 0.2 -- so pooling them
    would manufacture a percentile out of two incompatible populations.  Requiring the
    metric to be present is the same guard for the v1 documents, which recorded only the
    worst marker.
    """
    # .iloc[0:0] on a column-less empty frame yields something whose [metric] raises
    # KeyError rather than returning an empty Series; give callers a shaped frame
    if pop.empty or metric not in pop.columns:
        return pd.DataFrame(columns=list(pop.columns) or [metric])
    ok = pop[metric].notna()
    if "p_cut" in pop.columns:
        ok &= pop["p_cut"].apply(
            lambda c: c is not None and abs(float(c) - float(p_cut)) < 1e-9)
    if track_p_cut is not None and metric.startswith("track") \
            and "track_p_cut" in pop.columns:
        ok &= pop["track_p_cut"].apply(
            lambda c: c is not None and abs(float(c) - float(track_p_cut)) < 1e-9)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 redefined every
    # spread/offset quantity (self-referential head-fix fit); a fresh result percentiled
    # against v4 rows would compare two different measurements. Tracking metrics are
    # version-stable since v3 and stay unfiltered. | date: 2026-08-18]
    if not metric.startswith("track") and "qc_version" in pop.columns:
        from nems_lbhb.motor.dlc_fit_qc import QC_MODULE_VERSION
        ok &= pop["qc_version"].apply(
            lambda v: v is not None and np.isfinite(float(v))
            and int(float(v)) >= QC_MODULE_VERSION)
    # [AGENT EDIT END]
    return pop[ok]


def _ordinal(n):
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def qc_percentile(values, value, higher_is_better):
    """Where ``value`` falls in ``values``, as a 0-100 percentile of QUALITY.

    Inverted for ratio_spread, where smaller is tighter tracking, so that 90 always
    reads as "better than 90% of recorded fits" regardless of the metric's direction.
    """
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0 or value is None or not np.isfinite(value):
        return None
    frac = float((vals <= value).mean())
    return 100.0 * (frac if higher_is_better else 1.0 - frac)


def _hfqc_population_figure(all_pop, alias_pop, metric, value, alias_label, title,
                            height=225):
    """Every geometry behind this alias's own, with this session marked.

    Two overlaid histograms rather than one: the alias-only view answers "is this session
    normal for this model?", but cannot answer "is this MODEL any good?" -- if a geometry
    tracks poorly everywhere, a session at its median looks fine while sitting well down
    the all-geometry distribution.  Both questions matter and they have different answers,
    so both are on the axis.
    """
    fig = go.Figure()

    # an EMPTY DataFrame has no columns at all, so pop[metric] raises KeyError rather
    # than returning an empty Series -- which killed the whole histogram (and with it the
    # spread panel) whenever the routed alias had no comparably-scored fits
    def _vals(pop):
        if pop is None or pop.empty or metric not in pop.columns:
            return pd.Series(dtype=float)
        return pop[metric].dropna().astype(float)

    all_vals, alias_vals = _vals(all_pop), _vals(alias_pop)
    if all_vals.empty and alias_vals.empty:
        return None

    # min() over an empty Series is NaN, and NaN bin edges render a silently BLANK plot
    # rather than raising -- take the range from whichever populations actually have data
    finite = pd.concat([all_vals, alias_vals])
    lo, hi = float(finite.min()), float(finite.max())
    span = (hi - lo) or 1.0
    bins = dict(start=lo - 0.02 * span, end=hi + 0.02 * span, size=span / 28.0)

    # histnorm so the two populations are comparable despite very different n
    if not all_vals.empty:
        fig.add_trace(go.Histogram(
            x=all_vals, xbins=bins, histnorm="probability",
            name=f"all ({len(all_vals)})",
            marker=dict(color="rgba(150,160,175,0.55)"),
            hovertemplate="%{x:.2f}<extra>all geometries</extra>"))
    if not alias_vals.empty:                  # an empty trace draws a misleading "(0)"
        fig.add_trace(go.Histogram(
            x=alias_vals, xbins=bins, histnorm="probability",
            name=f"{alias_label} ({len(alias_vals)})",
            marker=dict(color="rgba(60,130,200,0.75)"),
            hovertemplate="%{x:.2f}<extra>" + str(alias_label) + "</extra>"))

    if value is not None and np.isfinite(value):
        fig.add_vline(x=float(value), line=dict(color="crimson", width=3),
                      annotation_text="this session",
                      annotation_font=dict(size=10, color="crimson"))
    fig.update_layout(
        barmode="overlay", title=dict(text=title, font=dict(size=13)),
        height=height, margin=dict(l=8, r=8, t=32, b=8), bargap=0.02,
        xaxis_title=None, yaxis_title=None,
        yaxis=dict(showticklabels=False),
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the legend sat top-left, on the title | date: 2026-09-28]
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right",
                    font=dict(size=10)))
        # [AGENT EDIT END]
    return fig
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: ferret-head glyph
# underlay for the spread scatter -- rendered once from nems_lbhb.ferret_glyph to a
# transparent PNG and anchored to the front/headpost markers, so the clouds read as
# points on an animal's head rather than abstract geometry | date: 2026-08-19]
# Glyph unit frame (draw_ferret_head at size=1): nose tip y=+1.5, eyes at
# (±0.56, 0.76), back of skull y=-1.0, ears to |x|~1.35, neck to y=-1.5. The PNG is
# rendered over the fixed box below so data-space placement in plotly is a pure
# scale + shift. Anchors (user-approved 2026-08-19): the FRONT marker tracks a point
# between the eyes -- not the nose tip -- so it staples to (0, 0.76); the back/headpost
# marker sits a hair anterior of the skull rim, at (0, -0.85).
_GLYPH_BOX = (1.6, 1.8)          # half-width, half-height of the rendered box
_GLYPH_FRONT_Y, _GLYPH_BACK_Y = 0.76, -0.85
_ferret_png_cache = {}


def _ferret_head_png():
    """Base64 PNG of the top-down ferret head, transparent background. Cached."""
    if "png" not in _ferret_png_cache:
        import io
        import matplotlib
        from matplotlib.figure import Figure
        from nems_lbhb.ferret_glyph import draw_ferret_head
        w, h = _GLYPH_BOX
        fig = Figure(figsize=(4 * w / h, 4), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-w, w); ax.set_ylim(-h, h)
        ax.set_aspect("equal"); ax.axis("off")
        draw_ferret_head(ax, 0.0, 0.0, 1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=150)
        _ferret_png_cache["png"] = ("data:image/png;base64,"
                                    + base64.b64encode(buf.getvalue()).decode())
    return _ferret_png_cache["png"]


def _add_ferret_underlay(fig, role_xy, opacity=1.0):
    """Anchor the glyph: between-the-eyes on front, back of skull on headpost."""
    front = role_xy["front"]
    head = role_xy["headpost"]
    span = front[1] - head[1]                 # plot-units between the anchors
    if not np.isfinite(span) or span <= 0:
        return
    s = span / (_GLYPH_FRONT_Y - _GLYPH_BACK_Y)     # glyph-unit -> plot-unit scale
    cx = float((front[0] + head[0]) / 2.0)
    cy = float(head[1] - _GLYPH_BACK_Y * s)         # glyph origin in plot coords
    w, h = _GLYPH_BOX
    fig.add_layout_image(
        source=_ferret_head_png(), xref="x", yref="y",
        x=cx - w * s, y=cy + h * s, sizex=2 * w * s, sizey=2 * h * s,
        sizing="stretch", opacity=opacity, layer="below")
# [AGENT EDIT END]


def _hfqc_figure(result):
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v5 -- the scatter
    # lives in the head-fix frame (each frame aligned by its own front/back axis), the
    # frame rms_spread is now measured in. Rigid markers only (a body marker's motion
    # relative to the head is posture, not tracking noise). The origin marker sits at
    # (0,0) and the front on x=0 BY CONSTRUCTION -- the caption owns that. Training-label
    # 2sd ellipses (hf_cov) are context when a reference exists; the black quad is the
    # VIDEO'S OWN mean head shape. | date: 2026-08-18]
    fig = go.Figure()
    palette = plt.get_cmap("tab10")
    rigid_parts = [p for p in result.table.index
                   if result.table.loc[p, "role"] == "rigid"]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: plot the
    # self-Procrustes frame when available (QC v6) -- all four rigid markers share the
    # per-frame fit there, so headpost and front show their own jitter instead of being
    # pinned by the axis-frame construction. Training-label ellipses are hf-frame
    # quantities (origin = headpost) and would sit at the wrong origin here, so they
    # draw only on the hf fallback. | date: 2026-08-19]
    proc = getattr(result, "scatter_proc", None)
    scatter = proc or getattr(result, "scatter_aligned", None) or result.scatter_hf
    ref = result.reference
    ref_cov = (getattr(ref, "hf_cov", None)
               if (ref is not None and not proc) else None)
    ref_mean = (getattr(ref, "hf_mean", None)
                if (ref is not None and not proc) else None)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: a handful of gross
    # outliers stretched the axes to ~10+ head-lengths and crushed the clouds; the view
    # is fixed at ±3 head-lengths and outliers are CLAMPED to the border so they stay
    # visible (as edge points) without owning the scale | date: 2026-08-19]
    LIM = 3.0
    n_clipped = 0
    for i, part in enumerate(rigid_parts):
        rgb = palette(i % 10)
        color = f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
        pts = scatter[part]
        out = np.abs(pts) > LIM
        n_clipped += int(np.any(out, axis=1).sum())
        pts = np.clip(pts, -LIM, LIM)
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: SVG needs no WebGL; bound display points without changing QC calculations | date: 2026-09-10]
        if len(pts) > 1000:
            pts = pts[np.linspace(0, len(pts) - 1, 1000, dtype=int)]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=part,
                                   marker=dict(size=3, color=color, opacity=0.35)))
        # [AGENT EDIT END]
    # [AGENT EDIT END]
        if ref_cov is not None and part in ref_cov and ref_mean is not None:
            ell = hfqc.ellipse_path(np.asarray(ref_mean[part], dtype=float),
                                    ref_cov[part], n_sd=2.0)
            if np.isfinite(ell).all():
                fig.add_trace(go.Scatter(
                    x=np.append(ell[:, 0], ell[0, 0]),
                    y=np.append(ell[:, 1], ell[0, 1]),
                    mode="lines", name=f"{part} (train 2sd)",
                    line=dict(color=color, width=2), showlegend=False))

    # the video's own mean head shape in black, rows in RIGID_ROLES order,
    # drawn as the quad front -> rightear -> headpost -> leftear -- taken from the
    # same frame as the scatter (proc mean when the proc scatter is drawn)
    role_xy = {}
    try:
        # explicit None checks -- `ndarray or fallback` raises on numpy arrays and the
        # except below silently ate the quad
        ms = getattr(result, "mean_shape_proc", None) if proc else None
        if ms is None:
            ms = result.mean_shape
        if ms is not None:
            role_xy = {role: np.asarray(xy, dtype=float) for role, xy
                       in zip(hfqc.RIGID_ROLES, np.asarray(ms))}
    except Exception:                                         # noqa: BLE001
        role_xy = {}
    if (len(role_xy) == 4
            and all(np.isfinite(v).all() for v in role_xy.values())):
        try:
            _add_ferret_underlay(fig, role_xy)
        except Exception:                                     # noqa: BLE001
            pass                                              # underlay is decoration
        loop = ["front", "rightear", "headpost", "leftear", "front"]
        fig.add_trace(go.Scatter(
            x=[role_xy[r][0] for r in loop], y=[role_xy[r][1] for r in loop],
            mode="lines+markers+text", name="mean head shape",
            text=["front", "", "", "", ""], textposition="top center",
            textfont=dict(size=10, color="black"),
            line=dict(color="black", width=2),
            marker=dict(color="black", size=8, symbol="x")))

    frame = "self-Procrustes" if proc else "head-fixed"
    if n_clipped:
        fig.add_annotation(x=LIM, y=-LIM, xanchor="right", yanchor="bottom",
                           text=f"{n_clipped} outlier pts clamped to border",
                           showarrow=False, font=dict(size=9, color="#888"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title=f"{frame} x (head-lengths)",
                      yaxis_title=f"{frame} y (head-lengths)",
                      xaxis=dict(range=[-LIM, LIM]),
                      yaxis=dict(range=[-LIM, LIM]),
                      legend=dict(font=dict(size=9)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _hfqc_frame_caption(result):
    """Terse frame tag for under the scatter; the explanation lives in the help."""
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: disclose display-only SVG subsampling | date: 2026-09-10]
    frame = "self-Procrustes" if getattr(result, "scatter_proc", None) else "head-fix"
    return (f"{frame} frame · black = mean head shape · SVG: up to 1,000 points/marker "
            "(QC statistics unchanged)")
    # [AGENT EDIT END]


def _hfqc_frame_help(result):
    """Hover-help detail behind the terse frame tag."""
    origin = getattr(result, "origin_part", None) or "the origin marker"
    if getattr(result, "scatter_proc", None):
        return ("Each frame is rigidly fit (rotation + translation, no scaling) onto "
                f"the video's own mean shape, so every rigid marker — {origin} and "
                "front included — shows its own jitter (rms_spread_proc). The "
                "axis-frame rms_spread in the table is the pipeline-noise companion. "
                "Points beyond ±3 head-lengths are clamped to the border.")
    return (f"Each frame is aligned by its own front–{origin} axis: {origin} sits at "
            f"(0, 0) and front on x = 0 by construction, so {origin} is excluded from "
            "the spread headline and front's cloud is pure head-length jitter. "
            "Ellipses are the training labels' 2sd in the same frame.")
    # [AGENT EDIT END]


def _hfqc_bar_figure(result):
    table = result.table
    parts = list(table.index)[::-1]
    # spread-vs-training is a RIGID-marker quantity (same reasoning as the scatter: a
    # body marker's motion relative to the head is posture, not tracking noise); the
    # tracking row below keeps every part, since detection rate is meaningful anywhere
    rigid = [p for p in parts if table.loc[p, "role"] == "rigid"]
    show_spread = bool(result.has_reference)
    n_rows = 2 if show_spread else 1
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False, vertical_spacing=0.13,
                        subplot_titles=(["Spread vs training (rigid)", "Frames tracked"]
                                        if show_spread else ["Frames tracked"]))
    row = 1
    if show_spread:
        fig.add_trace(go.Bar(
            y=rigid, x=[float(table.loc[p, "ratio_spread"]) for p in rigid],
            orientation="h", name="ratio_spread (flagged)",
            marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag"], "#9e9e9e")
                          for p in rigid]), row=1, col=1)
        fig.add_trace(go.Bar(
            y=rigid, x=[float(table.loc[p, "ratio_offset"]) for p in rigid],
            orientation="h", name="ratio_offset", marker_color="#bdbdbd"), row=1, col=1)
        for xval, color in ((hfqc.RATIO_WARN, "#ef6c00"), (hfqc.RATIO_FAIL, "#c62828")):
            fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2), row=1, col=1)
        fig.update_xaxes(title_text="ratio vs training spread", row=1, col=1)
        row = 2

    fig.add_trace(go.Bar(
        y=parts, x=[float(table.loc[p, "frac_tracked"]) for p in parts],
        orientation="h", name="frames tracked",
        marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag_track"], "#9e9e9e")
                      for p in parts]), row=row, col=1)
    fig.add_trace(go.Scatter(
        y=parts, x=[float(table.loc[p, "p_median"]) for p in parts], mode="markers",
        name="median likelihood",
        marker=dict(symbol="diamond", size=9, color="#212121")), row=row, col=1)
    for xval, color in ((hfqc.TRACK_WARN, "#ef6c00"), (hfqc.TRACK_FAIL, "#c62828")):
        fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2), row=row, col=1)
    fig.update_xaxes(title_text="fraction of frames", range=[0, 1.02], row=row, col=1)
    fig.update_layout(height=270 * n_rows, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(font=dict(size=9), orientation="h",
                                  yanchor="bottom", y=-0.22))
    return fig


# [AGENT EDIT START | agent: claude | user: wingertj | reason: A/B compare rendered a full
# stack per side in side-by-side columns -- two scatters, two bar charts, two tables. One
# figure carrying both fits is half the width and makes the comparison something you SEE
# rather than reconstruct by looking back and forth | date: 2026-09-16]

# Which fit a point belongs to is the marker SYMBOL, never its colour: colour already
# means bodypart here, and that is the question this plot is usually asked ("which marker
# is drifting?"). A second meaning would cost the first.
_COMPARE_SYMBOLS = ("circle", "x", "diamond-open", "square-open")
# Line style carries the side for the things drawn as lines: the head-shape quad
# and the training ellipses.
_COMPARE_DASH = ("solid", "dash", "dot", "dashdot")

# Two modes. With ONE fit nothing is being compared, so colour keeps its usual meaning:
# a hue per bodypart, outline in black -- exactly what _hfqc_figure draws.
#
# With TWO, colour switches to meaning MODEL, because two clouds in the same ten hues
# cannot be told apart. Each model gets a PAIR: its points in the first colour, and the
# things drawn on top of them -- the head-shape quad, the training ellipses -- in the
# second, so an outline never sinks into its own cloud. The pair stays within one family
# (A cool, B warm) so the model still reads at a glance. Okabe-Ito throughout, so the
# palette survives the common colour-vision deficiencies.
_COMPARE_PALETTE = (
    {"points": "#0072B2", "lines": "#009E73"},    # A: blue points, bluish-green lines
    {"points": "#D55E00", "lines": "#CC79A7"},    # B: vermillion points, purple lines
    {"points": "#E69F00", "lines": "#F0E442"},    # C: orange points, yellow lines
    {"points": "#000000", "lines": "#999999"},    # D: black points, grey lines
)


def _side_colors(index):
    return _COMPARE_PALETTE[index % len(_COMPARE_PALETTE)]


def _rigid_parts(table):
    return [p for p in table.index if table.loc[p, "role"] == "rigid"]


def _part_color(index):
    rgb = plt.get_cmap("tab10")(index % 10)
    return f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"


def _mean_head_shape(result, proc):
    """``{role: xy}`` for the video's own mean head shape, or ``{}`` if unusable.

    Explicit None checks rather than ``ms or fallback``: ``ndarray or x`` raises, and an
    except around it silently ate the quad once already.
    """
    try:
        ms = getattr(result, "mean_shape_proc", None) if proc else None
        if ms is None:
            ms = getattr(result, "mean_shape", None)
        if ms is None:
            return {}
        role_xy = {role: np.asarray(xy, dtype=float)
                   for role, xy in zip(hfqc.RIGID_ROLES, np.asarray(ms))}
    except Exception:                                         # noqa: BLE001
        return {}
    if len(role_xy) != 4 or not all(np.isfinite(v).all() for v in role_xy.values()):
        return {}
    return role_xy


def hfqc_compare_figure(pairs, limit=3.0):
    """One head-fix scatter carrying every selected fit, with its anatomy.

    ``pairs`` is ``[(side_label, SpreadResult), ...]``.  Outliers are clamped to the
    border rather than dropped -- a gross outlier is worth seeing, but not worth
    stretching the axes to ten head-lengths for -- and the count is stated, because a
    point pinned to the border is a lie unless it is declared.

    The mean head-shape quad is drawn per fit (they are two fits of the SAME video, so a
    disagreement between their shapes is itself the finding) but the ferret glyph is
    drawn once: a second would be the same animal traced twice.
    """
    fig = go.Figure()
    n_clipped = 0
    frame = "head-fixed"
    # one fit = nothing to contrast, so colour goes back to meaning bodypart
    compare = len([1 for _s, r in pairs if r is not None]) > 1
    for index, (side, result) in enumerate(pairs):
        if result is None:
            continue
        symbol = _COMPARE_SYMBOLS[index % len(_COMPARE_SYMBOLS)]
        dash = _COMPARE_DASH[index % len(_COMPARE_DASH)]
        palette = _side_colors(index)
        point_colour = palette["points"] if compare else None
        line_colour = palette["lines"] if compare else "black"
        proc = getattr(result, "scatter_proc", None)
        scatter = (proc or getattr(result, "scatter_aligned", None)
                   or result.scatter_hf)
        if proc is not None:
            frame = "self-Procrustes"
        # training-label ellipses are hf-frame quantities (origin = headpost) and would
        # sit at the wrong origin on the proc scatter, so they draw only on the fallback
        ref = getattr(result, "reference", None)
        ref_cov = getattr(ref, "hf_cov", None) if (ref is not None and not proc) else None
        ref_mean = getattr(ref, "hf_mean", None) if (ref is not None and not proc) else None

        for i, part in enumerate(_rigid_parts(result.table)):
            pts = np.asarray(scatter[part], dtype=float)
            n_clipped += int(np.any(np.abs(pts) > limit, axis=1).sum())
            pts = np.clip(pts, -limit, limit)
            if len(pts) > 1000:            # display-only subsample; SVG needs no WebGL
                pts = pts[np.linspace(0, len(pts) - 1, 1000, dtype=int)]
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{side} {part}",
                legendgroup=part,
                marker=dict(color=point_colour or _part_color(i),
                            symbol=symbol, size=4, opacity=0.6)))
            if ref_cov is not None and ref_mean is not None and part in ref_cov:
                ell = hfqc.ellipse_path(np.asarray(ref_mean[part], dtype=float),
                                        ref_cov[part], n_sd=2.0)
                if np.isfinite(ell).all():
                    fig.add_trace(go.Scatter(
                        x=np.append(ell[:, 0], ell[0, 0]),
                        y=np.append(ell[:, 1], ell[0, 1]), mode="lines",
                        name=f"{side} {part} (train 2sd)", legendgroup=part,
                        line=dict(color=line_colour if compare
                                  else _part_color(i), width=2, dash=dash),
                        showlegend=False))

        role_xy = _mean_head_shape(result, proc)
        if not role_xy:
            continue
        if index == 0:
            try:
                _add_ferret_underlay(fig, role_xy)
            except Exception:                                 # noqa: BLE001
                pass                                          # underlay is decoration
        loop = ["front", "rightear", "headpost", "leftear", "front"]
        fig.add_trace(go.Scatter(
            x=[role_xy[r][0] for r in loop], y=[role_xy[r][1] for r in loop],
            mode="lines+markers+text", name=f"{side} mean head shape",
            text=(["front", "", "", "", ""] if index == 0 else [""] * 5),
            textposition="top center", textfont=dict(size=10, color=line_colour),
            line=dict(color=line_colour, width=2.5, dash=dash),
            marker=dict(color=line_colour, size=7, symbol="x")))

    if n_clipped:
        fig.add_annotation(x=limit, y=-limit, xanchor="right", yanchor="bottom",
                           text=f"{n_clipped} outlier pts clamped to border",
                           showarrow=False, font=dict(size=9, color="#888"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title=f"{frame} x (head-lengths)",
                      yaxis_title=f"{frame} y (head-lengths)",
                      xaxis=dict(range=[-limit, limit]),
                      yaxis=dict(range=[-limit, limit]),
                      legend=dict(font=dict(size=9)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def hfqc_compare_bars(pairs):
    """One bar chart carrying every selected fit, grouped per bodypart.

    The training-ratio row is drawn only when some fit HAS a reference: without labels
    the ratio is undefined, and an empty panel reads as a zero rather than as "unknown".
    """
    pairs = [(side, r) for side, r in pairs if r is not None]
    if not pairs:
        return go.Figure()
    show_spread = any(r.has_reference for _side, r in pairs)
    n_rows = 2 if show_spread else 1
    titles = (["Spread vs training (rigid)", "Frames tracked"] if show_spread
              else ["Frames tracked"])
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: at 0.13 the first panel's x-axis title ran into the second panel's title | date: 2026-09-28]
    fig = make_subplots(rows=n_rows, cols=1, vertical_spacing=0.22, subplot_titles=titles)
    # [AGENT EDIT END]

    for index, (side, result) in enumerate(pairs):
        table = result.table
        parts = list(table.index)[::-1]
        rigid = [p for p in parts if table.loc[p, "role"] == "rigid"]
        # colour here means ok/warn/fail per marker, which is what the bars are FOR;
        # the model is carried by grouping, opacity and the legend
        opacity = 1.0 if index == 0 else 0.55
        row = 1
        if show_spread and result.has_reference:
            fig.add_trace(go.Bar(
                y=rigid, x=[float(table.loc[p, "ratio_spread"]) for p in rigid],
                orientation="h", name=f"{side} ratio_spread", opacity=opacity,
                marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag"], "#9e9e9e")
                              for p in rigid]), row=1, col=1)
            fig.add_trace(go.Bar(
                y=rigid, x=[float(table.loc[p, "ratio_offset"]) for p in rigid],
                orientation="h", name=f"{side} ratio_offset", opacity=opacity,
                marker_color="#bdbdbd"), row=1, col=1)
        if show_spread:
            row = 2
        fig.add_trace(go.Bar(
            y=parts, x=[float(table.loc[p, "frac_tracked"]) for p in parts],
            orientation="h", name=f"{side} tracked", opacity=opacity,
            marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag_track"], "#9e9e9e")
                          for p in parts]), row=row, col=1)
        fig.add_trace(go.Scatter(
            y=parts, x=[float(table.loc[p, "p_median"]) for p in parts], mode="markers",
            name=f"{side} likelihood",
            marker=dict(symbol=_COMPARE_SYMBOLS[index % len(_COMPARE_SYMBOLS)],
                        size=9, color="#212121")), row=row, col=1)

    if show_spread:
        for xval, color in ((hfqc.RATIO_WARN, "#ef6c00"), (hfqc.RATIO_FAIL, "#c62828")):
            fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2), row=1, col=1)
        fig.update_xaxes(title_text="ratio vs training spread", row=1, col=1)
    tracked_row = 2 if show_spread else 1
    for xval, color in ((hfqc.TRACK_WARN, "#ef6c00"), (hfqc.TRACK_FAIL, "#c62828")):
        fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2),
                      row=tracked_row, col=1)
    fig.update_xaxes(title_text="fraction of frames", range=[0, 1.02], row=tracked_row, col=1)
    fig.update_layout(height=270 * n_rows, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(font=dict(size=9), orientation="h",
                                  yanchor="bottom", y=-0.22))
    return fig


def hfqc_summary_chips(result):
    """The one line that replaced the per-marker table in the default view.

    Exactly the three numbers the table was actually consulted for. The per-marker
    breakdown is a question you ask AFTER one of these looks wrong, so it moved behind
    the settings popover rather than costing width on every look.
    """
    if result is None:
        return ""
    table = result.table
    rigid = _rigid_parts(table)
    origin = getattr(result, "origin_part", None)
    spread_parts = [p for p in rigid if p != origin] or rigid

    def _mean(column, parts):
        if column not in table.columns:
            return None
        values = pd.to_numeric(table.loc[parts, column], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None

    spread = _mean("rms_spread_proc", spread_parts)
    if spread is None:
        spread = _mean("rms_spread", spread_parts)
    tracked = _mean("frac_tracked", rigid)
    conf = _mean("p_median", rigid)
    bits = [f"spread {spread:.2f}" if spread is not None else "spread —",
            f"tracked {tracked:.0%}" if tracked is not None else "tracked —",
            f"conf {conf:.3f}" if conf is not None else "conf —"]
    return " · ".join(bits)
# [AGENT EDIT END]


def _hfqc_style_table(table, has_reference=True):
    def _row_style(row):
        color = _HFQC_FLAG_COLOR.get(row.get("flag", ""), "#9e9e9e")
        if row["role"] == "nonrigid":
            return ["color: #9e9e9e"] * len(row)
        return [f"color: {color}"] * len(row)

    if not has_reference:
        table = table.drop(columns=[c for c in ("ratio_spread", "ratio_offset",
                                                "bias_x", "bias_y", "flag")
                                    if c in table.columns])
    numeric = ["frac_valid", "rms_spread", "rms_spread_px", "ratio_spread",
               "ratio_offset", "bias_x", "bias_y", "hf_cx", "hf_cy", "hf_sd",
               "radius", "frac_tracked", "p_median"]
    numeric = [c for c in numeric if c in table.columns]
    return table.style.apply(_row_style, axis=1).format({c: "{:.3f}" for c in numeric})


def hfqc_available():
    return hfqc is not None


def hfqc_summary_rows(result):
    """Compact per-marker rows for the tab-1 summary card."""
    rows = []
    table = result.table
    for part in table.index:
        rows.append({
            "marker": part,
            "spread": (float(table.loc[part, "ratio_spread"])
                       if "ratio_spread" in table.columns else float("nan")),
            "tracked": float(table.loc[part, "frac_tracked"]),
            "flag": table.loc[part, "flag"] if "flag" in table.columns else "",
            "role": table.loc[part, "role"],
        })
    return rows


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: poll QC only while its viewer is selected | date: 2026-09-09]
# Polling is owned by tabs.model._viewer.
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: where this session sits
# against every other fit of the SAME alias -- the only way a lone ratio becomes a
# judgement. | date: 2026-08-17]
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: compare inspected fits against their own aliases with collision-free A/B chart keys | date: 2026-09-10]
def _hfqc_population_section(result, p_cut, track_cut=None, stacked=False,
                             alias=None, key_prefix="hfqc_pop", fig_height=None):
# [AGENT EDIT END]
    """This session against its own geometry AND against every geometry.

    The alias-only comparison cannot detect a model that tracks poorly everywhere: a
    session at the median of a bad geometry looks unremarkable. Both populations are
    shown so "normal for this model" and "good in absolute terms" stay separable.

    ``stacked=True`` renders the two comparisons vertically (for a side column)
    instead of side by side.
    """
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: an explicit inspected alias must not be replaced by current routing | date: 2026-09-10]
    if alias is None:
        alias = production_alias() or (active_model() or {}).get("alias")
    # [AGENT EDIT END]
    all_pop = qc_population(None)
    alias_pop = qc_population(alias) if alias else pd.DataFrame()

    if all_pop.empty:
        st.caption("No recorded QC in gDLCFit to compare against yet.")
        return

    # an older dlc_headfix_qc has no summarize(); the panel below only needs the rigid
    # means and worsts, which the older worst_rigid helpers can still supply
    if hasattr(hfqc, "summarize"):
        summary = hfqc.summarize(result)
    else:
        rigid = result.table[result.table["role"] == "rigid"]
        part_r, ratio_w = hfqc.worst_rigid(result)
        part_t, track_w = hfqc.worst_rigid_tracking(result)

        def _m(col):
            vals = rigid[col].astype(float)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if not vals.empty else None

        summary = {"ratio_mean": _m("ratio_spread"), "track_mean": _m("frac_tracked"),
                   "ratio_worst": ratio_w if np.isfinite(ratio_w) else None,
                   "ratio_worst_part": part_r,
                   "track_worst": track_w if np.isfinite(track_w) else None,
                   "track_worst_part": part_t}

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: stacked mode is a
    # side column beside the scatter -- the metric widget, heading and prose captions
    # made the pair much taller than the scatter it sits next to. The value moves into
    # each histogram's title, the caption reduces to the two percentiles, and the full
    # detail (worst marker, medians, n) lives in the hover help | date: 2026-08-19]
    if not stacked:
        st.markdown("**Against the rest of the database**")
        left, right = st.columns(2)
    else:
        left, right = st.container(), st.container()
    for col, metric, value, higher, label, fmt, worst_key in (
            (left, "ratio_mean", summary["ratio_mean"], False,
             "Mean rigid spread", "{:.2f}×", "ratio_worst"),
            (right, "track_mean", summary["track_mean"], True,
             "Mean rigid tracking", "{:.0%}", "track_worst")):
        with col:
            if value is None:
                st.caption(f"{label}: —")
                continue

            worst_part = summary.get(f"{worst_key}_part")
            worst_val = summary.get(worst_key)
            worst_note = (f"Worst marker: {worst_part} "
                          f"{fmt.format(worst_val)}." if worst_val is not None else "")

            # filter on the cut this result was ACTUALLY measured at, so a degraded
            # fallback run is compared against nothing rather than against fits that
            # used the strict bar
            cut = dict(track_p_cut=track_cut if track_cut is not None else TRACK_P_CUT,
                       metric=metric)
            comp_all = qc_comparable(all_pop, p_cut, **cut)
            comp_alias = qc_comparable(alias_pop, p_cut, **cut)

            if not stacked:
                st.metric(label, fmt.format(value))
            if comp_all.empty:
                st.caption(f"{worst_note} :orange[No comparably-scored fits on record "
                           f"yet.]",
                           help="Recorded fits were scored under an older QC version "
                                "(worst-marker only, spread at p_cut 0.9). Re-score "
                                "with: scripts/dlc_sync_fits.py --go --rescore-qc")
                continue

            # say the DIRECTION rather than a bare percentile: "91st pct" under a
            # heading about spread reads as "91st percentile of spread" (bad) when it
            # means the opposite
            def _line(sub, name, short):
                if sub.empty:
                    return None
                pct = qc_percentile(sub[metric], value, higher_is_better=higher)
                if pct is None:
                    return None
                return (f"better than {pct:.0f}% of {len(sub)} {name}",
                        f"**{pct:.0f}%** {short}",
                        float(sub[metric].dropna().median()))

            bits, shorts, helps = [], [], []
            for sub, name, short in ((comp_all, "fits (all geometries)", "all"),
                                     (comp_alias, f"{alias} fits", str(alias))):
                got = _line(sub, name, short)
                if got:
                    bits.append(got[0])
                    shorts.append(got[1])
                    helps.append(f"{name}: median {fmt.format(got[2])}")
            if not stacked:
                st.caption(f"{worst_note} " + " · ".join(bits),
                           help="\n\n".join(helps))

            # stacked: the metric widget is gone, so the value rides in the title
            fig = _hfqc_population_figure(comp_all, comp_alias, metric, value,
                                          alias or "this alias",
                                          f"{label}  {fmt.format(value)}"
                                          if stacked else label,
                                          height=fig_height or (150 if stacked else 225))
            if fig is not None:
                # [AGENT EDIT START | agent: Codex | user: wingertj | reason: unique keys for inspected A/B populations | date: 2026-09-10]
                st.plotly_chart(fig, use_container_width=True,
                                key=f"{key_prefix}_{metric}")
                # [AGENT EDIT END]
            if stacked:
                # two numbers only, under the plot they qualify; everything else
                # (worst marker, medians, n) rides in the hover help
                st.caption(" · ".join(shorts),
                           help=("Percentile of QUALITY — higher is better. "
                                 f"{worst_note}\n\n" + "\n\n".join(helps)))
# [AGENT EDIT END]


def render_headfix_spread_section(compact=False):
    """Head-aligned marker spread QC, gated on a model-matched analysis existing."""
    if hfqc is None:
        st.caption("dlc_headfix_qc unavailable — spread QC disabled.")
        return
    video_path = st.session_state.get("video_path", "")
    config_path = st.session_state.get("selected_dlc_model_config", "")
    if not video_path or not config_path or not os.path.exists(config_path):
        return

    bodyparts = st.session_state.get("dlc_bodyparts", [])
    missing = hfqc.missing_rigid_roles(bodyparts) if bodyparts else list(hfqc.RIGID_ROLES)
    if missing:
        st.info(f"Model has no {', '.join(missing)} marker(s); spread QC needs "
                f"{', '.join(hfqc.RIGID_ROLES)}.")
        return

    candidates = _hfqc_candidate_h5(video_path, config_path)
    if not candidates:
        st.info("Analyze → temp with this model first.")
        return

    options = [str(p) for p in candidates]
    if st.session_state.get("hfqc_h5_select") not in options:
        st.session_state.pop("hfqc_h5_select", None)

    pick_col, cut_col, go_col = st.columns([1.6, 1.1, 0.7])
    with pick_col:
        choice = st.selectbox("Analysis file", options,
                              format_func=lambda s: Path(s).name, key="hfqc_h5_select")
    with cut_col:
        p_cut = st.slider("Likelihood cutoff", 0.0, 1.0, float(SPREAD_P_CUT), 0.05,
                          key="hfqc_p_cut",
                          help=f"Gates both spread and tracking, and defaults to "
                               f"{SPREAD_P_CUT:.2f} — baphy_io's own `dlc_threshold` — so "
                               f"QC describes the frames analyses actually load.\n\n"
                               f"Moving it off {SPREAD_P_CUT:.2f} makes the result "
                               f"incomparable to the recorded population.")
    with go_col:
        st.write("")
        compute = st.button("Compute spread", key="hfqc_compute")

    h5_path = Path(choice)
    result_key = "hfqc_result"
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: do not display QC computed for older predictions or a different reference model | date: 2026-09-09]
    try:
        labels_fp = hfqc.label_files_fingerprint(config_path)
        request_key = (file_stamp(h5_path), file_stamp(config_path), labels_fp, float(p_cut))
    except OSError as exc:
        st.warning(f"QC input unavailable: {exc}")
        return
    # [AGENT EDIT END]

    if compute:
        with st.spinner("Head-fixing frames..."):
            try:
                result, track_cut = _hfqc_score(
                    str(h5_path), h5_path.stat().st_mtime,
                    str(config_path), os.path.getmtime(config_path),
                    labels_fp, float(p_cut))
            except Exception as exc:                          # noqa: BLE001
                st.error(f"Spread QC failed: {exc}")
                st.session_state.pop(result_key, None)
                return
            st.session_state[result_key] = (request_key, result, track_cut)

    stored = st.session_state.get(result_key)
    # a 2-tuple is a result cached before track_cut was carried; drop it rather than
    # unpacking it wrong
    if stored is None or len(stored) != 3 or stored[0] != request_key:
        return
    _key, result, track_cut = stored

    if not result.has_reference:
        st.caption(":orange[no training labels — within-video spread only]",
                   help="No CollectedData_*.h5 in this project, so there is no "
                        "ratio against label scatter and no reference ellipses.")
    for msg in result.warnings:
        st.warning(msg)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: no full sentences in
    # the GUI (user directive) -- fragments up front, prose detail in the hover help
    # | date: 2026-08-19]
    part, ratio = hfqc.worst_rigid(result)
    track_part, track_frac = hfqc.worst_rigid_tracking(result)
    bits = []
    if part is not None:
        bits.append(f"worst spread **{part}** {ratio:.2f}× "
                    f"({hfqc.flag_for_ratio(ratio)})")
    if track_part is not None:
        bits.append(f"worst tracking **{track_part}** {100 * track_frac:.0f}% "
                    f"({hfqc.flag_for_fraction(track_frac)})")
    bits.append(f"{100 * result.frac_valid:.0f}% frames pass")
    bits.append(f"1 HL = {result.scale_px:.0f} px")
    ref = result.reference
    help_lines = [f"{result.n_valid}/{result.n_total} frames passed p>{p_cut:.2f} "
                  f"(baphy's loader cut)."]
    if ref is not None:
        help_lines.append(f"Reference: {ref.n_frames} labeled frames from "
                          f"{len(ref.sources)} labeled-data folder(s).")
    help_lines.append("Spread is in head-lengths (HL): 1 HL = the median "
                      "front-to-back marker distance, "
                      f"{result.scale_px:.1f} px in this video.")
    st.caption(" · ".join(bits), help="\n\n".join(help_lines))
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the table led and the
    # scatter hid in an expander, so the view opened on 17 columns of numbers instead of
    # the picture that shows at a glance which marker is loose. Scatter first, numbers
    # behind it, and the population comparison that makes a lone ratio interpretable.
    # | date: 2026-08-17]
    if compact:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the scatter and
        # the two database comparisons each took a full row; scatter now sits smaller on
        # the left with the two population histograms stacked beside it | date: 2026-08-19]
        scatter_col, pop_col = st.columns([1.15, 1])
        with scatter_col:
            st.plotly_chart(_hfqc_figure(result), use_container_width=True,
                            key="hfqc_scatter_compact")
            st.caption(_hfqc_frame_caption(result), help=_hfqc_frame_help(result))
        with pop_col:
            _hfqc_population_section(result, p_cut, track_cut, stacked=True)
        # [AGENT EDIT END]
        with st.expander("Per-marker numbers", expanded=False):
            st.dataframe(_hfqc_style_table(result.table,
                                           has_reference=result.has_reference),
                         use_container_width=True)
            st.plotly_chart(_hfqc_bar_figure(result), use_container_width=True,
                            key="hfqc_bars_compact")
        return
    # [AGENT EDIT END]

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(_hfqc_figure(result), use_container_width=True)
        st.caption(_hfqc_frame_caption(result), help=_hfqc_frame_help(result))
    with right:
        st.plotly_chart(_hfqc_bar_figure(result), use_container_width=True)
    st.dataframe(_hfqc_style_table(result.table, has_reference=result.has_reference),
                 use_container_width=True)


# ── geometry helpers (arena tab) ───────────────────────────────────────────────
def recompute_pixels_per_meter():
    """Refresh distances/ppm bookkeeping from the current marker set."""
    if not st.session_state.points:
        return {}, None
    marker_names = sorted(st.session_state.points.keys())
    current_pairs = [f"{a}_{b}" for a, b in combinations(marker_names, 2)]
    for pair in list(st.session_state.distances.keys()):
        if pair not in current_pairs:
            del st.session_state.distances[pair]
    for pair in current_pairs:
        st.session_state.distances.setdefault(pair, 0.0)

    ppm_dict = {}
    for pair_name, dist_m in st.session_state.distances.items():
        m1, m2 = pair_name.split("_", 1)
        if m1 not in st.session_state.points or m2 not in st.session_state.points:
            continue
        px1, py1 = st.session_state.points[m1]
        px2, py2 = st.session_state.points[m2]
        pixel_dist = math.hypot(px2 - px1, py2 - py1)
        if dist_m and dist_m > 0:
            ppm_dict[pair_name] = pixel_dist / dist_m
    if ppm_dict:
        avg = float(np.average(list(ppm_dict.values())))
        st.session_state.ppm_avg = avg
        st.session_state.pixels_per_meter = ppm_dict
        return ppm_dict, avg
    return ppm_dict, None


def speaker_layout_figure():
    """Speakers in metres around the fixation point, in the rotated frame."""
    orientation = st.session_state.get("speaker_orientation", "Top")
    base_deg = {"Top": 90.0, "Right": 0.0, "Bottom": -90.0, "Left": 180.0}.get(orientation, 90.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-0.6, 0.6], y=[0, 0], mode="lines",
                            line=dict(color="#2c455d", width=2), name="x-axis"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                            marker=dict(color="#c62828", size=13, symbol="x"),
                            text=["fixation"], textposition="bottom center",
                            showlegend=False))
    for name, spec in (st.session_state.get("speakers") or {}).items():
        try:
            dist = float(spec.get("distance", 0.0))
            angle = float(spec.get("angle", 0.0))
        except (TypeError, ValueError):
            continue
        theta = math.radians(base_deg + angle)
        fig.add_trace(go.Scatter(
            x=[dist * math.cos(theta)], y=[dist * math.sin(theta)],
            mode="markers+text", marker=dict(color="#5980a6", size=12),
            text=[f"{name} {angle:+.0f}°"], textposition="top center", showlegend=False))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="metres", yaxis_title="metres",
                      showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
