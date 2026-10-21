"""Tab bodies for the tabbed DLC Arena Labeler."""
import importlib

import streamlit as st

_PANEL_ROOTS = (
    "nems_lbhb.motor.streamlit_vid_labeler.panels",
    "panels",
)


def load_panel(name):
    """Import a panel module however this app happens to be on the path.

    Run as a package (``nems_lbhb.motor.streamlit_vid_labeler``) or straight from
    its own directory by ``streamlit run``; try both, return None if neither works.
    """
    for root in _PANEL_ROOTS:
        try:
            return importlib.import_module(f"{root}.{name}")
        except Exception:                                     # noqa: BLE001
            continue
    return None


def panel_or_note(name):
    module = load_panel(name)
    if module is None:
        st.caption(f"{name} panel unavailable")
    return module


def section(title, help_text=None):
    st.markdown(f"##### {title}")
    if help_text:
        st.caption(help_text)
