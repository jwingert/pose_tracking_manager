# [AGENT EDIT START | agent: claude | user: wingertj | reason: keep new GUI surface out of the
# 2400-line app file -- panels are thin renderers over the logic modules | date: 2026-08-12]
"""Streamlit panels for the DLC labeler.

Each module here renders ONE panel and holds no logic of its own: the rules live in
``nems_lbhb/motor/dlc_*.py`` so the CLI, the queue jobs and the GUI cannot drift apart,
and so the rules stay testable without Streamlit.
"""
# [AGENT EDIT END]
