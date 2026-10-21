"""Pure state transitions for the arena labeler (no Streamlit or DLC imports)."""
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: refresh routing without losing manual selections | date: 2026-09-09]
def apply_routing(state, context, result):
    """Follow fresh suggestions unless the operator explicitly selected a model.

    A different video/method starts in automatic mode. A failed refresh must never
    clear a previous selection, or arm a model inferred from unavailable metadata.
    """
    changed_context = state.get("routing_stamp") != context
    if changed_context:
        state["routing_stamp"] = context
        state["routing_manual"] = False
        state.pop("routing_alias", None)
    suggested = result.get("alias") if result else None
    failed = bool(result and result.get("refresh_error"))
    if not failed and not state.get("routing_manual", False):
        if suggested:
            state["routing_alias"] = suggested
        else:
            state.pop("routing_alias", None)
    state["routing_declined"] = not bool(state.get("routing_alias"))
    state["routing_last_suggestion"] = suggested
    return state.get("routing_alias")
# [AGENT EDIT END]
