import streamlit as st

def section_title(text, subtitle=None):
    """Large section title with optional subtitle."""
    st.markdown(f"## {text}")
    if subtitle:
        st.markdown(f"**{subtitle}**")

def two_column(left_fn, right_fn, left_width=2, right_width=1):
    """
    Create a two-column layout and run provided callables inside each column.
    Example:
        two_column(lambda: st.write('L'), lambda: st.write('R'))
    """
    cols = st.columns([left_width, right_width])
    with cols[0]:
        left_fn()
    with cols[1]:
        right_fn()

def metric_row(metrics):
    """
    Display a row of metric cards.
    metrics: list of tuples -> (label, value, delta_text (optional))
    """
    cols = st.columns(len(metrics))
    for c, (label, value, *rest) in zip(cols, metrics):
        delta = rest[0] if rest else ""
        with c:
            if delta:
                st.metric(label, value, delta)
            else:
                st.metric(label, value)
