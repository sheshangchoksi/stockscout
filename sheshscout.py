"""
sheshscout.py — Indian Stock Scout: entry point.

Single mode: Positional Scanner — long-term value investing. Every
scoring threshold is adjustable from the sidebar (see mode_positional.py's
_params_ui). Rate limiting, checkpointing, and the sidebar shell come from
scanner_common.
"""

import warnings
import logging

import streamlit as st

import scanner_common as sc
import mode_positional

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

st.set_page_config(page_title="Indian Stock Scout", page_icon="🎯", layout="wide")
sc.inject_base_css()

st.markdown('<p class="main-header">🎯 Indian Stock Scout</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#666;'>NSE & BSE Positional Scanner — long-term value investing</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

mode_positional.render()
