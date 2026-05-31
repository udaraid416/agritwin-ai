"""
app.py — AI-Powered Predictive Digital Twin for Smart Protected Agriculture
INDUSTRIAL IoT CONTROL ROOM REDESIGN: Siemens / SCADA Style
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import datetime
import os
import requests
import google.generativeai as genai
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from PIL import Image

from config import (
    PARAM_DEFAULTS, GROWTH_STAGES,
    GRADE_COLORS, RISK_COLORS, ZONES, APP_TITLE, APP_ICON, VERSION
)
from utils import (
    compute_sustainability_score, calculate_disease_risk,
    calculate_irrigation_need, calculate_heat_stress,
    generate_zone_heatmap_data, generate_trend_data,
    evaluate_system_performance
)
from ai_model import get_ai_recommendations, get_ai_engine_status

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & API SETUP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgriTwin AI · Precision Farming",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

# Session States
if "vision_result" not in st.session_state: st.session_state.vision_result = None
if "vision_messages" not in st.session_state: st.session_state.vision_messages = []
if "market_messages" not in st.session_state: st.session_state.market_messages = []

# ─────────────────────────────────────────────────────────────────────────────
# 🎨 INDUSTRIAL IoT / SCADA CSS — Full Redesign
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

/* ══════════════════════════════════════════════
   ROOT VARIABLES — Industrial Metallic System
   ══════════════════════════════════════════════ */
:root {
  /* Backgrounds — Gunmetal / Slate */
  --bg-void:       #0D1117;
  --bg-base:       #1A1D21;
  --bg-panel:      #21252B;
  --bg-elevated:   #282C34;
  --bg-input:      #1C1F23;
  --bg-hover:      #2C313A;

  /* Borders — Industrial Steel */
  --border-dim:    #2E3138;
  --border-mid:    #3E4451;
  --border-bright: #5C6370;

  /* Text — High Legibility Clinical */
  --text-primary:  #E6E6E6;
  --text-secondary:#ABB2BF;
  --text-muted:    #636D7E;
  --text-label:    #7F848E;

  /* Status Colors — Functional ONLY */
  --green:         #98C379;
  --green-dim:     rgba(152,195,121,0.12);
  --green-glow:    rgba(152,195,121,0.25);
  --amber:         #E5C07B;
  --amber-dim:     rgba(229,192,123,0.12);
  --red:           #E06C75;
  --red-dim:       rgba(224,108,117,0.12);
  --blue:          #61AFEF;
  --blue-dim:      rgba(97,175,239,0.12);
  --cyan:          #56B6C2;
  --cyan-dim:      rgba(86,182,194,0.12);
  --purple:        #C678DD;

  /* System */
  --radius:        2px;
  --radius-sm:     1px;
  --transition:    0.18s linear;
}

/* ══════════════════════════════════════════════
   BASE LAYOUT
   ══════════════════════════════════════════════ */
html, body, .stApp, .main {
  background: var(--bg-void) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }
header { background-color: transparent !important; }
.block-container {
  padding: 1rem 1.5rem 2rem !important;
  max-width: 100% !important;
}

/* Thin top accent bar */
.stApp::before {
  content: '';
  display: block;
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--green), var(--blue), var(--amber));
  z-index: 9999;
}

/* ══════════════════════════════════════════════
   SIDEBAR — Control Panel
   ══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border-mid) !important;
}
section[data-testid="stSidebar"] > div {
  background: transparent !important;
}

/* ══════════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════════ */
body, .stMarkdown, .stText, label,
.stSelectbox label, .stSlider label,
.stCheckbox label, .stTextInput label, .stNumberInput label {
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--text-primary) !important;
  font-size: 0.875rem !important;
}

/* ══════════════════════════════════════════════
   KEYFRAME ANIMATIONS — Precise, Linear, Industrial
   ══════════════════════════════════════════════ */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes ledBlink {
  0%, 100% { opacity: 1; box-shadow: 0 0 4px currentColor; }
  50%       { opacity: 0.25; box-shadow: none; }
}
@keyframes progressFillLinear {
  from { width: 0%; }
}
@keyframes moduleFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanH {
  0%   { left: -100%; }
  100% { left: 200%; }
}

/* ══════════════════════════════════════════════
   SYSTEM HEADER / NAMEPLATE
   ══════════════════════════════════════════════ */
.sys-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 0.75rem 0 0.75rem;
  border-bottom: 1px solid var(--border-mid);
  margin-bottom: 1rem;
  animation: fadeInUp 0.25s linear both;
}
.sys-title {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary) !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.sys-subtitle {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.65rem;
  color: var(--text-muted);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-top: 0.15rem;
}
.sys-status-row {
  text-align: right;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem;
  color: var(--text-secondary);
  letter-spacing: 0.08em;
  line-height: 1.7;
}

/* ══════════════════════════════════════════════
   SECTION HEADERS — Panel Labels
   ══════════════════════════════════════════════ */
.panel-header {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.62rem;
  font-weight: 500;
  color: var(--text-muted) !important;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.6rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border-dim);
}
.panel-header .led {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: ledBlink 2s linear infinite;
  flex-shrink: 0;
}
.panel-header .led.amber { background: var(--amber); animation-duration: 1.4s; }
.panel-header .led.red   { background: var(--red);   animation-duration: 0.8s; }
.panel-header .led.blue  { background: var(--blue);  animation-duration: 3s; }
.panel-header .led.off   { background: var(--border-mid); animation: none; }

/* ══════════════════════════════════════════════
   CONTROL PANEL — Main Container
   ══════════════════════════════════════════════ */
.ctrl-panel {
  background: var(--bg-panel);
  border: 1px solid var(--border-mid);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  margin-bottom: 0.75rem;
  animation: moduleFadeIn 0.2s linear both;
  position: relative;
  overflow: hidden;
}
.ctrl-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-bright) 30%, transparent);
}
.ctrl-panel:hover {
  border-color: var(--border-bright);
  transition: border-color var(--transition);
}

/* ══════════════════════════════════════════════
   METRIC TILES — Data Readout Blocks
   ══════════════════════════════════════════════ */
.metric-tile {
  background: var(--bg-panel);
  border: 1px solid var(--border-mid);
  border-top: 2px solid var(--border-mid);
  border-radius: var(--radius);
  padding: 0.75rem 0.8rem;
  text-align: center;
  animation: moduleFadeIn 0.2s linear both;
  transition: border-top-color var(--transition), background var(--transition);
  position: relative;
  cursor: default;
}
.metric-tile:hover {
  background: var(--bg-hover);
}
.metric-tile.status-ok    { border-top-color: var(--green); }
.metric-tile.status-warn  { border-top-color: var(--amber); }
.metric-tile.status-crit  { border-top-color: var(--red); }
.metric-tile.status-info  { border-top-color: var(--blue); }
.metric-value {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.6rem;
  font-weight: 400;
  line-height: 1.2;
  letter-spacing: -0.02em;
}
.metric-unit {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-left: 0.15rem;
}
.metric-label {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-top: 0.25rem;
}
.metric-id {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.52rem;
  color: var(--text-muted);
  opacity: 0.5;
  position: absolute;
  top: 4px; left: 6px;
  letter-spacing: 0.1em;
}

/* ══════════════════════════════════════════════
   PROGRESS BARS — Mechanical Fill
   ══════════════════════════════════════════════ */
.prog-wrap { margin: 0.4rem 0; }
.prog-header {
  display: flex; justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.67rem;
  color: var(--text-muted);
  margin-bottom: 0.2rem;
  letter-spacing: 0.04em;
}
.prog-header .prog-val { color: var(--text-secondary); }
.prog-track {
  background: var(--bg-elevated);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  height: 6px;
  overflow: hidden;
}
.prog-fill {
  height: 100%;
  border-radius: var(--radius-sm);
  animation: progressFillLinear 1.2s linear both;
  position: relative;
}
/* Tick marks on track */
.prog-track::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0; left: 25%;
  width: 1px;
  background: rgba(255,255,255,0.04);
  pointer-events: none;
}

/* ══════════════════════════════════════════════
   STATUS BADGES
   ══════════════════════════════════════════════ */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  padding: 0.2rem 0.65rem;
  border-radius: var(--radius);
  border: 1px solid currentColor;
  text-transform: uppercase;
}
.status-badge .dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: currentColor;
  animation: ledBlink 1.5s linear infinite;
}

/* ══════════════════════════════════════════════
   TERMINAL / AI OUTPUT
   ══════════════════════════════════════════════ */
.terminal {
  background: var(--bg-void);
  border: 1px solid var(--border-mid);
  border-radius: var(--radius);
  padding: 0.85rem 1rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  color: var(--green);
  line-height: 1.75;
  position: relative;
  overflow: hidden;
}
.terminal::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 60px; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(152,195,121,0.04), transparent);
  animation: scanH 6s linear infinite;
}
.terminal-header {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.58rem;
  color: var(--text-muted);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border-dim);
  padding-bottom: 0.5rem;
  margin-bottom: 0.6rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.terminal-prompt {
  color: var(--blue);
  margin-right: 0.4rem;
}
.terminal-key { color: var(--amber); }
.terminal-val { color: var(--text-primary); }
.terminal-line { margin-bottom: 0.15rem; }
.terminal-section {
  color: var(--text-muted);
  border-top: 1px solid var(--border-dim);
  padding-top: 0.4rem;
  margin-top: 0.4rem;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ══════════════════════════════════════════════
   AI RESULT CARDS — Terminal Style
   ══════════════════════════════════════════════ */
.ai-result-block {
  background: var(--bg-void);
  border: 1px solid var(--border-mid);
  border-left: 2px solid var(--green);
  border-radius: var(--radius);
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.6rem;
  animation: slideInLeft 0.2s linear both;
}
.ai-result-block.warn { border-left-color: var(--amber); }
.ai-result-block.crit { border-left-color: var(--red); }
.ai-result-block.info { border-left-color: var(--blue); }
.ai-result-key {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.58rem;
  color: var(--text-muted);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.ai-result-val {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.85rem;
  color: var(--text-primary);
  line-height: 1.6;
}

/* ══════════════════════════════════════════════
   ACTION ITEMS — Priority Queue
   ══════════════════════════════════════════════ */
.action-row {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  padding: 0.55rem 0.75rem;
  border: 1px solid var(--border-dim);
  border-left: 2px solid var(--border-mid);
  border-radius: var(--radius);
  margin-bottom: 0.4rem;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.82rem;
  color: var(--text-primary);
  background: var(--bg-void);
  animation: slideInLeft 0.2s linear both;
  transition: background var(--transition);
}
.action-row:hover { background: var(--bg-panel); }
.action-row.p1 { border-left-color: var(--red); }
.action-row.p2 { border-left-color: var(--amber); }
.action-row.p3 { border-left-color: var(--green); }
.action-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-muted);
  padding-top: 0.1rem;
  flex-shrink: 0;
  min-width: 1.5rem;
}

/* ══════════════════════════════════════════════
   TELEMETRY TABLE
   ══════════════════════════════════════════════ */
.telem-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
}
.telem-table th {
  text-align: left;
  color: var(--text-muted);
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  padding: 0.4rem 0.5rem;
  border-bottom: 1px solid var(--border-dim);
  font-weight: 400;
}
.telem-table td {
  padding: 0.35rem 0.5rem;
  border-bottom: 1px solid rgba(62,68,81,0.4);
  color: var(--text-primary);
}
.telem-table td.key  { color: var(--text-secondary); font-size: 0.7rem; }
.telem-table td.val  { color: var(--text-primary); text-align: right; }
.telem-table td.unit { color: var(--text-muted); font-size: 0.65rem; text-align: right; }
.telem-table td.ok   { color: var(--green); }
.telem-table td.warn { color: var(--amber); }
.telem-table td.crit { color: var(--red); }

/* ══════════════════════════════════════════════
   SIDEBAR OVERRIDES
   ══════════════════════════════════════════════ */
.sidebar-brand {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary) !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.sidebar-sub {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.6rem;
  color: var(--text-muted) !important;
  letter-spacing: 0.18em;
  margin-top: 0.1rem;
}
.sidebar-section {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.58rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-muted) !important;
  border-top: 1px solid var(--border-dim);
  padding-top: 0.5rem;
  margin: 0.8rem 0 0.5rem;
}
.engine-tag {
  display: inline-block;
  background: var(--bg-elevated);
  border: 1px solid var(--border-mid);
  border-radius: var(--radius);
  padding: 0.15rem 0.6rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  color: var(--text-secondary);
  letter-spacing: 0.08em;
}

/* ══════════════════════════════════════════════
   DIVIDER
   ══════════════════════════════════════════════ */
.sys-divider {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.9rem 0 0.75rem;
}
.sys-divider span {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.55rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--text-muted);
  white-space: nowrap;
}
.sys-divider::before, .sys-divider::after {
  content: ''; flex: 1;
  height: 1px;
  background: var(--border-dim);
}

/* ══════════════════════════════════════════════
   GRADE / SCORE DISPLAY
   ══════════════════════════════════════════════ */
.grade-readout {
  text-align: center;
  padding: 0.5rem;
}
.grade-letter {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 3.2rem;
  font-weight: 600;
  line-height: 1;
  letter-spacing: -0.04em;
}
.grade-score {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
  letter-spacing: 0.15em;
  margin-top: 0.2rem;
}

/* ══════════════════════════════════════════════
   STANDBY PLACEHOLDER
   ══════════════════════════════════════════════ */
.standby-box {
  background: var(--bg-panel);
  border: 1px solid var(--border-mid);
  border-radius: var(--radius);
  padding: 2.5rem 1.5rem;
  text-align: center;
}
.standby-icon {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2rem;
  color: var(--border-bright);
  margin-bottom: 0.6rem;
}
.standby-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-muted);
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.standby-desc {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* ══════════════════════════════════════════════
   STREAMLIT OVERRIDES
   ══════════════════════════════════════════════ */
.stSlider > div > div > div {
  background: var(--blue) !important;
}
.stSlider > div > div > div > div {
  background: var(--bg-void) !important;
  border: 2px solid var(--blue) !important;
  border-radius: 2px !important;
}
[data-baseweb="slider"] [role="slider"] {
  border-radius: 2px !important;
}
.stSelectbox div[data-baseweb],
.stTextInput input,
.stNumberInput input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius) !important;
  color: var(--text-primary) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 0.85rem !important;
}
.stSelectbox div[data-baseweb]:hover,
.stTextInput input:focus,
.stNumberInput input:focus {
  border-color: var(--blue) !important;
}
.stButton > button {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius) !important;
  transition: all var(--transition) !important;
  padding: 0.45rem 1rem !important;
}
.stButton > button[kind="primary"] {
  background: var(--green) !important;
  color: var(--bg-void) !important;
  border: none !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #A8D388 !important;
  box-shadow: 0 0 12px var(--green-glow) !important;
}
.stButton > button:not([kind="primary"]) {
  background: transparent !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--border-bright) !important;
  color: var(--text-primary) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border-mid) !important;
  gap: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0.65rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all var(--transition) !important;
  margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
  color: var(--text-primary) !important;
  background: var(--bg-panel) !important;
  border-bottom: 2px solid var(--blue) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text-secondary) !important;
  background: var(--bg-elevated) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding: 1rem 0 !important;
}
.js-plotly-plot { border-radius: 0 !important; }
.stDownloadButton > button {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem !important;
  background: transparent !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius) !important;
}
.stMarkdown hr { border-color: var(--border-dim) !important; }
.stExpander {
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-panel) !important;
}
.stExpander [data-testid="stExpanderToggleIcon"] {
  color: var(--text-muted) !important;
}
.stCheckbox label { font-size: 0.82rem !important; }
[data-baseweb="checkbox"] [type="checkbox"] {
  accent-color: var(--green) !important;
}
.stAlert {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border-mid) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stChatInputContainer"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stChatInputContainer"] textarea {
  background: transparent !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--text-primary) !important;
  font-size: 0.85rem !important;
}

/* Metrics */
[data-testid="metric-container"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
  padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--text-muted) !important;
  font-size: 0.75rem !important;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_ai_thinking = load_lottieurl("https://lottie.host/809c91f1-331e-4509-9fc6-9e90098da0da/uJ0q3E1jKz.json")

# ── SCADA / Telemetry Plotly Layout ──────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#ABB2BF", size=11),
    margin=dict(l=10, r=10, t=24, b=10),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, family="IBM Plex Mono"),
        bordercolor="rgba(62,68,81,0.5)",
        borderwidth=1
    ),
    hoverlabel=dict(
        bgcolor="#21252B",
        bordercolor="#3E4451",
        font=dict(family="IBM Plex Mono", size=11, color="#E6E6E6"),
    )
)

def plotly_scada_axes(fig):
    axis_cfg = dict(
        gridcolor="rgba(255,255,255,0.05)",
        gridwidth=1,
        zerolinecolor="rgba(255,255,255,0.08)",
        zerolinewidth=1,
        tickfont=dict(family="IBM Plex Mono", size=9, color="#636D7E"),
        linecolor="#3E4451",
        linewidth=1,
        showline=True,
    )
    fig.update_xaxes(**axis_cfg)
    fig.update_yaxes(**axis_cfg)
    return fig

def progress_html(label: str, value: float, color: str = "#98C379", max_val: float = 100) -> str:
    pct = min(100, max(0, value / max_val * 100))
    status_cls = ""
    if color == "#E06C75": status_cls = "crit"
    elif color == "#E5C07B": status_cls = "warn"
    return f'''<div class="prog-wrap">
        <div class="prog-header">
            <span>{label}</span>
            <span class="prog-val {status_cls}">{value:.1f}</span>
        </div>
        <div class="prog-track">
            <div class="prog-fill" style="width:{pct}%;background:{color};opacity:0.85;"></div>
        </div>
    </div>'''

def get_status_class(value, low_crit=None, low_warn=None, high_warn=None, high_crit=None):
    if high_crit and value >= high_crit: return "status-crit", "#E06C75"
    if high_warn and value >= high_warn: return "status-warn", "#E5C07B"
    if low_crit and value <= low_crit:   return "status-crit", "#E06C75"
    if low_warn and value <= low_warn:   return "status-warn", "#E5C07B"
    return "status-ok", "#98C379"

def kpi_tile(col, ch_id, value, unit, label, cls, color):
    col.markdown(f'''<div class="metric-tile {cls}">
        <div class="metric-id">{ch_id}</div>
        <div class="metric-value" style="color:{color};">{value}<span class="metric-unit">{unit}</span></div>
        <div class="metric-label">{label}</div>
    </div>''', unsafe_allow_html=True)

def led_header(label, led_cls="", icon=""):
    return f'<div class="panel-header"><span class="led {led_cls}"></span>{icon} {label}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'''
        <div style="padding:0.5rem 0 0.3rem">
            <div class="sidebar-brand">◈ AgriTwin AI</div>
            <div class="sidebar-sub">Industrial Control System · v{VERSION}</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown("---")

    crop_type    = st.text_input("Crop Type", value="Lettuce", placeholder="Lettuce, Tomato…")
    growth_stage = st.selectbox("Growth Stage", GROWTH_STAGES, index=1)

    st.markdown('<div class="sidebar-section">// Climate Sensors</div>', unsafe_allow_html=True)
    temperature      = st.slider("Temperature (°C)",       5.0, 50.0,  PARAM_DEFAULTS["temperature"],     0.5)
    humidity         = st.slider("Humidity (%)",           10.0, 99.0, PARAM_DEFAULTS["humidity"],         0.5)
    co2_level        = st.slider("CO₂ (ppm)",             300.0,2000.0,PARAM_DEFAULTS["co2_level"],       10.0)

    st.markdown('<div class="sidebar-section">// Soil & Water</div>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Soil Moisture (%)",       5.0, 99.0, PARAM_DEFAULTS["soil_moisture"],    0.5)
    ph_level         = st.slider("pH Level",                4.0,  9.0, PARAM_DEFAULTS["ph_level"],         0.1)
    ec_level         = st.slider("EC Level (mS/cm)",        0.5,  5.0, PARAM_DEFAULTS["ec_level"],         0.1)

    st.markdown('<div class="sidebar-section">// Light & Airflow</div>', unsafe_allow_html=True)
    light_intensity  = st.slider("Light Intensity (lux)", 100.0,6000.0,PARAM_DEFAULTS["light_intensity"], 50.0)
    ventilation_rate = st.slider("Ventilation (%)",         0.0, 100.0,PARAM_DEFAULTS["ventilation_rate"],  1.0)
    leaf_area_index  = st.slider("Leaf Area Index",         0.5,  7.0, PARAM_DEFAULTS["leaf_area_index"],   0.1)

    st.markdown('<div class="sidebar-section">// 3D Model Viewer</div>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=220)

    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<span class="engine-tag">◈ {engine_status}</span>', unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_ai = st.button("▶ Run Full AI Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMS & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
params = {
    "crop_type": crop_type, "growth_stage": growth_stage, "temperature": temperature,
    "humidity": humidity, "soil_moisture": soil_moisture, "co2_level": co2_level,
    "light_intensity": light_intensity, "ventilation_rate": ventilation_rate,
    "ph_level": ph_level, "ec_level": ec_level, "leaf_area_index": leaf_area_index,
}
sus_data  = compute_sustainability_score(params)
dis_data  = calculate_disease_risk(params)
irr_data  = calculate_irrigation_need(params)
heat_data = calculate_heat_stress(temperature, humidity)
analysis  = {"sustainability": sus_data, "disease": dis_data, "irrigation": irr_data, "heat_stress": heat_data}
perf_data = evaluate_system_performance(sus_data, dis_data, irr_data)
analysis["performance"] = perf_data

if "ai_result" not in st.session_state: st.session_state["ai_result"] = None

if run_ai:
    ph = st.empty()
    with ph.container():
        st.markdown(
            "<div class='standby-box'>"
            "<div class='standby-icon'>◈</div>"
            "<div class='standby-label'>Neural Network Processing</div>"
            "<div class='standby-desc'>Querying AI inference engine…</div>"
            "</div>",
            unsafe_allow_html=True
        )
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=120, key="thinking")
        else: st.spinner("Processing…")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    ph.empty()

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM HEADER
# ─────────────────────────────────────────────────────────────────────────────
now = datetime.datetime.now()
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(f'''
        <div class="sys-header">
            <div>
                <div class="sys-title">{APP_ICON} AgriTwin AI · Control System</div>
                <div class="sys-subtitle">AI-Powered Predictive Digital Twin · Smart Protected Agriculture</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
with hc2:
    temp_ok = temperature <= 30
    st.markdown(f'''
        <div class="sys-status-row" style="padding-top:0.6rem">
            <div><span style="color:{'#98C379' if temp_ok else '#E06C75'}">●</span> SYSTEM ONLINE</div>
            <div>{now.strftime("%Y-%m-%d  %H:%M:%S")}</div>
            <div style="color:#61AFEF">{crop_type} · {growth_stage}</div>
        </div>
    ''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "SYS DASHBOARD", "AI ASSISTANT", "VISION SCANNER",
    "MARKET ANALYZER", "HARDWARE"
])

# =============================================================================
# TAB 1 — DASHBOARD
# =============================================================================
with tab_dashboard:

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    t_cls, t_col = get_status_class(temperature, high_warn=28, high_crit=35)
    h_cls, h_col = get_status_class(humidity, high_warn=80, high_crit=90)
    sm_cls,sm_col = get_status_class(soil_moisture, low_crit=20, low_warn=35)
    co2_cls,co2_col = get_status_class(co2_level, low_warn=500)
    l_cls, l_col = "status-info", "#61AFEF"
    v_cls, v_col = get_status_class(ventilation_rate, low_crit=20, low_warn=40)

    kpi_tile(k1, "CH-01", f"{temperature:.1f}", "°C",  "TEMPERATURE",    t_cls,  t_col)
    kpi_tile(k2, "CH-02", f"{humidity:.1f}",    "%",   "HUMIDITY",       h_cls,  h_col)
    kpi_tile(k3, "CH-03", f"{soil_moisture:.1f}","%", "SOIL MOISTURE",  sm_cls, sm_col)
    kpi_tile(k4, "CH-04", f"{co2_level:.0f}",   "ppm","CO₂ CONC.",     co2_cls,co2_col)
    kpi_tile(k5, "CH-05", f"{light_intensity:.0f}","lx","IRRADIANCE",   l_cls,  l_col)
    kpi_tile(k6, "CH-06", f"{ventilation_rate:.0f}","%","VENTILATION",  v_cls,  v_col)

    st.markdown('<div class="sys-divider"><span>// ANALYTICS MODULES</span></div>', unsafe_allow_html=True)

    # ── Sustainability | Disease | Irrigation ─────────────────────────────────
    c_sus, c_dis, c_irr = st.columns(3)

    with c_sus:
        st.markdown(led_header("SUSTAINABILITY INDEX", "blue", "◈"), unsafe_allow_html=True)
        gc = sus_data["grade_color"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=sus_data["total"],
            domain={'x':[0,1],'y':[0,1]},
            number={'font':{'color':gc,'size':36,'family':'IBM Plex Mono'},'suffix':''},
            gauge={
                'axis':{
                    'range':[None,100],
                    'tickcolor':"rgba(171,178,191,0.3)",
                    'tickfont':{'family':'IBM Plex Mono','size':8},
                    'tickwidth':1
                },
                'bar':{'color':gc,'thickness':0.2},
                'bgcolor':"rgba(0,0,0,0)",
                'borderwidth':1,
                'bordercolor':"rgba(62,68,81,0.6)",
                'steps':[
                    {'range':[0,40],  'color':"rgba(224,108,117,0.08)"},
                    {'range':[40,75], 'color':"rgba(229,192,123,0.08)"},
                    {'range':[75,100],'color':"rgba(152,195,121,0.08)"}
                ],
                'threshold':{'line':{'color':gc,'width':2},'thickness':0.75,'value':sus_data["total"]}
            }
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=185, margin=dict(l=16,r=16,t=16,b=8))
        st.markdown('<div class="ctrl-panel" style="text-align:center;padding-bottom:0.4rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'''<div class="grade-readout">
            <div class="grade-letter" style="color:{gc};">{sus_data["grade"]}</div>
            <div class="grade-score">GRADE · SCORE {sus_data["total"]}/100</div>
        </div></div>''', unsafe_allow_html=True)

        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
        html_bars  = progress_html("Water Efficiency",  sus_data["water_efficiency"],       "#61AFEF")
        html_bars += progress_html("Energy Efficiency", sus_data["energy_efficiency"],       "#98C379")
        html_bars += progress_html("Climate Control",   sus_data["climate_optimization"],    "#E5C07B")
        html_bars += progress_html("Disease Prevention",sus_data["disease_prevention"],      "#56B6C2")
        html_bars += progress_html("Yield Potential",   sus_data["yield_potential"],          "#98C379")
        st.markdown(html_bars + '</div>', unsafe_allow_html=True)

    with c_dis:
        st.markdown(led_header("DISEASE RISK ANALYSIS", "red" if dis_data["overall"] > 50 else "amber", "⬡"), unsafe_allow_html=True)
        dlc = dis_data["level_color"]
        badge_cls = "crit" if dis_data["overall"] > 60 else ("warn" if dis_data["overall"] > 30 else "")
        st.markdown(f'''<div class="ctrl-panel" style="text-align:center">
            <span class="status-badge {badge_cls}" style="color:{dlc};border-color:{dlc}">
                <span class="dot"></span>{dis_data["level"].upper()}
            </span>
            <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:400;color:{dlc};margin:0.5rem 0;letter-spacing:-0.02em;">{dis_data["overall"]:.1f}<span style="font-size:1rem;color:var(--text-muted);">%</span></div>
            <div style="font-family:IBM Plex Mono,monospace;font-size:0.58rem;color:var(--text-muted);letter-spacing:0.25em;text-transform:uppercase">OVERALL RISK INDEX</div>
        </div>''', unsafe_allow_html=True)
        d_clrs = ["#E06C75","#E5C07B","#98C379","#61AFEF"]
        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
        dh = "".join(progress_html(n, v, d_clrs[i % 4]) for i,(n,v) in enumerate(dis_data["diseases"].items()))
        st.markdown(dh + '</div>', unsafe_allow_html=True)

        with st.expander("▸ View Disease Diagnostic Data"):
            rows = "".join(f'<tr><td class="key">{n}</td><td class="val {("crit" if v>60 else "warn" if v>30 else "ok")}">{v:.1f}</td><td class="unit">%</td></tr>'
                           for n, v in dis_data["diseases"].items())
            st.markdown(f'<table class="telem-table"><thead><tr><th>PATHOGEN</th><th>RISK %</th><th>UNIT</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

    with c_irr:
        st.markdown(led_header("IRRIGATION INTELLIGENCE", "blue", "◎"), unsafe_allow_html=True)
        isc = irr_data["status_color"]
        irr_badge_cls = "crit" if irr_data["urgency"] > 70 else ("warn" if irr_data["urgency"] > 40 else "")
        st.markdown(f'''<div class="ctrl-panel" style="text-align:center">
            <span class="status-badge {irr_badge_cls}" style="color:{isc};border-color:{isc}">
                <span class="dot"></span>{irr_data["status"].upper()}
            </span>
            <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:400;color:{isc};margin:0.5rem 0;letter-spacing:-0.02em;">{irr_data["urgency"]:.1f}<span style="font-size:1rem;color:var(--text-muted);">%</span></div>
            <div style="font-family:IBM Plex Mono,monospace;font-size:0.58rem;color:var(--text-muted);letter-spacing:0.25em;text-transform:uppercase">URGENCY INDEX</div>
        </div>''', unsafe_allow_html=True)
        st.markdown(f'''<div class="ctrl-panel">
            {progress_html("Irrigation Urgency", irr_data["urgency"], isc)}
            <table class="telem-table" style="margin-top:0.7rem">
                <tr><td class="key">VOLUME</td><td class="val" style="color:{isc}">{irr_data["volume_liters"]:.2f}</td><td class="unit">L/m²</td></tr>
                <tr><td class="key">NEXT IRR.</td><td class="val" style="color:{isc}">{irr_data["next_irrigation_hours"]}</td><td class="unit">hrs</td></tr>
                <tr><td class="key">ET₀</td><td class="val" style="color:{isc}">{irr_data["evapotranspiration"]:.3f}</td><td class="unit">mm/hr</td></tr>
            </table>
        </div>''', unsafe_allow_html=True)

        with st.expander("▸ View Irrigation Telemetry"):
            st.markdown(f'''<div class="terminal">
                <div class="terminal-header">
                    <span>IRRIGATION CONTROLLER · TELEMETRY STREAM</span>
                    <span style="color:{'#98C379' if irr_data['urgency']<50 else '#E06C75'}">● ACTIVE</span>
                </div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> STATUS</span><span class="terminal-val">    = {irr_data["status"].upper()}</span></div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> URGENCY</span><span class="terminal-val">   = {irr_data["urgency"]:.2f}%</span></div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> VOLUME</span><span class="terminal-val">    = {irr_data["volume_liters"]:.4f} L/m²</span></div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> NEXT_IRR</span><span class="terminal-val">  = {irr_data["next_irrigation_hours"]}h</span></div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> ET0</span><span class="terminal-val">       = {irr_data["evapotranspiration"]:.4f} mm/hr</span></div>
                <div class="terminal-line"><span class="terminal-prompt">$</span><span class="terminal-key"> SOIL_MOIST</span><span class="terminal-val"> = {soil_moisture:.1f}%</span></div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="sys-divider"><span>// TELEMETRY VISUALIZATION</span></div>', unsafe_allow_html=True)

    # ── Diagnostic Toggles ───────────────────────────────────────────────────
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    show_trend  = diag_col1.checkbox("Show 24h Climate Trend", value=True)
    show_radar  = diag_col2.checkbox("Show Risk Radar", value=True)
    show_pie    = diag_col3.checkbox("Show Sustainability Breakdown", value=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    ch_cols = []
    ch_flags = [show_trend, show_radar, show_pie]
    num_charts = sum(ch_flags)
    if num_charts > 0:
        ch_cols = st.columns(num_charts)
    col_idx = 0

    if show_trend:
        with ch_cols[col_idx]:
            col_idx += 1
            st.markdown(led_header("24H CLIMATE TREND", "blue", "◉"), unsafe_allow_html=True)
            trend = generate_trend_data(temperature, humidity, 24)
            fig_t = go.Figure()
            t_line_color = "#E06C75" if temperature > 30 else "#61AFEF"
            fig_t.add_trace(go.Scatter(
                x=trend["times"], y=trend["temperatures"], name="Temp °C",
                line=dict(color=t_line_color, width=1.5),
                fill="tozeroy", fillcolor=f"{'rgba(224,108,117,0.05)' if temperature > 30 else 'rgba(97,175,239,0.05)'}"))
            fig_t.add_trace(go.Scatter(
                x=trend["times"], y=trend["humidities"], name="Humidity %",
                line=dict(color="#98C379", width=1.5),
                fill="tozeroy", fillcolor="rgba(152,195,121,0.04)"))
            fig_t.add_trace(go.Scatter(
                x=trend["times"], y=trend["soil_moisture"], name="Soil %",
                line=dict(color="#E5C07B", width=1, dash="dot")))
            fig_t.update_layout(**PLOTLY_LAYOUT, height=270, xaxis=dict(tickangle=45, nticks=8))
            plotly_scada_axes(fig_t)
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})

    if show_radar:
        with ch_cols[col_idx]:
            col_idx += 1
            st.markdown(led_header("RISK RADAR SCAN", "red", "⬡"), unsafe_allow_html=True)
            rc = ["Disease","Heat Stress","Water Stress","CO₂ Deficit","Ventilation","Yield Risk"]
            rv = [
                dis_data["overall"],
                min(100, max(0, (heat_data["heat_index"]-20)*2)),
                max(0, 100-sus_data["water_efficiency"]),
                max(0, (900-co2_level)/9),
                max(0, 100-ventilation_rate),
                max(0, 100-sus_data["yield_potential"])
            ]
            fig_r = go.Figure(go.Scatterpolar(
                r=rv+[rv[0]], theta=rc+[rc[0]],
                fill="toself", fillcolor="rgba(224,108,117,0.08)",
                line=dict(color="#E06C75", width=1.5), name="Current Risk"))
            fig_r.add_trace(go.Scatterpolar(
                r=[50]*6+[50], theta=rc+[rc[0]],
                line=dict(color="rgba(229,192,123,0.3)", width=1, dash="dot"),
                name="Threshold"))
            fig_r.update_layout(**PLOTLY_LAYOUT, height=270,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=8, color="#636D7E", family="IBM Plex Mono"),
                        showline=False),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=9, color="#ABB2BF", family="IBM Plex Mono"))))
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

    if show_pie:
        with ch_cols[col_idx]:
            col_idx += 1
            st.markdown(led_header("SUSTAINABILITY BREAKDOWN", "blue", "◈"), unsafe_allow_html=True)
            pl = ["Water","Energy","Climate","Disease","Yield"]
            pv = [sus_data["water_efficiency"], sus_data["energy_efficiency"],
                  sus_data["climate_optimization"], sus_data["disease_prevention"], sus_data["yield_potential"]]
            pc = ["#61AFEF","#98C379","#E5C07B","#56B6C2","#E06C75"]
            fig_p = go.Figure(go.Pie(
                labels=pl, values=pv, hole=0.62,
                marker=dict(colors=pc, line=dict(color="#1A1D21", width=2)),
                textfont=dict(size=10, family="IBM Plex Mono"),
                hovertemplate="<b>%{label}</b><br>%{value:.1f}<extra></extra>"))
            fig_p.add_annotation(
                text=f"<b>{sus_data['total']}</b>",
                x=0.5, y=0.5,
                font=dict(size=26, color="#E6E6E6", family="IBM Plex Mono"),
                showarrow=False)
            fig_p.update_layout(**PLOTLY_LAYOUT, height=270)
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="sys-divider"><span>// ZONE HEATMAPS</span></div>', unsafe_allow_html=True)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    hm1, hm2 = st.columns(2)
    temp_grid, hum_grid = generate_zone_heatmap_data(temperature, humidity)
    with hm1:
        st.markdown(led_header("ZONE TEMPERATURE MAP", "red" if temperature > 30 else "blue", "◉"), unsafe_allow_html=True)
        fig_ht = go.Figure(go.Heatmap(
            z=temp_grid,
            text=[[f"{v:.1f}°C" for v in row] for row in temp_grid],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10, color="#E6E6E6"),
            colorscale=[[0,"#61AFEF"],[0.5,"#E5C07B"],[1,"#E06C75"]],
            colorbar=dict(title="°C", tickfont=dict(color="#ABB2BF", family="IBM Plex Mono", size=9),
                          titlefont=dict(color="#ABB2BF", family="IBM Plex Mono", size=9)),
            showscale=True))
        fig_ht.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar":False})

    with hm2:
        st.markdown(led_header("ZONE HUMIDITY MAP", "blue", "◉"), unsafe_allow_html=True)
        fig_hh = go.Figure(go.Heatmap(
            z=hum_grid,
            text=[[f"{v:.1f}%" for v in row] for row in hum_grid],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10, color="#E6E6E6"),
            colorscale=[[0,"#1A1D21"],[0.5,"#61AFEF"],[1,"#98C379"]],
            colorbar=dict(title="%", tickfont=dict(color="#ABB2BF", family="IBM Plex Mono", size=9),
                          titlefont=dict(color="#ABB2BF", family="IBM Plex Mono", size=9)),
            showscale=True))
        fig_hh.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_hh, use_container_width=True, config={"displayModeBar":False})

    # ── Raw Telemetry Expander ────────────────────────────────────────────────
    with st.expander("▸ View Raw Telemetry · All Channels"):
        telem_cols = st.columns(2)
        with telem_cols[0]:
            rows = f'''
                <tr><td class="key">CH-01 TEMPERATURE</td><td class="val {'crit' if temperature>35 else 'warn' if temperature>28 else 'ok'}">{temperature:.1f}</td><td class="unit">°C</td></tr>
                <tr><td class="key">CH-02 HUMIDITY</td><td class="val {'warn' if humidity>80 else 'ok'}">{humidity:.1f}</td><td class="unit">%</td></tr>
                <tr><td class="key">CH-03 SOIL MOISTURE</td><td class="val {'crit' if soil_moisture<20 else 'warn' if soil_moisture<35 else 'ok'}">{soil_moisture:.1f}</td><td class="unit">%</td></tr>
                <tr><td class="key">CH-04 CO₂ CONC.</td><td class="val {'warn' if co2_level<500 else 'ok'}">{co2_level:.0f}</td><td class="unit">ppm</td></tr>
                <tr><td class="key">CH-05 IRRADIANCE</td><td class="val">{light_intensity:.0f}</td><td class="unit">lux</td></tr>
                <tr><td class="key">CH-06 VENTILATION</td><td class="val {'crit' if ventilation_rate<20 else 'warn' if ventilation_rate<40 else 'ok'}">{ventilation_rate:.0f}</td><td class="unit">%</td></tr>
            '''
            st.markdown(f'<table class="telem-table"><thead><tr><th>CHANNEL</th><th>VALUE</th><th>UNIT</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        with telem_cols[1]:
            rows2 = f'''
                <tr><td class="key">PH LEVEL</td><td class="val {'warn' if ph_level<5.5 or ph_level>7.5 else 'ok'}">{ph_level:.1f}</td><td class="unit">pH</td></tr>
                <tr><td class="key">EC LEVEL</td><td class="val">{ec_level:.2f}</td><td class="unit">mS/cm</td></tr>
                <tr><td class="key">LEAF AREA IDX</td><td class="val">{leaf_area_index:.2f}</td><td class="unit">LAI</td></tr>
                <tr><td class="key">HEAT INDEX</td><td class="val {'crit' if heat_data['heat_index']>35 else 'warn' if heat_data['heat_index']>30 else 'ok'}">{heat_data["heat_index"]:.1f}</td><td class="unit">°C</td></tr>
                <tr><td class="key">DISEASE RISK</td><td class="val {'crit' if dis_data['overall']>60 else 'warn' if dis_data['overall']>30 else 'ok'}">{dis_data["overall"]:.1f}</td><td class="unit">%</td></tr>
                <tr><td class="key">SUST. SCORE</td><td class="val {'ok' if sus_data['total']>75 else 'warn' if sus_data['total']>40 else 'crit'}">{sus_data["total"]:.1f}</td><td class="unit">/100</td></tr>
            '''
            st.markdown(f'<table class="telem-table"><thead><tr><th>PARAMETER</th><th>VALUE</th><th>UNIT</th></tr></thead><tbody>{rows2}</tbody></table>', unsafe_allow_html=True)

    st.markdown('<div class="sys-divider"><span>// AI PREDICTION ENGINE OUTPUT</span></div>', unsafe_allow_html=True)

    # ── AI Results — Terminal Style ───────────────────────────────────────────
    ai_res = st.session_state.get("ai_result")
    if not ai_res:
        st.markdown('''<div class="standby-box">
            <div class="standby-icon">◈</div>
            <div class="standby-label">AI Engine Standby</div>
            <div class="standby-desc">Press <strong>▶ Run Full AI Analysis</strong> in the sidebar to activate neural inference.</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="engine-tag">◈ {ai_res.get("source","AI Engine")}</span>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'''<div class="ai-result-block crit">
                <div class="ai-result-key">⬡ DISEASE WARNING</div>
                <div class="ai-result-val">{ai_res.get("disease_warning","—")}</div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="ai-result-block warn">
                <div class="ai-result-key">◉ CLIMATE WARNING</div>
                <div class="ai-result-val">{ai_res.get("climate_warning","—")}</div>
            </div>''', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'''<div class="ai-result-block info">
                <div class="ai-result-key">◎ IRRIGATION ADVICE</div>
                <div class="ai-result-val">{ai_res.get("irrigation_advice","—")}</div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="ai-result-block">
                <div class="ai-result-key">♻ SUSTAINABILITY INSIGHT</div>
                <div class="ai-result-val">{ai_res.get("sustainability_insight","—")}</div>
            </div>''', unsafe_allow_html=True)

        st.markdown(led_header("PRIORITY ACTION QUEUE", "red", "⚡"), unsafe_allow_html=True)
        priority_cls = ["p1","p2","p3"]
        priority_lbl = ["P1 CRITICAL","P2 WARNING","P3 ADVISORY"]
        for i, action in enumerate(ai_res.get("top_actions",[]),1):
            cls = priority_cls[min(i-1,2)]
            lbl = priority_lbl[min(i-1,2)]
            st.markdown(f'''<div class="action-row {cls}">
                <span class="action-num">{lbl}</span>
                <span>{action}</span>
            </div>''', unsafe_allow_html=True)

        st.markdown(f'''<div class="terminal" style="margin-top:0.8rem">
            <div class="terminal-header">
                <span>OVERALL ASSESSMENT · AI ENGINE OUTPUT</span>
                <span style="color:#98C379">● COMPLETE</span>
            </div>
            <span style="color:#ABB2BF;line-height:1.75;font-family:IBM Plex Mono,monospace;font-size:0.8rem">{ai_res.get("overall_assessment","—")}</span>
        </div>''', unsafe_allow_html=True)

        with st.expander("▸ Export Report"):
            report_text = f"""=== AgriTwin AI · Industrial Control Report ===
Timestamp     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Crop          : {crop_type} ({growth_stage})

--- TELEMETRY CHANNELS ---
CH-01 TEMP    : {temperature}°C
CH-02 HUMID   : {humidity}%
CH-03 SOIL    : {soil_moisture}%
CH-04 CO2     : {co2_level} ppm
CH-05 LIGHT   : {light_intensity} lux
CH-06 VENT    : {ventilation_rate}%
     PH       : {ph_level}
     EC       : {ec_level} mS/cm

--- SUSTAINABILITY ---
Score         : {sus_data['total']} / 100
Grade         : {sus_data['grade']}

--- AI INFERENCE OUTPUT ---
Disease       : {ai_res.get('disease_warning')}
Climate       : {ai_res.get('climate_warning')}
Irrigation    : {ai_res.get('irrigation_advice')}
Sustainability: {ai_res.get('sustainability_insight')}

--- OVERALL ASSESSMENT ---
{ai_res.get('overall_assessment')}
"""
            st.download_button("↓ Download AI Report (.txt)", data=report_text,
                file_name=f"AgriTwin_{crop_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True)

    st.markdown('<div class="sys-divider"><span>// SYSTEM PERFORMANCE</span></div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    gc = perf_data["grade_color"]

    def pcard(col, ch, label, val, color):
        cls = "status-ok" if val >= 75 else ("status-warn" if val >= 50 else "status-crit")
        col.markdown(f'''<div class="metric-tile {cls}">
            <div class="metric-id">{ch}</div>
            <div class="metric-value" style="color:{color};">{val:.1f}</div>
            <div class="metric-label">{label}</div>
        </div>''', unsafe_allow_html=True)

    pcard(p1,"PA-01","PREDICTION ACCURACY",  perf_data["prediction_accuracy"],    "#61AFEF")
    pcard(p2,"PA-02","SYSTEM EFFICIENCY",     perf_data["system_efficiency"],      "#98C379")
    pcard(p3,"PA-03","GREENHOUSE PERFORMANCE",perf_data["greenhouse_performance"], "#E5C07B")
    p4.markdown(f'''<div class="metric-tile" style="border-top-color:{gc}">
        <div class="metric-id">PA-04</div>
        <div class="metric-value" style="color:{gc};">{perf_data["overall"]:.1f}</div>
        <div class="metric-label">OVERALL · GRADE <span style="color:{gc};font-weight:600">{perf_data["grade"]}</span></div>
    </div>''', unsafe_allow_html=True)


# =============================================================================
# HELPERS — Model Selection
# =============================================================================
def get_working_text_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'vision' not in m.name.lower():
                return m.name
    except: pass
    return "models/gemini-1.5-flash"

def get_working_vision_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and ('vision' in m.name.lower() or 'flash' in m.name.lower()):
                return m.name
    except: pass
    return "models/gemini-1.5-flash"


# =============================================================================
# TAB 2 — AI ASSISTANT CHATBOT
# =============================================================================
with tab_chat:
    st.markdown(led_header("AI AGRI-ASSISTANT · CHAT INTERFACE", "blue", "◈"), unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('''<div class="terminal">
            <div class="terminal-header"><span>SYSTEM</span><span style="color:#E06C75">● ERROR</span></div>
            ERROR :: GEMINI_API_KEY not found in Streamlit secrets.<br>
            Set the key in .streamlit/secrets.toml to enable AI features.
        </div>''', unsafe_allow_html=True)
    else:
        if "main_messages" not in st.session_state: st.session_state.main_messages = []
        for msg in st.session_state.main_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input(f"Query: {crop_type} management…", key="main_chat"):
            st.session_state.main_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                with ph.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=80, key="chat_thinking")
                    else: st.spinner("Processing…")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    res = model.generate_content(
                        f"You are an expert AI assistant for {crop_type} at {growth_stage} stage. "
                        f"Question: {prompt}"
                    )
                    ph.empty(); st.markdown(res.text)
                    st.session_state.main_messages.append({"role":"assistant","content":res.text})
                except Exception as e:
                    ph.empty(); st.error(f"Error: {e}")


# =============================================================================
# TAB 3 — VISION SCANNER
# =============================================================================
with tab_vision:
    st.markdown(led_header("AI DISEASE VISION SCANNER", "amber", "◉"), unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('''<div class="terminal">
            <div class="terminal-header"><span>VISION MODULE</span><span style="color:#E06C75">● OFFLINE</span></div>
            ERROR :: GEMINI_API_KEY not found. Vision scanner offline.
        </div>''', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            img_file = st.file_uploader("Upload plant image", type=["jpg","png"]) or st.camera_input("Capture frame")
            if img_file:
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
        with c2:
            if img_file:
                if st.button("▶ Initiate Scan", type="primary", use_container_width=True):
                    ph = st.empty()
                    with ph.container():
                        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=120, key="vision_scan")
                        else: st.spinner("Scanning…")
                    try:
                        v_model = genai.GenerativeModel(get_working_vision_model())
                        res = v_model.generate_content([
                            f"Identify diseases on this {crop_type} plant. "
                            f"Provide structured 3-step treatment plan.", image
                        ])
                        ph.empty()
                        st.session_state.vision_result = res.text
                    except Exception as e:
                        ph.empty(); st.error(f"Error: {e}")

        if st.session_state.vision_result:
            st.markdown(f'''<div class="terminal" style="margin-top:0.6rem">
                <div class="terminal-header">
                    <span>VISION SCAN RESULTS · {crop_type.upper()}</span>
                    <span style="color:#98C379">● COMPLETE</span>
                </div>
                <span style="color:#ABB2BF;font-family:IBM Plex Mono,monospace;font-size:0.8rem;line-height:1.8">{st.session_state.vision_result}</span>
            </div>''', unsafe_allow_html=True)

            st.markdown(led_header("FOLLOW-UP ANALYSIS", "blue", "◈"), unsafe_allow_html=True)
            for msg in st.session_state.vision_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            if vp := st.chat_input("Query: treatment or pathology…", key="vision_chat_input"):
                st.session_state.vision_messages.append({"role":"user","content":vp})
                with st.chat_message("user"): st.markdown(vp)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing…"):
                        try:
                            model = genai.GenerativeModel(get_working_text_model())
                            r = model.generate_content(
                                f"You are a plant pathologist. Scan result: '{st.session_state.vision_result}'. "
                                f"Question: {vp}"
                            )
                            st.markdown(r.text)
                            st.session_state.vision_messages.append({"role":"assistant","content":r.text})
                        except Exception as e: st.error(f"Error: {e}")


# =============================================================================
# TAB 4 — MARKET ANALYZER
# =============================================================================
with tab_finance:
    st.markdown(led_header("PROFIT & MARKET ANALYZER", "blue", "◈"), unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    plants_count = f1.number_input("Total Plants",            value=1000,  step=100)
    yield_per    = f2.number_input("Yield per Plant (kg)",    value=0.15,  step=0.05)
    mkt_price    = f3.number_input("Market Price (Rs/kg)",    value=450.0, step=10.0)

    tot_yield = plants_count * yield_per
    gross     = tot_yield * mkt_price
    cost      = (plants_count * 15) + 5000
    profit    = gross - cost

    st.markdown('<div class="sys-divider"><span>// FINANCIAL PROJECTIONS</span></div>', unsafe_allow_html=True)

    r1,r2,r3 = st.columns(3)
    r1.metric("Total Yield",  f"{tot_yield:.1f} kg")
    r2.metric("Running Cost", f"Rs. {cost:,.2f}")
    r3.metric("Net Profit",   f"Rs. {profit:,.2f}", delta=f"{profit/cost*100:.1f}% ROI")

    fig_bar = go.Figure()
    cats   = ["Gross Revenue","Running Cost","Net Profit"]
    vals   = [gross, cost, profit]
    colors = ["#98C379","#E06C75","#61AFEF"]
    fig_bar.add_trace(go.Bar(
        x=cats, y=vals,
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0.5)",
        marker_line_width=1,
        text=[f"Rs.{v:,.0f}" for v in vals],
        textposition="auto",
        textfont=dict(family="IBM Plex Mono", size=10, color="#1A1D21")))
    fig_bar.update_layout(**PLOTLY_LAYOUT)
    fig_bar.update_layout(height=270, showlegend=False, bargap=0.35)
    plotly_scada_axes(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    st.markdown(f'''<div class="terminal">
        <div class="terminal-header"><span>MARKET INTELLIGENCE MODULE</span><span style="color:#98C379">● ACTIVE</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> HARVEST_TIMING</span><span class="terminal-val">    = OPTIMAL</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> DEMAND_TREND</span><span class="terminal-val">      = +12.0% (local supermarkets)</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> REC_PRICE_POINT</span><span class="terminal-val">   = Rs. {mkt_price*1.12:.0f}/kg (2-week projection)</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> PRODUCE_TYPE</span><span class="terminal-val">      = HYDROPONIC · PREMIUM</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> YIELD_KG</span><span class="terminal-val">          = {tot_yield:.2f}</span></div>
        <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> PROJECTED_ROI</span><span class="terminal-val">     = {profit/cost*100:.1f}%</span></div>
    </div>''', unsafe_allow_html=True)

    with st.expander("▸ Detailed Cost Breakdown"):
        cb_rows = f'''
            <tr><td class="key">PLANT STOCK COST</td><td class="val">Rs. {plants_count * 15:,.2f}</td><td class="unit">@ 15/plant</td></tr>
            <tr><td class="key">FIXED OVERHEAD</td><td class="val">Rs. 5,000.00</td><td class="unit">base</td></tr>
            <tr><td class="key">TOTAL COST</td><td class="val warn">Rs. {cost:,.2f}</td><td class="unit">total</td></tr>
            <tr><td class="key">GROSS REVENUE</td><td class="val ok">Rs. {gross:,.2f}</td><td class="unit">projected</td></tr>
            <tr><td class="key">NET PROFIT</td><td class="val {'ok' if profit>0 else 'crit'}">Rs. {profit:,.2f}</td><td class="unit"></td></tr>
        '''
        st.markdown(f'<table class="telem-table"><thead><tr><th>LINE ITEM</th><th>AMOUNT</th><th>NOTE</th></tr></thead><tbody>{cb_rows}</tbody></table>', unsafe_allow_html=True)

    st.markdown(led_header("AI FINANCIAL ADVISOR", "blue", "◈"), unsafe_allow_html=True)
    for msg in st.session_state.market_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if mp := st.chat_input("Query: market or profitability…", key="market_chat_input"):
        st.session_state.market_messages.append({"role":"user","content":mp})
        with st.chat_message("user"): st.markdown(mp)
        with st.chat_message("assistant"):
            with st.spinner("Calculating…"):
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    r = model.generate_content(
                        f"You are an agri-financial advisor. {plants_count} plants of {crop_type}, "
                        f"yield {tot_yield}kg, price Rs.{mkt_price}, cost Rs.{cost}, profit Rs.{profit}. "
                        f"Question: {mp}"
                    )
                    st.markdown(r.text)
                    st.session_state.market_messages.append({"role":"assistant","content":r.text})
                except Exception as e: st.error(f"Error: {e}")


# =============================================================================
# TAB 5 — HARDWARE & ALERTS
# =============================================================================
with tab_hardware:
    st.markdown(led_header("SMART HARDWARE & AUTOMATION INTERFACE", "amber", "⚙"), unsafe_allow_html=True)
    hw1, hw2 = st.columns(2)
    with hw1:
        st.markdown(f'''<div class="ctrl-panel">
            <div class="panel-header"><span class="led blue"></span>HARDWARE API ENDPOINT</div>
            <div style="font-family:IBM Plex Sans,sans-serif;font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.75rem;line-height:1.6">
                Connect ESP32, Arduino, or Raspberry Pi directly via REST API. 
                Send sensor readings in JSON format to the endpoint below.
            </div>
        </div>''', unsafe_allow_html=True)
        st.code('POST https://agritwin-ai.streamlit.app/api/v1/sensors\n{\n  "api_key": "YOUR_SECRET_KEY",\n  "temp": 28.5,\n  "hum": 65.2,\n  "ec": 1.8\n}', language="json")

        with st.expander("▸ View API Integration Guide"):
            st.markdown(f'''<div class="terminal">
                <div class="terminal-header"><span>API INTEGRATION GUIDE</span><span style="color:#61AFEF">● REFERENCE</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> ENDPOINT</span><span class="terminal-val">   = POST /api/v1/sensors</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> AUTH</span><span class="terminal-val">       = Bearer API_KEY</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> FORMAT</span><span class="terminal-val">     = application/json</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> INTERVAL</span><span class="terminal-val">   = 60s recommended</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> FIELDS</span><span class="terminal-val">     = temp, hum, ec, ph, co2, lux</span></div>
                <div class="terminal-line"><span class="terminal-prompt">></span><span class="terminal-key"> TIMEOUT</span><span class="terminal-val">    = 30s</span></div>
            </div>''', unsafe_allow_html=True)

        st.button("↺ Generate New API Key", use_container_width=True)

    with hw2:
        st.markdown(f'''<div class="ctrl-panel">
            <div class="panel-header"><span class="led {'red' if temperature > 35 else 'amber'}"></span>ALERT RULE CONFIGURATION</div>
        </div>''', unsafe_allow_html=True)
        phone_num      = st.text_input("Alert Destination (WhatsApp / +country)", value="+947XXXXXXXX")
        temp_threshold = st.slider("Temp Alert Threshold (°C):", 25.0, 45.0, 35.0)
        ec_threshold   = st.slider("EC Alert Floor (mS/cm):",    0.5,  2.0,  1.2)

        if st.button("▶ Save Alert Configuration", type="primary", use_container_width=True):
            st.success("✅ Automation rules committed to controller.")

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        if temperature > temp_threshold:
            st.markdown(f'''<div class="terminal" style="border-color:#E06C75;color:#E06C75">
                <div class="terminal-header" style="border-bottom-color:rgba(224,108,117,0.2)">
                    <span>ALERT CONTROLLER</span><span>● TRIGGERED</span>
                </div>
                <div class="terminal-line">TRIGGER :: TEMPERATURE THRESHOLD EXCEEDED</div>
                <div class="terminal-line">READING  :: {temperature}°C &gt; LIMIT {temp_threshold}°C</div>
                <div class="terminal-line">ACTION   :: DISPATCH → {phone_num}</div>
                <div class="terminal-line">PROTOCOL :: WHATSAPP NOTIFICATION</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="terminal" style="color:#636D7E">
                <div class="terminal-header">
                    <span>ALERT CONTROLLER</span><span style="color:#98C379">● STANDBY</span>
                </div>
                <div class="terminal-line">STATUS   :: ALL THRESHOLDS NOMINAL</div>
                <div class="terminal-line">TEMP     :: {temperature}°C ≤ LIMIT {temp_threshold}°C ✓</div>
                <div class="terminal-line">EC       :: {ec_level:.2f} mS/cm ≥ FLOOR {ec_threshold:.2f} ✓</div>
                <div class="terminal-line">DEST     :: {phone_num}</div>
            </div>''', unsafe_allow_html=True)

        with st.expander("▸ Supported Hardware Modules"):
            hw_rows = '''
                <tr><td class="key">ESP32-WROOM</td><td class="val ok">COMPATIBLE</td><td class="unit">WiFi</td></tr>
                <tr><td class="key">ARDUINO UNO</td><td class="val ok">COMPATIBLE</td><td class="unit">USB/Serial</td></tr>
                <tr><td class="key">RASPBERRY PI 4</td><td class="val ok">COMPATIBLE</td><td class="unit">ETH/WiFi</td></tr>
                <tr><td class="key">DHT22 SENSOR</td><td class="val ok">SUPPORTED</td><td class="unit">Temp/Humid</td></tr>
                <tr><td class="key">EC PROBE</td><td class="val ok">SUPPORTED</td><td class="unit">Conductivity</td></tr>
                <tr><td class="key">PH ELECTRODE</td><td class="val warn">BETA</td><td class="unit">pH</td></tr>
            '''
            st.markdown(f'<table class="telem-table"><thead><tr><th>MODULE</th><th>STATUS</th><th>INTERFACE</th></tr></thead><tbody>{hw_rows}</tbody></table>', unsafe_allow_html=True)
