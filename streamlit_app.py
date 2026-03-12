import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import warnings
warnings.filterwarnings('ignore')

# ── Global Matplotlib Style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.facecolor': '#070b14',
    'axes.facecolor': '#0d1220',
    'axes.edgecolor': 'none',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#1a2540',
    'axes.grid': False,
})

st.set_page_config(page_title="March Madness · 2026 Predictor", layout="wide", initial_sidebar_state="expanded")

# ── CSS Injection ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

/* Base */
.stApp {
    background-color: #070b14;
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(255,107,0,0.07) 0%, transparent 40%),
        radial-gradient(ellipse at 85% 100%, rgba(220,38,38,0.05) 0%, transparent 40%);
    font-family: 'Barlow', sans-serif;
}
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

/* Headings */
h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important; font-size: 2.6rem !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    background: linear-gradient(135deg, #ffffff 0%, #ffa500 55%, #ff6b00 100%);
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; line-height: 1.1 !important; margin-bottom: 0 !important;
}
h2 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; font-size: 1.4rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #e2e8f0 !important;
    border-bottom: 1px solid rgba(255,165,0,0.25) !important;
    padding-bottom: 0.35rem !important; margin-bottom: 1rem !important;
}
h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; color: #ffa500 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid rgba(255,165,0,0.2) !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #64748b !important; transition: color 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #ffa500 !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ffa500 0%, #ff6b00 100%) !important;
    color: #000 !important; font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 6px !important;
    padding: 0.55rem 1.8rem !important;
    box-shadow: 0 4px 18px rgba(255,165,0,0.28) !important;
    transition: all 0.18s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 22px rgba(255,165,0,0.42) !important;
}
/* Secondary button */
.stButton > button:not([kind="primary"]) {
    background: transparent !important; color: #94a3b8 !important;
    border: 1px solid rgba(148,163,184,0.28) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; border-radius: 6px !important;
    transition: all 0.18s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #ffa500 !important; color: #ffa500 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1220, #111827) !important;
    border: 1px solid rgba(255,165,0,0.22) !important;
    border-radius: 10px !important; padding: 1rem 1.2rem !important;
    transition: border-color 0.2s, transform 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255,165,0,0.5) !important; transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Barlow Condensed', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 700 !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.9rem !important; font-weight: 700 !important; color: #ffa500 !important;
}

/* Progress bar */
.stProgress > div > div { background: #1e293b !important; border-radius: 4px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ffa500, #ff6b00) !important; border-radius: 4px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #ffa500 !important; border-color: #ffa500 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1220 !important;
    border: 2px dashed rgba(255,165,0,0.3) !important;
    border-radius: 10px !important; transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(255,165,0,0.6) !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0d1220 !important; border: 1px solid rgba(255,165,0,0.2) !important;
    border-radius: 6px !important; color: #e2e8f0 !important;
}

/* Code blocks */
.stCode, code, pre {
    background: #060a12 !important; border: 1px solid rgba(255,165,0,0.12) !important;
    border-radius: 8px !important; color: #4ade80 !important; font-size: 0.82rem !important;
}

/* Alert/info */
[data-testid="stInfo"] {
    background: rgba(255,165,0,0.07) !important; border-left: 3px solid #ffa500 !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.07) !important; border-left: 3px solid #f59e0b !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stSuccess"] {
    background: rgba(34,197,94,0.07) !important; border-left: 3px solid #22c55e !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stError"] {
    background: rgba(220,38,38,0.07) !important; border-left: 3px solid #dc2626 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #ffa500 !important; }

/* Dividers */
hr { border-color: rgba(255,165,0,0.12) !important; }

/* Caption */
[data-testid="stCaptionContainer"] p { color: #475569 !important; font-size: 0.78rem !important; }

/* Hide chrome */
#MainMenu, footer, header { visibility: hidden; }

/* SVG icon utility */
.mm-icon { display:inline-flex; align-items:center; justify-content:center; }
.mm-icon svg { display:block; }
</style>
""", unsafe_allow_html=True)

# ── SVG Icon Library ──────────────────────────────────────────────────────
def svg_icon(key, size=20, color="currentColor"):
    """Return an inline SVG string for use in st.markdown HTML."""
    icons = {
        "ball": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M4.93 4.93c4.24 4.24 9.9 4.24 14.14 0"/>
            <path d="M4.93 19.07c4.24-4.24 9.9-4.24 14.14 0"/>
            <line x1="2" y1="12" x2="22" y2="12"/>
            <line x1="12" y1="2" x2="12" y2="22"/>
        </svg>''',
        "upload": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>''',
        "settings": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06
                     a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09
                     A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83
                     l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09
                     A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83
                     l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09
                     a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83
                     l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09
                     a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>''',
        "versus": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="16 3 21 3 21 8"/>
            <line x1="4" y1="20" x2="21" y2="3"/>
            <polyline points="21 16 21 21 16 21"/>
            <line x1="15" y1="15" x2="21" y2="21"/>
        </svg>''',
        "trophy": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>
            <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>
            <path d="M4 22h16"/>
            <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>
            <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>
            <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>
        </svg>''',
        "dice": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="2" width="20" height="20" rx="3" ry="3"/>
            <circle cx="8" cy="8" r="1.2" fill="{c}" stroke="none"/>
            <circle cx="16" cy="8" r="1.2" fill="{c}" stroke="none"/>
            <circle cx="8" cy="16" r="1.2" fill="{c}" stroke="none"/>
            <circle cx="16" cy="16" r="1.2" fill="{c}" stroke="none"/>
            <circle cx="12" cy="12" r="1.2" fill="{c}" stroke="none"/>
        </svg>''',
        "rocket": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91
                     a2.18 2.18 0 0 0-2.91-.09z"/>
            <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11
                     a22.35 22.35 0 0 1-4 2z"/>
            <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/>
            <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>
        </svg>''',
        "calendar": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
            <line x1="16" y1="2" x2="16" y2="6"/>
            <line x1="8" y1="2" x2="8" y2="6"/>
            <line x1="3" y1="10" x2="21" y2="10"/>
        </svg>''',
        "check": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"/>
        </svg>''',
        "x": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>''',
        "warning": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3
                     L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>''',
        "clock": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
        </svg>''',
        "medal1": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="15" r="7"/>
            <polyline points="8.56 2.75 4.5 9 8 9 9.5 12 12 5 14.5 12 16 9 19.5 9 15.44 2.75"/>
            <text x="12" y="19" text-anchor="middle" font-size="7" font-weight="800"
                  stroke="none" fill="{c}" font-family="sans-serif">1</text>
        </svg>''',
        "chevron_up": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="18 15 12 9 6 15"/>
        </svg>''',
    }
    path = icons.get(key, icons["ball"])
    filled = path.replace("{c}", color)
    return f'<span class="mm-icon" style="width:{size}px;height:{size}px;">{filled}</span>'

def svg_btn_icon(key, size=16):
    """SVG for use inside button labels (black text context)."""
    return svg_icon(key, size=size, color="#000000")

def svg_header_icon(key, size=24):
    """SVG for page header icon boxes."""
    return svg_icon(key, size=size, color="#ffffff")


# ── Page Header Helper ────────────────────────────────────────────────────
def page_header(icon_html, title, subtitle, accent="#ffa500"):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:28px;
                padding-bottom:20px; border-bottom:1px solid rgba(255,165,0,0.12);">
        <div style="background:linear-gradient(135deg,{accent},{accent}99);
                    width:48px; height:48px; border-radius:12px; display:flex;
                    align-items:center; justify-content:center;
                    box-shadow:0 4px 16px {accent}44; flex-shrink:0;
                    padding:10px;">{icon_html}</div>
        <div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.9rem;
                        font-weight:800; letter-spacing:0.05em; text-transform:uppercase;
                        background:linear-gradient(135deg,#fff 0%,{accent} 60%,#ff6b00 100%);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; line-height:1.05;">{title}</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.72rem;
                        color:#475569; letter-spacing:0.16em; text-transform:uppercase;
                        margin-top:2px;">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Claude Analysis Helper ────────────────────────────────────────────────
def get_analysis(prompt: str) -> str:
    """Call Claude Haiku with web search. Returns plain text analysis."""
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        return " ".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text
        ).strip() or "No analysis returned."
    except KeyError:
        return "Add ANTHROPIC_API_KEY to your Streamlit secrets to enable analysis."
    except Exception as e:
        return f"Analysis unavailable — {e}"


def _gs(obj, key):
    """Safe stat getter for dict or pandas Series."""
    return obj.get(key, 0) if isinstance(obj, dict) else getattr(obj, key, 0)


def h2h_analysis_prompt(team1, team2, prob_t1, seed1, seed2, s1, s2, auc):
    """Build the H2H matchup analysis prompt."""
    favored = team1 if prob_t1 >= 0.5 else team2
    fav_pct = f"{max(prob_t1, 1 - prob_t1):.0%}"
    s1_str = f"#{int(seed1)}" if seed1 else "unseeded"
    s2_str = f"#{int(seed2)}" if seed2 else "unseeded"
    return f"""You are a concise college basketball analyst for the 2026 NCAA Tournament.

Matchup: {team1} ({s1_str} seed) vs {team2} ({s2_str} seed)
Model prediction: {favored} favored at {fav_pct} win probability (LightGBM, AUC {auc:.3f})

Key stats ({team1} vs {team2}):
- Win %: {_gs(s1,'win_pct'):.0%} vs {_gs(s2,'win_pct'):.0%}
- Scoring margin: {_gs(s1,'margin'):+.1f} vs {_gs(s2,'margin'):+.1f} pts/game
- Offensive efficiency: {_gs(s1,'off_efficiency'):.1f} vs {_gs(s2,'off_efficiency'):.1f}
- Defensive efficiency: {_gs(s1,'def_efficiency'):.1f} vs {_gs(s2,'def_efficiency'):.1f}

Use web search to find current analyst, bracketologist, and ESPN/CBS Sports predictions for this matchup. Write exactly 3-4 sentences: first explain why the model favors {favored} based on the stats, then summarize what experts are currently saying. Be specific and direct — no filler."""


def team_analysis_prompt(name, seed, prob, n_sims, s, auc):
    """Build the championship contender analysis prompt."""
    s_str = f"#{int(seed)}" if seed else "unseeded"
    return f"""You are a concise college basketball analyst for the 2026 NCAA Tournament.

Team: {name} ({s_str} seed)
Model championship probability: {prob:.1%} across {n_sims:,} Monte Carlo simulations (AUC {auc:.3f})

Key stats:
- Win %: {_gs(s,'win_pct'):.0%}
- Scoring margin: {_gs(s,'margin'):+.1f} pts/game
- Offensive efficiency: {_gs(s,'off_efficiency'):.1f}
- Defensive efficiency: {_gs(s,'def_efficiency'):.1f}

Use web search to find what analysts, bracketologists, and ESPN/CBS Sports are currently saying about {name}'s tournament chances. Write exactly 3-4 sentences: explain what makes {name} a title contender based on the stats, then summarize current expert opinion. Be specific and direct — no filler."""


def render_analysis_card(text: str):
    """Render the analysis result in a styled card."""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(255,165,0,0.05),rgba(255,107,0,0.03));
                border:1px solid rgba(255,165,0,0.25); border-left:3px solid #ffa500;
                border-radius:8px; padding:16px 20px; margin-top:16px;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; font-weight:700;
                    letter-spacing:0.18em; color:#ffa500; margin-bottom:10px;">ANALYST INTELLIGENCE</div>
        <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.65;">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "uploaded_data"
MODEL_PATH = "trained_model.txt"
DATA_PKL_PATH = "app_data.pkl"
os.makedirs(DATA_DIR, exist_ok=True)

REQUIRED_KAGGLE = [
    "MRegularSeasonDetailedResults.csv",
    "MNCAATourneyDetailedResults.csv",
    "MTeams.csv",
    "MNCAATourneySeeds.csv",
    "MTeamConferences.csv",
]
TORVIK_FILES = ["cbb.csv", "cbb26.csv"]

BEST_PARAMS = {
    'n_estimators': 181,
    'max_depth': 3,
    'learning_rate': 0.010705,
    'subsample': 0.888,
    'colsample_bytree': 0.852,
    'min_child_samples': 21,
    'reg_alpha': 0.000247,
    'reg_lambda': 0.032461,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

BRACKET_2026 = [
    # EAST
    "Duke", "American Univ",
    "Mississippi St", "Boise St",
    "Wisconsin", "Montana St",
    "Kansas", "Howard",
    "Michigan St", "Bryant",
    "St John's", "NE Omaha",
    "Memphis", "Colorado St",
    "Auburn", "Alabama St",
    # WEST
    "Florida", "Norfolk St",
    "Connecticut", "New Hampshire",
    "Gonzaga", "McNeese St",
    "Arizona", "Akron",
    "Marquette", "Vermont",
    "Texas Tech", "NC Wilmington",
    "Missouri", "Drake",
    "Houston", "SIUE",
    # SOUTH
    "Tennessee", "Winthrop",
    "Michigan", "CS Bakersfield",
    "Iowa St", "Lipscomb",
    "Alabama", "Mt St Mary's",
    "BYU", "VCU",
    "Oregon", "Liberty",
    "Kentucky", "Troy",
    "Iowa", "S Dakota St",
    # MIDWEST
    "Kansas St", "Montana",
    "Purdue", "TX Southern",
    "Baylor", "Colgate",
    "Maryland", "Grand Canyon",
    "Clemson", "New Mexico St",
    "UCLA", "UNC Asheville",
    "Mississippi", "Yale",
    "Texas A&M", "Morehead St",
]

# ── Sidebar ───────────────────────────────────────────────────────────────
kaggle_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_KAGGLE)
torvik_ok = any(os.path.exists(os.path.join(DATA_DIR, f)) for f in TORVIK_FILES)
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(DATA_PKL_PATH)

st.sidebar.markdown("""
<div style="padding:16px 4px 20px 4px; border-bottom:1px solid rgba(255,165,0,0.15); margin-bottom:16px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="background:linear-gradient(135deg,#ffa500,#ff6b00); width:36px; height:36px;
                    border-radius:8px; display:flex; align-items:center; justify-content:center;
                    padding:6px; box-shadow:0 3px 10px rgba(255,165,0,0.35);"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="width:20px;height:20px;display:block;"><circle cx="12" cy="12" r="10"/><path d="M4.93 4.93c4.24 4.24 9.9 4.24 14.14 0"/><path d="M4.93 19.07c4.24-4.24 9.9-4.24 14.14 0"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/></svg></div>
        <div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-weight:800;
                        font-size:1.05rem; letter-spacing:0.08em; color:#e2e8f0;">MARCH MADNESS</div>
            <div style="font-size:0.65rem; color:#475569; letter-spacing:0.1em;">2026 PREDICTOR</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "Data Upload",
    "Train Model",
    "Head to Head",
    "Bracket Simulator",
], label_visibility="collapsed")

_gc = lambda ok, warn=False: '#22c55e' if ok else ('#f59e0b' if warn else '#dc2626')
st.sidebar.markdown(f"""
<div style="margin-top:20px; padding:14px; background:#0d1220; border:1px solid rgba(255,255,255,0.06);
            border-radius:10px;">
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; font-weight:700;
                letter-spacing:0.18em; color:#334155; margin-bottom:10px;">SYSTEM STATUS</div>
    <div style="font-size:0.82rem; margin-bottom:6px; color:#94a3b8;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                     background:{_gc(kaggle_ok)};margin-right:8px;vertical-align:middle;"></span>Kaggle Data</div>
    <div style="font-size:0.82rem; margin-bottom:6px; color:#94a3b8;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                     background:{_gc(torvik_ok, warn=True)};margin-right:8px;vertical-align:middle;"></span>Torvik Data</div>
    <div style="font-size:0.82rem; color:#94a3b8;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                     background:{_gc(model_ok)};margin-right:8px;vertical-align:middle;"></span>ML Model</div>
</div>
""", unsafe_allow_html=True)


# ── Helper: load trained model + data ─────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    if not model_ok:
        return None, None
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(DATA_PKL_PATH, 'rb') as f:
        d = pickle.load(f)
    return model, d


# ── Feature engineering ────────────────────────────────────────────────────
def build_team_stats(season_df, teams_df, conf_df, torvik_df=None):
    log = []
    log.append("Building team stats...")

    wins = season_df.groupby(['Season', 'WTeamID']).agg(
        W=('WTeamID', 'count'),
        pts_for=('WScore', 'mean'),
        pts_against=('LScore', 'mean'),
        fg_pct=('WFGM', lambda x: x.sum() / season_df.loc[x.index, 'WFGA'].sum()),
        fg3_pct=('WFGM3', lambda x: x.sum() / season_df.loc[x.index, 'WFGA3'].sum()),
        reb=('WOR', lambda x: (x + season_df.loc[x.index, 'WDR']).mean()),
        ast=('WAst', 'mean'),
        to=('WTO', 'mean'),
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})

    losses = season_df.groupby(['Season', 'LTeamID']).agg(
        L=('LTeamID', 'count'),
        pts_for_l=('LScore', 'mean'),
        pts_against_l=('WScore', 'mean'),
        fg_pct_l=('LFGM', lambda x: x.sum() / season_df.loc[x.index, 'LFGA'].sum()),
        fg3_pct_l=('LFGM3', lambda x: x.sum() / season_df.loc[x.index, 'LFGA3'].sum()),
        reb_l=('LOR', lambda x: (x + season_df.loc[x.index, 'LDR']).mean()),
        ast_l=('LAst', 'mean'),
        to_l=('LTO', 'mean'),
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    stats = wins.merge(losses, on=['Season', 'TeamID'], how='outer').fillna(0)
    stats['G'] = stats['W'] + stats['L']
    stats['win_pct'] = stats['W'] / stats['G'].clip(lower=1)
    stats['pts_for'] = (stats['pts_for'] * stats['W'] + stats['pts_for_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['pts_against'] = (stats['pts_against'] * stats['W'] + stats['pts_against_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['margin'] = stats['pts_for'] - stats['pts_against']
    stats['fg_pct'] = (stats['fg_pct'] * stats['W'] + stats['fg_pct_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['fg3_pct'] = (stats['fg3_pct'] * stats['W'] + stats['fg3_pct_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['reb'] = (stats['reb'] * stats['W'] + stats['reb_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['ast'] = (stats['ast'] * stats['W'] + stats['ast_l'] * stats['L']) / stats['G'].clip(lower=1)
    stats['to'] = (stats['to'] * stats['W'] + stats['to_l'] * stats['L']) / stats['G'].clip(lower=1)

    # Efficiency
    poss_w = season_df['WFGA'] - season_df['WOR'] + season_df['WTO'] + 0.44 * season_df['WFTA']
    poss_l = season_df['LFGA'] - season_df['LOR'] + season_df['LTO'] + 0.44 * season_df['LFTA']
    season_df = season_df.copy()
    season_df['poss'] = (poss_w + poss_l) / 2
    season_df['off_eff_w'] = season_df['WScore'] / season_df['poss'].clip(lower=1) * 100
    season_df['off_eff_l'] = season_df['LScore'] / season_df['poss'].clip(lower=1) * 100
    season_df['tempo'] = season_df['poss']

    eff_w = season_df.groupby(['Season', 'WTeamID']).agg(
        off_efficiency=('off_eff_w', 'mean'),
        def_efficiency=('off_eff_l', 'mean'),
        tempo=('tempo', 'mean'),
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})
    eff_l = season_df.groupby(['Season', 'LTeamID']).agg(
        off_efficiency_l=('off_eff_l', 'mean'),
        def_efficiency_l=('off_eff_w', 'mean'),
        tempo_l=('tempo', 'mean'),
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    eff = eff_w.merge(eff_l, on=['Season', 'TeamID'], how='outer').fillna(0)
    g_w = season_df.groupby(['Season', 'WTeamID']).size().reset_index(name='gw').rename(columns={'WTeamID': 'TeamID'})
    g_l = season_df.groupby(['Season', 'LTeamID']).size().reset_index(name='gl').rename(columns={'LTeamID': 'TeamID'})
    eff = eff.merge(g_w, on=['Season', 'TeamID'], how='left').fillna(0)
    eff = eff.merge(g_l, on=['Season', 'TeamID'], how='left').fillna(0)
    eff['G'] = eff['gw'] + eff['gl']
    eff['off_efficiency'] = (eff['off_efficiency'] * eff['gw'] + eff['off_efficiency_l'] * eff['gl']) / eff['G'].clip(lower=1)
    eff['def_efficiency'] = (eff['def_efficiency'] * eff['gw'] + eff['def_efficiency_l'] * eff['gl']) / eff['G'].clip(lower=1)
    eff['tempo'] = (eff['tempo'] * eff['gw'] + eff['tempo_l'] * eff['gl']) / eff['G'].clip(lower=1)

    stats = stats.merge(eff[['Season', 'TeamID', 'off_efficiency', 'def_efficiency', 'tempo']], on=['Season', 'TeamID'], how='left')

    # Conference features
    stats = stats.merge(conf_df[['Season', 'TeamID', 'ConfAbbrev']], on=['Season', 'TeamID'], how='left')
    conf_stats = stats.groupby(['Season', 'ConfAbbrev']).agg(
        conf_avg_margin=('margin', 'mean'),
        conf_avg_win_pct=('win_pct', 'mean'),
    ).reset_index()
    stats = stats.merge(conf_stats, on=['Season', 'ConfAbbrev'], how='left')

    # Neutral court
    neutral = season_df[season_df['WLoc'] == 'N']
    n_wins = neutral.groupby(['Season', 'WTeamID']).size().reset_index(name='nw').rename(columns={'WTeamID': 'TeamID'})
    n_loss = neutral.groupby(['Season', 'LTeamID']).size().reset_index(name='nl').rename(columns={'LTeamID': 'TeamID'})
    neutral_stats = n_wins.merge(n_loss, on=['Season', 'TeamID'], how='outer').fillna(0)
    neutral_stats['neutral_win_pct'] = neutral_stats['nw'] / (neutral_stats['nw'] + neutral_stats['nl']).clip(lower=1)
    stats = stats.merge(neutral_stats[['Season', 'TeamID', 'neutral_win_pct']], on=['Season', 'TeamID'], how='left').fillna(0)

    log.append(f"[OK] Base stats built: {len(stats)} team-seasons")

    # Torvik merge
    torvik_features = ['ADJOE', 'ADJDE', 'BARTHAG', 'ADJ_T', 'WAB', 'EFG_O', 'EFG_D', 'TOR', 'TORD']
    has_torvik = False

    if torvik_df is not None:
        log.append("Merging Torvik data...")
        available_torvik_cols = [f for f in torvik_features if f in torvik_df.columns]
        stats = stats.merge(
            torvik_df[['Season', 'TeamID'] + available_torvik_cols],
            on=['Season', 'TeamID'], how='left'
        )
        # Add any missing Torvik cols as 0
        for col in torvik_features:
            if col not in stats.columns:
                stats[col] = 0.0

        # Stage 1: fill with season average (handles teams missing from Torvik within covered seasons)
        for col in torvik_features:
            season_avg = stats.groupby('Season')[col].transform('mean')
            stats[col] = stats[col].fillna(season_avg)

        # Stage 2: fill remaining with global average (handles pre-2013 seasons entirely)
        for col in torvik_features:
            global_avg = stats[col].mean()
            stats[col] = stats[col].fillna(global_avg)

        missing_after = stats['BARTHAG'].isna().sum()
        log.append(f"[OK] Torvik features merged — {missing_after} missing after imputation (should be 0)")
        has_torvik = True
    else:
        for col in torvik_features:
            stats[col] = 0.0
        log.append("[WARN] No Torvik data — using zeros for Torvik features (lower AUC expected)")

    log.append(f"[OK] Final stats shape: {stats.shape}")
    return stats, log, has_torvik


def build_matchups(tourney_df, team_stats, seeds_df):
    log = []
    seeds_df = seeds_df.copy()
    seeds_df['SeedNum'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)
    stats_index = {(int(r.Season), int(r.TeamID)): r for _, r in team_stats.iterrows()}
    seeds_index = {(int(r.Season), int(r.TeamID)): r.SeedNum for _, r in seeds_df.iterrows()}

    features_base = [
        'seed_diff', 'margin_diff', 'win_pct_diff', 'pts_for_diff', 'pts_against_diff',
        'fg_pct_diff', 'fg3_pct_diff', 'reb_diff', 'ast_diff', 'to_diff',
        'off_eff_diff', 'def_eff_diff', 'tempo_diff', 'seed_t1', 'seed_t2',
        'conf_margin_diff', 'conf_win_pct_diff', 'neutral_win_pct_diff',
    ]
    features_torvik = [
        'adjoe_diff', 'adjde_diff', 'barthag_diff', 'adj_t_diff', 'wab_diff',
        'efg_o_diff', 'efg_d_diff', 'tor_diff', 'tord_diff',
    ]
    all_features = features_base + features_torvik

    rows = []
    for _, game in tourney_df.iterrows():
        s = int(game.Season)
        w, l = int(game.WTeamID), int(game.LTeamID)
        sw = stats_index.get((s, w))
        sl = stats_index.get((s, l))
        seedw = seeds_index.get((s, w))
        seedl = seeds_index.get((s, l))
        if sw is None or sl is None or seedw is None or seedl is None:
            continue

        def diff(a, b, col, alt=0):
            return a.get(col, alt) - b.get(col, alt) if hasattr(a, 'get') else getattr(a, col, alt) - getattr(b, col, alt)

        for t1, t2, seed1, seed2, label in [(sw, sl, seedw, seedl, 1), (sl, sw, seedl, seedw, 0)]:
            row = {
                'seed_diff': seed1 - seed2,
                'margin_diff': diff(t1, t2, 'margin'),
                'win_pct_diff': diff(t1, t2, 'win_pct'),
                'pts_for_diff': diff(t1, t2, 'pts_for'),
                'pts_against_diff': diff(t1, t2, 'pts_against'),
                'fg_pct_diff': diff(t1, t2, 'fg_pct'),
                'fg3_pct_diff': diff(t1, t2, 'fg3_pct'),
                'reb_diff': diff(t1, t2, 'reb'),
                'ast_diff': diff(t1, t2, 'ast'),
                'to_diff': diff(t1, t2, 'to'),
                'off_eff_diff': diff(t1, t2, 'off_efficiency'),
                'def_eff_diff': diff(t1, t2, 'def_efficiency'),
                'tempo_diff': diff(t1, t2, 'tempo'),
                'seed_t1': seed1,
                'seed_t2': seed2,
                'conf_margin_diff': diff(t1, t2, 'conf_avg_margin'),
                'conf_win_pct_diff': diff(t1, t2, 'conf_avg_win_pct'),
                'neutral_win_pct_diff': diff(t1, t2, 'neutral_win_pct'),
                'adjoe_diff': diff(t1, t2, 'ADJOE'),
                'adjde_diff': diff(t1, t2, 'ADJDE'),
                'barthag_diff': diff(t1, t2, 'BARTHAG'),
                'adj_t_diff': diff(t1, t2, 'ADJ_T'),
                'wab_diff': diff(t1, t2, 'WAB'),
                'efg_o_diff': diff(t1, t2, 'EFG_O'),
                'efg_d_diff': diff(t1, t2, 'EFG_D'),
                'tor_diff': diff(t1, t2, 'TOR'),
                'tord_diff': diff(t1, t2, 'TORD'),
                'label': label,
            }
            rows.append(row)

    matchups = pd.DataFrame(rows)
    log.append(f"[OK] Built {len(matchups)} matchup rows from {len(tourney_df)} tournament games")
    return matchups, all_features, log


def train_model(matchups, features, log_list):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    X = matchups[features].values.astype(np.float32)
    y = matchups['label'].values

    # Diagnose and fix NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        log_list.append(f"[WARN] Found {nan_count} NaN and {inf_count} inf values — replacing with 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log_list.append(f"[OK] Feature matrix: {X.shape[0]} rows × {X.shape[1]} features, all finite")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LGBMClassifier(**BEST_PARAMS, verbosity=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    log_list.append(f"[OK] Model trained — Accuracy: {acc:.3f} | AUC: {auc:.3f}")
    return clf, acc, auc, log_list


# ── Monte Carlo ────────────────────────────────────────────────────────────
def predict_winner_prob(model, t1_id, t2_id, season, stats_index, seeds_index, feature_order):
    s1 = stats_index.get((int(season), int(t1_id)))
    s2 = stats_index.get((int(season), int(t2_id)))
    seed1 = seeds_index.get((int(season), int(t1_id)))
    seed2 = seeds_index.get((int(season), int(t2_id)))
    if s1 is None or s2 is None or seed1 is None or seed2 is None:
        return 0.5

    def g(obj, key):
        return obj.get(key, 0) if isinstance(obj, dict) else getattr(obj, key, 0)

    row = {
        'seed_diff': seed1 - seed2,
        'margin_diff': g(s1,'margin') - g(s2,'margin'),
        'win_pct_diff': g(s1,'win_pct') - g(s2,'win_pct'),
        'pts_for_diff': g(s1,'pts_for') - g(s2,'pts_for'),
        'pts_against_diff': g(s1,'pts_against') - g(s2,'pts_against'),
        'fg_pct_diff': g(s1,'fg_pct') - g(s2,'fg_pct'),
        'fg3_pct_diff': g(s1,'fg3_pct') - g(s2,'fg3_pct'),
        'reb_diff': g(s1,'reb') - g(s2,'reb'),
        'ast_diff': g(s1,'ast') - g(s2,'ast'),
        'to_diff': g(s1,'to') - g(s2,'to'),
        'off_eff_diff': g(s1,'off_efficiency') - g(s2,'off_efficiency'),
        'def_eff_diff': g(s1,'def_efficiency') - g(s2,'def_efficiency'),
        'tempo_diff': g(s1,'tempo') - g(s2,'tempo'),
        'seed_t1': seed1, 'seed_t2': seed2,
        'conf_margin_diff': g(s1,'conf_avg_margin') - g(s2,'conf_avg_margin'),
        'conf_win_pct_diff': g(s1,'conf_avg_win_pct') - g(s2,'conf_avg_win_pct'),
        'neutral_win_pct_diff': g(s1,'neutral_win_pct') - g(s2,'neutral_win_pct'),
        'adjoe_diff': g(s1,'ADJOE') - g(s2,'ADJOE'),
        'adjde_diff': g(s1,'ADJDE') - g(s2,'ADJDE'),
        'barthag_diff': g(s1,'BARTHAG') - g(s2,'BARTHAG'),
        'adj_t_diff': g(s1,'ADJ_T') - g(s2,'ADJ_T'),
        'wab_diff': g(s1,'WAB') - g(s2,'WAB'),
        'efg_o_diff': g(s1,'EFG_O') - g(s2,'EFG_O'),
        'efg_d_diff': g(s1,'EFG_D') - g(s2,'EFG_D'),
        'tor_diff': g(s1,'TOR') - g(s2,'TOR'),
        'tord_diff': g(s1,'TORD') - g(s2,'TORD'),
    }
    X = np.array([[row[f] for f in feature_order]], dtype=np.float32)
    return float(model.predict(X)[0])


def simulate_tournament(model, bracket_ids, season, stats_index, seeds_index, feature_order):
    teams = bracket_ids.copy()
    while len(teams) > 1:
        next_round = []
        if len(teams) % 2 != 0:
            next_round.append(teams.pop())
        for i in range(0, len(teams), 2):
            p = predict_winner_prob(model, teams[i], teams[i+1], season, stats_index, seeds_index, feature_order)
            next_round.append(teams[i] if random.random() < p else teams[i+1])
        teams = next_round
    return teams[0]


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════
if page == "Data Upload":
    page_header(svg_header_icon("upload"), "Data Repository", "Upload historical performance & efficiency datasets")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.18); border-top:3px solid #ffa500;
                    border-radius:10px; padding:18px 20px 8px 20px; margin-bottom:12px;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.78rem;
                        letter-spacing:0.14em; text-transform:uppercase; color:#ffa500; margin-bottom:4px;">
                KAGGLE FILES <span style="color:#dc2626;">&#9679; REQUIRED</span></div>
            <div style="font-size:0.78rem; color:#475569; margin-bottom:12px;">
                kaggle.com/competitions/march-machine-learning-mania-2026</div>
        </div>""", unsafe_allow_html=True)
        kaggle_uploads = st.file_uploader(
            "Upload all 5 Kaggle CSV files",
            type="csv", accept_multiple_files=True, key="kaggle", label_visibility="collapsed"
        )
        if kaggle_uploads:
            saved = []
            for f in kaggle_uploads:
                if f.name in REQUIRED_KAGGLE:
                    with open(os.path.join(DATA_DIR, f.name), 'wb') as out: out.write(f.read())
                    saved.append(f.name)
            for fname in saved:
                st.markdown(f"<span style='background:#14532d;color:#4ade80;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-block;margin:2px 3px;'>{fname}</span>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        for f in REQUIRED_KAGGLE:
            exists = os.path.exists(os.path.join(DATA_DIR, f))
            dot = "<span style='color:#22c55e;'>&#10003;</span>" if exists else "<span style='color:#dc2626;'>&#10007;</span>"
            st.markdown(f"<div style='font-size:0.8rem;color:#64748b;padding:2px 0;'>{dot} {f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.18); border-top:3px solid #ff6b00;
                    border-radius:10px; padding:18px 20px 8px 20px; margin-bottom:12px;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.78rem;
                        letter-spacing:0.14em; text-transform:uppercase; color:#ff6b00; margin-bottom:4px;">
                TORVIK EFFICIENCY <span style="color:#f59e0b;">&#9679; OPTIONAL</span></div>
            <div style="font-size:0.78rem; color:#475569; margin-bottom:12px;">
                kaggle.com/datasets/nishaanamin/march-madness-data</div>
        </div>""", unsafe_allow_html=True)
        torvik_uploads = st.file_uploader(
            "Upload Torvik CSVs",
            type="csv", accept_multiple_files=True, key="torvik", label_visibility="collapsed"
        )
        if torvik_uploads:
            saved = []
            for f in torvik_uploads:
                if f.name in TORVIK_FILES:
                    with open(os.path.join(DATA_DIR, f.name), 'wb') as out: out.write(f.read())
                    saved.append(f.name)
            for fname in saved:
                st.markdown(f"<span style='background:#14532d;color:#4ade80;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-block;margin:2px 3px;'>{fname}</span>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        for f in TORVIK_FILES:
            exists = os.path.exists(os.path.join(DATA_DIR, f))
            dot = "<span style='color:#22c55e;'>&#10003;</span>" if exists else "<span style='color:#f59e0b;'>&#9675;</span>"
            st.markdown(f"<div style='font-size:0.8rem;color:#64748b;padding:2px 0;'>{dot} {f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
    if kaggle_ok:
        st.success("All required files present — head to **Train Model** to continue!")
    else:
        missing = [f for f in REQUIRED_KAGGLE if not os.path.exists(os.path.join(DATA_DIR, f))]
        st.warning(f"Required files still missing: {', '.join(missing)}")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════
elif page == "Train Model":
    page_header(svg_header_icon("settings"), "Engine Room", "LightGBM training · Optuna-tuned hyperparameters")

    if not kaggle_ok:
        st.error("Missing Kaggle files — go to Data Upload first.")
        st.stop()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"""
        <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.15); border-radius:10px; padding:18px;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700;
                        letter-spacing:0.16em; color:#475569; margin-bottom:12px;">TRAINING CONFIG</div>
            <div style="display:grid; grid-template-columns:auto 1fr; gap:6px 16px; font-size:0.83rem; color:#94a3b8;">
                <span style="color:#64748b;">Algorithm</span><span style="color:#e2e8f0; font-weight:600;">LightGBM</span>
                <span style="color:#64748b;">Tuning</span><span style="color:#e2e8f0; font-weight:600;">100 Optuna trials</span>
                <span style="color:#64748b;">n_estimators</span><span style="color:#ffa500; font-weight:700;">{BEST_PARAMS['n_estimators']}</span>
                <span style="color:#64748b;">learning_rate</span><span style="color:#ffa500; font-weight:700;">{BEST_PARAMS['learning_rate']:.5f}</span>
                <span style="color:#64748b;">max_depth</span><span style="color:#ffa500; font-weight:700;">{BEST_PARAMS['max_depth']}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        expected_auc = "~0.771" if torvik_ok else "~0.755"
        torvik_label = "WITH TORVIK" if torvik_ok else "WITHOUT TORVIK"
        torvik_color = "#22c55e" if torvik_ok else "#f59e0b"
        st.markdown(f"""
        <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.15); border-radius:10px; padding:18px;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700;
                        letter-spacing:0.16em; color:#475569; margin-bottom:12px;">EXPECTED PERFORMANCE</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:2.2rem; font-weight:800; color:#ffa500;">{expected_auc}</div>
            <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.1em; margin-bottom:8px;">AUC SCORE</div>
            <span style="background:{torvik_color}22; color:{torvik_color}; padding:3px 10px;
                         border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:0.08em;">{torvik_label}</span>
            <div style="font-size:0.78rem; color:#475569; margin-top:10px;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="width:13px;height:13px;display:inline-block;vertical-align:middle;margin-right:5px;"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>~30–60 seconds</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
    if st.button("Execute Training", type="primary"):
        log_box = st.empty()
        log_lines = []

        def update_log(lines):
            log_box.code('\n'.join(lines))

        with st.spinner("Loading data..."):
            season_df = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
            tourney_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyDetailedResults.csv"))
            teams_df = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
            seeds_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
            conf_df = pd.read_csv(os.path.join(DATA_DIR, "MTeamConferences.csv"))
            log_lines.append("[OK] Kaggle files loaded")

            torvik_df = None
            if torvik_ok:
                parts = []
                for tf in TORVIK_FILES:
                    path = os.path.join(DATA_DIR, tf)
                    if os.path.exists(path):
                        parts.append(pd.read_csv(path))
                if parts:
                    torvik_raw = pd.concat(parts, ignore_index=True).drop_duplicates()
                    torvik_raw.columns = [c.upper() for c in torvik_raw.columns]
                    year_col = next((c for c in torvik_raw.columns if 'YEAR' in c), None)
                    team_col = next((c for c in torvik_raw.columns if c in ['TEAM', 'TEAM NAME']), None)

                    if year_col and team_col:
                        torvik_raw = torvik_raw.rename(columns={year_col: 'Season', team_col: 'TEAM'})
                        torvik_raw['Season'] = torvik_raw['Season'].astype(int)

                        # Full name mapping — Torvik name → Kaggle name (v2 + v3 combined)
                        name_map_v2 = {
                            'Alabama St.': 'Alabama St', 'Albany': 'SUNY Albany',
                            'Alcorn St.': 'Alcorn St', 'Appalachian St.': 'Appalachian St',
                            'Arizona St.': 'Arizona St', 'Arkansas Little Rock': 'Ark Little Rock',
                            'Arkansas St.': 'Arkansas St', 'Ball St.': 'Ball St',
                            'Boise St.': 'Boise St', 'Central Arkansas': 'Cent Arkansas',
                            'Central Michigan': 'C Michigan', 'Chicago St.': 'Chicago St',
                            'Cleveland St.': 'Cleveland St', 'Colorado St.': 'Colorado St',
                            'Coppin St.': 'Coppin St', 'Delaware St.': 'Delaware St',
                            'Dixie St.': 'Utah Tech', 'East Tennessee St.': 'ETSU',
                            'Florida Gulf Coast': 'FGCU', 'Florida St.': 'Florida St',
                            'Fort Wayne': 'PFW', 'Fresno St.': 'Fresno St',
                            'Georgia St.': 'Georgia St', 'Houston Baptist': 'Houston Chr',
                            'IPFW': 'PFW', 'Idaho St.': 'Idaho St',
                            'Illinois St.': 'Illinois St', 'Indiana St.': 'Indiana St',
                            'Iowa St.': 'Iowa St', 'Jackson St.': 'Jackson St',
                            'Jacksonville St.': 'Jacksonville St', 'Kansas St.': 'Kansas St',
                            'Kennesaw St.': 'Kennesaw', 'Little Rock': 'Ark Little Rock',
                            'Long Beach St.': 'Long Beach St', 'Louisiana Monroe': 'ULM',
                            'Loyola Marymount': 'Loy Marymount', 'McNeese St.': 'McNeese St',
                            'Michigan St.': 'Michigan St', 'Middle Tennessee': 'MTSU',
                            'Mississippi St.': 'Mississippi St', 'Mississippi Valley St.': 'MS Valley St',
                            'Missouri St.': 'Missouri St', 'Montana St.': 'Montana St',
                            'Morehead St.': 'Morehead St', 'Morgan St.': 'Morgan St',
                            'Murray St.': 'Murray St', 'Nebraska Omaha': 'NE Omaha',
                            'New Mexico St.': 'New Mexico St', 'Nicholls St.': 'Nicholls St',
                            'Norfolk St.': 'Norfolk St', 'North Dakota St.': 'N Dakota St',
                            'Northwestern St.': 'Northwestern LA', 'Ohio St.': 'Ohio St',
                            'Oklahoma St.': 'Oklahoma St', 'Oregon St.': 'Oregon St',
                            'Penn St.': 'Penn St', 'Portland St.': 'Portland St',
                            'SIU Edwardsville': 'SIUE', 'Sacramento St.': 'CS Sacramento',
                            "Saint Joseph's": "St Joseph's PA", 'Saint Louis': 'St Louis',
                            "Saint Mary's": "St Mary's CA", "Saint Peter's": "St Peter's",
                            'Sam Houston St.': 'Sam Houston St', 'San Diego St.': 'San Diego St',
                            'San Jose St.': 'San Jose St', 'Savannah St.': 'Savannah St',
                            'South Carolina St.': 'S Carolina St', 'South Dakota St.': 'S Dakota St',
                            'Southeast Missouri St.': 'SE Missouri St', 'Tarleton St.': 'Tarleton St',
                            'Tennessee Martin': 'TN Martin', 'Tennessee St.': 'Tennessee St',
                            'Texas A&M Commerce': 'East Texas A&M', 'Texas A&M Corpus Chris': 'TAM C. Christi',
                            'Texas St.': 'Texas St', 'UMKC': 'Missouri KC',
                            'Utah St.': 'Utah St', 'Washington St.': 'Washington St',
                            'Weber St.': 'Weber St', 'Western Kentucky': 'WKU',
                            'Wichita St.': 'Wichita St', 'Wright St.': 'Wright St',
                            'Youngstown St.': 'Youngstown St',
                        }
                        name_map_v3 = {
                            'Abilene Christian': 'Abilene Chr', 'American': 'American Univ',
                            'Arkansas Pine Bluff': 'Ark Pine Bluff', 'Bethune Cookman': 'Bethune-Cookman',
                            'Boston University': 'Boston Univ', 'Cal St. Bakersfield': 'CS Bakersfield',
                            'Cal St. Fullerton': 'CS Fullerton', 'Cal St. Northridge': 'CS Northridge',
                            'Central Connecticut': 'Central Conn', 'Charleston Southern': 'Charleston So',
                            'Coastal Carolina': 'Coastal Car', 'College of Charleston': 'Col Charleston',
                            'Eastern Illinois': 'E Illinois', 'Eastern Kentucky': 'E Kentucky',
                            'Eastern Michigan': 'E Michigan', 'Eastern Washington': 'E Washington',
                            'FIU': 'Florida Intl', 'Fairleigh Dickinson': 'F Dickinson',
                            'Florida Atlantic': 'FL Atlantic', 'George Washington': 'G Washington',
                            'Georgia Southern': 'Ga Southern', 'Grambling St.': 'Grambling',
                            'Green Bay': 'WI Green Bay', 'Houston Christian': 'Houston Chr',
                            'Illinois Chicago': 'IL Chicago', 'Kent St.': 'Kent',
                            'Louisiana Lafayette': 'Louisiana', 'Loyola Chicago': 'Loyola-Chicago',
                            'Maryland Eastern Shore': 'MD E Shore', 'Milwaukee': 'WI Milwaukee',
                            'Monmouth': 'Monmouth NJ', "Mount St. Mary's": "Mt St Mary's",
                            'North Carolina A&T': 'NC A&T', 'North Carolina Central': 'NC Central',
                            'North Carolina St.': 'NC State', 'Northern Colorado': 'N Colorado',
                            'Northern Illinois': 'N Illinois', 'Northern Kentucky': 'N Kentucky',
                            'Prairie View A&M': 'Prairie View', 'Queens': 'Queens NC',
                            'Southeastern Louisiana': 'SE Louisiana', 'Southern': 'Southern Univ',
                            'Southern Illinois': 'S Illinois', 'St. Bonaventure': 'St Bonaventure',
                            'St. Francis NY': 'St Francis NY', 'St. Francis PA': 'St Francis PA',
                            "St. John's": "St John's", 'St. Thomas': 'St Thomas MN',
                            'Stephen F. Austin': 'SF Austin', 'Texas Southern': 'TX Southern',
                            'The Citadel': 'Citadel', 'UMass Lowell': 'MA Lowell',
                            'USC Upstate': 'SC Upstate', 'UT Rio Grande Valley': 'UTRGV',
                            'UTSA': 'UT San Antonio', 'Western Carolina': 'W Carolina',
                            'Western Illinois': 'W Illinois', 'Western Michigan': 'W Michigan',
                        }
                        full_name_map = {**name_map_v2, **name_map_v3}

                        # Drop existing ID cols if present from prior merge
                        for col in ['TeamID', 'TeamName']:
                            if col in torvik_raw.columns:
                                torvik_raw = torvik_raw.drop(columns=[col])

                        torvik_raw['TeamName'] = torvik_raw['TEAM'].map(full_name_map).fillna(torvik_raw['TEAM'])
                        torvik_raw = torvik_raw.merge(teams_df[['TeamID', 'TeamName']], on='TeamName', how='left')

                        unmatched = torvik_raw[torvik_raw['TeamID'].isna()]['TEAM'].nunique()
                        matched = torvik_raw['TeamID'].notna().sum()
                        log_lines.append(f"[OK] Torvik name mapping: {matched} rows matched, {unmatched} unique teams unmatched")

                        torvik_raw = torvik_raw.dropna(subset=['TeamID'])
                        torvik_raw['TeamID'] = torvik_raw['TeamID'].astype(int)
                        torvik_df = torvik_raw
                        log_lines.append(f"[OK] Torvik data loaded: {len(torvik_df)} rows, seasons {sorted(torvik_df['Season'].unique())[:3]}...{sorted(torvik_df['Season'].unique())[-1]}")
                    else:
                        log_lines.append("[WARN] Could not find Year/Team columns in Torvik files")
            update_log(log_lines)

        with st.spinner("Building team stats..."):
            team_stats, stats_log, has_torvik = build_team_stats(season_df, teams_df, conf_df, torvik_df)
            log_lines += stats_log
            update_log(log_lines)

        with st.spinner("Building matchup features..."):
            matchups, features, match_log = build_matchups(tourney_df, team_stats, seeds_df)
            log_lines += match_log
            update_log(log_lines)

        with st.spinner("Training LightGBM model..."):
            clf, acc, auc, log_lines = train_model(matchups, features, log_lines)
            update_log(log_lines)

        with st.spinner("Saving model and data..."):
            clf.booster_.save_model(MODEL_PATH)

            seeds_df2 = seeds_df.copy()
            seeds_df2['SeedNum'] = seeds_df2['Seed'].str.extract(r'(\d+)').astype(int)
            # Use the most recent season that has seed data — NOT +1
            latest_season = int(seeds_df2['Season'].max())

            stats_index_save = {
                (int(r.Season), int(r.TeamID)): dict(r)
                for _, r in team_stats.iterrows()
            }
            seeds_index_save = {
                (int(r.Season), int(r.TeamID)): r.SeedNum
                for _, r in seeds_df2.iterrows()
            }

            app_data = {
                'feature_order': features,
                'teams': teams_df.to_dict(),
                'latest_season': latest_season,
                'stats_index': stats_index_save,
                'seeds_index': seeds_index_save,
                'has_torvik': has_torvik,
                'auc': auc,
                'accuracy': acc,
            }
            with open(DATA_PKL_PATH, 'wb') as f:
                pickle.dump(app_data, f)

            log_lines.append(f"[OK] Model saved to {MODEL_PATH}")
            log_lines.append(f"[OK] App data saved to {DATA_PKL_PATH}")
            log_lines.append(f"")
            log_lines.append(f"[READY] Head to Head and Bracket Simulator are now available.")
            update_log(log_lines)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:20px;">
            <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.25); border-radius:10px; padding:16px; text-align:center;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; font-weight:700;
                            letter-spacing:0.16em; color:#475569; margin-bottom:6px;">MODEL AUC</div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:800; color:#ffa500;">{auc:.4f}</div>
            </div>
            <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.25); border-radius:10px; padding:16px; text-align:center;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; font-weight:700;
                            letter-spacing:0.16em; color:#475569; margin-bottom:6px;">ACCURACY</div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:800; color:#ffa500;">{acc:.2%}</div>
            </div>
            <div style="background:#0d1220; border:1px solid rgba(255,165,0,0.25); border-radius:10px; padding:16px; text-align:center;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; font-weight:700;
                            letter-spacing:0.16em; color:#475569; margin-bottom:6px;">MATCHUPS</div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:800; color:#ffa500;">{len(matchups):,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Model trained and saved — Head to Head and Bracket Simulator are ready!")
        st.cache_resource.clear()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — HEAD TO HEAD
# ══════════════════════════════════════════════════════════════════════════
elif page == "Head to Head":
    page_header(svg_header_icon("versus"), "Matchup Lab", "Direct win probability comparison · 27-feature LightGBM model")

    if not model_ok:
        st.error("No trained model found — go to Train Model first.")
        st.stop()

    model, d = load_model_and_data()
    if model is None:
        st.error("Could not load model.")
        st.stop()

    feature_order = d['feature_order']
    teams_dict = d['teams']
    stats_index = d['stats_index']
    seeds_index = d['seeds_index']
    latest_season = d['latest_season']
    auc = d.get('auc', 0.771)

    teams_df2 = pd.DataFrame.from_dict(teams_dict)
    team_lookup = dict(zip(teams_df2['TeamID'], teams_df2['TeamName']))
    name_to_id = {v: k for k, v in team_lookup.items()}
    all_team_names = sorted(name_to_id.keys())

    col1, col2, col3 = st.columns([5, 1, 5])
    default_idx1 = all_team_names.index("Duke") if "Duke" in all_team_names else 0
    default_idx2 = all_team_names.index("Kansas") if "Kansas" in all_team_names else 1
    with col1:
        st.markdown("<div style='font-family:Barlow Condensed,sans-serif;font-size:0.7rem;font-weight:700;letter-spacing:0.16em;color:#475569;margin-bottom:4px;'>TEAM A</div>", unsafe_allow_html=True)
        team1_name = st.selectbox("Team A", all_team_names, index=default_idx1, label_visibility="collapsed")
    with col2:
        st.markdown("<div style='height:38px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Barlow Condensed,sans-serif;font-size:1.1rem;font-weight:800;color:#334155;text-align:center;padding-top:8px;'>VS</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='font-family:Barlow Condensed,sans-serif;font-size:0.7rem;font-weight:700;letter-spacing:0.16em;color:#475569;margin-bottom:4px;'>TEAM B</div>", unsafe_allow_html=True)
        team2_name = st.selectbox("Team B", all_team_names, index=default_idx2, label_visibility="collapsed")

    st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
    if st.button("Calculate Win Probability", type="primary"):
        if team1_name == team2_name:
            st.error("Please select two different teams.")
        else:
            t1_id = name_to_id.get(team1_name)
            t2_id = name_to_id.get(team2_name)

            prob_t1 = predict_winner_prob(model, t1_id, t2_id, latest_season,
                                          stats_index, seeds_index, feature_order)
            seed1 = seeds_index.get((int(latest_season), int(t1_id)))
            seed2 = seeds_index.get((int(latest_season), int(t2_id)))
            s1 = stats_index.get((int(latest_season), int(t1_id)), {})
            s2 = stats_index.get((int(latest_season), int(t2_id)), {})

            # Store everything needed to re-render after the analysis button is clicked
            st.session_state['h2h_result'] = {
                'team1': team1_name, 'team2': team2_name,
                'prob_t1': prob_t1, 'seed1': seed1, 'seed2': seed2,
                's1': s1, 's2': s2,
            }
            st.session_state.pop('h2h_analysis', None)  # clear stale analysis on new calc

    # Render results outside the button block so they survive re-renders
    if 'h2h_result' in st.session_state:
        r = st.session_state['h2h_result']
        prob_t1 = r['prob_t1']
        prob_t2 = 1 - prob_t1
        team1_name = r['team1']
        team2_name = r['team2']
        seed1, seed2 = r['seed1'], r['seed2']
        seed1_str = f"#{int(seed1)} seed" if seed1 else "unseeded"
        seed2_str = f"#{int(seed2)} seed" if seed2 else "unseeded"
        favored = team1_name if prob_t1 > 0.5 else team2_name
        fav_prob = max(prob_t1, prob_t2)

        st.markdown(f"""
        <div style="text-align:center; margin:30px 0 10px 0;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:4.5rem; font-weight:800;
                        color:#ffa500; line-height:1; letter-spacing:-0.01em;">{prob_t1:.1%}</div>
            <div style="font-size:0.72rem; color:#475569; letter-spacing:0.18em; margin-top:4px;">
                WIN PROBABILITY &mdash; {team1_name.upper()}</div>
        </div>

        <div style="display:flex; height:40px; border-radius:20px; overflow:hidden;
                    margin:20px 0; border:1px solid rgba(255,255,255,0.06);">
            <div style="width:{prob_t1*100:.1f}%; background:linear-gradient(90deg,#ffa500,#ff6b00);
                        display:flex; align-items:center; padding-left:16px; overflow:hidden;">
                <span style="font-family:'Barlow Condensed',sans-serif; font-weight:800;
                             color:#000; font-size:0.85rem; white-space:nowrap;">{team1_name.upper()}</span>
            </div>
            <div style="width:{prob_t2*100:.1f}%; background:#1a2540; display:flex;
                        align-items:center; justify-content:flex-end; padding-right:16px; overflow:hidden;">
                <span style="font-family:'Barlow Condensed',sans-serif; font-weight:800;
                             color:#94a3b8; font-size:0.85rem; white-space:nowrap;">{team2_name.upper()}</span>
            </div>
        </div>

        <div style="display:flex; justify-content:space-between; margin-bottom:24px;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.72rem; color:#64748b;">
                {seed1_str.upper()}</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.78rem; font-weight:700;
                        color:#22c55e;"><svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#22c55e' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' style='width:11px;height:11px;display:inline-block;vertical-align:middle;margin-right:4px;'><polyline points='18 15 12 9 6 15'/></svg>{favored.upper()} FAVORED &mdash; {fav_prob:.1%}</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.72rem; color:#64748b;
                        text-align:right;">{seed2_str.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.barh([team2_name, team1_name], [prob_t2*100, prob_t1*100],
                 color=['#1d4ed8', '#ffa500'], height=0.45)
        ax1.set_xlim(0, 115)
        ax1.set_xlabel('Win Probability (%)')
        ax1.set_title('Win Probability Comparison', fontweight='bold', pad=12)
        for val, nm in [(prob_t2, team2_name), (prob_t1, team1_name)]:
            ax1.text(val*100+1.8, [team2_name, team1_name].index(nm),
                     f'{val:.1%}', va='center', fontweight='bold', fontsize=11)
        ax1.spines[:].set_visible(False)

        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=model.feature_name()
        ).sort_values(ascending=False).head(8)
        bar_colors = ['#ffa500' if i == 0 else '#ff6b00' if i < 3 else '#1d4ed8' for i in range(8)]
        ax2.barh(importance.index[::-1], importance.values[::-1], color=bar_colors[::-1], height=0.45)
        ax2.set_title('Top Feature Importances', fontweight='bold', pad=12)
        ax2.spines[:].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig)

        # Get Analysis button
        if st.button("Get Analysis", key="h2h_get_analysis"):
            with st.spinner("Searching for expert analysis..."):
                prompt = h2h_analysis_prompt(
                    r['team1'], r['team2'], r['prob_t1'],
                    r['seed1'], r['seed2'], r['s1'], r['s2'], auc
                )
                st.session_state['h2h_analysis'] = get_analysis(prompt)

        if 'h2h_analysis' in st.session_state:
            render_analysis_card(st.session_state['h2h_analysis'])

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — BRACKET SIMULATOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "Bracket Simulator":
    page_header(svg_header_icon("trophy"), "The Gauntlet", "Monte Carlo tournament simulation · 2026 bracket")

    if not model_ok:
        st.error("No trained model found — go to Train Model first.")
        st.stop()

    model, d = load_model_and_data()
    if model is None:
        st.error("Could not load model.")
        st.stop()

    feature_order = d['feature_order']
    teams_dict = d['teams']
    stats_index = d['stats_index']
    seeds_index = d['seeds_index']
    latest_season = d['latest_season']
    auc = d.get('auc', 0.771)

    teams_df2 = pd.DataFrame.from_dict(teams_dict)
    team_lookup = dict(zip(teams_df2['TeamID'], teams_df2['TeamName']))
    name_to_id = {v: k for k, v in team_lookup.items()}
    all_team_names = sorted(name_to_id.keys())

    st.info("Bracket finalizes on Selection Sunday, March 15 — update BRACKET_2026 then retrain for live predictions.")

    # Styled bracket preview
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700;
        letter-spacing:0.18em; color:#475569; margin:16px 0 10px 0;">2026 BRACKET PREVIEW</div>""",
        unsafe_allow_html=True)

    region_colors = {"East": "#ffa500", "West": "#3b82f6", "South": "#22c55e", "Midwest": "#dc2626"}
    seed_order_map = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
    regions = ["East", "West", "South", "Midwest"]
    cols = st.columns(4)
    missing_teams = []
    for i, (region, col) in enumerate(zip(regions, cols)):
        region_teams = BRACKET_2026[i*16:(i+1)*16]
        accent = region_colors[region]
        matchup_rows = ""
        for j in range(0, 16, 2):
            t1, t2 = region_teams[j], region_teams[j+1]
            ok1 = name_to_id.get(t1) is not None
            ok2 = name_to_id.get(t2) is not None
            s1 = seed_order_map[j]; s2 = seed_order_map[j+1]
            c1 = "#e2e8f0" if ok1 else "#dc2626"
            c2 = "#94a3b8" if ok2 else "#dc2626"
            if not ok1: missing_teams.append(t1)
            if not ok2: missing_teams.append(t2)
            matchup_rows += f"""
            <div style="display:flex; align-items:center; gap:4px; padding:3px 0;
                        border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.75rem;">
                <span style="background:{accent}22; color:{accent}; border-radius:3px; padding:0 4px;
                             font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.65rem;
                             min-width:16px; text-align:center;">{s1}</span>
                <span style="color:{c1}; flex:1; overflow:hidden; text-overflow:ellipsis;
                             white-space:nowrap; font-family:'Barlow Condensed',sans-serif;">{t1}</span>
                <span style="color:#334155; font-size:0.65rem;">vs</span>
                <span style="color:{c2}; flex:1; overflow:hidden; text-overflow:ellipsis;
                             white-space:nowrap; text-align:right; font-family:'Barlow Condensed',sans-serif;">{t2}</span>
                <span style="background:{accent}22; color:{accent}; border-radius:3px; padding:0 4px;
                             font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:0.65rem;
                             min-width:16px; text-align:center;">{s2}</span>
            </div>"""
        with col:
            st.markdown(f"""
            <div style="background:#0d1220; border:1px solid rgba(255,255,255,0.06);
                        border-top:3px solid {accent}; border-radius:10px; padding:14px;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700;
                            letter-spacing:0.14em; color:{accent}; margin-bottom:10px;">{region.upper()}</div>
                {matchup_rows}
            </div>""", unsafe_allow_html=True)

    if missing_teams:
        st.warning(f"{len(set(missing_teams))} teams not found in dataset (50/50 odds): {', '.join(sorted(set(missing_teams)))}")

    st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
    n_sims = st.slider("Number of simulations", min_value=100, max_value=2000, value=1000, step=100)

    if st.button(f"Run {n_sims:,} Simulations", type="primary"):
        bracket_ids = [name_to_id.get(t) for t in BRACKET_2026]
        # Replace None with a dummy ID that returns 0.5 win prob
        bracket_ids = [b if b is not None else -1 for b in bracket_ids]

        # Build seeds directly from bracket position (1=best, 16=worst per region)
        # BRACKET_2026 is ordered: pairs within each region in seed order
        # positions 0,2,4,6,8,10,12,14 = seeds 1,2,3,4,5,6,7,8 (higher seeds)
        # positions 1,3,5,7,9,11,13,15 = seeds 16,15,14,13,12,11,10,9 (lower seeds)
        seed_assignments = {}
        seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        for region_idx in range(4):
            region_teams = BRACKET_2026[region_idx * 16: (region_idx + 1) * 16]
            for slot_idx, team_name in enumerate(region_teams):
                tid = name_to_id.get(team_name)
                if tid is not None:
                    seed_assignments[int(tid)] = seed_order[slot_idx]

        # Override predict_winner_prob to use bracket seeds when historical seeds missing
        def predict_with_bracket_seeds(t1_id, t2_id):
            if t1_id == -1 or t2_id == -1:
                return 0.5
            s1 = stats_index.get((int(latest_season), int(t1_id)))
            s2 = stats_index.get((int(latest_season), int(t2_id)))
            seed1 = seeds_index.get((int(latest_season), int(t1_id))) or seed_assignments.get(int(t1_id))
            seed2 = seeds_index.get((int(latest_season), int(t2_id))) or seed_assignments.get(int(t2_id))
            if s1 is None or s2 is None or seed1 is None or seed2 is None:
                return 0.5

            def g(obj, key):
                return obj.get(key, 0) if isinstance(obj, dict) else getattr(obj, key, 0)

            row = {
                'seed_diff': seed1 - seed2,
                'margin_diff': g(s1,'margin') - g(s2,'margin'),
                'win_pct_diff': g(s1,'win_pct') - g(s2,'win_pct'),
                'pts_for_diff': g(s1,'pts_for') - g(s2,'pts_for'),
                'pts_against_diff': g(s1,'pts_against') - g(s2,'pts_against'),
                'fg_pct_diff': g(s1,'fg_pct') - g(s2,'fg_pct'),
                'fg3_pct_diff': g(s1,'fg3_pct') - g(s2,'fg3_pct'),
                'reb_diff': g(s1,'reb') - g(s2,'reb'),
                'ast_diff': g(s1,'ast') - g(s2,'ast'),
                'to_diff': g(s1,'to') - g(s2,'to'),
                'off_eff_diff': g(s1,'off_efficiency') - g(s2,'off_efficiency'),
                'def_eff_diff': g(s1,'def_efficiency') - g(s2,'def_efficiency'),
                'tempo_diff': g(s1,'tempo') - g(s2,'tempo'),
                'seed_t1': seed1, 'seed_t2': seed2,
                'conf_margin_diff': g(s1,'conf_avg_margin') - g(s2,'conf_avg_margin'),
                'conf_win_pct_diff': g(s1,'conf_avg_win_pct') - g(s2,'conf_avg_win_pct'),
                'neutral_win_pct_diff': g(s1,'neutral_win_pct') - g(s2,'neutral_win_pct'),
                'adjoe_diff': g(s1,'ADJOE') - g(s2,'ADJOE'),
                'adjde_diff': g(s1,'ADJDE') - g(s2,'ADJDE'),
                'barthag_diff': g(s1,'BARTHAG') - g(s2,'BARTHAG'),
                'adj_t_diff': g(s1,'ADJ_T') - g(s2,'ADJ_T'),
                'wab_diff': g(s1,'WAB') - g(s2,'WAB'),
                'efg_o_diff': g(s1,'EFG_O') - g(s2,'EFG_O'),
                'efg_d_diff': g(s1,'EFG_D') - g(s2,'EFG_D'),
                'tor_diff': g(s1,'TOR') - g(s2,'TOR'),
                'tord_diff': g(s1,'TORD') - g(s2,'TORD'),
            }
            X = np.array([[row[f] for f in feature_order]], dtype=np.float32)
            return float(model.predict(X)[0])

        def simulate_tournament_2026(bracket_ids):
            teams = bracket_ids.copy()
            while len(teams) > 1:
                next_round = []
                if len(teams) % 2 != 0:
                    next_round.append(teams.pop())
                for i in range(0, len(teams), 2):
                    p = predict_with_bracket_seeds(teams[i], teams[i+1])
                    next_round.append(teams[i] if random.random() < p else teams[i+1])
                teams = next_round
            return teams[0]

        progress = st.progress(0, text="Starting simulations...")
        champ_counts = {}
        for i in range(n_sims):
            champ = simulate_tournament_2026(bracket_ids)
            champ_counts[champ] = champ_counts.get(champ, 0) + 1
            if i % 50 == 0:
                progress.progress(i / n_sims, text=f"Simulating... {i}/{n_sims}")
        progress.progress(1.0, text="Done!")

        results = {
            tid: count / n_sims
            for tid, count in sorted(champ_counts.items(), key=lambda x: -x[1])
            if tid != -1
        }
        top15 = list(results.items())[:15]

        # Store results so they survive the analysis button re-renders
        st.session_state['sim_results'] = {
            'top15': top15, 'n_sims': n_sims, 'seed_assignments': seed_assignments
        }
        # Clear any stale per-team analyses from a previous run
        for k in [f'sim_analysis_{j}' for j in range(3)]:
            st.session_state.pop(k, None)

    # ── Render chart + leaderboard outside the button block ──────────────
    if 'sim_results' in st.session_state:
        sr = st.session_state['sim_results']
        top15 = sr['top15']
        n_sims_stored = sr['n_sims']
        seed_assignments = sr['seed_assignments']

        names = [team_lookup.get(tid, str(tid)) for tid, _ in top15]
        probs = [p * 100 for _, p in top15]
        colors = []
        for i in range(len(top15)):
            if i == 0: colors.append('#ffd700')
            elif i < 3: colors.append('#ffa500')
            elif i < 7: colors.append('#1d4ed8')
            else: colors.append('#334155')

        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(names[::-1], probs[::-1], color=colors[::-1], height=0.52)
        for bar, p in zip(bars, probs[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{p:.1f}%', va='center', fontweight='bold', fontsize=10)
        ax.set_xlabel('Championship Probability (%)')
        ax.set_title(f'2026 Championship Odds  |  {n_sims_stored:,} Monte Carlo Simulations  |  AUC {auc:.3f}',
                     fontweight='bold', fontsize=11, pad=14)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines[:].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig)

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700;
            letter-spacing:0.18em; color:#475569; margin:24px 0 10px 0;">CHAMPIONSHIP LEADERBOARD</div>""",
            unsafe_allow_html=True)

        max_prob = top15[0][1] if top15 else 1
        rank_styles = [
            ("#ffd700", "rgba(255,215,0,0.12)", "rgba(255,215,0,0.28)"),
            ("#c0c0c0", "rgba(192,192,192,0.08)", "rgba(192,192,192,0.18)"),
            ("#cd7f32", "rgba(205,127,50,0.08)", "rgba(205,127,50,0.18)"),
        ]
        for i, (tid, prob) in enumerate(top15):
            name = team_lookup.get(tid, str(tid))
            bar_w = int((prob / max_prob) * 120)
            if i < 3:
                rc, bg, border = rank_styles[i]
                rank_label = ["1","2","3"][i]
            else:
                rc, bg, border = "#475569", "rgba(255,255,255,0.03)", "rgba(255,255,255,0.06)"
                rank_label = str(i + 1)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:9px 16px;margin-bottom:4px;
                        background:{bg};border:1px solid {border};border-radius:8px;
                        border-left:3px solid {rc};">
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;
                             color:{rc};min-width:28px;text-align:center;">{rank_label}</span>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;font-weight:600;
                             color:#e2e8f0;flex:1;">{name}</span>
                <div style="background:{rc}44;border-radius:3px;height:5px;width:{bar_w}px;min-width:4px;"></div>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:800;
                             color:{rc};min-width:52px;text-align:right;">{prob:.1%}</span>
            </div>""", unsafe_allow_html=True)

            # Get Analysis button for top-3 only
            if i < 3:
                btn_key = f"sim_get_analysis_{i}"
                analysis_key = f"sim_analysis_{i}"
                if st.button(f"Get Analysis — {name}", key=btn_key):
                    team_seed = (
                        seeds_index.get((int(latest_season), int(tid)))
                        or seed_assignments.get(int(tid))
                    )
                    team_stats = stats_index.get((int(latest_season), int(tid)), {})
                    with st.spinner(f"Searching for analyst takes on {name}..."):
                        prompt = team_analysis_prompt(
                            name, team_seed, prob, n_sims_stored, team_stats, auc
                        )
                        st.session_state[analysis_key] = get_analysis(prompt)
                if analysis_key in st.session_state:
                    render_analysis_card(st.session_state[analysis_key])
