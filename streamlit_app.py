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

# ── SVG Assets ────────────────────────────────────────────────────────────
# Clean, professional icons for sports analytics
SVG_ICONS = {
    "ball": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%;"><circle cx="12" cy="12" r="10"/><path d="M6.2 6.2c3.2 3.2 8.4 3.2 11.6 0"/><path d="M6.2 17.8c3.2-3.2 8.4-3.2 11.6 0"/><path d="M2 12h20"/><path d="M12 2v20"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "settings": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "versus": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "trophy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%;"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>'
}

# ── Global Matplotlib Style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.facecolor': '#070b14',
    'axes.facecolor': '#0d1220',
    'axes.edgecolor': 'none',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#1e293b', 
    'axes.grid': True
})

st.set_page_config(page_title="March Madness Predictor", layout="wide", initial_sidebar_state="expanded")

# ── Global CSS Injection ────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500&family=Barlow+Condensed:wght@700;800&display=swap');

.stApp {{
    background-color: #070b14;
    background-image: 
        radial-gradient(circle at 0% 0%, rgba(255, 107, 0, 0.05) 0%, transparent 25%),
        radial-gradient(circle at 100% 100%, rgba(220, 38, 38, 0.05) 0%, transparent 25%);
    color: #e2e8f0;
    font-family: 'Barlow', sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Barlow Condensed', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 800 !important;
}}

h1 {{
    background: linear-gradient(90deg, #ffa500, #ff6b00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important;
}}

section[data-testid="stSidebar"] {{
    background-color: #0d1220 !important;
    border-right: 2px solid #ff6b00;
}}

.data-card {{
    background-color: #0d1220;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}}

.svg-icon-container {{
    width: 24px;
    height: 24px;
    color: #ffa500;
}}

.stButton>button {{
    background: linear-gradient(90deg, #ffa500, #ff6b00) !important;
    color: #000000 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 15px rgba(255, 107, 0, 0.3) !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background-color: #0d1220 !important;
    border: 1px solid rgba(255, 165, 0, 0.2) !important;
    border-radius: 8px !important;
    padding: 15px !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Rest of Constants & Logic (Unchanged) ──────────────────────────────────
DATA_DIR = "uploaded_data"
MODEL_PATH = "trained_model.txt"
DATA_PKL_PATH = "app_data.pkl"
os.makedirs(DATA_DIR, exist_ok=True)
REQUIRED_KAGGLE = ["MRegularSeasonDetailedResults.csv", "MNCAATourneyDetailedResults.csv", "MTeams.csv", "MNCAATourneySeeds.csv", "MTeamConferences.csv"]
TORVIK_FILES = ["cbb.csv", "cbb26.csv"]
BEST_PARAMS = {'n_estimators': 181, 'max_depth': 3, 'learning_rate': 0.010705, 'subsample': 0.888, 'colsample_bytree': 0.852, 'min_child_samples': 21, 'reg_alpha': 0.000247, 'reg_lambda': 0.032461, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
BRACKET_2026 = ["Duke", "American Univ", "Mississippi St", "Boise St", "Wisconsin", "Montana St", "Kansas", "Howard", "Michigan St", "Bryant", "St John's", "NE Omaha", "Memphis", "Colorado St", "Auburn", "Alabama St", "Florida", "Norfolk St", "Connecticut", "New Hampshire", "Gonzaga", "McNeese St", "Arizona", "Akron", "Marquette", "Vermont", "Texas Tech", "NC Wilmington", "Missouri", "Drake", "Houston", "SIUE", "Tennessee", "Winthrop", "Michigan", "CS Bakersfield", "Iowa St", "Lipscomb", "Alabama", "Mt St Mary's", "BYU", "VCU", "Oregon", "Liberty", "Kentucky", "Troy", "Iowa", "S Dakota St", "Kansas St", "Montana", "Purdue", "TX Southern", "Baylor", "Colgate", "Maryland", "Grand Canyon", "Clemson", "New Mexico St", "UCLA", "UNC Asheville", "Mississippi", "Yale", "Texas A&M", "Morehead St"]

# ── Sidebar Navigation ─────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
    <div style='background: linear-gradient(135deg, #ffa500, #ff6b00); padding: 8px; border-radius: 8px; width: 40px; height: 40px; color: black;'>
        {SVG_ICONS['ball']}
    </div>
    <div style='font-family: "Barlow Condensed"; font-weight: 800; font-size: 20px; letter-spacing: 1px; color: #e2e8f0;'>MARCH MADNESS</div>
</div>
""", unsafe_allow_html=True)

# Using internal keys for radio to avoid emoji-dependency
page_options = {
    "upload": "DATA UPLOAD",
    "train": "TRAIN MODEL",
    "h2h": "HEAD TO HEAD",
    "bracket": "BRACKET SIMULATOR"
}
page_key = st.sidebar.radio("NAVIGATE", list(page_options.keys()), format_func=lambda x: page_options[x])

# Status indicators
kaggle_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_KAGGLE)
torvik_ok = any(os.path.exists(os.path.join(DATA_DIR, f)) for f in TORVIK_FILES)
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(DATA_PKL_PATH)

st.sidebar.markdown("---")
def status_item(label, state):
    dot_color = "#22c55e" if state == "ok" else "#ffa500" if state == "warn" else "#dc2626"
    return f"<div style='margin-bottom: 8px; font-size: 13px;'><span style='height:8px; width:8px; background-color:{dot_color}; border-radius:50%; display:inline-block; margin-right:8px;'></span>{label}</div>"

st.sidebar.markdown(f"""
<div class='data-card' style='padding: 12px; border-color: rgba(255,255,255,0.1);'>
    <div style='font-family: "Barlow Condensed"; color: #64748b; font-size: 11px; margin-bottom: 10px; letter-spacing: 1px;'>SYSTEM STATUS</div>
    {status_item("Kaggle Data", "ok" if kaggle_ok else "off")}
    {status_item("Torvik Data", "ok" if torvik_ok else "warn")}
    {status_item("ML Model", "ok" if model_ok else "off")}
</div>
""", unsafe_allow_html=True)

# ── Page Header Helper ─────────────────────────────────────────────────────
def page_header(title, subtitle, svg_key):
    st.markdown(f"""
    <div style='margin-bottom: 40px;'>
        <div style='display: flex; align-items: center; gap: 15px;'>
            <div style='background: linear-gradient(135deg, #ffa500, #ff6b00); width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border-radius: 10px; color: black; padding: 8px;'>
                {SVG_ICONS[svg_key]}
            </div>
            <h1>{title}</h1>
        </div>
        <div style='color: #64748b; font-family: "Barlow Condensed"; font-weight: 500; letter-spacing: 2px; margin-top: -10px;'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# [REDACTED Logic Functions: build_team_stats, train_model, etc. are identical to original]

@st.cache_resource
def load_model_and_data():
    if not model_ok: return None, None
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(DATA_PKL_PATH, 'rb') as f: d = pickle.load(f)
    return model, d

def build_team_stats(season_df, teams_df, conf_df, torvik_df=None):
    log = []
    log.append("Building team stats...")
    wins = season_df.groupby(['Season', 'WTeamID']).agg(W=('WTeamID', 'count'), pts_for=('WScore', 'mean'), pts_against=('LScore', 'mean'), fg_pct=('WFGM', lambda x: x.sum() / season_df.loc[x.index, 'WFGA'].sum()), fg3_pct=('WFGM3', lambda x: x.sum() / season_df.loc[x.index, 'WFGA3'].sum()), reb=('WOR', lambda x: (x + season_df.loc[x.index, 'WDR']).mean()), ast=('WAst', 'mean'), to=('WTO', 'mean')).reset_index().rename(columns={'WTeamID': 'TeamID'})
    losses = season_df.groupby(['Season', 'LTeamID']).agg(L=('LTeamID', 'count'), pts_for_l=('LScore', 'mean'), pts_against_l=('WScore', 'mean'), fg_pct_l=('LFGM', lambda x: x.sum() / season_df.loc[x.index, 'LFGA'].sum()), fg3_pct_l=('LFGM3', lambda x: x.sum() / season_df.loc[x.index, 'LFGA3'].sum()), reb_l=('LOR', lambda x: (x + season_df.loc[x.index, 'LDR']).mean()), ast_l=('LAst', 'mean'), to_l=('LTO', 'mean')).reset_index().rename(columns={'LTeamID': 'TeamID'})
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
    poss_w = season_df['WFGA'] - season_df['WOR'] + season_df['WTO'] + 0.44 * season_df['WFTA']
    poss_l = season_df['LFGA'] - season_df['LOR'] + season_df['LTO'] + 0.44 * season_df['LFTA']
    season_df = season_df.copy()
    season_df['poss'] = (poss_w + poss_l) / 2
    season_df['off_eff_w'] = season_df['WScore'] / season_df['poss'].clip(lower=1) * 100
    season_df['off_eff_l'] = season_df['LScore'] / season_df['poss'].clip(lower=1) * 100
    season_df['tempo'] = season_df['poss']
    eff_w = season_df.groupby(['Season', 'WTeamID']).agg(off_efficiency=('off_eff_w', 'mean'), def_efficiency=('off_eff_l', 'mean'), tempo=('tempo', 'mean')).reset_index().rename(columns={'WTeamID': 'TeamID'})
    eff_l = season_df.groupby(['Season', 'LTeamID']).agg(off_efficiency_l=('off_eff_l', 'mean'), def_efficiency_l=('off_eff_w', 'mean'), tempo_l=('tempo', 'mean')).reset_index().rename(columns={'LTeamID': 'TeamID'})
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
    stats = stats.merge(conf_df[['Season', 'TeamID', 'ConfAbbrev']], on=['Season', 'TeamID'], how='left')
    conf_stats = stats.groupby(['Season', 'ConfAbbrev']).agg(conf_avg_margin=('margin', 'mean'), conf_avg_win_pct=('win_pct', 'mean')).reset_index()
    stats = stats.merge(conf_stats, on=['Season', 'ConfAbbrev'], how='left')
    neutral = season_df[season_df['WLoc'] == 'N']
    n_wins = neutral.groupby(['Season', 'WTeamID']).size().reset_index(name='nw').rename(columns={'WTeamID': 'TeamID'})
    n_loss = neutral.groupby(['Season', 'LTeamID']).size().reset_index(name='nl').rename(columns={'LTeamID': 'TeamID'})
    neutral_stats = n_wins.merge(n_loss, on=['Season', 'TeamID'], how='outer').fillna(0)
    neutral_stats['neutral_win_pct'] = neutral_stats['nw'] / (neutral_stats['nw'] + neutral_stats['nl']).clip(lower=1)
    stats = stats.merge(neutral_stats[['Season', 'TeamID', 'neutral_win_pct']], on=['Season', 'TeamID'], how='left').fillna(0)
    torvik_features = ['ADJOE', 'ADJDE', 'BARTHAG', 'ADJ_T', 'WAB', 'EFG_O', 'EFG_D', 'TOR', 'TORD']
    if torvik_df is not None:
        available_torvik_cols = [f for f in torvik_features if f in torvik_df.columns]
        stats = stats.merge(torvik_df[['Season', 'TeamID'] + available_torvik_cols], on=['Season', 'TeamID'], how='left')
        for col in torvik_features:
            if col not in stats.columns: stats[col] = 0.0
            season_avg = stats.groupby('Season')[col].transform('mean')
            stats[col] = stats[col].fillna(season_avg).fillna(stats[col].mean())
        has_torvik = True
    else:
        for col in torvik_features: stats[col] = 0.0
        has_torvik = False
    return stats, log, has_torvik

def build_matchups(tourney_df, team_stats, seeds_df):
    log = []
    seeds_df = seeds_df.copy()
    seeds_df['SeedNum'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)
    stats_index = {(int(r.Season), int(r.TeamID)): r for _, r in team_stats.iterrows()}
    seeds_index = {(int(r.Season), int(r.TeamID)): r.SeedNum for _, r in seeds_df.iterrows()}
    all_features = ['seed_diff', 'margin_diff', 'win_pct_diff', 'pts_for_diff', 'pts_against_diff', 'fg_pct_diff', 'fg3_pct_diff', 'reb_diff', 'ast_diff', 'to_diff', 'off_eff_diff', 'def_eff_diff', 'tempo_diff', 'seed_t1', 'seed_t2', 'conf_margin_diff', 'conf_win_pct_diff', 'neutral_win_pct_diff', 'adjoe_diff', 'adjde_diff', 'barthag_diff', 'adj_t_diff', 'wab_diff', 'efg_o_diff', 'efg_d_diff', 'tor_diff', 'tord_diff']
    rows = []
    for _, game in tourney_df.iterrows():
        s, w, l = int(game.Season), int(game.WTeamID), int(game.LTeamID)
        sw, sl = stats_index.get((s, w)), stats_index.get((s, l))
        seedw, seedl = seeds_index.get((s, w)), seeds_index.get((s, l))
        if sw is None or sl is None or seedw is None or seedl is None: continue
        def diff(a, b, col): return getattr(a, col, 0) - getattr(b, col, 0)
        for t1, t2, seed1, seed2, label in [(sw, sl, seedw, seedl, 1), (sl, sw, seedl, seedw, 0)]:
            row = {'seed_diff': seed1 - seed2, 'margin_diff': diff(t1, t2, 'margin'), 'win_pct_diff': diff(t1, t2, 'win_pct'), 'pts_for_diff': diff(t1, t2, 'pts_for'), 'pts_against_diff': diff(t1, t2, 'pts_against'), 'fg_pct_diff': diff(t1, t2, 'fg_pct'), 'fg3_pct_diff': diff(t1, t2, 'fg3_pct'), 'reb_diff': diff(t1, t2, 'reb'), 'ast_diff': diff(t1, t2, 'ast'), 'to_diff': diff(t1, t2, 'to'), 'off_eff_diff': diff(t1, t2, 'off_efficiency'), 'def_eff_diff': diff(t1, t2, 'def_efficiency'), 'tempo_diff': diff(t1, t2, 'tempo'), 'seed_t1': seed1, 'seed_t2': seed2, 'conf_margin_diff': diff(t1, t2, 'conf_avg_margin'), 'conf_win_pct_diff': diff(t1, t2, 'conf_avg_win_pct'), 'neutral_win_pct_diff': diff(t1, t2, 'neutral_win_pct'), 'adjoe_diff': diff(t1, t2, 'ADJOE'), 'adjde_diff': diff(t1, t2, 'ADJDE'), 'barthag_diff': diff(t1, t2, 'BARTHAG'), 'adj_t_diff': diff(t1, t2, 'ADJ_T'), 'wab_diff': diff(t1, t2, 'WAB'), 'efg_o_diff': diff(t1, t2, 'EFG_O'), 'efg_d_diff': diff(t1, t2, 'EFG_D'), 'tor_diff': diff(t1, t2, 'TOR'), 'tord_diff': diff(t1, t2, 'TORD'), 'label': label}
            rows.append(row)
    return pd.DataFrame(rows), all_features, log

def train_model(matchups, features, log_list):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    X, y = matchups[features].values.astype(np.float32), matchups['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = lgb.LGBMClassifier(**BEST_PARAMS, verbosity=-1)
    clf.fit(X_train, y_train)
    y_pred, y_prob = clf.predict(X_test), clf.predict_proba(X_test)[:, 1]
    return clf, accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob), log_list

def predict_winner_prob(model, t1_id, t2_id, season, stats_index, seeds_index, feature_order):
    s1, s2 = stats_index.get((int(season), int(t1_id))), stats_index.get((int(season), int(t2_id)))
    seed1, seed2 = seeds_index.get((int(season), int(t1_id))), seeds_index.get((int(season), int(t2_id)))
    if not all([s1, s2, seed1, seed2]): return 0.5
    def g(obj, k): return obj.get(k, 0) if isinstance(obj, dict) else getattr(obj, k, 0)
    row = {'seed_diff': seed1 - seed2, 'margin_diff': g(s1,'margin')-g(s2,'margin'), 'win_pct_diff': g(s1,'win_pct')-g(s2,'win_pct'), 'pts_for_diff': g(s1,'pts_for')-g(s2,'pts_for'), 'pts_against_diff': g(s1,'pts_against')-g(s2,'pts_against'), 'fg_pct_diff': g(s1,'fg_pct')-g(s2,'fg_pct'), 'fg3_pct_diff': g(s1,'fg3_pct')-g(s2,'fg3_pct'), 'reb_diff': g(s1,'reb')-g(s2,'reb'), 'ast_diff': g(s1,'ast')-g(s2,'ast'), 'to_diff': g(s1,'to')-g(s2,'to'), 'off_eff_diff': g(s1,'off_efficiency')-g(s2,'off_efficiency'), 'def_eff_diff': g(s1,'def_efficiency')-g(s2,'def_efficiency'), 'tempo_diff': g(s1,'tempo')-g(s2,'tempo'), 'seed_t1': seed1, 'seed_t2': seed2, 'conf_margin_diff': g(s1,'conf_avg_margin')-g(s2,'conf_avg_margin'), 'conf_win_pct_diff': g(s1,'conf_avg_win_pct')-g(s2,'conf_avg_win_pct'), 'neutral_win_pct_diff': g(s1,'neutral_win_pct')-g(s2,'neutral_win_pct'), 'adjoe_diff': g(s1,'ADJOE')-g(s2,'ADJOE'), 'adjde_diff': g(s1,'ADJDE')-g(s2,'ADJDE'), 'barthag_diff': g(s1,'BARTHAG')-g(s2,'BARTHAG'), 'adj_t_diff': g(s1,'ADJ_T')-g(s2,'ADJ_T'), 'wab_diff': g(s1,'WAB')-g(s2,'WAB'), 'efg_o_diff': g(s1,'EFG_O')-g(s2,'EFG_O'), 'efg_d_diff': g(s1,'EFG_D')-g(s2,'EFG_D'), 'tor_diff': g(s1,'TOR')-g(s2,'TOR'), 'tord_diff': g(s1,'TORD')-g(s2,'TORD')}
    return float(model.predict(np.array([[row[f] for f in feature_order]], dtype=np.float32))[0])

# ── Page: Data Upload ──────────────────────────────────────────────────────
if page_key == "upload":
    page_header("Data Repository", "HISTORICAL PERFORMANCE & EFFICIENCY DATASET", "upload")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='data-card' style='border-top: 4px solid #ffa500;'>", unsafe_allow_html=True)
        st.subheader("Kaggle Authority Files")
        uploads = st.file_uploader("Upload required CSVs", type="csv", accept_multiple_files=True, key="k")
        if uploads:
            for f in uploads:
                if f.name in REQUIRED_KAGGLE:
                    with open(os.path.join(DATA_DIR, f.name), 'wb') as out: out.write(f.read())
                    st.markdown(f"<span style='background:#22c55e; color:black; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:5px; font-weight:700;'>{f.name}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='data-card' style='border-top: 4px solid #ff6b00;'>", unsafe_allow_html=True)
        st.subheader("Torvik Efficiency Logs")
        t_uploads = st.file_uploader("Upload optional metrics", type="csv", accept_multiple_files=True, key="t")
        if t_uploads:
            for f in t_uploads:
                if f.name in TORVIK_FILES:
                    with open(os.path.join(DATA_DIR, f.name), 'wb') as out: out.write(f.read())
                    st.markdown(f"<span style='background:#22c55e; color:black; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:5px; font-weight:700;'>{f.name}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Page: Train Model ──────────────────────────────────────────────────────
elif page_key == "train":
    page_header("Engine Room", "LIGHTGBM MODEL TRAINING & HYPERPARAMETER TUNING", "settings")
    
    if st.button("EXECUTE TRAINING CYCLE"):
        with st.spinner("Processing..."):
            season_df = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
            tourney_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyDetailedResults.csv"))
            teams_df = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
            seeds_df = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
            conf_df = pd.read_csv(os.path.join(DATA_DIR, "MTeamConferences.csv"))
            
            team_stats, _, has_torvik = build_team_stats(season_df, teams_df, conf_df)
            matchups, features, _ = build_matchups(tourney_df, team_stats, seeds_df)
            clf, acc, auc, _ = train_model(matchups, features, [])
            
            clf.booster_.save_model(MODEL_PATH)
            with open(DATA_PKL_PATH, 'wb') as f:
                pickle.dump({'feature_order': features, 'teams': teams_df.to_dict(), 'latest_season': int(seeds_df['Season'].max()), 'stats_index': {(int(r.Season), int(r.TeamID)): dict(r) for _, r in team_stats.iterrows()}, 'seeds_index': {(int(r.Season), int(r.TeamID)): int(s.split()[0][1:]) for _, (s, r) in seeds_df[['Seed', 'TeamID']].iterrows()}, 'has_torvik': has_torvik, 'auc': auc, 'accuracy': acc}, f)
            
            st.markdown("<div class='data-card' style='font-family: monospace; color: #22c55e;'>", unsafe_allow_html=True)
            st.text(">>> Training initialized...\n>>> Feature engineering complete.\n>>> LightGBM convergence reached.\n>>> Artifacts saved.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("MODEL AUC", f"{auc:.4f}")
            m2.metric("ACCURACY", f"{acc:.2%}")
            m3.metric("MATCHUPS", f"{len(matchups):,}")

# ── Page: Head to Head ─────────────────────────────────────────────────────
elif page_key == "h2h":
    page_header("Matchup Lab", "DIRECT PROBABILITY COMPARISON", "versus")
    model, d = load_model_and_data()
    if not model: st.stop()
    
    teams_df = pd.DataFrame.from_dict(d['teams'])
    team_list = sorted(teams_df['TeamName'].tolist())
    
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("SELECT TEAM A", team_list, index=team_list.index("Duke") if "Duke" in team_list else 0)
    t2 = col2.selectbox("SELECT TEAM B", team_list, index=team_list.index("Kansas") if "Kansas" in team_list else 0)
    
    id1 = teams_df[teams_df['TeamName'] == t1]['TeamID'].values[0]
    id2 = teams_df[teams_df['TeamName'] == t2]['TeamID'].values[0]
    
    prob = predict_winner_prob(model, id1, id2, d['latest_season'], d['stats_index'], d['seeds_index'], d['feature_order'])
    
    st.markdown(f"""
    <div style='text-align: center; margin-top: 50px;'>
        <div style='font-family: "Barlow Condensed"; font-size: 80px; font-weight: 800; color: #ffa500;'>{prob:.1%}</div>
        <div style='color: #64748b; letter-spacing: 5px; font-family: "Barlow Condensed"; font-weight: 500;'>WIN PROBABILITY FOR {t1.upper()}</div>
    </div>
    
    <div style='display: flex; height: 44px; border-radius: 22px; overflow: hidden; margin: 40px 0; border: 1px solid rgba(255,255,255,0.05);'>
        <div style='width: {prob*100}%; background: linear-gradient(90deg, #ffa500, #ff6b00); display: flex; align-items: center; padding-left: 24px; font-weight: 800; color: black; font-family: "Barlow Condensed"; font-size: 16px;'>{t1.upper()}</div>
        <div style='width: {(1-prob)*100}%; background: #1e293b; display: flex; align-items: center; justify-content: flex-end; padding-right: 24px; font-weight: 800; font-family: "Barlow Condensed"; font-size: 16px; color: #e2e8f0;'>{t2.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Page: Bracket Simulator ────────────────────────────────────────────────
else:
    page_header("The Gauntlet", "MONTE CARLO TOURNAMENT SIMULATION", "trophy")
    st.markdown("<div class='data-card'>Ready to simulate the 2026 bracket. Run Monte Carlo trials to identify championship contenders.</div>", unsafe_allow_html=True)
