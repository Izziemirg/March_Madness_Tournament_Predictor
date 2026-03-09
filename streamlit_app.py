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

st.set_page_config(page_title="🏀 March Madness Predictor", layout="wide", initial_sidebar_state="expanded")

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
    "St John's", "Omaha",
    "Memphis", "Colorado St",
    "Auburn", "Alabama St",
    # WEST
    "Florida", "Norfolk St",
    "UConn", "New Hampshire",
    "Gonzaga", "McNeese St",
    "Arizona", "Akron",
    "Marquette", "Vermont",
    "Texas Tech", "UNCW",
    "Missouri", "Drake",
    "Houston", "SIU Edwardsvll",
    # SOUTH
    "Tennessee", "Winthrop",
    "Michigan", "CS Bakersfield",
    "Iowa St", "Lipscomb",
    "Alabama", "Mount St. Mary's",
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
    "Ole Miss", "Yale",
    "Texas A&M", "Morehead St",
]

# ── Sidebar nav ────────────────────────────────────────────────────────────
st.sidebar.title("🏀 March Madness")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📁 Data Upload",
    "⚙️ Train Model",
    "🆚 Head to Head",
    "🏆 Bracket Simulator",
])

# Status indicators in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
kaggle_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_KAGGLE)
torvik_ok = any(os.path.exists(os.path.join(DATA_DIR, f)) for f in TORVIK_FILES)
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(DATA_PKL_PATH)

st.sidebar.markdown(f"{'✅' if kaggle_ok else '❌'} Kaggle data")
st.sidebar.markdown(f"{'✅' if torvik_ok else '⚠️'} Torvik data (optional)")
st.sidebar.markdown(f"{'✅' if model_ok else '❌'} Model trained")


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

    log.append(f"✅ Base stats built: {len(stats)} team-seasons")

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
        log.append(f"✅ Torvik features merged — {missing_after} missing after imputation (should be 0)")
        has_torvik = True
    else:
        for col in torvik_features:
            stats[col] = 0.0
        log.append("⚠️ No Torvik data — using zeros for Torvik features (lower AUC expected)")

    log.append(f"✅ Final stats shape: {stats.shape}")
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
    log.append(f"✅ Built {len(matchups)} matchup rows from {len(tourney_df)} tournament games")
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
        log_list.append(f"⚠️ Found {nan_count} NaN and {inf_count} inf values — replacing with 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log_list.append(f"✅ Feature matrix: {X.shape[0]} rows × {X.shape[1]} features, all finite")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LGBMClassifier(**BEST_PARAMS, verbosity=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    log_list.append(f"✅ Model trained — Accuracy: {acc:.3f} | AUC: {auc:.3f}")
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
if page == "📁 Data Upload":
    st.title("📁 Data Upload")
    st.markdown("Upload your data files once — they'll be saved and remembered across sessions.")

    st.markdown("---")
    st.subheader("Kaggle Files (Required)")
    st.caption("Download from: kaggle.com/competitions/march-machine-learning-mania-2026")

    kaggle_uploads = st.file_uploader(
        "Upload all 5 Kaggle CSV files",
        type="csv",
        accept_multiple_files=True,
        key="kaggle"
    )

    if kaggle_uploads:
        saved = []
        for f in kaggle_uploads:
            if f.name in REQUIRED_KAGGLE:
                path = os.path.join(DATA_DIR, f.name)
                with open(path, 'wb') as out:
                    out.write(f.read())
                saved.append(f.name)
        if saved:
            st.success(f"✅ Saved: {', '.join(saved)}")

    st.markdown("---")
    st.subheader("Torvik Files (Optional — improves AUC)")
    st.caption("Download cbb.csv + cbb26.csv from: kaggle.com/datasets/nishaanamin/march-madness-data")

    torvik_uploads = st.file_uploader(
        "Upload Torvik CSV files (cbb.csv and/or cbb26.csv)",
        type="csv",
        accept_multiple_files=True,
        key="torvik"
    )

    if torvik_uploads:
        saved = []
        for f in torvik_uploads:
            if f.name in TORVIK_FILES:
                path = os.path.join(DATA_DIR, f.name)
                with open(path, 'wb') as out:
                    out.write(f.read())
                saved.append(f.name)
        if saved:
            st.success(f"✅ Saved: {', '.join(saved)}")

    st.markdown("---")
    st.subheader("Current File Status")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Kaggle Files**")
        for f in REQUIRED_KAGGLE:
            exists = os.path.exists(os.path.join(DATA_DIR, f))
            st.markdown(f"{'✅' if exists else '❌'} {f}")
    with col2:
        st.markdown("**Torvik Files**")
        for f in TORVIK_FILES:
            exists = os.path.exists(os.path.join(DATA_DIR, f))
            st.markdown(f"{'✅' if exists else '⬜'} {f}")

    if kaggle_ok:
        st.success("✅ All required files present — head to **⚙️ Train Model** to continue!")
    else:
        missing = [f for f in REQUIRED_KAGGLE if not os.path.exists(os.path.join(DATA_DIR, f))]
        st.warning(f"Still needed: {', '.join(missing)}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Train Model":
    st.title("⚙️ Train Model")

    if not kaggle_ok:
        st.error("❌ Missing Kaggle files. Go to **📁 Data Upload** first.")
        st.stop()

    st.markdown("Trains a LightGBM model on historical tournament data using your best tuned hyperparameters.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training config**")
        st.markdown(f"- Algorithm: LightGBM")
        st.markdown(f"- Hyperparameters: Pre-tuned (100 Optuna trials)")
        st.markdown(f"- n_estimators: {BEST_PARAMS['n_estimators']}")
        st.markdown(f"- learning_rate: {BEST_PARAMS['learning_rate']:.4f}")
        st.markdown(f"- max_depth: {BEST_PARAMS['max_depth']}")
    with col2:
        st.markdown("**Expected performance**")
        if torvik_ok:
            st.markdown("- With Torvik: ~AUC 0.771")
        else:
            st.markdown("- Without Torvik: ~AUC 0.755")
        st.markdown("- Training time: ~30–60 seconds")

    st.markdown("---")

    if st.button("🚀 Train Model", type="primary"):
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
            log_lines.append("✅ Kaggle files loaded")

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
                        log_lines.append(f"✅ Torvik name mapping: {matched} rows matched, {unmatched} unique teams unmatched")

                        torvik_raw = torvik_raw.dropna(subset=['TeamID'])
                        torvik_raw['TeamID'] = torvik_raw['TeamID'].astype(int)
                        torvik_df = torvik_raw
                        log_lines.append(f"✅ Torvik data loaded: {len(torvik_df)} rows, seasons {sorted(torvik_df['Season'].unique())[:3]}...{sorted(torvik_df['Season'].unique())[-1]}")
                    else:
                        log_lines.append("⚠️ Could not find Year/Team columns in Torvik files")
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

            log_lines.append(f"✅ Model saved to {MODEL_PATH}")
            log_lines.append(f"✅ App data saved to {DATA_PKL_PATH}")
            log_lines.append(f"")
            log_lines.append(f"🏀 Ready! Head to Head and Bracket Simulator are now available.")
            update_log(log_lines)

        st.success(f"🎉 Training complete! Accuracy: {acc:.3f} | AUC: {auc:.3f}")
        st.cache_resource.clear()
        st.balloons()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — HEAD TO HEAD
# ══════════════════════════════════════════════════════════════════════════
elif page == "🆚 Head to Head":
    st.title("🆚 Head to Head Predictor")

    if not model_ok:
        st.error("❌ No trained model found. Go to **⚙️ Train Model** first.")
        st.stop()

    model, d = load_model_and_data()
    if model is None:
        st.error("❌ Could not load model.")
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

    st.caption(f"Model AUC: {auc:.3f} · Season: {latest_season} · {'With Torvik' if d.get('has_torvik') else 'Without Torvik'}")

    col1, col2 = st.columns(2)
    default_idx1 = all_team_names.index("Duke") if "Duke" in all_team_names else 0
    default_idx2 = all_team_names.index("Kansas") if "Kansas" in all_team_names else 1
    with col1:
        team1_name = st.selectbox("Team A", all_team_names, index=default_idx1)
    with col2:
        team2_name = st.selectbox("Team B", all_team_names, index=default_idx2)

    if st.button("Predict 🏀", type="primary"):
        if team1_name == team2_name:
            st.error("Please select two different teams.")
        else:
            t1_id = name_to_id.get(team1_name)
            t2_id = name_to_id.get(team2_name)

            prob_t1 = predict_winner_prob(model, t1_id, t2_id, latest_season,
                                          stats_index, seeds_index, feature_order)
            prob_t2 = 1 - prob_t1

            seed1 = seeds_index.get((int(latest_season), int(t1_id)))
            seed2 = seeds_index.get((int(latest_season), int(t2_id)))
            seed1_str = f"#{int(seed1)} seed" if seed1 else "unseeded"
            seed2_str = f"#{int(seed2)} seed" if seed2 else "unseeded"
            favored = team1_name if prob_t1 > 0.5 else team2_name

            st.markdown(f"### {team1_name} ({seed1_str}) vs {team2_name} ({seed2_str})")

            col_a, col_b, col_c = st.columns([2, 1, 2])
            with col_a:
                st.metric(team1_name, f"{prob_t1:.1%}")
            with col_b:
                st.markdown("<div style='text-align:center;padding-top:30px;font-size:24px'>VS</div>",
                            unsafe_allow_html=True)
            with col_c:
                st.metric(team2_name, f"{prob_t2:.1%}")

            emoji = "🔵" if prob_t1 > 0.5 else "🟠"
            st.markdown(f"### {emoji} **{favored} is favored**")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor('#0f172a')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#1e293b')

            ax1.barh([team2_name, team1_name],
                     [prob_t2 * 100, prob_t1 * 100],
                     color=['#f97316', '#3b82f6'], height=0.5)
            ax1.set_xlim(0, 110)
            ax1.set_xlabel('Win Probability (%)', color='white')
            ax1.set_title('Win Probability', color='white', fontweight='bold')
            ax1.tick_params(colors='white')
            for val, name in zip([prob_t2, prob_t1], [team2_name, team1_name]):
                ax1.text(val * 100 + 1.5, [team2_name, team1_name].index(name),
                         f'{val:.1%}', va='center', color='white', fontweight='bold')
            ax1.spines[:].set_visible(False)

            importance = pd.Series(
                model.feature_importance(importance_type='gain'),
                index=model.feature_name()
            ).sort_values(ascending=False).head(8)
            ax2.barh(importance.index[::-1], importance.values[::-1], color='#3b82f6')
            ax2.set_title('Top 8 Features (Model-wide)', color='white', fontweight='bold')
            ax2.tick_params(colors='white')
            ax2.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — BRACKET SIMULATOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏆 Bracket Simulator":
    st.title("🏆 Bracket Simulator")

    if not model_ok:
        st.error("❌ No trained model found. Go to **⚙️ Train Model** first.")
        st.stop()

    model, d = load_model_and_data()
    if model is None:
        st.error("❌ Could not load model.")
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

    st.caption(f"Model AUC: {auc:.3f} · 1,000 simulations · Update BRACKET_2026 after Selection Sunday (March 15)")
    st.info("🗓️ Bracket will be finalized on Selection Sunday, March 15, 2026")

    # Show bracket by region
    st.markdown("### 2026 Bracket")
    regions = ["East", "West", "South", "Midwest"]
    cols = st.columns(4)
    missing_teams = []
    for i, (region, col) in enumerate(zip(regions, cols)):
        region_teams = BRACKET_2026[i*16:(i+1)*16]
        with col:
            st.markdown(f"**{region}**")
            for j in range(0, 16, 2):
                t1, t2 = region_teams[j], region_teams[j+1]
                ok1 = name_to_id.get(t1) is not None
                ok2 = name_to_id.get(t2) is not None
                e1 = "✅" if ok1 else "⚠️"
                e2 = "✅" if ok2 else "⚠️"
                st.markdown(f"{e1} {t1} vs {e2} {t2}")
                if not ok1:
                    missing_teams.append(t1)
                if not ok2:
                    missing_teams.append(t2)

    if missing_teams:
        st.warning(f"⚠️ Teams not found in dataset (will use 50/50 odds): {', '.join(set(missing_teams))}")

    st.markdown("---")
    n_sims = st.slider("Number of simulations", min_value=100, max_value=2000, value=1000, step=100)

    if st.button(f"🎲 Run {n_sims:,} Simulations", type="primary"):
        bracket_ids = [name_to_id.get(t) for t in BRACKET_2026]
        # Replace None with a dummy ID that returns 0.5 win prob
        bracket_ids = [b if b is not None else -1 for b in bracket_ids]

        progress = st.progress(0, text="Starting simulations...")
        champ_counts = {}

        for i in range(n_sims):
            champ = simulate_tournament(model, bracket_ids, latest_season,
                                        stats_index, seeds_index, feature_order)
            champ_counts[champ] = champ_counts.get(champ, 0) + 1
            if i % 50 == 0:
                progress.progress(i / n_sims, text=f"Simulating... {i}/{n_sims}")

        progress.progress(1.0, text="Done!")

        results = {
            tid: count / n_sims
            for tid, count in sorted(champ_counts.items(), key=lambda x: -x[1])
            if tid != -1  # exclude dummy ID for unmatched teams
        }

        top15 = list(results.items())[:15]
        names = [team_lookup.get(tid, str(tid)) for tid, _ in top15]
        probs = [p * 100 for _, p in top15]
        colors = ['#f59e0b' if i == 0 else '#3b82f6' if i < 4 else '#64748b'
                  for i in range(len(top15))]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        bars = ax.barh(names[::-1], probs[::-1], color=colors[::-1], height=0.6)
        for bar, prob in zip(bars, probs[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{prob:.1f}%', va='center', color='white', fontweight='bold')
        ax.set_xlabel('Championship Probability (%)', color='white')
        ax.set_title(f'🏆 2026 March Madness Championship Odds\n({n_sims:,} simulations · LightGBM + Torvik · AUC {auc:.3f})',
                     color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Leaderboard
        st.markdown("### 🏆 Top Championship Contenders")
        medals = ["🥇", "🥈", "🥉"]
        lcols = st.columns(3)
        for i, (tid, prob) in enumerate(top15):
            name = team_lookup.get(tid, str(tid))
            medal = medals[i] if i < 3 else f"{i+1}."
            with lcols[i % 3]:
                st.metric(f"{medal} {name}", f"{prob:.1%}")
