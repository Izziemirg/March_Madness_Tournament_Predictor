import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏀 March Madness Predictor 2026", layout="wide")

# ── 2026 Bracket — update after Selection Sunday (March 15) ───────────────
# Format: pairs in matchup order, region by region
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


@st.cache_resource
def load_model_and_data():
    model = lgb.Booster(model_file='lgbm_model.txt')
    with open('app_data.pkl', 'rb') as f:
        d = pickle.load(f)
    return model, d

model, d = load_model_and_data()

feature_order = d['feature_order']
teams_dict = d['teams']
stats_index = d['stats_index']
seeds_index = d['seeds_index']
latest_season = d['latest_season']

teams_df = pd.DataFrame.from_dict(teams_dict)
team_lookup = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
name_to_id = {v: k for k, v in team_lookup.items()}
all_team_names = sorted(name_to_id.keys())


# ── Core prediction functions ──────────────────────────────────────────────
def get_matchup_features(t1_id, t2_id, season):
    s1 = stats_index.get((int(season), int(t1_id)))
    s2 = stats_index.get((int(season), int(t2_id)))
    seed1 = seeds_index.get((int(season), int(t1_id)))
    seed2 = seeds_index.get((int(season), int(t2_id)))

    if s1 is None or s2 is None or seed1 is None or seed2 is None:
        return None, None, None

    row = {
        'seed_diff': seed1 - seed2,
        'margin_diff': s1.get('margin', 0) - s2.get('margin', 0),
        'win_pct_diff': s1.get('win_pct', 0) - s2.get('win_pct', 0),
        'pts_for_diff': s1.get('pts_for', 0) - s2.get('pts_for', 0),
        'pts_against_diff': s1.get('pts_against', 0) - s2.get('pts_against', 0),
        'fg_pct_diff': s1.get('fg_pct', 0) - s2.get('fg_pct', 0),
        'fg3_pct_diff': s1.get('fg3_pct', 0) - s2.get('fg3_pct', 0),
        'reb_diff': s1.get('reb', 0) - s2.get('reb', 0),
        'ast_diff': s1.get('ast', 0) - s2.get('ast', 0),
        'to_diff': s1.get('to', 0) - s2.get('to', 0),
        'off_eff_diff': s1.get('off_efficiency', 0) - s2.get('off_efficiency', 0),
        'def_eff_diff': s1.get('def_efficiency', 0) - s2.get('def_efficiency', 0),
        'tempo_diff': s1.get('tempo', 0) - s2.get('tempo', 0),
        'seed_t1': seed1,
        'seed_t2': seed2,
        'conf_margin_diff': s1.get('conf_avg_margin', 0) - s2.get('conf_avg_margin', 0),
        'conf_win_pct_diff': s1.get('conf_avg_win_pct', 0) - s2.get('conf_avg_win_pct', 0),
        'neutral_win_pct_diff': s1.get('neutral_win_pct', 0) - s2.get('neutral_win_pct', 0),
        'adjoe_diff': s1.get('ADJOE', 0) - s2.get('ADJOE', 0),
        'adjde_diff': s1.get('ADJDE', 0) - s2.get('ADJDE', 0),
        'barthag_diff': s1.get('BARTHAG', 0) - s2.get('BARTHAG', 0),
        'adj_t_diff': s1.get('ADJ_T', 0) - s2.get('ADJ_T', 0),
        'wab_diff': s1.get('WAB', 0) - s2.get('WAB', 0),
        'efg_o_diff': s1.get('EFG_O', 0) - s2.get('EFG_O', 0),
        'efg_d_diff': s1.get('EFG_D', 0) - s2.get('EFG_D', 0),
        'tor_diff': s1.get('TOR', 0) - s2.get('TOR', 0),
        'tord_diff': s1.get('TORD', 0) - s2.get('TORD', 0),
    }
    X = np.array([[row[f] for f in feature_order]], dtype=np.float32)
    return X, seed1, seed2


def predict_winner_prob(t1_id, t2_id, season):
    X, _, _ = get_matchup_features(t1_id, t2_id, season)
    if X is None:
        return 0.5
    return float(model.predict(X)[0])


def simulate_tournament(bracket_ids, season):
    teams = bracket_ids.copy()
    while len(teams) > 1:
        next_round = []
        if len(teams) % 2 != 0:
            next_round.append(teams.pop())
        for i in range(0, len(teams), 2):
            p = predict_winner_prob(teams[i], teams[i+1], season)
            next_round.append(teams[i] if random.random() < p else teams[i+1])
        teams = next_round
    return teams[0]


def run_monte_carlo_live(bracket_ids, season, n_sims=1000, progress_bar=None):
    champ_counts = {}
    for i in range(n_sims):
        champ = simulate_tournament(bracket_ids, season)
        champ_counts[champ] = champ_counts.get(champ, 0) + 1
        if progress_bar and i % 50 == 0:
            progress_bar.progress(i / n_sims, text=f"Simulating... {i}/{n_sims}")
    if progress_bar:
        progress_bar.progress(1.0, text="Done!")
    return {
        tid: count / n_sims
        for tid, count in sorted(champ_counts.items(), key=lambda x: -x[1])
    }


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🏀 March Madness Predictor 2026")
st.caption("LightGBM + Bart Torvik efficiency ratings · AUC: 0.771 · Update bracket after Selection Sunday (March 15)")

tab1, tab2 = st.tabs(["🏆 Bracket Simulator", "🆚 Head to Head"])


# ── Tab 1: Bracket Simulator ───────────────────────────────────────────────
with tab1:
    st.subheader("2026 NCAA Tournament Bracket Simulator")
    st.info("🗓️ Bracket will be updated after Selection Sunday on March 15, 2026")

    # Validate bracket teams
    missing = [t for t in BRACKET_2026 if name_to_id.get(t) is None]
    if missing:
        st.warning(f"⚠️ These teams aren't in the dataset: {', '.join(missing)}")

    st.markdown("**2026 Bracket Teams:**")
    cols = st.columns(4)
    regions = ["East", "West", "South", "Midwest"]
    for i, region in enumerate(regions):
        region_teams = BRACKET_2026[i*16:(i+1)*16]
        with cols[i]:
            st.markdown(f"**{region}**")
            for j in range(0, 16, 2):
                st.markdown(f"- {region_teams[j]} vs {region_teams[j+1]}")

    if st.button("🎲 Run 1,000 Simulations", type="primary"):
        bracket_ids = [name_to_id.get(t) for t in BRACKET_2026]

        if None in bracket_ids:
            st.error("Some teams not found — check team names match the dataset exactly.")
        else:
            progress = st.progress(0, text="Starting simulations...")

            with st.spinner("Running Monte Carlo simulation..."):
                results = run_monte_carlo_live(
                    bracket_ids, latest_season,
                    n_sims=1000, progress_bar=progress
                )

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
            ax.set_title('🏆 2026 March Madness Championship Odds\n(1,000 simulations · LightGBM + Torvik · AUC 0.771)',
                         color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🏆 Top Championship Contenders")
            medals = ["🥇", "🥈", "🥉"]
            cols2 = st.columns(3)
            for i, (tid, prob) in enumerate(top15):
                name = team_lookup.get(tid, str(tid))
                medal = medals[i] if i < 3 else f"{i+1}."
                col_idx = i % 3
                with cols2[col_idx]:
                    st.metric(f"{medal} {name}", f"{prob:.1%}")


# ── Tab 2: Head to Head ────────────────────────────────────────────────────
with tab2:
    st.subheader("Head to Head Matchup Predictor")
    col1, col2 = st.columns(2)
    with col1:
        team1_name = st.selectbox("Team A", all_team_names, index=all_team_names.index("Duke"))
    with col2:
        team2_name = st.selectbox("Team B", all_team_names, index=all_team_names.index("Kansas"))

    if st.button("Predict 🏀", type="primary"):
        if team1_name == team2_name:
            st.error("Please select two different teams.")
        else:
            t1_id = name_to_id.get(team1_name)
            t2_id = name_to_id.get(team2_name)
            X, seed1, seed2 = get_matchup_features(t1_id, t2_id, latest_season)

            if X is None:
                st.error("Could not find stats for one or both teams.")
            else:
                prob_t1 = float(model.predict(X)[0])
                prob_t2 = 1 - prob_t1

                seed1_str = f"#{int(seed1)} seed" if seed1 else "unseeded"
                seed2_str = f"#{int(seed2)} seed" if seed2 else "unseeded"
                favored = team1_name if prob_t1 > 0.5 else team2_name
                emoji = "🔵" if prob_t1 > 0.5 else "🟠"

                st.markdown(f"### {team1_name} ({seed1_str}) vs {team2_name} ({seed2_str})")
                st.markdown(f"{emoji} **{favored} is favored**")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(f"{team1_name}", f"{prob_t1:.1%}", "Win Probability")
                with col_b:
                    st.metric(f"{team2_name}", f"{prob_t2:.1%}", "Win Probability")

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
                ).sort_values(ascending=False).head(5)

                ax2.barh(importance.index[::-1], importance.values[::-1], color='#3b82f6')
                ax2.set_title('Top 5 Features', color='white', fontweight='bold')
                ax2.tick_params(colors='white')
                ax2.spines[:].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
