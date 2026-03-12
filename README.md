## March Madness 2026 Predictor

A LightGBM-powered NCAA tournament bracket predictor with a Streamlit interface. Trains on historical Kaggle game data and Bart Torvik efficiency ratings to simulate championship odds via Monte Carlo.

Live app: https://tournament-predictor-95wc43ckrqd.streamlit.app

Features

1.) Head to Head — win probability for any two teams using a 27-feature model
2.) Bracket Simulator — Monte Carlo simulation (100–2000 runs) with championship odds leaderboard
3.) Data Upload — drag-and-drop CSV ingestion, persisted across sessions
4.) Train Model — full pipeline in-app: feature engineering, LightGBM training, model save


Data Requirements

Required — Kaggle NCAA March Madness:

MRegularSeasonDetailedResults.csv
MNCAATourneyDetailedResults.csv
MTeams.csv
MNCAATourneySeeds.csv
MTeamConferences.csv

Source: kaggle.com/competitions/march-machine-learning-mania-2026
Optional — Bart Torvik efficiency ratings (improves AUC ~1.5%):

cbb.csv
cbb26.csv

Source: kaggle.com/datasets/nishaanamin/march-madness-data

Usage

1.) Open the app and go to Data Upload — upload the Kaggle CSVs (and optionally the Torvik files)
2.) Go to Train Model and click Execute Training (~30–60 seconds)
3.) Use Head to Head to compare any two teams
4.) Use Bracket Simulator to run Monte Carlo simulations on the 2026 bracket

Tech Stack
Python, Streamlit, LightGBM, scikit-learn, pandas, NumPy, Matplotlib
