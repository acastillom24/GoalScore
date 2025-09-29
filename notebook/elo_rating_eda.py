# %% Load libraries
import pandas as pd

from functions.utils import load_config

# %% Load config
config = load_config("conf/config_dev.yaml")

# %% Load data
data = pd.read_csv(config["files"]["datasets"]["input"]["spain_league_csv"])
data.head()

# %% Prepare data
df = data.copy()
# Rename columns
column_mapping = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HG": "home_goals",  # Alternative naming
    "AG": "away_goals",  # Alternative naming
}
df = df.rename(columns=column_mapping)
# Standardize column names
df.columns = [col.lower() for col in df.columns]
# Parse dates
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
# Remove rows with missing essential data
df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

print(f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}")
print(
    f"Teams found {len(df['home_team'].unique())}: {sorted(df['home_team'].unique())}"
)

# %%
