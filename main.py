
# %% Libraries
from functions.football_match_prediction import FootballPredictor
from functions.utils import load_config

# Initialize predictor
predictor = FootballPredictor()

# %% Load config file
config = load_config("conf/config_dev.yaml")

# %% Load data (replace with your actual path)
predictor.load_data(config["files"]["datasets"]["input"]["spain_league_csv"])  # or whatever your file is called

# %% Train models
predictor.train_all_models()

# %% Make predictions
TEAM_NAME_MAPPING = {
    "Real Oviedo": "Oviedo",
    "RCD Espanyol": "Espanyol",
    "R.C.D. Mallorca": "Mallorca",
    "Real Betis": "Betis",
    "Atlético Madrid": "Ath Madrid",
    "Elche C. F.": "Elche",
    "Athletic": "Ath Bilbao",
    "Celta de Vigo": "Celta",
    "Real Sociedad": "Sociedad",
    "Alavés": "Alaves",
    "Valencia C. F.": "Valencia",
}

spain_matches = [
    {"home_team": "Elche C. F.", "away_team": "Real Sociedad"},
    {"home_team": "Girona", "away_team": "Alavés"},
    {"home_team": "Sevilla", "away_team": "Osasuna"},
    {"home_team": "Atlético Madrid", "away_team": "Levante"},
    {"home_team": "RCD Espanyol", "away_team": "Villareal"},
    {"home_team": "Athletic", "away_team": "Real Oviedo"},
    {"home_team": "Rayo Vallecano", "away_team": "Real Madrid"},
    {"home_team": "R.C.D. Mallorca", "away_team": "Getafe"},
    {"home_team": "Valencia C. F.", "away_team": "Real Betis"},
    {"home_team": "Celta de Vigo", "away_team": "Barcelona"},
]

def clean_team_name(team_name):
    """Retorna el nombre limpio del equipo o el original si no está en el mapeo."""
    return TEAM_NAME_MAPPING.get(team_name, team_name)

def print_match_prediction(home_team, away_team):
    """Imprime la predicción formateada de un partido."""
    home_clean = clean_team_name(home_team)
    away_clean = clean_team_name(away_team)
    
    print(f"\n{'='*60}")
    print(f"PREDICTION: {home_team} vs {away_team}")
    print(f"{'='*60}")
    
    return home_clean, away_clean

# Procesar cada partido
for match in spain_matches:
    home_clean, away_clean = print_match_prediction(match["home_team"], match["away_team"])
    # Get detailed prediction
    prediction = predictor.predict_match(home_clean, away_clean)

    print(f"\nExpected Goals:")
    print(f"  {home_clean}: {prediction['expected_goals']['home']}")
    print(f"  {away_clean}: {prediction['expected_goals']['away']}")
    print(f"  Total: {prediction['expected_goals']['total']}")

    print(f"\nOutcome Probabilities:")
    print(f"  Home Win: {prediction['outcome_probabilities']['home_win']:.1%}")
    print(f"  Draw: {prediction['outcome_probabilities']['draw']:.1%}")
    print(f"  Away Win: {prediction['outcome_probabilities']['away_win']:.1%}")

    print(f"\nGoal Intervals:")
    intervals = prediction["goal_intervals"]
    print(f"  Over 1.5 goals: {intervals['over_1_5']:.1%}")
    print(f"  Over 2.5 goals: {intervals['over_2_5']:.1%}")
    print(f"  Over 3.5 goals: {intervals['over_3_5']:.1%}")

    print(f"\nConfidence Interval:")
    ci = prediction["confidence_interval"]
    print(f"  Expected: {ci['total_goals_mean']} ± {ci['total_goals_std']:.1f}")
    print(f"  Likely range: {ci['likely_range'][0]} - {ci['likely_range'][1]} goals")

    # Get betting recommendations
    recommendations = predictor.get_goal_recommendations(home_clean, away_clean)

    print(f"\nBetting Recommendations:")
    for bet, tip in recommendations["betting_tips"].items():
        prob = recommendations["probabilities"].get(bet, "N/A")
        print(f"  {bet.upper()}: {tip} ({prob})")


# %%
