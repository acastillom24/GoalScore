# %% Libraries
from loguru import logger

from functions.football_match_prediction import football_predictor as predictor
from functions.utils import load_config

# %% Load config file
config = load_config()

# %% Load data (replace with your actual path)
predictor.load_data(
    config["files"]["datasets"]["input"]["spain_league_csv"]
)  # or whatever your file is called

# %% Train models
predictor.train_all_models()

# %% Make predictions
MAPPING_SPAIN = config["mapping_teams"]["spain"]
MATCHES_SPAIN = config["matches_teams"]["spain"]


def clean_team_name(team_name):
    """Retorna el nombre limpio del equipo o el original si no está en el mapeo."""
    return MAPPING_SPAIN.get(team_name, team_name)


def print_match_prediction(home_team, away_team):
    """Imprime la predicción formateada de un partido."""
    home_clean = clean_team_name(home_team)
    away_clean = clean_team_name(away_team)

    logger.info(f"\n{'='*80}")
    logger.info(f"PREDICTION: {home_team} vs {away_team}")
    logger.info(f"{'='*80}")

    return home_clean, away_clean


# Procesar cada partido
for i in range(len(MATCHES_SPAIN["home_team"])):
    home_team = MATCHES_SPAIN["home_team"][i]
    away_team = MATCHES_SPAIN["away_team"][i]
    home_clean, away_clean = print_match_prediction(home_team, away_team)
    # Get detailed prediction
    prediction = predictor.predict_match(home_clean, away_clean)

    logger.info(f"\nExpected Goals:")
    logger.info(f"  {home_clean}: {prediction['expected_goals']['home']}")
    logger.info(f"  {away_clean}: {prediction['expected_goals']['away']}")
    logger.info(f"  Total: {prediction['expected_goals']['total']}")

    logger.info(f"\nOutcome Probabilities:")
    logger.info(f"  Home Win: {prediction['outcome_probabilities']['home_win']:.1%}")
    logger.info(f"  Draw: {prediction['outcome_probabilities']['draw']:.1%}")
    logger.info(f"  Away Win: {prediction['outcome_probabilities']['away_win']:.1%}")

    logger.info(f"\nGoal Intervals:")
    intervals = prediction["goal_intervals"]
    logger.info(f"  Over 1.5 goals: {intervals['over_1_5']:.1%}")
    logger.info(f"  Over 2.5 goals: {intervals['over_2_5']:.1%}")
    logger.info(f"  Over 3.5 goals: {intervals['over_3_5']:.1%}")

    logger.info(f"\nConfidence Interval:")
    ci = prediction["confidence_interval"]
    logger.info(f"  Expected: {ci['total_goals_mean']} ± {ci['total_goals_std']:.1f}")
    logger.info(
        f"  Likely range: {ci['likely_range'][0]} - {ci['likely_range'][1]} goals"
    )

    # Get betting recommendations
    recommendations = predictor.get_goal_recommendations(home_clean, away_clean)

    logger.info(f"\nBetting Recommendations:")
    for bet, tip in recommendations["betting_tips"].items():
        prob = recommendations["probabilities"].get(bet, "N/A")
        logger.info(f"  {bet.upper()}: {tip} ({prob})")

# %%
