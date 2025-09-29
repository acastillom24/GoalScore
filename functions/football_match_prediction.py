"""
Modular Football Match Prediction Toolkit
=========================================

Structure:
- data/: Contains CSV files with match data
- models/: Contains individual prediction model classes
- utils/: Utility functions for data processing
- predictor.py: Main class that orchestrates all predictions
- examples/: Usage examples

Usage:
    from predictor import FootballPredictor

    predictor = FootballPredictor('data/matches.csv')
    predictor.train_all_models()

    prediction = predictor.predict_match('Real Madrid', 'Barcelona')
    print(prediction)
"""

# =============================================================================
# UTILS MODULE
# =============================================================================

import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataProcessor:
    """Handles data loading and preprocessing for football matches."""

    @staticmethod
    def load_matches(path: str) -> pd.DataFrame:
        """Load and preprocess match data from CSV file.

        Expected columns: Date, HomeTeam, AwayTeam, FTHG, FTAG
        (Based on football-data.co.uk format)
        """
        try:
            df = pd.read_csv(path)

            # Standardize column names
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

            # Parse dates
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            # Remove rows with missing essential data
            df = df.dropna(
                subset=["date", "home_team", "away_team", "home_goals", "away_goals"]
            )

            print(
                f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}"
            )
            print(f"Teams found: {sorted(df['home_team'].unique())}")

            return df

        except Exception as e:
            raise ValueError(f"Error loading data from {path}: {str(e)}")

    @staticmethod
    def get_team_stats(matches: pd.DataFrame, team: str, last_n: int = 10) -> Dict:
        """Get recent statistics for a team."""
        home_matches = matches[matches["home_team"] == team].tail(last_n // 2)
        away_matches = matches[matches["away_team"] == team].tail(last_n // 2)

        home_goals_scored = (
            home_matches["home_goals"].mean() if len(home_matches) > 0 else 1.0
        )
        home_goals_conceded = (
            home_matches["away_goals"].mean() if len(home_matches) > 0 else 1.0
        )
        away_goals_scored = (
            away_matches["away_goals"].mean() if len(away_matches) > 0 else 1.0
        )
        away_goals_conceded = (
            away_matches["home_goals"].mean() if len(away_matches) > 0 else 1.0
        )

        return {
            "goals_scored_home": home_goals_scored,
            "goals_conceded_home": home_goals_conceded,
            "goals_scored_away": away_goals_scored,
            "goals_conceded_away": away_goals_conceded,
            "total_goals_scored": (home_goals_scored + away_goals_scored) / 2,
            "total_goals_conceded": (home_goals_conceded + away_goals_conceded) / 2,
        }


# =============================================================================
# MODELS MODULE
# =============================================================================

import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression


class EloRatingModel:
    """Elo rating system for football teams."""

    def __init__(
        self,
        k_factor: float = 20,
        initial_rating: float = 1500,
        home_advantage: float = 50,
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings = defaultdict(lambda: initial_rating)
        self.history = []

    def update_ratings(self, matches: pd.DataFrame):
        """Update Elo ratings based on match results."""
        self.ratings.clear()
        self.history.clear()

        for _, match in matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_goals = match["home_goals"]
            away_goals = match["away_goals"]

            # Current ratings
            home_rating = self.ratings[home_team]
            away_rating = self.ratings[away_team]

            # Expected scores
            expected_home = 1 / (
                1 + 10 ** (-(home_rating - away_rating + self.home_advantage) / 400)
            )

            # Actual scores
            if home_goals > away_goals:
                actual_home = 1.0
            elif home_goals < away_goals:
                actual_home = 0.0
            else:
                actual_home = 0.5

            # Update ratings
            rating_change = self.k_factor * (actual_home - expected_home)
            self.ratings[home_team] += rating_change
            self.ratings[away_team] -= rating_change

            # Store history
            self.history.append(
                {
                    "date": match["date"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_rating_before": home_rating,
                    "away_rating_before": away_rating,
                    "expected_home": expected_home,
                    "actual_home": actual_home,
                    "rating_change": rating_change,
                }
            )

    def get_rating(self, team: str) -> float:
        """Get current rating for a team."""
        return self.ratings.get(team, self.initial_rating)

    def predict_match_outcome(self, home_team: str, away_team: str) -> Dict:
        """Predict match outcome probabilities based on Elo ratings."""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        expected_home = 1 / (
            1 + 10 ** (-(home_rating - away_rating + self.home_advantage) / 400)
        )
        expected_draw = 0.3  # Simple approximation
        expected_away = 1 - expected_home

        # Adjust for draw probability
        expected_home *= 0.7
        expected_away *= 0.7

        return {
            "home_win_prob": expected_home,
            "draw_prob": expected_draw,
            "away_win_prob": expected_away,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "rating_difference": home_rating - away_rating,
        }


class PoissonGoalsModel:
    """Poisson regression model for predicting goal counts."""

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.teams = None
        self.team_index = None
        self.is_fitted = False

    def prepare_data(self, matches: pd.DataFrame):
        """Prepare design matrices for Poisson regression."""
        self.teams = sorted(
            pd.unique(matches[["home_team", "away_team"]].values.ravel())
        )
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)
        n_matches = len(matches)

        # Design matrix for home goals: intercept + home attack + away defense
        X_home = np.zeros((n_matches, 2 * n_teams + 1))
        y_home = matches["home_goals"].values

        # Design matrix for away goals: intercept + away attack + home defense
        X_away = np.zeros((n_matches, 2 * n_teams + 1))
        y_away = matches["away_goals"].values

        for i, (_, match) in enumerate(matches.iterrows()):
            home_idx = self.team_index[match["home_team"]]
            away_idx = self.team_index[match["away_team"]]

            # Home goals model
            X_home[i, 0] = 1  # intercept
            X_home[i, 1 + home_idx] = 1  # home attack
            X_home[i, 1 + n_teams + away_idx] = 1  # away defense

            # Away goals model
            X_away[i, 0] = 1  # intercept
            X_away[i, 1 + away_idx] = 1  # away attack
            X_away[i, 1 + n_teams + home_idx] = 1  # home defense

        return X_home, y_home, X_away, y_away

    def fit(self, matches: pd.DataFrame):
        """Fit Poisson regression models for home and away goals."""
        X_home, y_home, X_away, y_away = self.prepare_data(matches)

        try:
            self.home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
            self.away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
            self.is_fitted = True

            print(f"Poisson model fitted successfully")
            print(f"Home model AIC: {self.home_model.aic:.2f}")
            print(f"Away model AIC: {self.away_model.aic:.2f}")

        except Exception as e:
            print(f"Error fitting Poisson model: {str(e)}")
            self.is_fitted = False

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict:
        """Predict goals and match outcomes for a specific match."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if home_team not in self.team_index or away_team not in self.team_index:
            # Use league averages for unknown teams
            lambda_home = 1.4
            lambda_away = 1.1
        else:
            # Build design matrices for this match
            n_teams = len(self.teams)
            home_idx = self.team_index[home_team]
            away_idx = self.team_index[away_team]

            X_home = np.zeros((1, 2 * n_teams + 1))
            X_home[0, 0] = 1
            X_home[0, 1 + home_idx] = 1
            X_home[0, 1 + n_teams + away_idx] = 1

            X_away = np.zeros((1, 2 * n_teams + 1))
            X_away[0, 0] = 1
            X_away[0, 1 + away_idx] = 1
            X_away[0, 1 + n_teams + home_idx] = 1

            lambda_home = float(self.home_model.predict(X_home)[0])
            lambda_away = float(self.away_model.predict(X_away)[0])

        # Calculate match outcome probabilities
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(
                    j, lambda_away
                )

        # Calculate outcome probabilities
        p_home_win = np.sum(
            [
                prob_matrix[i, j]
                for i in range(max_goals + 1)
                for j in range(max_goals + 1)
                if i > j
            ]
        )
        p_draw = np.sum([prob_matrix[i, i] for i in range(max_goals + 1)])
        p_away_win = 1 - p_home_win - p_draw

        # Calculate total goals probabilities
        total_goals_probs = {}
        for total in range(2 * max_goals + 1):
            prob = np.sum(
                [
                    prob_matrix[i, j]
                    for i in range(max_goals + 1)
                    for j in range(max_goals + 1)
                    if i + j == total
                ]
            )
            total_goals_probs[total] = prob

        return {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "expected_total_goals": lambda_home + lambda_away,
            "home_win_prob": p_home_win,
            "draw_prob": p_draw,
            "away_win_prob": p_away_win,
            "prob_matrix": prob_matrix,
            "total_goals_probs": total_goals_probs,
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for match outcomes."""

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_match(self, lambda_home: float, lambda_away: float) -> Dict:
        """Simulate match outcomes using Poisson distributions."""
        rng = np.random.default_rng()

        home_goals = rng.poisson(lambda_home, self.n_simulations)
        away_goals = rng.poisson(lambda_away, self.n_simulations)
        total_goals = home_goals + away_goals

        # Calculate probabilities
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)

        # Goal interval analysis
        goal_intervals = {
            "under_1_5": np.mean(total_goals < 1.5),
            "under_2_5": np.mean(total_goals < 2.5),
            "under_3_5": np.mean(total_goals < 3.5),
            "over_1_5": np.mean(total_goals > 1.5),
            "over_2_5": np.mean(total_goals > 2.5),
            "over_3_5": np.mean(total_goals > 3.5),
        }

        return {
            "home_win_prob": home_wins,
            "draw_prob": draws,
            "away_win_prob": away_wins,
            "expected_home_goals": home_goals.mean(),
            "expected_away_goals": away_goals.mean(),
            "expected_total_goals": total_goals.mean(),
            "goal_intervals": goal_intervals,
            "total_goals_std": total_goals.std(),
            "simulated_scores": list(
                zip(home_goals[:10], away_goals[:10])
            ),  # Sample results
        }


# =============================================================================
# MAIN PREDICTOR CLASS
# =============================================================================


class FootballPredictor:
    """Main class that orchestrates all prediction models."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.matches = None
        self.elo_model = EloRatingModel()
        self.poisson_model = PoissonGoalsModel()
        self.monte_carlo = MonteCarloSimulator()
        self.is_trained = False

        if data_path:
            self.load_data(data_path)

    def load_data(self, path: str):
        """Load match data from CSV file."""
        self.matches = DataProcessor.load_matches(path)
        self.data_path = path

    def train_all_models(self):
        """Train all prediction models."""
        if self.matches is None:
            raise ValueError("No data loaded. Use load_data() first.")

        print("Training Elo ratings...")
        self.elo_model.update_ratings(self.matches)

        print("Training Poisson model...")
        self.poisson_model.fit(self.matches)

        self.is_trained = True
        print("All models trained successfully!")

    def predict_match(
        self, home_team: str, away_team: str, detailed: bool = True
    ) -> Dict:
        """Predict match outcome with goal intervals."""
        if not self.is_trained:
            raise ValueError("Models must be trained first. Use train_all_models().")

        # Get predictions from all models
        elo_pred = self.elo_model.predict_match_outcome(home_team, away_team)
        poisson_pred = self.poisson_model.predict_match(home_team, away_team)

        # Monte Carlo simulation
        mc_pred = self.monte_carlo.simulate_match(
            poisson_pred["lambda_home"], poisson_pred["lambda_away"]
        )

        # Combine predictions
        prediction = {
            "match": f"{home_team} vs {away_team}",
            "expected_goals": {
                "home": round(poisson_pred["lambda_home"], 2),
                "away": round(poisson_pred["lambda_away"], 2),
                "total": round(poisson_pred["expected_total_goals"], 2),
            },
            "outcome_probabilities": {
                "home_win": round(
                    (elo_pred["home_win_prob"] + mc_pred["home_win_prob"]) / 2, 3
                ),
                "draw": round((elo_pred["draw_prob"] + mc_pred["draw_prob"]) / 2, 3),
                "away_win": round(
                    (elo_pred["away_win_prob"] + mc_pred["away_win_prob"]) / 2, 3
                ),
            },
            "goal_intervals": mc_pred["goal_intervals"],
            "confidence_interval": {
                "total_goals_mean": round(mc_pred["expected_total_goals"], 2),
                "total_goals_std": round(mc_pred["total_goals_std"], 2),
                "likely_range": (
                    max(
                        0,
                        round(
                            mc_pred["expected_total_goals"]
                            - mc_pred["total_goals_std"],
                            1,
                        ),
                    ),
                    round(
                        mc_pred["expected_total_goals"] + mc_pred["total_goals_std"], 1
                    ),
                ),
            },
        }

        if detailed:
            prediction["detailed"] = {
                "elo_ratings": {
                    "home": round(elo_pred["home_rating"], 0),
                    "away": round(elo_pred["away_rating"], 0),
                    "difference": round(elo_pred["rating_difference"], 0),
                },
                "poisson_lambdas": {
                    "home": poisson_pred["lambda_home"],
                    "away": poisson_pred["lambda_away"],
                },
                "sample_scores": mc_pred["simulated_scores"],
            }

        return prediction

    def get_goal_recommendations(self, home_team: str, away_team: str) -> Dict:
        """Get betting recommendations based on goal predictions."""
        pred = self.predict_match(home_team, away_team)

        intervals = pred["goal_intervals"]
        expected_total = pred["expected_goals"]["total"]

        recommendations = {
            "most_likely_total_goals": round(expected_total),
            "confidence_range": pred["confidence_interval"]["likely_range"],
            "betting_tips": {
                "over_2_5": "YES" if intervals["over_2_5"] > 0.55 else "NO",
                "under_2_5": "YES" if intervals["under_2_5"] > 0.55 else "NO",
                "over_1_5": "YES" if intervals["over_1_5"] > 0.70 else "NO",
                "btts": (
                    "YES"
                    if pred["expected_goals"]["home"] > 0.8
                    and pred["expected_goals"]["away"] > 0.8
                    else "NO"
                ),
            },
            "probabilities": {
                "over_2_5": f"{intervals['over_2_5']:.1%}",
                "under_2_5": f"{intervals['under_2_5']:.1%}",
                "over_1_5": f"{intervals['over_1_5']:.1%}",
                "under_1_5": f"{intervals['under_1_5']:.1%}",
            },
        }

        return recommendations


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def example_usage():
    """Example of how to use the FootballPredictor."""

    # Initialize predictor
    predictor = FootballPredictor()

    # Load data (replace with your actual path)
    predictor.load_data("data/spain.csv")  # or whatever your file is called

    # Train models
    predictor.train_all_models()

    # Make predictions
    home_team = "Real Madrid"
    away_team = "Barcelona"

    print(f"\n{'='*60}")
    print(f"PREDICTION: {home_team} vs {away_team}")
    print(f"{'='*60}")

    # Get detailed prediction
    prediction = predictor.predict_match(home_team, away_team)

    print(f"\nExpected Goals:")
    print(f"  {home_team}: {prediction['expected_goals']['home']}")
    print(f"  {away_team}: {prediction['expected_goals']['away']}")
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
    recommendations = predictor.get_goal_recommendations(home_team, away_team)

    print(f"\nBetting Recommendations:")
    for bet, tip in recommendations["betting_tips"].items():
        prob = recommendations["probabilities"].get(bet, "N/A")
        print(f"  {bet.upper()}: {tip} ({prob})")


if __name__ == "__main__":
    print(__doc__)
    print("\nTo use this toolkit:")
    print("1. Save your CSV data file")
    print("2. Import and create a FootballPredictor instance")
    print("3. Load data and train models")
    print("4. Make predictions!")
    print("\nExample:")
    print("  from predictor import FootballPredictor")
    print("  predictor = FootballPredictor('your_data.csv')")
    print("  predictor.train_all_models()")
    print("  result = predictor.predict_match('Team A', 'Team B')")
