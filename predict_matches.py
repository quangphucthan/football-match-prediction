"""
Predict football match outcomes using XGBoost and Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def preprocess_data(matches, countries):
    print("Loading dataset...")
    
    # Load matches and countries data into DataFrames
    df_matches = pd.DataFrame(matches)
    df_countries = pd.DataFrame(countries)
    
    print("Cleaning data...")
    
    # Drop matches with missing scores
    df_matches = df_matches.dropna(subset=['home_score', 'away_score']).copy()
    
    # To ensure we have the most recent data, we will go with data from the year 2000 onwards, as soccer has evolved a lot
    # Also converting the date column to datetime format for easier filtering
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_matches = df_matches[df_matches['date'].dt.year >= 2000].copy()
    
    # Standardize country names in matches dataset using the mapping from countries dataset
    country_map = dict(zip(df_countries['original_name'], df_countries['current_name']))
    df_matches['home_team'] = df_matches['home_team'].replace(country_map)
    df_matches['away_team'] = df_matches['away_team'].replace(country_map)
    
    # Create the target variable named outcome: 2 for home win, 1 for draw, 0 for away win
    df_matches['outcome'] = np.where(
        df_matches['home_score'] > df_matches['away_score'], 2,
        np.where(df_matches['home_score'] == df_matches['away_score'], 1, 0)
    )
    
    # Change team name as string to numerical values
    le_team = LabelEncoder()
    all_teams = pd.concat([df_matches['home_team'], df_matches['away_team']]).unique()
    le_team.fit(all_teams)

    df_matches['home_team_encoded'] = pd.Series(le_team.transform(df_matches['home_team']), index=df_matches.index)
    df_matches['away_team_encoded'] = pd.Series(le_team.transform(df_matches['away_team']), index=df_matches.index)
    
    # Change tournament name as string to numerical values
    le_tournament = LabelEncoder()
    df_matches['tournament_encoded'] = pd.Series(le_tournament.fit_transform(df_matches['tournament']), index=df_matches.index)
    
    # Create features for whether the match was a friendly or played on neutral venue
    df_matches['is_friendly'] = (df_matches['tournament'] == 'Friendly').astype(int)
    df_matches['is_neutral'] = (df_matches['neutral'] == True).astype(int)
    
    # Select features and target variable
    features = [
        'home_team_encoded', 'away_team_encoded', 'tournament_encoded', 'is_friendly', 'is_neutral'
    ]
    
    X = df_matches[features]
    y = df_matches['outcome']
    
    return X, y

def main():
    print("Initializing...")
    X, y = preprocess_data('dataset/all_matches.csv', 'dataset/countries_names.csv')
    
if __name__ == "__main__":
    main()