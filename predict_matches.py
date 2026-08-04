"""
Benchmark script: XGBoost and Random Forest on team/tournament identity alone.

This is not what the app serves -- see model.py, which blends a Poisson goal
model with XGBoost and scores better. Keep this around to document the baseline
those numbers are measured against.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def preprocess_data(matches, countries):
    print("Loading dataset...")
    
    # Load matches and countries data into DataFrames
    df_matches = pd.read_csv(matches)
    df_countries = pd.read_csv(countries)
    
    print("Cleaning data...")
    
    # Drop matches with missing scores
    df_matches = df_matches.dropna(subset=['home_score', 'away_score']).copy()
    
    # To ensure we have the most recent data, we will go with data from the year 2000 onwards, as soccer has evolved a lot
    # Also converting the date column to datetime format for easier filtering
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_matches = df_matches[df_matches['date'].dt.year >= 2000].copy()

    # Sorted by date so train_models can cut chronologically rather than shuffling.
    df_matches = df_matches.sort_values('date').reset_index(drop=True)
    
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

    # np.asarray keeps the type checker happy: LabelEncoder is unannotated, so the
    # stubs give transform() numpy's very broad ArrayLike -- which admits Buffer and
    # str -- and pandas rejects that. This narrows it to the ndarray it already is.
    # The rows were reindexed above, so assigning the array positionally is the same
    # as the pd.Series(..., index=...) wrapper this replaces.
    df_matches['home_team_encoded'] = np.asarray(le_team.transform(df_matches['home_team']))
    df_matches['away_team_encoded'] = np.asarray(le_team.transform(df_matches['away_team']))

    # Change tournament name as string to numerical values
    le_tournament = LabelEncoder()
    df_matches['tournament_encoded'] = np.asarray(le_tournament.fit_transform(df_matches['tournament']))
    
    # Create features for whether the match was a friendly or played on neutral venue
    df_matches['is_friendly'] = (df_matches['tournament'] == 'Friendly').astype(int)
    df_matches['is_neutral'] = (df_matches['neutral'] == True).astype(int)
    
    # Select features and target variable
    features = [
        'home_team_encoded', 
        'away_team_encoded', 
        'tournament_encoded', 
        'is_friendly', 
        'is_neutral'
    ]
    
    X = df_matches[features]
    y = df_matches['outcome']
    
    return X, y

def train_models(X, y):
    # Chronological 80/20 cut, not a random shuffle. Forecasting is a forward-in-time
    # job, so the test set has to sit entirely after the training set -- a shuffled
    # split lets 2020 matches inform a prediction about a 2004 one.
    cut = int(len(X) * 0.8)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    
    # Training Random Forest Classifier
    print("Training Random Forrest Classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators = 100, 
        max_depth = 10, 
        random_state = 42
    )
    
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    # Training XGBoost Classifier
    print("Training XGBoost Classifier...")
    
    xgb_model = XGBClassifier(
        n_estimators = 100,
        max_depth = 6,
        learning_rate = 0.1,
        random_state = 42,
        eval_metric = 'mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    
    os.makedirs('results', exist_ok=True)

    # Save the results to a text file
    with open('results/model_performance.txt', 'w') as f:
        f.write("Random Forest Classifier Performance:\n")
        f.write(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report(y_test, rf_predictions, target_names=['Away Win', 'Draw', 'Home Win'])))
        
        f.write("\n\nXGBoost Classifier Performance:\n")
        f.write(f"Accuracy: {accuracy_score(y_test, xgb_predictions):.4f}\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report(y_test, xgb_predictions, target_names=['Away Win', 'Draw', 'Home Win'])))
        
    print("Model training and evaluation completed. Results saved to results/model_performance.txt")

def main():
    print("Initializing...")
    X, y = preprocess_data('dataset/all_matches.csv', 'dataset/countries_names.csv')
    train_models(X, y)
    
if __name__ == "__main__":
    main()