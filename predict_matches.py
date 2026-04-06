"""
Predict football match outcomes using XGBoost and Random Forest.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
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
    # Splitting the data into training and testing sets, with 80% for training and 20% (0.2) for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
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