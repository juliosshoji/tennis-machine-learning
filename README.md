# tennis-machine-learning

A comprehensive machine learning project for predicting ATP tennis match outcomes and analyzing feature importance. This project implements multiple models to identify which player attributes most impact victory.

## Project Overview

**Objective:** Build ML models to predict tennis match outcomes and rank match attributes by their importance in determining winners at the professional level.

**Models Implemented:**
- **Decision Tree** - Provides interpretable rules and feature importance rankings
- **Neural Network (MLP)** - Captures complex relationships and interactions between features
- **Random Forest** - Ensemble method for robust predictions
- **K-means Clustering** - Exploratory analysis to discover playing styles/tactics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download the ATP tennis dataset from Kaggle:
```bash
kaggle datasets download -d dissfya/atp-tennis-2000-2023daily-pull -p data
```

Or use the provided `atp_tennis.csv` directly.

### 3. Train Models

#### Full Pipeline (All Models)
```bash
python -m tennis_ml \
  --data atp_tennis.csv \
  --model-type all \
  --output-dir models/
```

#### Specific Model Types
```bash
# Decision Tree only
python -m tennis_ml --data atp_tennis.csv --model-type tree --output-dir models/

# Neural Network only
python -m tennis_ml --data atp_tennis.csv --model-type neural --output-dir models/

# Random Forest only
python -m tennis_ml --data atp_tennis.csv --model-type forest --output-dir models/
```

#### Basic Mode (Simple Baseline)
```bash
python -m tennis_ml \
  --data atp_tennis.csv \
  --model-type baseline \
  --output-dir models/ \
  --feature-columns Rank_1 Rank_2 Pts_1 Pts_2 \
  --target-column Player_1_Wins
```

### 4. View Results

All training results are saved to `models/training_results.json`:
```bash
cat models/training_results.json
```

This includes:
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Top 20 most important features for each model
- Paths to saved model files

## Project Structure

```
tennis-machine-learning/
├── atp_tennis.csv                 # ATP match data (2000-2023)
├── MINERAÇÃO DE DADOS_ EM PARTIDAS DE TÊNIS.pdf  # Requirements document (Portuguese)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── tennis_ml/
│   ├── __init__.py               # Package exports
│   ├── __main__.py               # CLI entrypoint
│   ├── preprocessing.py          # Data loading & feature engineering
│   ├── training.py               # Model training functions
│   └── usage.py                  # Model loading & prediction
└── tests/
    ├── __init__.py
    └── test_ml_pipeline.py       # Unit tests for all models
```

## Preprocessing Pipeline

The `load_and_preprocess_tennis_data()` function:

1. **Loads** tennis match data from CSV or ZIP archives
2. **Cleans** data (replaces -1 with NaN)
3. **Creates target** variable: `Player_1_Wins` (1 if Player_1 won, 0 otherwise)
4. **Engineers features:**
   - Date features: Year, Month, Day
   - Player name encoding (Label Encoder)
   - Numerical features: Rankings, Points, Odds, Best Of
   - Categorical features: Tournament, Series, Court, Surface, Round
5. **Imputes** missing values (median for numeric, mode for categorical)
6. **Scales** numeric features (StandardScaler)
7. **Encodes** categorical features (One-Hot Encoding)
8. **Splits** data: 80% train, 20% test (stratified, random_state=42)

## Feature Categories

### Numerical Features
- `Rank_1`, `Rank_2` - Player rankings
- `Pts_1`, `Pts_2` - ATP points
- `Odd_1`, `Odd_2` - Betting odds
- `Best of` - Number of sets in match
- Date features: `Year`, `Month`, `Day`
- Encoded players: `Player_1`, `Player_2`

### Categorical Features
- `Tournament` - Tournament name
- `Series` - Tournament series (ATP 250, 500, etc.)
- `Court` - Indoor/Outdoor
- `Surface` - Hard, Clay, Grass
- `Round` - Match stage (1st Round, QF, SF, Final, etc.)

## Model Performance

Each trained model produces:

### Classification Metrics
- **Accuracy** - Percentage of correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of Precision and Recall
- **ROC-AUC** - Area under the ROC curve

### Feature Importance
Top 20 features ranked by their impact on predictions:
```json
{
  "decision_tree": [
    ["Rank_1", 0.235],
    ["Odd_1", 0.189],
    ["Pts_1", 0.156],
    ...
  ]
}
```

## Exploratory Analysis: K-means Clustering

The K-means clustering identifies natural groupings of matches without using the win/loss label:
- Discovers 3 distinct playing style profiles
- Helps understand match characteristics and tactics
- Reveals whether certain player/surface combinations form clusters

## Testing

Run the full test suite:
```bash
python -m unittest discover -v
```

Run specific test file:
```bash
python -m unittest tests.test_ml_pipeline -v
```

## File Formats

### Input: CSV Tennis Data
```csv
Tournament,Date,Series,Court,Surface,Round,Best of,Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score
Australian Hardcourt Championships,2000-01-03,International,Outdoor,Hard,1st Round,3,Dosedel S.,Ljubicic I.,Dosedel S.,63,77,-1,-1,-1,-1.0,6-4 6-2
...
```

### Output: Model Files
- `*.pkl` - Scikit-learn models (pickled for Decision Tree, Neural Network, Random Forest, K-means)
- `training_results.json` - Comprehensive results summary with metrics and feature importance

### Results JSON Structure
```json
{
  "models": {
    "decision_tree": {
      "metrics": {
        "accuracy": 0.752,
        "precision": 0.731,
        "recall": 0.698,
        "f1": 0.714,
        "roc_auc": 0.821
      },
      "model_path": "models/decision_tree_model.pkl"
    },
    ...
  },
  "feature_importance": {
    "decision_tree": [
      ["Rank_1", 0.235],
      ["Odd_1", 0.189],
      ...
    ]
  }
}
```

## Module Usage

### In Python Code

```python
from tennis_ml import (
    load_and_preprocess_tennis_data,
    train_decision_tree,
    train_neural_network,
    evaluate_model,
    get_feature_importance,
    save_model,
    load_model,
    predict,
)

# Load and preprocess data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data("atp_tennis.csv")

# Train Decision Tree
dt_model = train_decision_tree(X_train, y_train, max_depth=10)

# Evaluate
metrics = evaluate_model(dt_model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Get feature importance
importance = get_feature_importance(dt_model, feature_names, top_n=20)
for feature, score in importance:
    print(f"{feature}: {score:.4f}")

# Save and load model
save_model(dt_model, "models/my_model.pkl")
loaded_model = load_model("models/my_model.pkl")

# Make predictions
predictions = predict(loaded_model, X_test)
```

## Data Source

**Dataset:** ATP Tennis matches 2000-2023
**Source:** Kaggle - https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull

**Notes:**
- Contains ~180,000 historical ATP matches
- Missing odds data for earlier years (pre-2015)
- Some ranking data gaps (more complete from 2000+)
- Surface and court information complete for most records

## Requirements Document

The project requirements and objectives are detailed in:
`MINERAÇÃO DE DADOS_ EM PARTIDAS DE TÊNIS.pdf` (Portuguese)

Key requirements:
- Predict match outcomes (victory/defeat) using player statistics
- Rank attributes by importance in determining winners
- Implement Decision Tree (for interpretability) and Neural Network (for complexity)
- Perform exploratory clustering analysis
- Identify which playing styles have higher win rates

## Dependencies

- **pandas** - Data loading and manipulation
- **numpy** - Numerical operations
- **scikit-learn** - ML algorithms and preprocessing
- **scipy** - Scientific computing utilities

See `requirements.txt` for version specifications.
