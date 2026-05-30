# tennis-machine-learning

Basic Python structure for a machine learning project with:

- **pre-processing** (`/tennis_ml/preprocessing.py`)
- **training** (`/tennis_ml/training.py`)
- **use/inference** (`/tennis_ml/usage.py`)

## Kaggle dataset usage

Dataset page:
`https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull?resource=download`

You can prepare the data locally with Kaggle CLI (after configuring Kaggle credentials):

```bash
kaggle datasets download -d dissfya/atp-tennis-2000-2023daily-pull -p data
```

Then call preprocessing with either:

- zip file path (auto-picks `atp_tennis.csv` when available):
  `data/atp-tennis-2000-2023daily-pull.zip`
- zip member path format:
  `data/atp-tennis-2000-2023daily-pull.zip/atp_tennis.csv`
- plain CSV path:
  `data/atp_tennis.csv`

## Preprocessing phase

Use `load_and_preprocess_tennis_data(...)` from `/tennis_ml/preprocessing.py`.
It uses `pandas`, `numpy`, and `scikit-learn` for:

- missing-value replacement (`-1` to `NaN`)
- target creation (`Player_1_Wins`)
- date feature engineering (`Year`, `Month`, `Day`)
- player name label encoding
- numeric imputation + scaling
- categorical imputation + one-hot encoding
- train/test split and transformed outputs

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests with:

```bash
python -m unittest discover -v
```
