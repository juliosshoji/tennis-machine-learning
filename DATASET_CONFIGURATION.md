# Dataset Configuration - Tennis Match Charting Project Only

This project now uses exclusively the **Tennis Match Charting Project** dataset.

---

## 📊 Current Dataset

### **Tennis Match Charting Project** (Active)
- **Location:** `new-dataset/tennis_MatchChartingProject-master/`
- **Files:**
  - `charting-m-matches.csv` - 7,566 men's matches
  - `charting-w-matches.csv` - 4,080 women's matches
- **Total Players:** 1,734 (1,003 men, 732 women)
- **Total Matches:** 11,648
- **Data Type:** Point-by-point match charting with detailed statistics
- **Date Range:** Various eras, charted by community volunteers
- **Features:**
  - Player names (full names like "Jannik Sinner")
  - Handedness (Right/Left/Unknown)
  - Tournament, Round, Surface, Court
  - Date, Best of, Umpire, Charted by

---

## 🚀 Using the Dataset

### Command Line (Simplified)

```bash
# Search for players
python player_search.py "Sinner"
python player_search.py "Alcaraz"

# Interactive exploration
python player_search.py

# Make predictions
python quick_predict.py --player1 "Jannik Sinner" --player2 "Carlos Alcaraz" \
  --odds1 1.90 --odds2 1.95 --rank1 2 --rank2 3
```

### Python API

```python
# Load the dataset
from tennis_ml.preprocessing_new_dataset import load_and_preprocess_charting_data

# Men's matches
X_train, X_test, y_train, y_test, features = \
    load_and_preprocess_charting_data(dataset_type="men")

# Women's matches  
X_train_w, X_test_w, y_train_w, y_test_w, features_w = \
    load_and_preprocess_charting_data(dataset_type="women")

# Train model
from tennis_ml.training import train_random_forest
model = train_random_forest(X_train, y_train)
```

---

## 📊 Dataset Specifications

| Feature | Value |
|---------|-------|
| **Total Players** | 1,734 |
| **Total Matches** | 11,648 |
| **Men's Matches** | 7,566 |
| **Women's Matches** | 4,080 |
| **Player Name Format** | Full names |
| **Point Data** | ✅ Available |
| **Handedness** | ✅ Included |
| **Statistical Files** | ✅ Included |
| **Surface Types** | Clay, Grass, Hard, Carpet |

---

## 📁 Directory Structure

```
tennis-machine-learning/
├── new-dataset/                             ← Tennis Match Charting Project
│   └── tennis_MatchChartingProject-master/
│       ├── charting-m-matches.csv           (7,566 matches)
│       ├── charting-w-matches.csv           (4,080 matches)
│       ├── charting-m-points-*.csv          (Point-by-point data)
│       ├── charting-m-stats-*.csv           (Statistics)
│       └── data_dictionary.txt
│
├── PLAYER_LIST.txt                          ← Players (1,734)
│
├── tennis_ml/
│   ├── preprocessing_new_dataset.py         ← New dataset preprocessing
│   ├── training.py                          ← Model training
│   ├── usage.py                             ← Predictions
│   └── __init__.py
│
├── player_search.py                         ← Search tool
├── predict_match.py                         ← Prediction functions
├── quick_predict.py                         ← Quick CLI predictions
└── README.md                                ← Main documentation
```

---

## 🔍 Searching for Players

```bash
# Search for a specific player
python player_search.py "Sinner"
# Output: Jannik Sinner

python player_search.py "Alcaraz"
# Output: Carlos Alcaraz

python player_search.py "Federer"
# Output: Not found (older players not in this dataset)
```

---

## 📚 Preprocessing Pipeline

The dataset is automatically preprocessed with:

1. **Temporal Features:** Year, Month, Day
2. **Player Features:** Handedness (encoded)
3. **Match Features:** Surface, Round, Best of
4. **Tournament Features:** One-hot encoded (top 20 tournaments)
5. **Court Features:** One-hot encoded
6. **Total Features Generated:** 79

**Output:**
- Training samples: 6,052
- Testing samples: 1,514
- Features: 79
- Target: Balanced (49.4% / 50.6%)

---

## ✨ Key Benefits

✅ **Simplified:** Single dataset to manage  
✅ **Modern:** Community-curated current data  
✅ **Detailed:** Point-by-point charting available  
✅ **Players:** 1,734 current professional players  
✅ **Consistent:** Full names, standardized format  
✅ **Rich:** Tournament, surface, handedness data  

---

## 📚 Related Documentation

- [TENNIS_CHARTING_GUIDE.md](TENNIS_CHARTING_GUIDE.md) - Dataset details
- [PLAYER_DATABASE.md](PLAYER_DATABASE.md) - Player search guide
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [README.md](README.md) - Main documentation

---

## 💡 What Changed

**Removed:**
- ❌ Old ATP database (`old-dataset/`)
- ❌ Archive files
- ❌ `--dataset old` flag
- ❌ Backward compatibility code

**Kept:**
- ✅ Tennis Match Charting Project
- ✅ All models and predictions
- ✅ All documentation
- ✅ Player search and tools
- ✅ Full preprocessing pipeline

**Result:** Cleaner codebase, single dataset focus, easier to maintain!

