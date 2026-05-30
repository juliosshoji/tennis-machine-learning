# Tennis Match Charting Project Dataset Guide

Complete guide to understanding and using the Tennis Match Charting Project dataset.

---

## 📊 Dataset Overview

The Tennis Match Charting Project is a community-driven initiative to provide detailed, point-by-point charting of professional tennis matches.

- **Website:** https://www.tennisexplorer.com/
- **Project:** Open-source match charting archive
- **Coverage:** Various matches across different eras
- **Data Quality:** Manually charted by community volunteers
- **Files in This Project:**
  - Men's matches: **7,566 matches**
  - Women's matches: **4,080 matches**
  - Total: **11,648 matches**
  - Total players: **1,734**

---

## 📁 Files in New Dataset

### Match Files
- `charting-m-matches.csv` - Men's match metadata (7,566 rows)
- `charting-w-matches.csv` - Women's match metadata (4,080 rows)

### Point-by-Point Data
- `charting-m-points-to-2009.csv` - Men's points before 2010
- `charting-m-points-2010s.csv` - Men's points 2010-2019
- `charting-m-points-2020s.csv` - Men's points 2020+
- `charting-w-points-*.csv` - Women's equivalent files

### Statistical Aggregations
- `charting-m-stats-Overview.csv` - Overall match statistics
- `charting-m-stats-ServeBasics.csv` - Serve statistics
- `charting-m-stats-ReturnOutcomes.csv` - Return statistics
- `charting-m-stats-RallyCount.csv` - Rally length statistics
- Plus 15+ more detailed statistical files

### Documentation
- `data_dictionary.txt` - Complete field descriptions
- `README.md` - Project information

---

## 🔍 Match File Structure (charting-m-matches.csv)

### Columns

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| **match_id** | String | `20260517-M-Rome_Masters-F-Casper_Ruud-Jannik_Sinner` | Unique match identifier |
| **Player 1** | String | `Casper Ruud` | First player (served first) |
| **Player 2** | String | `Jannik Sinner` | Second player |
| **Pl 1 hand** | String | `R` | Player 1 handedness (R=Right, L=Left, U=Unknown) |
| **Pl 2 hand** | String | `R` | Player 2 handedness |
| **Date** | Integer | `20260517` | Match date (YYYYMMDD format) |
| **Tournament** | String | `Rome Masters` | Tournament name |
| **Round** | String | `F` | Match round (F=Final, SF=Semifinal, etc.) |
| **Time** | String | `5pm` | Match start time (often empty) |
| **Court** | String | `Centre` | Court name |
| **Surface** | String | `Clay` | Court surface (Clay, Grass, Hard, Carpet) |
| **Umpire** | String | `Renaud Lichtenstein` | Umpire name (often empty) |
| **Best of** | Integer | `3` | Best of X sets (3 or 5) |
| **Final TB?** | Binary | `1` | Was there a final set tiebreak? (1=Yes, 0=No) |
| **Charted by** | String | `Edo` | Person who charted this match |

### Sample Data

```
match_id: 20260517-M-Rome_Masters-F-Casper_Ruud-Jannik_Sinner
Player 1: Casper Ruud
Player 2: Jannik Sinner
Date: 20260517 (May 17, 2026)
Tournament: Rome Masters
Round: F (Final)
Surface: Clay
Best of: 3
Final TB?: 1 (Yes)
```

---

## 🎾 Point-by-Point Data Structure

Each charting file contains detailed point data with fields like:

- `match_id` - Links to matches file
- `Pt` - Point number
- `Set1/Set2` - Sets won by each player so far
- `Gm1/Gm2` - Games won in current set
- `Pts` - Point score (0, 15, 30, 40, D)
- `Svr` - Who is serving (1=Player1, 2=Player2)
- `1st`/`2nd` - Coded serve and rally information
- `1stIn` - Did first serve go in?
- `2ndIn` - Did second serve go in?
- `isAce` - Was this an ace?
- `isUnret` - Was this unreturned?
- `isRallyWinner` - Did rally end with winner?
- `isUnforced` - Unforced error?
- `PtWinner` - Who won the point (1 or 2)?
- `rallyCount` - Number of shots in rally

---

## 📋 Round Abbreviations

| Code | Meaning |
|------|---------|
| **Q3, Q2, Q1** | Qualifying rounds 3, 2, 1 |
| **R128** | Round of 128 |
| **R64** | Round of 64 |
| **R32** | Round of 32 |
| **R16** | Round of 16 |
| **QF** | Quarterfinal |
| **SF** | Semifinal |
| **F** | Final |

---

## 🏆 Tournament Names

Some examples of tournaments in the data:

```
Major Tournaments:
- Australian Open
- Roland Garros (French Open)
- Wimbledon
- US Open

Masters 1000:
- Rome Masters
- Miami Masters
- Cincinnati Masters
- Shanghai Masters
- Paris Masters

ATP 500:
- Dubai Duty Free Championships
- Halle Open
- Canada Masters

ATP 250:
- Sydney International
- Marseille Open
- Valencia Open
...and many others
```

---

## 🌍 Court Surfaces

| Surface | Characteristics | Famous Tournaments |
|---------|-----------------|-------------------|
| **Clay** | Slower game, high bounce | Roland Garros |
| **Grass** | Faster game, low bounce | Wimbledon |
| **Hard** | Medium speed | Australian Open, US Open |
| **Carpet** | Fast, artificial | Some older tournaments |

---

## ✋ Handedness Codes

| Code | Meaning |
|------|---------|
| **R** | Right-handed |
| **L** | Left-handed |
| **U** | Unknown/Ambidextrous |

---

## 📊 Statistics Files Explained

The dataset includes pre-computed statistics files for easy analysis:

### Overview Statistics (`charting-m-stats-Overview.csv`)
- Match duration
- Sets, games, points won
- Serve % success
- Break point conversions
- Unforced errors

### Serve Statistics (`charting-m-stats-ServeBasics.csv`)
- 1st serve in %
- 2nd serve in %
- Ace counts
- Double fault counts
- Points won on serve

### Rally Statistics (`charting-m-stats-Rally.csv`)
- Rally length distribution
- Winner types (ace, rally winner, unforced error)
- Net points statistics

### Shot Direction (`charting-m-stats-ShotDirection.csv`)
- Shots hit to each side
- Direction patterns
- Consistency metrics

---

## 🔄 Loading the Dataset in Python

### Load Matches File

```python
import pandas as pd

# Men's matches
df_men = pd.read_csv("new-dataset/tennis_MatchChartingProject-master/charting-m-matches.csv")
print(f"Loaded {len(df_men)} men's matches")

# Women's matches
df_women = pd.read_csv("new-dataset/tennis_MatchChartingProject-master/charting-w-matches.csv")
print(f"Loaded {len(df_women)} women's matches")

# Combined
df_all = pd.concat([df_men, df_women])
print(f"Total matches: {len(df_all)}")
```

### Load Point Data

```python
import pandas as pd

# Men's points (combined from multiple files)
points_pre_2010 = pd.read_csv("charting-m-points-to-2009.csv")
points_2010s = pd.read_csv("charting-m-points-2010s.csv")
points_2020s = pd.read_csv("charting-m-points-2020s.csv")

df_points = pd.concat([points_pre_2010, points_2010s, points_2020s])
print(f"Total points charted: {len(df_points)}")
```

### Filter by Tournament

```python
# Get all matches from Roland Garros
df_rg = df_all[df_all['Tournament'] == 'Roland Garros']
print(f"Roland Garros matches: {len(df_rg)}")

# Get all final matches
df_finals = df_all[df_all['Round'] == 'F']
print(f"Final matches: {len(df_finals)}")

# Get matches on clay courts
df_clay = df_all[df_all['Surface'] == 'Clay']
print(f"Clay court matches: {len(df_clay)}")
```

### Analyze Player Performance

```python
# Get all matches of a player
player = "Jannik Sinner"
player_matches_p1 = df_all[df_all['Player 1'] == player]
player_matches_p2 = df_all[df_all['Player 2'] == player]
player_matches = pd.concat([player_matches_p1, player_matches_p2])

print(f"{player} appears in {len(player_matches)} matches")
print(f"Matches as Player 1: {len(player_matches_p1)}")
print(f"Matches as Player 2: {len(player_matches_p2)}")
```

---

## 🎯 Typical Analysis Examples

### 1. Compare Clay vs Hard Court Performance

```python
# Clay matches
clay_matches = df_all[df_all['Surface'] == 'Clay']

# Hard matches
hard_matches = df_all[df_all['Surface'] == 'Hard']

# Analyze surface preference
print(f"Clay matches: {len(clay_matches)}")
print(f"Hard matches: {len(hard_matches)}")
```

### 2. Analyze Tournament Patterns

```python
# Most charted tournaments
tournament_counts = df_all['Tournament'].value_counts()
print("Most charted tournaments:")
print(tournament_counts.head(10))

# Masters vs regular events
masters = df_all[df_all['Tournament'].str.contains('Masters')]
print(f"Masters events: {len(masters)}")
```

### 3. Handedness Analysis

```python
# Righty vs Lefty matches
righties = df_all[(df_all['Pl 1 hand'] == 'R') & (df_all['Pl 2 hand'] == 'R')]
lefties = df_all[(df_all['Pl 1 hand'] == 'L') & (df_all['Pl 2 hand'] == 'L')]
mixed = df_all[df_all['Pl 1 hand'] != df_all['Pl 2 hand']]

print(f"Righty vs Righty: {len(righties)}")
print(f"Lefty vs Lefty: {len(lefties)}")
print(f"Mixed handedness: {len(mixed)}")
```

### 4. Recent Matches

```python
# Parse dates
df_all['Date_parsed'] = pd.to_datetime(df_all['Date'], format='%Y%m%d')

# Recent matches
recent = df_all[df_all['Date_parsed'] >= '2024-01-01']
print(f"Matches from 2024+: {len(recent)}")

# By year
by_year = df_all.groupby(df_all['Date_parsed'].dt.year).size()
print("Matches by year:")
print(by_year)
```

---

## 📈 Using with Our ML Pipeline

### Option 1: Use Our Preprocessing

```python
from tennis_ml.preprocessing_new_dataset import load_and_preprocess_charting_data

# Load men's matches
X_train, X_test, y_train, y_test, features = \
    load_and_preprocess_charting_data(dataset_type="men")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {len(features)}")
```

### Option 2: Custom Analysis

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("new-dataset/tennis_MatchChartingProject-master/charting-m-matches.csv")

# Your custom feature engineering
df['Year'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.year
df['is_clay'] = (df['Surface'] == 'Clay').astype(int)
df['is_final'] = (df['Round'] == 'F').astype(int)

# Scale features
scaler = StandardScaler()
features = ['Year', 'is_clay', 'is_final']
df[features] = scaler.fit_transform(df[features])

# Your model training here
```

---

## ⚠️ Data Quality Notes

- **Gaps:** Not all matches are charted. Coverage is sporadic
- **Volunteer-Based:** Quality depends on who charted each match
- **Missing Fields:** Some older matches have incomplete data
- **Date Variations:** Some matches may have incorrect dates
- **Player Names:** May vary (e.g., "Jannik Sinner" vs "Sinner J.")

---

## 🔗 Resources

- **Tennis Explorer:** https://www.tennisexplorer.com/
- **Data Dictionary:** `new-dataset/tennis_MatchChartingProject-master/data_dictionary.txt`
- **GitHub (if available):** Check project source
- **API Reference:** Use pandas documentation for data manipulation

---

## 💡 Tips for Using This Data

1. **Handle missing values:** Many fields are optional
2. **Date format:** Always YYYYMMDD integer format
3. **Filter by surface:** Most analysis is surface-specific
4. **Player name matching:** Use exact strings
5. **Tournament names:** Can vary (e.g., "Roland Garros" vs "French Open")
6. **Combine with statistics files:** For detailed performance analysis
7. **Cross-reference point data:** For detailed rally analysis

---

## ✨ Summary

The Tennis Match Charting Project provides:
- ✅ 11,648 matches with detailed metadata
- ✅ 1,734 unique players
- ✅ Point-by-point charting data
- ✅ Pre-computed statistics
- ✅ Surface, tournament, and round information
- ✅ Player handedness data

Perfect for analyzing professional tennis with granular detail!

See [DATASET_CONFIGURATION.md](DATASET_CONFIGURATION.md) for how to use both this and the old ATP dataset.
