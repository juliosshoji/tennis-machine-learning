# ATP Tennis Players Database

## Overview
- **Total Players:** 1,799 unique players
- **Period:** 2000-2023
- **Data Source:** ATP Match Data from CSV

---

## How to Find Players

### Option 1: Search by Name
```bash
# Interactive search
python player_search.py

# Quick search for specific player
python player_search.py "Nadal"      # Returns: Nadal R., Nadal-Parera R.
python player_search.py "Djokovic"   # Returns: Djokovic N., Djokovic M.
python player_search.py "Federer"    # Returns: Federer R.
```

### Option 2: View Full List
```bash
# View all 1,799 players
cat PLAYER_LIST.txt

# View top 50 players
head -55 PLAYER_LIST.txt

# Find a specific player name
grep "Alcaraz" PLAYER_LIST.txt
```

### Option 3: Use in Python
```python
import pandas as pd

df = pd.read_csv("atp_tennis.csv", low_memory=False)
players = sorted(list(set(df["Player_1"].unique()) | set(df["Player_2"].unique())))

# Find players matching pattern
results = [p for p in players if "Nadal" in p]
# Returns: ['Nadal R.', 'Nadal-Parera R.']
```

---

## Notable Players

### All-Time Greats (Found in Dataset)

| Name | Peak Era | Matches |
|------|----------|---------|
| **Federer R.** | 2000-2018 | 1,000+ |
| **Nadal R.** | 2005-2022 | 800+ |
| **Djokovic N.** | 2008-2023 | 900+ |
| **Murray A.** | 2007-2019 | 700+ |
| **Sampras P.** | 1990-2002 | 250+ |
| **Agassi A.** | 1986-2003 | 300+ |

### Recent Champions

| Name | Notable |
|------|---------|
| **Alcaraz C.** | Rising star, ATP #1 |
| **Sinner J.** | Young Italian talent |
| **Medvedev D.** | US Open champion |
| **Thiem D.** | US Open champion |
| **Tsitsipas S.** | Top 5 player |

---

## Player Name Format

Players in the database use standardized naming:
```
Last Name First Initial(s)
```

**Examples:**
- `Federer R.` (Roger Federer)
- `Nadal R.` (Rafael Nadal)
- `Djokovic N.` (Novak Djokovic)
- `Alcaraz C.` (Carlos Alcaraz)
- `Alvarez Varona N.` (Multiple names)

**Important:** Use EXACT names when making predictions:
```bash
# ✅ Correct
python quick_predict.py --player1 "Federer R." --player2 "Nadal R." ...

# ❌ Wrong
python quick_predict.py --player1 "Roger Federer" --player2 "Rafael Nadal" ...
```

---

## Search Examples

### Finding Similar Names
```bash
# Find all Djokovic players
python player_search.py "Djokovic"
# Returns: Djokovic M., Djokovic N., Djokovic N. (duplicate)

# Find all de/van players
python player_search.py "van"
# Returns: van der Meer N., van Gemerden M., etc.
```

### Getting Player Statistics
```bash
# Federer's statistics
python player_search.py "Federer R."

# Get detailed stats (in interactive mode)
python player_search.py
# Then select option "Get player statistics"
```

---

## Complete Player List (A-Z)

The complete alphabetical list of all 1,799 players is in **[PLAYER_LIST.txt](PLAYER_LIST.txt)**.

### Sample Sections

**A names (sample):**
- Abdulla M.
- Acasuso J.
- Agassi A.
- Agostinelli B.
- Alami K.
- Alcaraz C.
- Almagro N.
- Anderson K.
- Andreev A.

**F names (sample):**
- Federer R.
- Ferrero J.C.
- Fish M.
- Fish P.

**N names (sample):**
- Nadal R.
- Nadal-Parera R.
- Nalbandian D.

**Z names (sample):**
- Zverev A.
- Zverev M.
- Zalopinski M.

---

## Database Statistics

- **Unique Player-1 entries:** 1,299
- **Unique Player-2 entries:** 1,502
- **Combined unique players:** 1,799
- **Date range:** 2000-2023
- **Total matches:** 67,572+

---

## Using Players in Predictions

### Quick Prediction with a Player

```bash
python quick_predict.py \
  --player1 "Nadal R." \
  --player2 "Djokovic N." \
  --odds1 3.5 \
  --odds2 1.35 \
  --rank1 3 \
  --rank2 1
```

### Using in Python

```python
from predict_match import predict_ensemble

result = predict_ensemble(
    player_1_name="Federer R.",
    player_2_name="Murray A.",
    player_1_rank=2,
    player_2_rank=4,
    player_1_odds=1.85,
    player_2_odds=2.05,
)

print(f"Winner: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## Tips for Finding Players

1. **Start with last name only:** `python player_search.py "Nadal"`
2. **Use partial matches:** Search works with any substring
3. **Check the exact name:** Look at search results carefully
4. **Handle special characters:** Names like `de`, `van`, `di` are included
5. **Account for duplicates:** Some names appear with slight variations

---

## Troubleshooting

### Player not found
- Try partial name: Instead of "Rafael Nadal", use "Nadal"
- Check spelling in PLAYER_LIST.txt
- Some players may only have first initial (e.g., "Nadal R.")

### Wrong player returned
- Search results show partial matches
- Use more specific search terms
- Check for name variations (e.g., "Djokovic N." vs "Djokovic N. ")

### Making predictions
```bash
# This works (uses exact name from search)
python quick_predict.py --player1 "Nadal R." --player2 "Federer R." --odds1 2.0 --odds2 1.85

# This won't work (wrong format)
python quick_predict.py --player1 "Nadal" --player2 "Federer" --odds1 2.0 --odds2 1.85
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `PLAYER_LIST.txt` | Complete alphabetical list of all 1,799 players |
| `player_search.py` | Search tool for finding players |
| `atp_tennis.csv` | Raw ATP match data |
| `PLAYER_DATABASE.md` | This file |

---

## Quick Links

- Search players: `python player_search.py`
- View all: `cat PLAYER_LIST.txt`
- Make predictions: `python quick_predict.py`
- Interactive predictions: `python player_search.py` → Option 3

