# Learning Implementation Guide

Practical examples of what to learn from each source and how to apply it to this project.

---

## 🎓 Learning Track by Complexity Level

### Level 1: Foundations (1-2 weeks)

#### 1A. Python Basics
**Source:** https://docs.python.org/3/tutorial/

**Learn:**
- Variables, data types, control flow
- Functions and modules
- List comprehensions
- File I/O

**Apply to this project:**
```python
# From player_search.py - search functionality
def search_players(query: str, players: list) -> list:
    """Search for players matching query."""
    query_lower = query.lower()
    results = [p for p in players if query_lower in p.lower()]
    return results
```

**Hands-on exercise:**
- Write a function that loads the CSV file
- Filter for matches from a specific year

#### 1B. NumPy Basics
**Source:** https://numpy.org/doc/stable/user/absolute_beginners.html

**Learn:**
- Creating arrays
- Array operations
- Indexing and slicing
- Basic statistics (mean, std)

**Apply to this project:**
```python
# From preprocessing.py - numerical operations
import numpy as np

# Calculate feature statistics
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

# Standardize features
X_scaled = (X_train - X_train_mean) / X_train_std
```

**Hands-on exercise:**
- Load ATP data as numpy array
- Calculate average rank and points
- Create simple numpy statistics

#### 1C. Pandas Basics
**Source:** https://pandas.pydata.org/docs/user_guide/10min.html

**Learn:**
- DataFrames and Series
- Reading CSV files
- Basic filtering and selection
- Data inspection (head, describe, info)

**Apply to this project:**
```python
# From preprocessing.py - data loading
import pandas as pd

# Load data
df = pd.read_csv("atp_tennis.csv", low_memory=False)

# Inspect
print(df.head())           # First 5 rows
print(df.info())           # Data types
print(df.describe())       # Statistics
print(df['Player_1'].unique())  # All unique values
```

**Hands-on exercise:**
```python
# Try these commands:
df = pd.read_csv("atp_tennis.csv", low_memory=False)
print(df.shape)  # How many rows and columns?
print(df.columns)  # Column names
print(df[df['Year'] == 2020])  # Matches from 2020
print(df['Winner'].value_counts())  # Most common winners
```

---

### Level 2: Machine Learning Foundations (2-4 weeks)

#### 2A. Andrew Ng's ML Course
**Source:** https://www.coursera.org/learn/machine-learning

**Key Concepts (in order):**
1. Linear regression
2. Logistic regression
3. Neural networks
4. Support vector machines
5. Decision trees, random forests
6. Anomaly detection
7. Recommendation systems

**Apply to this project:**
```python
# From training.py - the models used ARE from this course

# Decision Trees (Week 4)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Neural Networks (Week 5)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
model.fit(X_train, y_train)

# Random Forests (Week 6)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Clustering (Week 8)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X_train)
```

**Hands-on exercise:**
- Complete first 3 weeks of the course
- Implement logistic regression from scratch (bonus)
- Understand cost function and gradient descent

#### 2B. StatQuest YouTube Channel
**Source:** https://www.youtube.com/@statquest

**Watch (in order):**
1. "Logistic Regression" (understand the concept)
2. "Decision Trees" (15 min)
3. "Random Forests" (15 min)
4. "Neural Networks (MLP)" (15 min)
5. "K-means Clustering" (15 min)

**Apply to this project:**
- Understand why each model was chosen
- Understand how hyperparameters work (max_depth, n_estimators, etc.)

---

### Level 3: Scikit-Learn & Application (2-3 weeks)

#### 3A. Scikit-learn Documentation
**Source:** https://scikit-learn.org/stable/

**Learn in this order:**
1. Getting Started guide
2. Preprocessing: StandardScaler, OneHotEncoder, ColumnTransformer
3. Model Selection: train_test_split, cross_val_score
4. Model Evaluation: accuracy, precision, recall, F1, ROC-AUC
5. Each classifier: DecisionTree, RandomForest, MLP, KMeans

**Apply to this project:**
```python
# From preprocessing.py - The full pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# This is the complete preprocessing pipeline:
# 1. Define what to do to each type of column
# 2. Apply transformers in parallel
# 3. Split data 80/20
# 4. Train model on training data
# 5. Evaluate on test data
```

**Hands-on exercise:**
```python
# Build a simple pipeline from scratch
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_tennis_data("atp_tennis.csv")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Accuracy: {accuracy:.2%}")
```

#### 3B. Model Evaluation
**Source:** https://scikit-learn.org/stable/modules/model_evaluation.html

**Learn:**
- Confusion matrix
- Accuracy, Precision, Recall, F1
- ROC curves and ROC-AUC
- Cross-validation

**Apply to this project:**
```python
# From training.py - evaluate_model function
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1])
}
```

---

### Level 4: Advanced Topics (3-4 weeks)

#### 4A. Hyperparameter Tuning
**Source:** https://scikit-learn.org/stable/modules/grid_search.html

**Learn:**
- GridSearchCV
- RandomizedSearchCV
- Cross-validation strategies

**Apply to this project:**
```python
# Example: Tune Decision Tree
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5  # 5-fold cross-validation
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2%}")

# Use best model
best_model = grid_search.best_estimator_
```

**Hands-on exercise:**
- Improve the Random Forest model accuracy
- Try different hyperparameters
- Compare results

#### 4B. Feature Importance Analysis
**Source:** https://scikit-learn.org/stable/modules/ensemble.html#feature-importance

**Learn:**
- How decision trees measure importance
- How random forests aggregate importance
- How to visualize feature importance

**Apply to this project:**
```python
# From training.py - get_feature_importance function
import numpy as np

# Get importance scores
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Display top features
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Visualize (requires matplotlib)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices][:10])
plt.xticks(range(10), [feature_names[indices[i]] for i in range(10)], rotation=45)
plt.title("Top 10 Feature Importances")
plt.show()
```

#### 4C. Statistical Learning Theory
**Source:** https://www.statlearning.com/ (ISLR Book)

**Learn:**
- Bias-variance tradeoff
- Overfitting vs underfitting
- Model complexity
- Test error vs training error

**Apply to this project:**
- Understand why we use train_test_split
- Why max_depth=10 for trees (prevents overfitting)
- Why ensemble methods (RF) work better

---

## 🔨 Hands-On Projects to Learn

### Project 1: Simple Player Prediction (Beginner)
**Learn:** Data loading, basic filtering, simple statistics

```python
import pandas as pd

# Load data
df = pd.read_csv("atp_tennis.csv", low_memory=False)

# Get Nadal vs Federer matches
nadal_fed = df[(df['Player_1'] == 'Nadal R.') & (df['Player_2'] == 'Federer R.')]
print(f"Nadal vs Federer: {len(nadal_fed)} matches")
print(nadal_fed['Winner'].value_counts())

# Simple prediction: higher-ranked player wins
nadal_wins = len(nadal_fed[nadal_fed['Winner'] == 'Nadal R.'])
federer_wins = len(nadal_fed) - nadal_wins
print(f"Nadal wins: {nadal_wins / len(nadal_fed):.1%}")
```

### Project 2: Reproduce This Project (Intermediate)
**Learn:** Scikit-learn, pipelines, model evaluation

1. Load ATP data
2. Preprocess (scale, encode)
3. Train decision tree
4. Evaluate with metrics
5. Make predictions

### Project 3: Improve the Models (Advanced)
**Learn:** Hyperparameter tuning, cross-validation, ensemble methods

1. Use GridSearchCV to tune parameters
2. Implement cross-validation
3. Try additional models (SVM, GradientBoosting)
4. Compare all models
5. Create ensemble of best models

### Project 4: Build Your Own Dataset (Expert)
**Learn:** Data collection, cleaning, real-world ML

1. Scrape or download new tennis data
2. Preprocess and clean
3. Train models from scratch
4. Compare to ATP model
5. Deploy and make predictions

---

## 📖 Recommended Study Schedule

### Week 1-2: Foundations
- Python basics (Python docs)
- NumPy fundamentals (NumPy tutorial)
- Pandas basics (Pandas 10 minutes tutorial)

### Week 3-4: Machine Learning Theory
- Watch Andrew Ng videos (Week 1-3)
- Watch StatQuest decision trees + random forests

### Week 5-6: Scikit-learn
- Official scikit-learn getting started
- Reproduce this project
- Understand each algorithm

### Week 7-8: Advanced Topics
- GridSearchCV and hyperparameter tuning
- Feature importance analysis
- Watch remaining Andrew Ng course

### Week 9+: Apply & Create
- Build your own projects
- Modify this project
- Explore other datasets (Kaggle)

---

## 🎯 Key Milestones & Checkpoints

### ✅ Checkpoint 1: Can you load ATP data and explore it?
```python
import pandas as pd
df = pd.read_csv("atp_tennis.csv")
print(df.shape)
print(df.describe())
# If this works, you're ready for Level 2
```

### ✅ Checkpoint 2: Can you train a simple decision tree?
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
# If this works, you're ready for Level 3
```

### ✅ Checkpoint 3: Can you understand the preprocessing pipeline?
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# If you can build a ColumnTransformer without looking at examples,
# you're ready for Level 4
```

### ✅ Checkpoint 4: Can you tune model hyperparameters?
```python
from sklearn.model_selection import GridSearchCV

# If you can run GridSearchCV and interpret results,
# you're ready to build your own projects
```

---

## 💡 Pro Tips for Learning

1. **Learn by doing:** Don't just watch videos, code along
2. **Read official docs:** Often better than tutorials
3. **Experiment:** Try different parameters, see what breaks
4. **Read the source code:** scikit-learn code is readable
5. **Join communities:** Reddit r/MachineLearning, StackOverflow
6. **Build projects:** Best way to learn is building
7. **Ask questions:** Stack Overflow, GitHub discussions

---

## 🚀 Next Projects After This One

### Project A: Stock Price Prediction
- Similar classification approach
- Use historical price data
- Apply scikit-learn models

### Project B: Movie Rating Prediction
- Regression instead of classification
- Text preprocessing for reviews
- Use TF-IDF vectorizer

### Project C: Image Classification
- Introduction to deep learning
- Use TensorFlow/PyTorch
- MNIST or CIFAR dataset

### Project D: Natural Language Processing
- Sentiment analysis
- Text classification
- Use NLTK or spaCy

---

## 📚 Reference Sheet

**Quick reference for commands you'll use often:**

```python
# Data loading
import pandas as pd
df = pd.read_csv("file.csv")

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
accuracy = accuracy_score(y_test, model.predict(X_test))

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## 🎓 Final Advice

> **"The best way to learn machine learning is to build machine learning projects."** - Andrew Ng

Start small, build projects, learn from mistakes. This project you just completed demonstrates:
- ✅ Data loading and preprocessing
- ✅ Training multiple models
- ✅ Evaluating performance
- ✅ Making predictions
- ✅ Saving/loading models
- ✅ Testing and validation

You now have a solid foundation. Use the resources above to deepen your understanding and build more projects!

Good luck! 🚀
