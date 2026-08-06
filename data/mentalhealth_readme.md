# 🧠 Student Mental Health Level Classifier
### Predicting Mental Health Tiers from Social Media & Demographic Features

---

## 📌 What This Project Does

Social media use has become central to student life — but its relationship to mental health is poorly understood in a predictive sense. This project answers the question:

> **Can we reliably predict a student's mental health tier (Low / Moderate / High) from their social media habits and basic demographics alone — without asking them directly about their mental state?**

We build a full supervised ML pipeline that:
- Cleans and encodes a real student survey dataset
- Engineers a 3-class mental health target from raw scores
- Trains and compares three progressively powerful classifiers
- Evaluates them honestly with stratified cross-validation
- Identifies which features matter most and visualises all results

The best model achieves **90% test accuracy and 88% CV accuracy**, meaning this signal is genuinely learnable from behavioural data.

---

## 🗂️ Project Walkthrough — Step by Step

### Step 1 — Data Loading & Inspection
The dataset `students_socialmedia.csv` contains student-level survey responses covering social media usage, sleep, academics, and demographics. On load, the shape is printed to confirm it was read correctly.

### Step 2 — Missing Value Imputation
Missing entries are filled with the **column mode** (most frequent value). This is robust for categorical-heavy survey data where mean/median imputation would be semantically wrong (e.g., filling "Platform" with a mean makes no sense).

### Step 3 — Categorical Encoding
All object-type columns are passed through `LabelEncoder`, converting string labels (e.g., `"Instagram"`, `"Male"`) to integers. This makes the data compatible with all three sklearn-based models without expanding dimensionality (unlike one-hot encoding).

### Step 4 — Target Engineering
The raw `Mental_Health_Score` column (integer, 4–9) is binned into three ordinal classes:

| Class | Score Range | Meaning |
|---|---|---|
| 0 | 4–5 | Low mental health |
| 1 | 6–7 | Moderate mental health |
| 2 | 8–9 | High mental health |

This transforms the problem from a regression (predict a number) into a **meaningful classification** (predict a wellness tier), which is more actionable and interpretable.

### Step 5 — Feature Leakage Prevention
Several columns are explicitly dropped before any model sees the data:

| Dropped Column | Reason |
|---|---|
| `Mental_Health_Score` | Raw source of the target — direct leak |
| `MH_Level` | The target itself |
| `Student_ID` | Non-informative identifier |
| `Addicted_Score` | Consequence of the same behaviour, not a cause |
| `Affects_Academic_Performance` | Outcome of poor mental health, not predictor |
| `Conflicts_Over_Social_Media` | Downstream effect, would inflate accuracy artificially |

Skipping this step would cause the model to cheat, learning target-correlated noise instead of real behaviour patterns.

### Step 6 — Stratified Train / Test Split
Data is split 80/20 with `stratify=y`, ensuring all three mental health classes appear in both sets at the same proportion as the full dataset. Without stratification, rare classes (especially "High") could be underrepresented in the test set, giving misleadingly good results.

### Step 7 — Model Training
Three models are trained, each representing a different level of complexity:

**Logistic Regression** — linear baseline
- Wrapped in a `StandardScaler` pipeline (mandatory for regularised linear models)
- Low regularisation (`C=0.05`) to avoid overfitting on a small dataset
- Cannot capture non-linear interactions between features

**Random Forest** — non-linear ensemble
- 80 decision trees, each trained on a random feature subset
- `max_depth=5` and `min_samples_leaf=5` prevent individual trees from memorising training data
- Captures feature interactions that Logistic Regression misses
- Also provides feature importances as a bonus

**XGBoost (Gradient Boosting)** — sequential boosting ensemble
- Builds trees one at a time, each correcting the errors of the previous
- Conservative settings: `lr=0.05`, `max_depth=3`, `subsample=0.7` — reduces overfitting
- Best suited for structured tabular data; consistently outperforms random forests on small-to-medium datasets

### Step 8 — Evaluation
Each model is evaluated on three metrics:
- **Test Accuracy** — raw proportion correct on held-out test set
- **Macro F1** — average F1 across all three classes, treating them equally regardless of size (fairer than accuracy for imbalanced classes)
- **5-fold Stratified CV Accuracy** — most trustworthy metric; the model is trained and tested 5 times on different splits, and the mean ± std is reported

CV accuracy is used as the primary selection criterion because it reflects how the model generalises across the entire dataset, not just one lucky split.

### Step 9 — Visualisation
Three plots are generated and saved as `social_media_results.png`:
1. **CV vs Test Accuracy bar chart** — side-by-side comparison with error bars showing CV std
2. **Confusion Matrix** — for the best model (XGBoost), showing exactly where predictions succeed or fail
3. **Feature Importance chart** — Random Forest importances showing which input variables carry the most predictive signal

---

## 📊 Results

### Model Performance

| Model | CV Accuracy | CV Std | Test Accuracy | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.74 | ±~0.02 | 0.74 | Low |
| Random Forest | 0.83 | ±~0.02 | 0.84 | Moderate |
| **XGBoost (GBT)** | **0.88** | **±~0.01** | **0.90** | **Best** |

> 5-fold Stratified Cross-Validation · XGBoost selected as best model by CV Accuracy

### XGBoost Confusion Matrix (Test Set)

| | Predicted Low | Predicted Moderate | Predicted High |
|---|---|---|---|
| **Actual Low** | 32 | 8 | 1 |
| **Actual Moderate** | 1 | 77 | 1 |
| **Actual High** | 0 | 3 | 18 |

- **Moderate** class: near-perfect (77/79, 97.5% recall) — largest class, most training signal
- **High** class: 18/21 correct (85.7% recall) — 3 edge cases confused with Moderate
- **Low** class: 32/41 correct (78% recall) — 8 cases drift into Moderate, likely genuine score-boundary ambiguity

### Feature Importances (Random Forest)

| Rank | Feature | Approx. Importance | Insight |
|---|---|---|---|
| 1 | `Avg_Daily_Usage_Hours` | ~0.42 | By far the strongest signal — heavy usage strongly correlates with poor mental health |
| 2 | `Sleep_Hours_Per_Night` | ~0.20 | Sleep disruption is a well-known mediator between screen time and wellbeing |
| 3 | `Country` | ~0.16 | Cultural and systemic differences in mental health norms vary by geography |
| 4 | `Most_Used_Platform` | ~0.10 | Platform type matters — passive scrolling differs from active social use |
| 5 | `Age` | ~0.08 | Younger students may be more vulnerable to usage-related effects |
| 6 | `Academic_Level` | ~0.03 | University vs school-level stress differs slightly |
| 7 | `Relationship_Status` | ~0.02 | Minor but present social context signal |
| 8 | `Gender` | ~0.02 | Marginal demographic effect after controlling for usage |

### Output Plot

<img width="1880" height="707" alt="Screenshot 2026-05-09 113359" src="https://github.com/user-attachments/assets/a0a111ad-6d9e-48bd-8690-6b3fda708130" />


---

## 🔍 Why Such High Accuracy? — An Honest Analysis

90% accuracy on a 3-class problem is strong. Here is why it was achievable and what to keep in mind:

### ✅ Why It Works

**One dominant feature does the heavy lifting.**
`Avg_Daily_Usage_Hours` alone accounts for ~42% of the Random Forest's predictive power. It is a clean, continuous, low-noise variable that directly encodes behavioural intensity. When one feature is this discriminative, a well-regularised model can anchor most predictions on it reliably.

**Sleep compounds the signal.**
`Sleep_Hours_Per_Night` adds another ~20% importance. Sleep and screen time together are strongly correlated with the target because mental health — as a lived experience — manifests most visibly through these two behavioural channels. The model is essentially learning: *high usage + low sleep = poor mental health*, which matches established psychology literature.

**XGBoost is the right tool for this data shape.**
Sequential gradient boosting excels at tabular data with a mix of a few dominant features and several weaker ones. Its shallow trees (`max_depth=3`) avoid memorising the training set, while each round corrects prior errors — making it particularly effective at resolving the boundary cases (Low vs Moderate) that confuse single-tree models.

**The Moderate class is large and clean.**
The majority of students fall in the Moderate tier. Predicting this class well alone accounts for a large fraction of the overall accuracy. XGBoost achieves 97.5% recall on Moderate, which lifts the headline number.

**Stratification and CV prevent inflated estimates.**
Every metric is computed on data the model never trained on. The close match between CV accuracy (0.88) and test accuracy (0.90) confirms the model is genuinely generalising, not overfitting to a lucky split.

### ⚠️ Caveats

| Caveat | Detail |
|---|---|
| Low class is hardest | 78% recall — the Low/Moderate boundary is fuzzy in self-reported data, and the model reflects that human ambiguity |
| Single survey source | Performance may not generalise to different student populations, countries, or time periods |
| Correlation ≠ causation | The model learns co-occurrence patterns — it cannot confirm that reducing screen time *causes* better mental health |
| Class imbalance | Moderate dominates; Macro F1 is a better single-number summary than accuracy for this reason |

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

> No external XGBoost package needed — `GradientBoostingClassifier` from scikit-learn is used as the boosting model.

---

## 🚀 Usage

1. **Clone the repository**

```bash
git clone https://github.com/your-username/student-mental-health-classifier.git
cd student-mental-health-classifier
```

2. **Place your dataset** in the project root as `students_socialmedia.csv`

3. **Run the script**

```bash
python social_media_fixed.py
```

4. **Outputs**
   - Console: class distribution, per-model metrics, CV scores, best-model classification report
   - File: `social_media_results.png` — three-panel comparison figure

---

## 📁 Project Structure

```
├── social_media_fixed.py        # Main training & evaluation script
├── students_socialmedia.csv     # Dataset (not included — add your own)
├── social_media_results.png     # Output plot (generated on run)
└── README.md
```

---

## 🔬 Full Methodology Reference

| Step | Choice | Justification |
|---|---|---|
| Missing value imputation | Column mode | Robust for categorical survey data |
| Encoding | `LabelEncoder` | Compact; no dimensionality explosion vs one-hot |
| Train/test split | 80/20, stratified | Preserves class ratios in both sets |
| Cross-validation | 5-fold Stratified KFold | Reliable generalisation estimate |
| LR regularisation | `C=0.05` | Strong penalty prevents overfitting on small data |
| RF depth limit | `max_depth=5`, `min_samples_leaf=5` | Balances expressiveness and generalisation |
| GBT learning rate | `lr=0.05`, `subsample=0.7` | Slow learning + row sampling reduces overfitting |
| Model selection criterion | CV Accuracy | Less sensitive to test-set luck than a single split |

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.
