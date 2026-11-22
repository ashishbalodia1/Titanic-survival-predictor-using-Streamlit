# Titanic Survival Prediction & Explainable AI (Tuned Decision Tree)
**Project 04 | Data Science & AI**

This project builds, tunes, and explains a Decision Tree Classifier using Scikit-Learn to predict passenger survival on the RMS Titanic. Hyperparameters were systematically optimized using **GridSearchCV with 5-Fold Stratified Cross-Validation**, and the model features transparent **SHAP (SHapley Additive exPlanations)** reasoning for every prediction.

---

## Project Overview

- **Dataset:** Titanic dataset from Seaborn (891 passenger records).
- **Target:** `survived` (0 = Did not survive, 1 = Survived).
- **Features:** `pclass`, `sex`, `age`, `fare`.
- **Optimization:** `GridSearchCV` over `max_depth`, `min_samples_leaf`, `min_samples_split`, and `criterion`.
- **Best Hyperparameters:** `{'criterion': 'gini', 'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 2}`.
- **Explainable AI:** SHAP `TreeExplainer` providing feature-level attribution and reasoning for every prediction.
- **Interactive Web App:** Streamlit dashboard with real-time input sliders and visual SHAP contribution charts.
- **Split:** 80% train (571 samples), 20% test (143 samples), stratified on survival.

---

## Files in this Repository

- `app.py`: Interactive Streamlit web application.
- `model.joblib`: Serialized trained Decision Tree classifier.
- `titanic_classifier.ipynb`: Complete Jupyter Notebook with code, GridSearchCV tuning, evaluation outputs, and plots.
- `titanic.csv`: Local copy of the Titanic dataset.
- `requirements.txt`: Project dependencies for both training and the web app.
- `project-4-guide.pdf`: Project instructions and requirements.
- `README.md`: Project summary, hyperparameter tuning results, and baseline comparison.

---

## Step-by-Step Workflow

1. **Data Exploration:** Inspected shape (891 rows, 15 columns) and missing values across the 5 selected columns (`survived`, `pclass`, `sex`, `age`, `fare`).
2. **Preprocessing:** Dropped 177 rows with missing `age` values, leaving 714 clean rows. Encoded `sex` into numbers (0 for male, 1 for female).
3. **Train-Test Split:** Split into 80% training (571 rows) and 20% testing (143 rows) using `random_state=42` and `stratify=y` to preserve survival rates (~36.5% survivors).
4. **Baseline Model:** Calculated the majority-class baseline. Always predicting "Did not survive" gives **59.44% accuracy**.
5. **Model Tuning with GridSearchCV:** Evaluated 448 parameter combinations using 5-Fold Stratified Cross-Validation on the training set. The best tree reached **81.98% cross-validation accuracy**.
6. **Evaluation:** Tested the tuned model on the 143 unseen test passengers using accuracy, confusion matrix, precision, recall, specificity, and F1-score.
7. **Overfitting Analysis:** Tracked training vs. testing accuracy across depths 1 to 12.
8. **Feature Importance:** Analyzed Gini importances to determine the key factors driving survival decisions.

---

## Baseline Comparison & Evaluation Results

### Model vs. Baseline Comparison

| Metric | Majority Baseline | Tuned Decision Tree | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **5-Fold Cross-Validation Accuracy** | 59.44% | **81.98%** | **+22.54% points** |
| **Test Set Accuracy (Unseen Data)** | 59.44% | **77.62%** | **+18.18% points** |
| **Training Set Accuracy** | 59.44% | **90.37%** | **+30.93% points** |
| **Precision (Survivors)** | 0.00% | **76.00%** | **+76.00% points** |
| **Recall (Survivors)** | 0.00% | **65.52%** | **+65.52% points** |
| **Specificity (Non-Survivors)** | 100.00% | **85.88%** | -14.12% points |
| **F1-Score (Survivors)** | 0.0000 | **0.7037** | **+0.7037** |

*Note: The baseline model achieves 59.44% simply by guessing "Did not survive" for every passenger. It has 0% recall and 0% precision for actual survivors. The tuned Decision Tree correctly identifies survivors with 76% precision and 65.5% recall, adding substantial real-world predictive utility.*

---

### Confusion Matrix (Test Set, n = 143)

```
                      Predicted: Died    Predicted: Survived
Actual: Died                 73 (TN)            12 (FP)
Actual: Survived             20 (FN)            38 (TP)
```

- **True Negatives (TN = 73):** Passengers who died and were correctly identified as non-survivors (Specificity: **85.88%**).
- **False Positives (FP = 12):** Passengers who died, but the model mistakenly guessed they survived (Type I error).
- **False Negatives (FN = 20):** Actual survivors who were predicted to perish (Type II error).
- **True Positives (TP = 38):** Actual survivors who were correctly identified (Recall: **65.52%**).

---

### Classification Report

```text
                      precision    recall  f1-score   support

Did Not Survive (0)     0.7849    0.8588    0.8202        85
       Survived (1)     0.7600    0.6552    0.7037        58

           accuracy                         0.7762       143
          macro avg     0.7725    0.7570    0.7620       143
       weighted avg     0.7748    0.7762    0.7730       143
```

---

## GridSearchCV Hyperparameter Details

```python
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
```

- **`criterion='gini'`:** Evaluated split quality via Gini impurity.
- **`max_depth=7`:** Allowed the tree to explore nuanced interaction splits.
- **`min_samples_leaf=2`:** Constrained terminal leaves so no single outlier could create an isolated decision node, regularizing deeper branches.
- **`min_samples_split=2`:** Standard threshold for internal node partitioning.

---

## Feature Importances (Gini Impurity Reduction)

| Feature | Gini Importance | Share | Explanation |
| :--- | :---: | :---: | :--- |
| **`sex`** | **0.4090** | **40.9%** | Primary survival determinant. Women were given life-saving evacuation priority. |
| **`age`** | **0.2326** | **23.3%** | Significant weight in deeper splits, separating children and younger passengers. |
| **`pclass`** | **0.1946** | **19.5%** | 1st and 2nd class passengers had physical proximity to lifeboat launch stations. |
| **`fare`** | **0.1639** | **16.4%** | Economic indicator refining cabin location and accommodation deck. |

---

## Explainable AI (SHAP Reasoning)

To move beyond "black-box" predictions and understand *why* the model makes specific survival calls, we use **SHAP (SHapley Additive exPlanations)**.

### How SHAP Works in This Project
- **Baseline Starting Point:** The model begins with the historical cohort survival rate (**40.63%**).
- **Additive Contributions:** For any passenger, each attribute adds positive or negative percentage points to the baseline:
  - **`sex = female`:** Strongly pushes probability **UP (+43.1%)** reflecting the "women and children first" protocol.
  - **`sex = male`:** Pushes probability **DOWN (-11.4%)**.
  - **`pclass = 1`:** Pushes probability **UP (+24.5%)** due to boat deck proximity.
  - **`pclass = 3`:** Pushes probability **DOWN (-12.4%)** due to lower deck barriers.
  - **`age` & `fare`:** Refine the probability based on whether the passenger was a child, adult, or senior, and their ticket tier.
- **Interactive App Explanations:** The Streamlit app renders real-time SHAP contribution charts and plain-English justification cards for every single prediction.

---

## Overfitting Analysis (Depths 1 to 12)

| Depth | Training Accuracy | Testing Accuracy | Gap (Train - Test) | Observation |
| :---: | :---: | :---: | :---: | :--- |
| **1** | 77.93% | 78.32% | -0.39% | Single decision stump on gender |
| **2** | 79.51% | 77.62% | +1.89% | Underfitting |
| **3** | **81.79%** | **81.12%** | **+0.67%** | Peak test accuracy |
| **4** | 84.76% | 79.72% | +5.04% | Balanced default tree |
| **5** | 86.51% | 76.22% | +10.29% | Overfitting starts |
| **7 (Tuned)** | **90.37%** | **77.62%** | **+12.75%** | **Tuned model (CV: 81.98%)** |
| **10** | 95.62% | 76.22% | +19.40% | Severe overfitting |
| **12** | 96.85% | 77.62% | +19.23% | Tree memorizes noise |

*Why deeper trees overfit:* Without regularization, a decision tree isolates individual training records. In our tuned model, setting `min_samples_leaf=2` prevents the tree from splitting down to single individuals, helping it maintain an **81.98% cross-validation score**.

---

## Limitations

1. **Dropped missing values:** Removing 177 rows with missing `age` values discarded ~20% of the dataset. Conditional imputation could retain these records in future iterations.
2. **Small test set size:** With only 143 test passengers, each prediction error alters the accuracy score by ~0.7%.
3. **Single Decision Tree sensitivity:** Single trees use greedy orthogonal splits. An ensemble model (like Random Forest) can further smooth prediction boundaries.
4. **Omitted features:** Attributes like family size (`sibsp`, `parch`), cabin deck, and embarkation town were not included in the feature set.

---

## How to Run

1. Clone or open this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive Streamlit web app:
   ```bash
   streamlit run app.py
   ```
4. Or open and explore the training notebook:
   ```bash
   jupyter notebook titanic_classifier.ipynb
   ```
5. All cells in the notebook run sequentially with reproducible outputs (`random_state=42`).
