# Heart Disease Prediction Using Machine Learning

A beginner-friendly machine learning project that trains and compares four classification models (Logistic Regression, Random Forest, Support Vector Machine and XGBoost) on a public heart-health dataset. It was built as part of the **CodeAlpha Machine Learning Internship**.

> **Educational project only.** This work is not a clinically validated diagnostic tool and must not be used for medical decisions. See the [Disclaimer](#disclaimer).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives and Key Features](#objectives-and-key-features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Repository Structure](#repository-structure)
6. [Installation and Setup](#installation-and-setup)
7. [How to Run the Notebooks](#how-to-run-the-notebooks)
8. [Machine Learning Workflow](#machine-learning-workflow)
9. [Model Comparison](#model-comparison)
10. [Best-Performing Model](#best-performing-model)
11. [Key Observations and Learnings](#key-observations-and-learnings)
12. [Limitations and Future Improvements](#limitations-and-future-improvements)
13. [Disclaimer](#disclaimer)
14. [Author and Acknowledgement](#author-and-acknowledgement)

---

## Project Overview

**Problem statement:** Given a patient's recorded health measurements (age, blood pressure, cholesterol, maximum heart rate and similar), can a machine learning model predict the dataset's binary `target` label (`0` or `1`)?

This is a **binary classification** task. The project walks through a complete beginner-level workflow: loading data, checking its quality, handling duplicate records, splitting into training and test sets, training several models, and comparing how well each one predicts on unseen test data.

## Objectives and Key Features

- Explore the dataset and check for missing values and duplicate records.
- Remove exact duplicate rows before training.
- Train four different classifiers and evaluate each on the same kind of held-out test set.
- Report accuracy, confusion matrices and classification reports (precision, recall, F1-score).
- Compare the models side by side in a table and a bar chart.

**Implemented:** data inspection, duplicate removal, train/test split, four trained models, evaluation metrics, bar-chart comparison.
**Not implemented:** cross-validation, automated hyperparameter search, feature-importance analysis, model saving, or any web application or deployment. See [Future Improvements](#limitations-and-future-improvements).

## Technologies Used

| Purpose | Tool |
|---|---|
| Language | Python 3 |
| Environment | Jupyter notebooks (written and run in Google Colab) |
| Data handling | pandas, NumPy |
| Models and metrics | scikit-learn (Logistic Regression, Random Forest, SVM, `StandardScaler`, metrics) |
| Gradient boosting | XGBoost |
| Visualization | Matplotlib |

## Dataset

- **File:** `heart.csv` (Kaggle heart disease dataset, derived from the UCI Heart Disease data)
- **Source:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) · Original data: [UCI Heart Disease repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Size:** 1,025 rows × 14 columns (13 features + 1 target)
- **Missing values:** none (verified in `Heart_Disease_1.ipynb`)
- **Target column:** `target`, a binary label (`0` / `1`). The class distribution in the original file is 499 zeros and 526 ones.

### Feature columns

| Column | Typical meaning (per the UCI documentation) |
|---|---|
| `age` | Age in years |
| `sex` | Sex (encoded as 0/1) |
| `cp` | Chest pain type (coded category) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar above a threshold (0/1) |
| `restecg` | Resting electrocardiogram result (coded category) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (0/1) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (coded category) |
| `ca` | Number of major vessels (coded count) |
| `thal` | Thalassemia-related test result (coded category) |

### Important note on duplicates

The raw file contains **723 exact duplicate rows**. Removing them leaves **302 rows**, so the 1,025 rows should **not** be read as 1,025 unique patients. All notebooks remove duplicates before splitting the data, so the models are trained and tested on the 302 unique rows:

| Dataset version | Class 0 | Class 1 | Total |
|---|---:|---:|---:|
| Original | 499 (48.68%) | 526 (51.32%) | 1,025 |
| After removing duplicates | 138 (45.70%) | 164 (54.30%) | 302 |

Because every notebook splits **after** deduplication, identical rows cannot appear in both the training and test sets.

### Note on the meaning of `target`

The notebooks do not document what `0` and `1` mean, and this README does not assume it. Please consult the dataset's source page before interpreting predictions. In this file, rows with `target = 1` have on average a higher maximum heart rate and lower `oldpeak` than rows with `target = 0`, so the label encoding should be checked against the source documentation rather than assumed.

## Repository Structure

```text
CodeAlpha_Disease_Prediction/
├── datasets/
│   └── heart.csv
├── notebooks/
│   ├── Heart_Disease_1.ipynb
│   ├── Random_Forest_model_2.ipynb
│   ├── SVM_MODEL_3.ipynb
│   ├── XGBoost_4.ipynb
│   └── Model_Comparison_Table.ipynb
├── requirements.txt
└── README.md
```

| Notebook | Contents |
|---|---|
| [`Heart_Disease_1.ipynb`](notebooks/Heart_Disease_1.ipynb) | Dataset inspection, duplicate check, class balance, stratified train/test split, **Logistic Regression** |
| [`Random_Forest_model_2.ipynb`](notebooks/Random_Forest_model_2.ipynb) | Random Forest classifier |
| [`SVM_MODEL_3.ipynb`](notebooks/SVM_MODEL_3.ipynb) | Support Vector Machine with feature scaling |
| [`XGBoost_4.ipynb`](notebooks/XGBoost_4.ipynb) | XGBoost classifier with manually chosen, regularized settings |
| [`Model_Comparison_Table.ipynb`](notebooks/Model_Comparison_Table.ipynb) | Accuracy table and bar chart |

## Installation and Setup

The notebooks were written for **Google Colab**, which already includes most of the libraries. To run locally instead, you need Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/Someone18-6/CodeAlpha_Disease_Prediction.git
cd CodeAlpha_Disease_Prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install the libraries
pip install pandas numpy matplotlib scikit-learn xgboost jupyter

# 4. Start Jupyter
jupyter notebook
```

**Running locally requires small edits.** Several notebooks load the data through Google Colab (`from google.colab import files` followed by an upload prompt, or the path `/content/heart.csv`). Outside Colab, replace those lines with:

```python
df = pd.read_csv("../datasets/heart.csv")
```

and remove the `google.colab` import.

## How to Run the Notebooks

Run them in this order. Each notebook is self-contained and loads the data itself.

1. **`Heart_Disease_1.ipynb`**: understand the data and train Logistic Regression.
2. **`Random_Forest_model_2.ipynb`**: train Random Forest.
3. **`SVM_MODEL_3.ipynb`**: train the SVM.
4. **`XGBoost_4.ipynb`**: train XGBoost.
5. **`Model_Comparison_Table.ipynb`**: view the comparison chart.

> The comparison notebook uses **accuracy values typed in by hand**. It does not read results from the other notebooks, so re-running the models will not update it automatically.

## Machine Learning Workflow

### 1. Data loading and exploration
The CSV is loaded with pandas. `head()`, `tail()`, `shape`, `info()` and `isnull().sum()` confirm 1,025 rows, 14 columns, all numeric, and no missing values.

### 2. Data cleaning
Exact duplicate rows (723) are removed with `drop_duplicates()`, leaving 302 rows. No other cleaning, outlier handling or feature engineering is performed.

### 3. Exploratory data analysis
The EDA in this project is limited to summary tables and class-balance counts (before and after deduplication). No feature-distribution plots or correlation analysis are included.

### 4. Train/test splitting
The 302 unique rows are split 80% / 20% with `random_state=42`, giving **241 training rows and 61 test rows**.

| Notebook | Stratified? | Test set: class 0 / class 1 |
|---|---|---|
| Logistic Regression (`Heart_Disease_1`) | Yes (`stratify=y`) | 28 / 33 |
| Random Forest, SVM, XGBoost | No | 32 / 29 |

> **Comparability caveat:** Logistic Regression was tested on a *different* set of 61 rows than the other three models. This is why its class counts differ. Its accuracy is therefore not strictly like-for-like with the others.

### 5. Model training

| Model | Settings used in the notebook |
|---|---|
| Logistic Regression | `max_iter=1000`, no feature scaling |
| Random Forest | `n_estimators=100`, `random_state=42` |
| SVM | `SVC(kernel="rbf")`, features scaled with `StandardScaler` (fit on training data only) |
| XGBoost | `n_estimators=100`, `max_depth=2`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`, `random_state=42` |

### 6. Hyperparameter tuning
No automated search (such as grid search or cross-validation) is implemented. The XGBoost values above were **set by hand** to keep the model simple and limit overfitting. The other models use default or near-default settings.

### 7. Evaluation
Each model is evaluated on its test set with **accuracy**, a **confusion matrix** and a **classification report** (precision, recall, F1-score, support). Only these metrics are computed. ROC-AUC and cross-validation are not included.

## Model Comparison

All results are on a single 61-row test set per model, taken directly from the notebook outputs.

| Rank | Model | Accuracy | Correct / 61 | Macro F1 | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---:|---|---:|---:|---:|---|
| 1 | XGBoost | **85.25%** | 52 | 0.85 | `[[25, 7], [2, 27]]` |
| 2 | Random Forest | 83.61% | 51 | 0.84 | `[[25, 7], [3, 26]]` |
| 3 | Logistic Regression | 80.33% | 49 | 0.80 | `[[21, 7], [5, 28]]` |
| 4 | SVM (RBF) | 78.69% | 48 | 0.79 | `[[24, 8], [5, 24]]` |

The bar chart is in [`Model_Comparison_Table.ipynb`](notebooks/Model_Comparison_Table.ipynb).

## Best-Performing Model

**XGBoost** achieved the highest test accuracy (85.25%, 52 of 61 correct). Its classification report on that test set:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.93 | 0.78 | 0.85 | 32 |
| 1 | 0.79 | 0.93 | 0.86 | 29 |

In plain terms: when the model predicted class 0 it was usually right (0.93), and it found 93% of the true class-1 rows (recall 0.93). It made only 2 false negatives, though it produced 7 false positives.

**Please read this result carefully.** The test set has only 61 rows, so one single prediction changes accuracy by about 1.6 percentage points. XGBoost's lead over Random Forest is **one prediction**. With this little data, the ranking of the top two models could easily change with a different split, so the results do not prove that XGBoost is truly better. Neither do they show that any model would work on real patients.

## Key Observations and Learnings

- Removing duplicates cut the dataset from 1,025 rows to 302. Checking for duplicates before splitting avoids inflated scores caused by identical rows landing in both the training and test sets.
- The tree-based models (XGBoost, Random Forest) scored higher than Logistic Regression and the SVM on these test sets, but the gaps are only a few predictions wide.
- Accuracy alone hides how errors are distributed, so the confusion matrices and classification reports are worth reading, especially recall for each class.
- Small test sets give noisy results. Cross-validation would give a more trustworthy estimate.

## Limitations and Future Improvements

**Limitations**

- Only 302 unique rows are available, and only 61 are used for testing.
- Each model is evaluated on one train/test split. Results can change with a different random seed.
- The Logistic Regression test set differs from the other models' test set (see above).
- No automated hyperparameter tuning or cross-validation.
- The comparison notebook uses hard-coded numbers.
- The dataset is old, comes from a limited population, and its label meaning should be verified.

**Possible future improvements** (not yet implemented)

- Use the same stratified split for all models and compare them with repeated or stratified k-fold cross-validation.
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`, using validation data rather than the test set.
- Add ROC-AUC, feature-importance analysis and EDA plots.
- Wrap preprocessing and models in scikit-learn pipelines.
- Compute the comparison table from live results instead of typed-in values.

## Disclaimer

This project is for **educational purposes only**. The models are trained on a small public dataset and have not been clinically validated. Do not use them for medical diagnosis, treatment decisions or any real-world health assessment. Consult a qualified healthcare professional for medical concerns.

## Author and Acknowledgement

Created by [Someone18-6](https://github.com/Someone18-6) as part of the **CodeAlpha Machine Learning Internship**. Thanks to CodeAlpha for the internship opportunity, and to the UCI Machine Learning Repository and the Kaggle dataset contributors for making the data publicly available.
