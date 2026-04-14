# Adult Census Income Classification

## Overview

This project builds a clean end-to-end machine learning workflow for predicting whether an individual's annual income exceeds `$50K` using the Adult Census dataset from OpenML.

The project compares three baseline tabular classification models under the same preprocessing pipeline:

- Logistic Regression
- Random Forest
- Gradient Boosting

The goal is to create a portfolio-ready classification project that demonstrates practical preprocessing, multi-model evaluation, result tracking, and basic interpretability on structured census data.

## Dataset

- Source: OpenML Adult dataset
- Loader: `fetch_openml(name="adult", version=2, as_frame=True)`
- Samples: 48,842
- Features: 14 input features + 1 target column
- Target column: `class`
- Classes: `<=50K`, `>50K`

The dataset contains demographic, education, employment, and work-hour related attributes such as age, workclass, education, occupation, marital status, and hours worked per week.

## Problem Type

- Supervised learning
- Binary classification
- Structured tabular machine learning

## Methods Used

- `pandas` for data handling
- `scikit-learn` for preprocessing, modeling, and evaluation
- `ColumnTransformer` for mixed-type preprocessing
- `Pipeline` for unified preprocessing + model training
- Train/test split with stratification
- Multi-model comparison across linear and tree-based methods

## Preprocessing

The same preprocessing pipeline is applied to every model:

- Numeric features
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- Categorical features
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

This keeps preprocessing consistent across all experiments and makes model comparisons more reliable.

## Models

The following models are trained and evaluated:

- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=200, random_state=42)`
- `GradientBoostingClassifier(random_state=42)`

## Evaluation Metrics

Each model is evaluated on both the training set and the test set using:

- Accuracy
- ROC-AUC
- Classification report
- Confusion matrix
- Macro F1-score
- Weighted F1-score

## Results

The main comparison file is saved to:

```text
artifacts/metrics/model_comparison.csv
```

### Test Set Performance

| Model | Accuracy | ROC-AUC | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.8676 | 0.9214 | 0.8003 | 0.8608 |
| Random Forest | 0.8595 | 0.9055 | 0.7969 | 0.8557 |
| Logistic Regression | 0.8524 | 0.9042 | 0.7811 | 0.8462 |

### Key Takeaways

- `GradientBoostingClassifier` achieved the best overall test-set performance across all reported metrics.
- `RandomForestClassifier` performed well on the test set but showed near-perfect training performance, suggesting stronger overfitting.
- `LogisticRegression` provided a solid and interpretable baseline, but underperformed relative to the tree-based models on this dataset.

## Visualizations

The project saves the following plots for each model:

- ROC curve
- Confusion matrix

Saved in:

```text
artifacts/plots/
    roc_<model_name>.png
    confusion_<model_name>.png
```

## Feature Importance

Feature importance is exported for the tree-based models:

```text
artifacts/metrics/feature_importance_random_forest.csv
artifacts/metrics/feature_importance_gradient_boosting.csv
```

These files contain the top transformed features after preprocessing, including one-hot encoded categorical variables.

## Project Structure

```text
main.py
README.md
requirements.txt
artifacts/
    evaluation_results.json
    metrics/
        train_<model_name>.json
        test_<model_name>.json
        model_comparison.csv
        feature_importance_<model_name>.csv
    plots/
        roc_<model_name>.png
        confusion_<model_name>.png
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Run the project with:

```bash
python main.py
```

The script will:

1. Load the Adult dataset from OpenML
2. Inspect the dataset
3. Train three machine learning models
4. Evaluate each model on train and test data
5. Save metrics, comparison tables, visualizations, and feature importance outputs

## Resume-Style Summary

- Built an end-to-end income classification project on the Adult Census dataset using Python, `pandas`, and `scikit-learn`
- Implemented a unified preprocessing pipeline for mixed numeric and categorical features with model comparison across Logistic Regression, Random Forest, and Gradient Boosting
- Achieved best test performance with Gradient Boosting at `86.76%` accuracy and `0.921` ROC-AUC, with saved evaluation artifacts, ROC curves, confusion matrices, and feature importance outputs
