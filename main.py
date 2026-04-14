from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "class"
POSITIVE_CLASS = ">50K"
ARTIFACTS_DIR = Path("artifacts")
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EVALUATION_PATH = ARTIFACTS_DIR / "evaluation_results.json"
COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"


def load_data() -> pd.DataFrame:
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    if dataset.frame is None:
        raise ValueError("OpenML returned an empty dataset frame.")
    return dataset.frame.copy()


def inspect_data(df: pd.DataFrame) -> None:
    X = df.drop(columns=[TARGET_COLUMN])
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Missing value counts:\n{df.isna().sum().to_string()}")
    print(f"Target distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    print(f"Numeric columns detected: {numeric_columns}")
    print(f"Categorical columns detected: {categorical_columns}")


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Detected numeric columns: {numeric_features}")
    print(f"Detected categorical columns: {categorical_features}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    split_name: str,
) -> dict[str, object]:
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    report_text = classification_report(y, predictions)
    report_dict = classification_report(y, predictions, output_dict=True)
    confusion = confusion_matrix(y, predictions)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        positive_class = POSITIVE_CLASS if POSITIVE_CLASS in classes else classes[-1]
        positive_index = classes.index(positive_class)
        y_binary = (y == positive_class).astype(int)
        roc_auc = float(roc_auc_score(y_binary, probabilities[:, positive_index]))

    print(f"Model: {model_name} | Split: {split_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {f'{roc_auc:.4f}' if roc_auc is not None else 'not available'}")
    print(f"Classification report:\n{report_text}")
    print(f"Confusion matrix:\n{confusion}")

    return {
        "model_name": model_name,
        "dataset_name": split_name,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report_dict,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    inspect_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows")
    print(f"Training feature shape: {X_train.shape}")
    print(f"Test feature shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")

    models = [
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]

    all_results: list[dict[str, object]] = []
    test_results: list[dict[str, object]] = []

    for model_name, estimator in models:
        print(f"=== Training {model_name} ===")

        preprocessor = build_preprocessor(X_train)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        model.fit(X_train, y_train)

        print(f"=== Evaluating {model_name} on train ===")
        train_results = evaluate_model(model, X_train, y_train, model_name, "train")
        train_path = METRICS_DIR / f"train_{model_name}.json"
        train_path.write_text(json.dumps(train_results, indent=2), encoding="utf-8")
        print(f"Saved evaluation results to {train_path.resolve()}")
        all_results.append(train_results)

        print(f"=== Evaluating {model_name} on test ===")
        test_result = evaluate_model(model, X_test, y_test, model_name, "test")
        test_path = METRICS_DIR / f"test_{model_name}.json"
        test_path.write_text(json.dumps(test_result, indent=2), encoding="utf-8")
        print(f"Saved evaluation results to {test_path.resolve()}")
        all_results.append(test_result)
        test_results.append(test_result)

        if hasattr(model, "predict_proba"):
            roc_figure, roc_axis = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test,
                name=model_name,
                pos_label=POSITIVE_CLASS,
                ax=roc_axis,
            )
            roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            roc_axis.set_title(f"ROC Curve - {model_name}")
            roc_figure.tight_layout()
            roc_path = PLOTS_DIR / f"roc_{model_name}.png"
            roc_figure.savefig(roc_path, dpi=200, bbox_inches="tight")
            plt.close(roc_figure)
            print(f"Saved ROC curve to {roc_path.resolve()}")

        confusion_figure, confusion_axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            cmap="Blues",
            colorbar=False,
            ax=confusion_axis,
        )
        confusion_axis.set_title(f"Confusion Matrix - {model_name}")
        confusion_figure.tight_layout()
        confusion_path = PLOTS_DIR / f"confusion_{model_name}.png"
        confusion_figure.savefig(confusion_path, dpi=200, bbox_inches="tight")
        plt.close(confusion_figure)
        print(f"Saved confusion matrix plot to {confusion_path.resolve()}")

        fitted_estimator = model.named_steps["model"]
        if hasattr(fitted_estimator, "feature_importances_"):
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            importance_frame = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": fitted_estimator.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            importance_path = METRICS_DIR / f"feature_importance_{model_name}.csv"
            importance_frame.head(20).to_csv(importance_path, index=False)
            print(f"Saved feature importance to {importance_path.resolve()}")
            print(
                "Top 10 important features for "
                f"{model_name}:\n{importance_frame.head(10).to_string(index=False)}"
            )

    comparison_rows = []
    for result in test_results:
        report = result["classification_report"]
        comparison_rows.append(
            {
                "model_name": result["model_name"],
                "accuracy": result["accuracy"],
                "roc_auc": result["roc_auc"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }
        )

    comparison_table = pd.DataFrame(comparison_rows).sort_values(
        by=["roc_auc", "accuracy"],
        ascending=False,
        na_position="last",
    )
    comparison_table.to_csv(COMPARISON_PATH, index=False)
    print(f"Saved model comparison table to {COMPARISON_PATH.resolve()}")

    EVALUATION_PATH.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {EVALUATION_PATH.resolve()}")


if __name__ == "__main__":
    main()
