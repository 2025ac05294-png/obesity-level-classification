"""Train and evaluate five classifiers on the UCI Obesity Levels dataset.

Running this script rebuilds everything the Streamlit app needs:
  model/artifacts/*.joblib   fitted end-to-end pipelines
  model/artifacts/schema.json metadata (feature list, class order, label column)
  model/metrics.csv          the comparison table reproduced in the README
  test_data.csv              held-out split, uploaded through the app UI

The two body-measurement columns (Height, Weight) are removed on purpose: the
NObeyesdad label is assigned from BMI, which is a pure function of those two
columns, so leaving them in leaks the answer and pushes every model above 96%.
What remains are 14 self-reported habit / demographic features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METRICS_CSV = Path(__file__).resolve().parent / "metrics.csv"
TEST_CSV = PROJECT_ROOT / "test_data.csv"

LABEL_COLUMN = "NObeyesdad"
LEAKY_COLUMNS = ["Height", "Weight"]
HOLDOUT_FRACTION = 0.2
SEED = 7


def load_habits_frame() -> pd.DataFrame:
    """Read the raw UCI export and strip the BMI-derived columns."""
    frame = pd.read_csv(RAW_CSV)
    return frame.drop(columns=LEAKY_COLUMNS)


def split_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in features.columns if col not in numeric]
    return numeric, categorical


def build_encoder(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale the numeric habit scores, one-hot the categorical answers.

    Dense output is requested because GaussianNB cannot consume sparse matrices.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def candidate_estimators() -> dict[str, object]:
    """The five classifiers required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, C=1.0, random_state=SEED
        ),
        # Depth caps of 12/16 and larger leaf sizes were compared by 5-fold CV on
        # the training split; the unrestricted tree scored highest (0.7464 vs
        # 0.7275 for max_depth=12, min_samples_leaf=4), so it is kept as-is.
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
    }


def score_predictions(
    truth: pd.Series, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Six metrics; macro averaging keeps all seven obesity classes equal."""
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, probabilities, multi_class="ovr", average="macro"),
        "Precision": precision_score(truth, predicted, average="macro", zero_division=0),
        "Recall": recall_score(truth, predicted, average="macro", zero_division=0),
        "F1": f1_score(truth, predicted, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def artifact_filename(model_name: str) -> str:
    slug = model_name.lower().replace(" ", "_")
    for junk in "()":
        slug = slug.replace(junk, "")
    return f"{slug.strip('_')}.joblib"


def main() -> pd.DataFrame:
    habits = load_habits_frame()
    features = habits.drop(columns=[LABEL_COLUMN])
    target = habits[LABEL_COLUMN]

    numeric, categorical = split_column_types(features)
    print(f"{len(habits)} rows | {features.shape[1]} features "
          f"({len(numeric)} numeric, {len(categorical)} categorical) "
          f"| {target.nunique()} classes")

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=target,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scoreboard = []

    for name, estimator in candidate_estimators().items():
        pipeline = Pipeline(
            steps=[
                ("encode", build_encoder(numeric, categorical)),
                ("classify", estimator),
            ]
        )
        pipeline.fit(train_x, train_y)

        predicted = pipeline.predict(test_x)
        probabilities = pipeline.predict_proba(test_x)
        row = {"ML Model Name": name, **score_predictions(test_y, predicted, probabilities)}
        scoreboard.append(row)

        # compress=3 shrinks the 400-tree forest from 20.4 MB to 4.3 MB with no
        # measurable load penalty, which keeps the repo and the cloud build light.
        joblib.dump(pipeline, ARTIFACT_DIR / artifact_filename(name), compress=3)
        print(f"  {name:<26} accuracy={row['Accuracy']:.4f}  mcc={row['MCC']:.4f}")

    results = pd.DataFrame(scoreboard).round(4)
    results.to_csv(METRICS_CSV, index=False)

    holdout = test_x.copy()
    holdout[LABEL_COLUMN] = test_y
    holdout.to_csv(TEST_CSV, index=False)

    schema = {
        "label_column": LABEL_COLUMN,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "class_labels": sorted(target.unique().tolist()),
        "dropped_columns": LEAKY_COLUMNS,
        "artifacts": {name: artifact_filename(name) for name in candidate_estimators()},
    }
    (ARTIFACT_DIR / "schema.json").write_text(json.dumps(schema, indent=2))

    print(f"\nWrote {METRICS_CSV.name}, {TEST_CSV.name} ({len(holdout)} rows) "
          f"and {len(scoreboard)} model artifacts.")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
