"""Streamlit front-end for the obesity-level classifiers.

Upload the held-out test_data.csv, pick one of the five trained pipelines and
inspect its metrics, confusion matrix and per-class report, or switch to the
leaderboard tab to score every model at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "model" / "artifacts"

st.set_page_config(
    page_title="Obesity Level Classifier",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_schema() -> dict:
    return json.loads((ARTIFACT_DIR / "schema.json").read_text())


@st.cache_resource(show_spinner=False)
def load_pipeline(filename: str):
    return joblib.load(ARTIFACT_DIR / filename)


@st.cache_data(show_spinner=False)
def read_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def evaluate(pipeline, features: pd.DataFrame, truth: pd.Series) -> dict[str, float]:
    predicted = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, probabilities, multi_class="ovr", average="macro"),
        "Precision": precision_score(truth, predicted, average="macro", zero_division=0),
        "Recall": recall_score(truth, predicted, average="macro", zero_division=0),
        "F1": f1_score(truth, predicted, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def draw_confusion(truth, predicted, labels: list[str]):
    matrix = confusion_matrix(truth, predicted, labels=labels)
    short = [name.replace("_", "\n") for name in labels]
    figure, axis = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=short,
        yticklabels=short,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.tick_params(axis="both", labelsize=8)
    figure.tight_layout()
    return figure


schema = load_schema()
label_column = schema["label_column"]
feature_columns = schema["feature_columns"]
class_labels = schema["class_labels"]
artifacts = schema["artifacts"]

st.title("Obesity Level Classification")
st.caption(
    "Five classifiers trained on the UCI *Estimation of Obesity Levels* dataset, "
    "using 14 eating-habit and physical-condition features. "
    "Height and Weight are deliberately excluded — the label is derived from BMI, "
    "so keeping them would leak the answer."
)

with st.sidebar:
    st.header("Controls")
    upload = st.file_uploader("Test dataset (CSV)", type="csv")
    chosen_model = st.selectbox("Classification model", list(artifacts))
    st.divider()
    st.markdown(
        "**Expected columns**\n\n"
        + "\n".join(f"- `{column}`" for column in feature_columns)
        + f"\n- `{label_column}` *(optional — enables scoring)*"
    )

if upload is None:
    st.info(
        "Upload `test_data.csv` from the repository (423 held-out rows) to begin. "
        "Any CSV carrying the columns listed in the sidebar will work."
    )
    st.stop()

data = read_upload(upload)
missing = [column for column in feature_columns if column not in data.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

features = data[feature_columns]
has_labels = label_column in data.columns
truth = data[label_column] if has_labels else None

st.success(f"Loaded **{len(data)}** rows and **{len(feature_columns)}** feature columns.")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(15), use_container_width=True)

single_tab, board_tab, predict_tab = st.tabs(
    ["Selected model", "All models", "Predictions"]
)

pipeline = load_pipeline(artifacts[chosen_model])
predicted = pipeline.predict(features)

with single_tab:
    st.subheader(chosen_model)
    if not has_labels:
        st.warning(
            f"No `{label_column}` column found, so metrics cannot be computed. "
            "See the Predictions tab."
        )
    else:
        scores = evaluate(pipeline, features, truth)
        for column, (metric, value) in zip(st.columns(len(scores)), scores.items()):
            column.metric(metric, f"{value:.4f}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(draw_confusion(truth, predicted, class_labels))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                truth, predicted, labels=class_labels, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3), height=430)

with board_tab:
    st.subheader("Comparison across all five models")
    if not has_labels:
        st.warning(f"Add a `{label_column}` column to the CSV to build the leaderboard.")
    else:
        leaderboard = pd.DataFrame(
            [
                {"ML Model Name": name, **evaluate(load_pipeline(file), features, truth)}
                for name, file in artifacts.items()
            ]
        ).set_index("ML Model Name")
        st.dataframe(
            leaderboard.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
            use_container_width=True,
        )
        st.bar_chart(leaderboard[["Accuracy", "F1", "MCC"]])
        best = leaderboard["MCC"].idxmax()
        st.success(f"Best model on this upload by MCC: **{best}**")

with predict_tab:
    st.subheader("Row-level predictions")
    output = data.copy()
    output["Predicted"] = predicted
    if has_labels:
        output["Correct"] = output[label_column] == output["Predicted"]
        st.caption(f"{int(output['Correct'].sum())} of {len(output)} rows predicted correctly.")
    st.dataframe(output, use_container_width=True, height=420)
    st.download_button(
        "Download predictions as CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen_model.split()[0].lower()}.csv",
        mime="text/csv",
    )
