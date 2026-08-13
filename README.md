# Obesity Level Classification from Eating Habits and Physical Condition

Machine Learning Assignment 2 — M.Tech (AIML/DSE), WILP Division, BITS Pilani.

Live app: https://obesity-level-classification-oqharg5xe7g9a2f2yhmwbu.streamlit.app/

Repository: https://github.com/2025ac05294-png/obesity-level-classification

---

## a. Problem statement

Obesity is usually diagnosed after the fact, from a body measurement. The more
useful question for a preventive-health service is whether the **lifestyle** a
person reports — what they eat, how much water they drink, how often they
exercise, how they commute, whether obesity runs in the family — is on its own
enough to place them in the right weight category.

This project frames that as a **7-class supervised classification** problem.
Given 14 self-reported demographic and habit features, predict which of the
seven obesity levels a respondent belongs to:

`Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`,
`Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`.

Five classification algorithms are trained on an identical preprocessing
pipeline and identical train/test split, compared on six metrics, and served
through an interactive Streamlit application.

### A deliberate design decision: removing Height and Weight

The raw dataset ships with `Height` and `Weight` columns. The `NObeyesdad`
target label was itself assigned by the dataset authors from
**BMI = Weight / Height²**, so those two columns are not features — they *are*
the answer, arithmetically restated. Training on them is textbook **target
leakage**: every model lands in the 96–99% band, the comparison table flattens
out, and nothing is learned about the algorithms or the problem.

Both columns are therefore dropped in `model/train_models.py`. The remaining
**14 features** still satisfy the assignment's ≥12-feature requirement, the task
becomes genuinely hard, and the models separate cleanly — which is what makes
the comparison below worth reading.

## b. Dataset description

| Property | Value |
| --- | --- |
| Name | Estimation of Obesity Levels Based On Eating Habits and Physical Condition |
| Source | UCI Machine Learning Repository, dataset ID 544 |
| URL | https://archive.ics.uci.edu/dataset/544/ |
| Instances | 2,111 (requirement: ≥ 500) |
| Columns in raw file | 17 (16 features + 1 target) |
| Features used | 14, after dropping the BMI-derived `Height` and `Weight` (requirement: ≥ 12) |
| Target | `NObeyesdad`, 7 classes |
| Missing values | None |
| Origin | Survey responses collected in Colombia, Peru and Mexico; ~77% of records were synthetically balanced by the authors using SMOTE, ~23% are original responses |

Class distribution is close to balanced (272–351 records per class), so plain
accuracy is not misleading here — but macro-averaged metrics and MCC are still
reported so that no single class can dominate the score.

### Feature dictionary

| # | Feature | Type | Meaning |
| --- | --- | --- | --- |
| 1 | `Gender` | categorical | Male / Female |
| 2 | `Age` | numeric | Age in years |
| 3 | `family_history_with_overweight` | categorical | Family history of overweight (yes/no) |
| 4 | `FAVC` | categorical | Frequent consumption of high-calorie food |
| 5 | `FCVC` | numeric | Frequency of vegetable consumption (1–3) |
| 6 | `NCP` | numeric | Number of main meals per day |
| 7 | `CAEC` | categorical | Eating between meals (no / Sometimes / Frequently / Always) |
| 8 | `SMOKE` | categorical | Smoker (yes/no) |
| 9 | `CH2O` | numeric | Daily water intake (1–3) |
| 10 | `SCC` | categorical | Monitors calorie consumption (yes/no) |
| 11 | `FAF` | numeric | Physical activity frequency per week (0–3) |
| 12 | `TUE` | numeric | Time using technology devices (0–2) |
| 13 | `CALC` | categorical | Alcohol consumption frequency |
| 14 | `MTRANS` | categorical | Primary mode of transport |
| — | `NObeyesdad` | **target** | Obesity level (7 classes) |

### Methodology

- **Split** — stratified 80/20 hold-out, `random_state=7`; 1,688 training rows
  and 423 test rows. The test split is saved verbatim as `test_data.csv` and is
  the file uploaded to the Streamlit app.
- **Preprocessing** — one `ColumnTransformer` shared by all five models:
  `StandardScaler` on the 6 numeric features, `OneHotEncoder`
  (`handle_unknown="ignore"`, dense output) on the 8 categorical features,
  yielding 29 encoded columns (6 numeric + 23 one-hot). Encoder and estimator
  are wrapped in a single
  scikit-learn `Pipeline`, so the scaler is fitted on training data only and
  no test statistics leak backwards.
- **Metrics** — Precision, Recall and F1 are **macro-averaged**; AUC is
  macro-averaged **one-vs-rest** over `predict_proba`; MCC is the multiclass
  Matthews correlation coefficient. Every number below is computed on the
  untouched 423-row test split.

## c. GitHub repository link

https://github.com/2025ac05294-png/obesity-level-classification

```
project-folder/
├── app.py                       Streamlit application
├── requirements.txt             pinned dependencies
├── README.md                    this file
├── test_data.csv                423-row held-out test split
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv    raw UCI download
└── model/
    ├── train_models.py          preprocessing, training, evaluation, export
    ├── metrics.csv              generated comparison table
    └── artifacts/               5 fitted pipelines (.joblib) + schema.json
```

### Reproducing the results

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all 5 models, rewrites test_data.csv
streamlit run app.py             # launches the UI on http://localhost:8501
```

## d. Models used

All five classifiers share the same pipeline, the same split and the same seed;
only the estimator changes.

| Model | Key hyperparameters |
| --- | --- |
| Logistic Regression | `max_iter=3000`, `C=1.0`, multinomial |
| Decision Tree | unrestricted depth (chosen by 5-fold CV, see below) |
| kNN | `n_neighbors=11`, `weights="distance"` |
| Naive Bayes | `GaussianNB`, default smoothing |
| Random Forest | `n_estimators=400`, `min_samples_leaf=2` |

### Comparison table

Evaluated on the 423-row held-out test set. Best value per column in **bold**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6383 | 0.8895 | 0.6385 | 0.6321 | 0.6065 | 0.5845 |
| Decision Tree | 0.7754 | 0.8670 | 0.7730 | 0.7713 | 0.7699 | 0.7384 |
| kNN | 0.7518 | 0.9492 | 0.7713 | 0.7468 | 0.7300 | 0.7151 |
| Naive Bayes | 0.4326 | 0.8359 | 0.4043 | 0.4353 | 0.3568 | 0.3728 |
| **Random Forest (Ensemble)** | **0.8392** | **0.9733** | **0.8436** | **0.8372** | **0.8374** | **0.8132** |

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Clearly underfits at 63.8% accuracy, yet posts a respectable 0.8895 AUC — the ranking of classes is broadly right even when the arg-max decision is wrong. The gap is diagnostic: a linear decision boundary in the 29-dimensional encoded space cannot express rules such as "high vegetable intake helps *only if* physical activity is also high". Its F1 (0.6065) trails its precision and recall, meaning the errors are concentrated in a few classes rather than spread evenly. |
| Decision Tree | Jumps to 77.5% accuracy — the best non-ensemble model — by capturing exactly the feature interactions logistic regression cannot express, and it remains the most interpretable option here. It is also the clearest illustration that accuracy and AUC measure different things: despite beating kNN on accuracy, F1 and MCC, its AUC is the second *worst* (0.8670). A fully grown tree drives every leaf to purity, so `predict_proba` returns near-0/1 values — confident decisions with essentially no usable ranking between them. Depth caps of 12 and 16 and larger leaf sizes were compared by 5-fold CV on the training split; the unrestricted tree won (0.7464 vs 0.7275 for `max_depth=12, min_samples_leaf=4`), so no pruning was applied. |
| kNN | Slightly behind the decision tree on accuracy (75.2%) but far ahead on AUC (0.9492, second best overall) and holding the second-best precision (0.7713) — distance-weighted voting over 11 neighbours produces smooth, well-calibrated probabilities where the tree produces steps. It depends on the `StandardScaler` step: removing it drops test accuracy to 73.5%, as the wide raw `Age` range starts to dominate the Euclidean distance. The cost is inference-time, since all 1,688 training rows must be retained and searched. |
| Naive Bayes | The weakest model by a wide margin (43.3% accuracy, 0.3728 MCC) and a useful negative result. Its conditional-independence assumption is badly violated — the habit features are strongly correlated (frequent high-calorie food, low physical activity and family history co-occur), so the product of likelihoods double-counts the same evidence. It also models 23 one-hot dummy columns as Gaussians, which is the wrong distribution entirely. Per-class inspection shows it collapsing rather than degrading: it never once predicts `Obesity_Type_I` (F1 = 0.00) while over-predicting `Obesity_Type_II` (recall 0.97 at precision 0.39). Its 0.8359 AUC confirms the ranking signal survives; it is the calibration that fails. |
| Random Forest (Ensemble) | Best on every single metric — 83.9% accuracy, 0.9733 AUC, 0.8132 MCC. Averaging 400 decorrelated trees retains the interaction-capturing power of a single tree while cancelling its variance, worth ~6 accuracy points over the lone tree and, more strikingly, +0.106 AUC as the vote proportions restore the probability calibration a single tree destroys. Per-class results are strongest at the extremes (`Obesity_Type_III` F1 = 0.99, `Insufficient_Weight` = 0.91) and weakest across the genuinely ambiguous middle band, `Normal_Weight` vs `Overweight_Level_I` (F1 = 0.72 / 0.74), which is where the confusion matrix concentrates its errors. Feature importances rank `Age` (0.147), `FCVC` vegetable consumption (0.143) and `NCP` number of meals (0.088) highest. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it dominates all six metrics simultaneously, leads the runner-up (Decision Tree) by 0.0748 MCC, and degrades gracefully on the hard adjacent classes rather than collapsing. If interpretability were the priority, the single Decision Tree is the pragmatic fallback at ~6 points less accuracy, since it yields readable if-then rules a clinician could audit. The broader finding is that **lifestyle self-reports alone explain roughly 84% of obesity category** on this data — high enough to be useful for screening, and far more honest than the ~97% a leaky BMI-derived model would advertise. |

---

## Streamlit application

The deployed app implements every required feature:

| Requirement | Where it lives in the UI |
| --- | --- |
| Dataset upload (CSV) | Sidebar file uploader; upload `test_data.csv` |
| Model selection dropdown | Sidebar selectbox listing all five trained models |
| Display of evaluation metrics | **Selected model** tab — all six metrics as headline cards |
| Confusion matrix / classification report | **Selected model** tab — annotated 7×7 heatmap next to the full per-class report |

Two additions beyond the minimum:

- **All models** tab — scores every one of the five pipelines against the
  uploaded file in a single view, highlights the best cell per metric, charts
  Accuracy/F1/MCC side by side and names the winner by MCC. This is what makes
  "the results of different models on your test data" visible at a glance.
- **Predictions** tab — row-level predictions with a correctness flag and a CSV
  download.

The app validates the uploaded schema and fails with a clear message listing any
missing columns. If the `NObeyesdad` column is absent it degrades to
prediction-only mode instead of erroring, so unlabelled data can still be
scored.

## BITS Virtual Lab

The assignment was executed on the BITS Virtual Lab; the required screenshot is
included in the submitted PDF.
