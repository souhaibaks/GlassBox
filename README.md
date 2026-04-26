# GlassBox-AutoML

A simple **white-box Automated Machine Learning library** in Python,
using only NumPy at runtime. The goal is to keep everything easy to read
and understand.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [How to Run — Step by Step](#how-to-run--step-by-step)
   - [Step 1 — End-to-end AutoFit demo](#step-1--test-end-to-end-autofit-demo)
   - [Step 2 — IronClaw agent scenario](#step-2--ironclaw-agent-scenario-build-a-model-to-predict-target)
   - [Step 3 — sklearn benchmark](#step-3--sklearn-benchmark)
   - [Step 4 — Full test suite](#step-4--full-test-suite)
5. [API Reference](#api-reference)
6. [Project Structure](#project-structure)
7. [Benchmark vs scikit-learn](#benchmark-vs-scikit-learn)
8. [IronClaw MCP Server](#ironclaw-mcp-server)
9. [Class Diagram](#class-diagram-architecture)
10. [Authors](#authors)

---

## Features

- **Light dependencies** — only NumPy needed at runtime
- **EDA Inspector** — basic stats, correlation, outlier detection, column type detection
- **Preprocessing Pipeline** — imputation, scaling, encoding
- **9 ML algorithms** implemented from scratch:
  - `DecisionTreeClassifier` / `DecisionTreeRegressor` (Gini / Entropy criterion)
  - `RandomForestClassifier` / `RandomForestRegressor` (bagging + feature subspace)
  - `LogisticRegression` (multi-class softmax + SGD)
  - `LinearRegression` (OLS closed-form) and `GDLinearRegression` (gradient descent)
  - `NBClassifier` (Gaussian Naive Bayes with Laplace smoothing)
  - `KNNClassifier` / `KNNRegressor` (Euclidean or Manhattan distance)
- **Model Selection** — `GridSearch` and `RandomSearch` with K-Fold cross-validation
- **Utilities** — `KFold`, `cross_val_score`, `train_test_split` (with stratification)
- **AutoFit** — EDA -> Impute -> Scale -> GridSearch -> JSON report with feature importance
- **Mixed CSV AutoFit** — numeric, boolean, and text categorical columns are inferred and encoded
- **Reusable model artifacts** — `autofit_tool` returns a JSON artifact that `predict_tool` can use later
- **IronClaw MCP Server** — 4 tools (`eda_tool`, `autofit_tool`, `explain_tool`, `predict_tool`)
  over JSON-RPC 2.0 for NEAR AI agent integration

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.9 | Tested on 3.11 and 3.12 |
| NumPy | >= 1.21 | Only **required** runtime dependency |
| scikit-learn | >= 1.2 | Optional — only needed for `compare_sklearn.py` and some test files |

Check your Python version:

```bash
python --version
```

---

## Installation

### Option A — Run directly from source (recommended for development)

```bash
git clone https://github.com/..
cd GlassBox
pip install numpy          # only required dependency
pip install scikit-learn   # optional, for benchmark + some tests
```

No `pip install -e .` needed — all scripts use `sys.path.insert(0, '.')` automatically.

### Option B — Install as a package

```bash
cd GlassBox
pip install -e .
```

### Option C — From PyPI

```bash
pip install glassbox-automl
```

---

## How to Run — Step by Step

> All commands must be run from the **root of the repository** (`GlassBox/`).

### Step 1 — Run an end-to-end AutoFit demo

Runs the full pipeline (EDA → preprocessing → model search) on sample data:

```bash
python examples/run_autofit_demo.py
```

Expected output:

```
============================================================
GlassBox-AutoML - End-to-End Demo
============================================================
Dataset: 150 samples, 4 features, 3 classes

Best Model   : NaiveBayes
Best Params  : {'var_smoothing': 1e-05}
Best CV Score: 0.9800
Elapsed      : 6.1s

All Candidate Results:
  DecisionTree          : cv=0.9533  params={...}
  RandomForest          : cv=0.9600  params={...}
  LogisticRegression    : cv=0.9067  params={...}
  NaiveBayes            : cv=0.9800  params={...}
  KNN                   : cv=0.9733  params={...}

Evaluation Report (full training set):
  accuracy: 0.98
  macro avg: {'precision': 0.9811, 'recall': 0.98, 'f1-score': 0.98, ...}
```

---

### Step 2 — IronClaw agent scenario: Build a model to predict 'Target'

This is the main use-case from the project spec. The agent receives a natural-language
request, calls the GlassBox tools, and explains the result.

#### 2a — Run the Python scenario script

Save this as `run_agent_scenario.py` or run it inline:

```bash
python -c "
import sys, json
sys.path.insert(0, '.')
from ironclaw.tools import autofit_tool, explain_tool

# ---- The CSV the user provides ----
csv_data = '''Age,Income,Education,Target
25,30000,2,0
45,80000,4,1
30,45000,3,0
55,120000,5,1
22,25000,1,0
60,150000,5,1
35,60000,3,1
28,35000,2,0
48,95000,4,1
32,52000,3,0
50,110000,4,1
27,31000,2,0
40,75000,4,1
33,48000,3,0
58,140000,5,1
'''

# ---- Step 1: Agent triggers autofit_tool ----
print('Agent: calling autofit_tool...')
result = autofit_tool(csv_data, target_col='Target',
                      task_type='classification', cv=3)

report = result['report']
print(f'Best model  : {report[\"best_model\"]}')
print(f'CV score    : {report[\"best_cv_score\"]}')
print(f'Feature imp : {report[\"feature_importance\"]}')

# ---- Step 2: Agent calls explain_tool ----
print()
print('Agent: generating explanation...')
explanation = explain_tool(report)
print()
print('=== Agent response to user ===')
print(explanation['summary'])
print()
print('Decision factors:')
for f in explanation['decision_factors']:
    print(f'  {f[\"rank\"]}. {f[\"feature\"]:12} {f[\"importance_pct\"]:5.1f}%  ({f[\"method\"]})')
"
```

Expected output:

```
Agent: calling autofit_tool...
Best model  : DecisionTree
CV score    : 0.9333
Feature imp : [{'feature_index': 0, 'importance': 1.0, 'method': 'split_frequency', ...}]

Agent: generating explanation...

=== Agent response to user ===
GlassBox AutoFit analysed 15 samples with 3 features for a classification task.
After 5-fold cross-validation over 5 candidate model(s), the best model was
DecisionTree (CV score: 93.33%, params: max_depth=3, criterion=entropy).
The runner-up was RandomForest (93.33%).
On the full training set the model achieved accuracy = 100.00%.
The most influential feature was 'Age' (100.0% of split usage in decision tree nodes).
Top features by importance:
  1. 'Age' (100.0%, split usage in decision tree nodes)
  2. 'Income' (0.0%, split usage in decision tree nodes)
  3. 'Education' (0.0%, split usage in decision tree nodes)

Decision factors:
  1. Age          100.0%  (split_frequency)
  2. Income         0.0%  (split_frequency)
  3. Education      0.0%  (split_frequency)
```

#### 2b — Run via the MCP server (JSON-RPC, exactly as an IronClaw agent would)

**1 — List available tools:**

```bash
echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}" | python ironclaw/mcp_server.py
```

You should see 4 tools: `eda_tool`, `autofit_tool`, `explain_tool`, `predict_tool`.

**2 — Run AutoML on a CSV:**

```bash
echo "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"autofit_tool\",\"arguments\":{\"csv_data\":\"Age,Income,Target\n25,30000,0\n45,80000,1\n30,45000,0\n55,120000,1\n22,25000,0\n60,150000,1\",\"target_col\":\"Target\",\"task_type\":\"classification\",\"cv\":3}}}" | python ironclaw/mcp_server.py
```

**3 — Run EDA only:**

```bash
echo "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"eda_tool\",\"arguments\":{\"csv_data\":\"sepal_length,sepal_width,petal_length\n5.1,3.5,1.4\n4.9,3.0,1.4\n6.7,3.1,4.4\n5.8,2.7,5.1\"}}}" | python ironclaw/mcp_server.py
```

**4 — Predict with a specific model:**

```bash
echo "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"predict_tool\",\"arguments\":{\"model_name\":\"DecisionTreeClassifier\",\"model_params\":{\"max_depth\":3},\"X_train\":[[25,30000],[45,80000],[30,45000],[55,120000]],\"y_train\":[0,1,0,1],\"X_data\":[[28,35000],[52,110000]]}}}" | python ironclaw/mcp_server.py
```

---

### Step 3 — sklearn benchmark

Compares every GlassBox model against its scikit-learn equivalent on Iris and
California Housing datasets:

```bash
python examples/compare_sklearn.py
```

Expected output:

```
=================================================================
  GlassBox vs scikit-learn - Benchmark
=================================================================

-----------------------------------------------------------------
  CLASSIFICATION  Iris Dataset (150 samples, 4 features, 3 classes)
-----------------------------------------------------------------
  Model                           GlassBox CV  sklearn CV      Diff
  ----------------------------  -------------- ----------  --------
  DecisionTree (depth=5)       GB=0.9400  SK=0.9533  (-0.0133)
  RandomForest (100 trees)     GB=0.9400  SK=0.9667  (-0.0267)
  Logistic Regression          GB=0.9733  SK=0.9600  (+0.0133)
  Naive Bayes                  GB=0.9533  SK=0.9533  (+0.0000)
  KNN (k=5)                    GB=0.9533  SK=0.9600  (-0.0067)

-----------------------------------------------------------------
  REGRESSION  California Housing (500 samples, 8 features)
-----------------------------------------------------------------
  ...
```

> Requires `pip install scikit-learn`.

---

### Step 4 — Full test suite

Run each test module individually:

```bash
# Core algorithms
python test_dt.py        # DecisionTree (classifier + regressor)
python test_dtr.py       # DecisionTreeRegressor vs sklearn
python test_rf.py        # RandomForest
python test_nb.py        # Naive Bayes
python test_knn.py       # KNN
python test_lg.py        # Logistic Regression
python test_lgr.py       # Logistic Regression (gradient variant)
python test_cv.py        # Cross-validation

# Metrics
python test_metrics.py   # Accuracy, F1, Precision, Recall, R2, Confusion Matrix

# EDA
python test_eda.py       # EDA Inspector

# Pipeline + AutoFit
python test_pipeline.py  # Pipeline (regression + classification)
python test_autofit.py   # AutoFit end-to-end (classification + regression + timing)

# IronClaw MCP
python test_ironclaw.py  # All 4 MCP tools + server JSON-RPC roundtrip
```

Or run them all at once:

```bash
python -c "
import importlib, sys
sys.path.insert(0, '.')
tests = ['test_dt','test_dtr','test_nb','test_knn','test_rf',
         'test_metrics','test_eda','test_pipeline','test_autofit','test_ironclaw']
for t in tests:
    m = importlib.import_module(t)
    for name in sorted(dir(m)):
        if name.startswith('test_'):
            getattr(m, name)()
    print(f'PASS  {t}')
print('All tests passed.')
"
```

Expected output:

```
Your DecisionTreeClassifier Score (Digits): 0.858
...
PASS  test_dt
...
PASS  test_dtr
PASS  test_nb
PASS  test_knn
PASS  test_rf
PASS  test_metrics
PASS  test_eda
[OK] Regression pipeline passed: R2=0.9975
[OK] Classification pipeline passed: acc=1.0000
PASS  test_pipeline
[OK] AutoFit classification: best=LogisticRegression, cv_score=0.9801
[OK] AutoFit regression: best=LinearRegression, R2=0.9925
[OK] AutoFit timing: 4.7s (budget: 120s)
PASS  test_autofit
[OK] eda_tool passed
[OK] autofit_tool passed
[OK] predict_tool passed
[OK] MCP tools/list: ['eda_tool', 'autofit_tool', 'explain_tool', 'predict_tool']
[OK] MCP tools/call eda_tool via server
PASS  test_ironclaw
All tests passed.
```

---

## API Reference

### AutoFit

```python
from autofit import AutoFit

af = AutoFit(task_type='classification', cv=5, random_state=42)
af.fit(X, y)          # numpy arrays
report = af.report()  # JSON-serializable dict
```

`report` keys: `task_type`, `n_samples`, `n_features`, `eda`, `best_model`,
`best_params`, `best_cv_score`, `all_results`, `evaluation`, `feature_importance`,
`elapsed_seconds`.

Use `AutoFit(search_strategy='random', n_iter=10)` to switch tuning from
exhaustive grid search to stochastic random search.

### Pipeline

```python
from pipeline import Pipeline
from transformers import SimpleImputer, StandardScaler
from DecisionTree import DecisionTreeClassifier

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # handles NaN
    ('scaler',  StandardScaler()),
    ('model',   DecisionTreeClassifier(max_depth=5)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
pipe.score(X_test, y_test)
```

### GridSearch / RandomSearch

```python
from model_selection import GridSearch, RandomSearch

gs = GridSearch(model, {'max_depth': [3, 5, 10]}, cv=5)
gs.fit(X, y)
print(gs.best_params_, gs.best_score_)

rs = RandomSearch(model, {'max_depth': [3, 5, 10]}, n_iter=6, cv=5, random_state=0)
rs.fit(X, y)
```

### Metrics

```python
from metrics import Accuracy, F1Score, Precision, Recall
from metrics import MeanSquaredError, MeanAbsoluteError, R2Score, ConfusionMatrix
from metrics.metric import classification_report

# Classification
print(Accuracy().score(y_true, y_pred))
print(F1Score(average='macro').score(y_true, y_pred))
print(classification_report(y_true, y_pred))   # dict with per-class + macro

# Regression
print(R2Score().score(y_true, y_pred))
print(MeanSquaredError().score(y_true, y_pred))
```

### Transformers

```python
from transformers import SimpleImputer, StandardScaler, MinMaxScaler
from transformers import LabelEncoder, OneHotEncoder, OrdinalEncoder
from transformers import EDAInspector

imp = SimpleImputer(strategy='mean')   # or 'median', 'mode'
X_clean = imp.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


eda = EDAInspector()
eda.fit(X)
print(eda.report())   # stats, correlations, outlier bounds, column types
```

### Utilities

```python
from utils import cross_val_score, train_test_split, KFold


scores = cross_val_score(model, X, y, cv=5, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
    ...
```

---

## Project Structure

```
GlassBox/
|
+-- transformers/          # Data preprocessing
|   +-- EDA.py             # EDAInspector
|   +-- Imputers.py        # SimpleImputer, KNNImputer
|   +-- Scalers.py         # MinMaxScaler, StandardScaler
|   +-- Encoders.py        # OrdinalEncoder, OneHotEncoder, LabelEncoder
|   +-- Transfomer.py      # Transformer base class
|
+-- metrics/               # Evaluation metrics
|   +-- metric.py          # Accuracy, Precision, Recall, F1, MSE, MAE, R2,
|                          # ConfusionMatrix, classification_report
|
+-- model_selection/       # Hyperparameter tuning
|   +-- GridSearch.py      # Exhaustive grid search with K-Fold CV
|   +-- RandomSearch.py    # Random search with K-Fold CV
|
+-- models/                # ML models (trees, forests, linear models, NB, KNN)
|   +-- DecisionTree.py
|   +-- RandomForest.py
|   +-- LinearModel.py
|   +-- LogisticRegression.py
|   +-- NBClassifier.py
|   +-- KnnEstimator.py
|   +-- Estimator.py
|   +-- Predictor.py
|
+-- utils/                 # ML utilities
|   +-- cross_validation.py  cross_val_score
|   +-- Kfolds.py            KFold splitter
|   +-- train_test_split.py  Stratified train/test split
|
+-- ironclaw/              # NEAR AI MCP server integration
|   +-- mcp_server.py      # JSON-RPC 2.0 stdio/HTTP MCP server (4 tools)
|   +-- tools.py           # eda_tool, autofit_tool, explain_tool, predict_tool
|   +-- chat_sim.py        # Local chat-style scenario runner
|
+-- autofit.py             # AutoFit orchestrator (numeric matrices)
+-- pipeline.py            # Pipeline for transformers + final estimator
+-- examples/              # Organized runnable demos
+-- tests/                 # Test suite
+-- docs/                  # Documentation (currently deployment notes only)
```

The preferred imports are now the organized package paths:

```python
from autofit import AutoFit
from RandomForest import RandomForestClassifier
from transformers import SimpleImputer, StandardScaler
from ironclaw.tools import autofit_tool, explain_tool
```

Note: this repository currently exposes its APIs via the root modules
(`autofit.py`, `pipeline.py`) and the folders shown above. Some older docs and
paths may refer to a `glassbox/` package layout which is not present here.

---

## Benchmark vs scikit-learn

Results on standard datasets (5-fold cross-validation, standardized features):

### Classification — Iris (150 samples, 4 features, 3 classes)

| Model                  | GlassBox Accuracy | sklearn Accuracy | Difference |
|------------------------|:-----------------:|:----------------:|:----------:|
| DecisionTree (depth=5) | 0.9400            | 0.9533           | -0.0133    |
| RandomForest (100)     | 0.9400            | 0.9667           | -0.0267    |
| Logistic Regression    | 0.9733            | 0.9600           | **+0.0133**|
| Naive Bayes            | 0.9533            | 0.9533           | 0.0000     |
| KNN (k=5)              | 0.9533            | 0.9600           | -0.0067    |

### Regression — California Housing (500 samples, 8 features)

| Model                  | GlassBox R2 | sklearn R2 | Difference |
|------------------------|:-----------:|:----------:|:----------:|
| Decision Tree (depth=5)| 0.3400      | 0.2991     | **+0.0409**|
| Random Forest (50)     | 0.4199      | 0.5565     | -0.1366    |
| Linear Regression      | 0.4960      | 0.5206     | -0.0245    |
| KNN Regressor (k=5)    | 0.5735      | 0.5511     | **+0.0224**|

```bash
python examples/compare_sklearn.py
```

---

## IronClaw MCP Server

GlassBox ships with a JSON-RPC 2.0 MCP server for integration with NEAR AI agents.
The server reads newline-delimited JSON from **stdin** and writes responses to **stdout**.

### Available tools

| Tool | Purpose |
|---|---|
| `eda_tool` | Automated EDA report (stats, correlations, outlier bounds, column types) |
| `autofit_tool` | Full AutoML: EDA -> impute -> scale -> GridSearch -> JSON report |
| `explain_tool` | Natural language explanation of an AutoFit report (feature importance, data quality) |
| `predict_tool` | Predict from explicit training data or an AutoFit `model_artifact` |

### Running the server

```bash
python ironclaw/mcp_server.py
```

Then pipe JSON-RPC requests line by line.

For IronClaw's HTTP MCP client:

```bash
python ironclaw/mcp_server.py --http --host 0.0.0.0 --port 8765
ironclaw mcp add glassbox http://127.0.0.1:8765 \
  --description "GlassBox white-box AutoML tools"
ironclaw mcp test glassbox
```

When the user provides a file path instead of pasted CSV text, the agent can
call `autofit_tool` with `csv_path`. That path must exist on the machine running
the MCP server.

`autofit_tool` accepts mixed CSV data. Numeric and boolean features are converted
to numeric values, while text categorical features are one-hot encoded. Its
report includes `model_artifact`, which can be passed back into `predict_tool`
with raw row dictionaries:

```python
fit = autofit_tool(csv_path="titanic.csv", target_col="survived", cv=3)
artifact = fit["report"]["model_artifact"]
pred = predict_tool(
    model_artifact=artifact,
    X_data=[{"pclass": "3", "sex": "male", "age": "24", "fare": "7.25"}],
)
```

Without the hosted IronClaw chat UI, you can run the free chat-style simulator:

```bash
python ironclaw/chat_sim.py "Build a model to predict \"survived\" using titanic.csv"
```

See `ironclaw/DEPLOY.md` for the NEAR AI remote-host workflow.

### Feature importance methods per model

Because every model is built from scratch, the `explain_tool` can trace the exact
reason a prediction was made:

| Model | Importance method |
|---|---|
| DecisionTree / RandomForest | Split-frequency per feature across all tree nodes |
| LinearRegression | Absolute value of OLS coefficients (normalised) |
| LogisticRegression | L2 norm of softmax weight vectors per feature |
| NaiveBayes | Variance of class means per feature (discriminative power) |
| KNN | Not applicable (distance-based, uniform feature usage) |

---

## Class Diagram (Architecture)

```
Estimator (base)
    +-- Predictor (base)
        +-- DecisionTreeClassifier
        +-- DecisionTreeRegressor
        +-- RandomForestClassifier
        +-- RandomForestRegressor
        +-- LinearRegression
        +-- GDLinearRegression
        +-- LogisticRegression
        +-- NBClassifier
        +-- KNNClassifier  (_KNNBase)
        +-- KNNRegressor   (_KNNBase)

Transformer (base)
    +-- Scaler
    |   +-- MinMaxScaler
    |   +-- StandardScaler
    +-- Imputer
    |   +-- SimpleImputer
    |   +-- KNNImputer
    +-- Encoder
    |   +-- OrdinalEncoder
    |   +-- OneHotEncoder
    |   +-- LabelEncoder
    +-- EDAInspector

Pipeline        chains transformers + final estimator
AutoFit         EDA -> Preprocess -> GridSearch -> JSON report
GridSearch      exhaustive K-Fold CV hyperparameter search
RandomSearch    stochastic K-Fold CV hyperparameter search
KFold           K-fold index splitter with optional shuffle
```

---

## Authors

- Aksikas Mohamed Souhaib 
- Benali Amine 
- Boubrik Aymen
- Chahti Moad
