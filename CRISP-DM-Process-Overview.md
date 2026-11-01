# CRISP-DM Process Overview

This project adapts the original semester **Task 3: Modeling & Validation** into one compact
[notebook](data_science.ipynb). The original model families and experiment stages remain the
starting point. CRISP-DM provides the structure; Gradio is a small deployment extension.

![CRISP-DM process](docs/process.svg)

[D2 source](docs/process.d2)

## 1. Business Understanding

Classify `User of latest model`: **0 = older watch model, 1 = latest model**. The task concerns
observed usage, not future purchases or causal effects.

Select by validation **weighted F1**, retaining the semester workflow's main F1 definition.
Use accuracy as the first tie-breaker and the smaller absolute train–validation F1 gap as the
second. Weighted F1 weights each class by its frequency; the classification report also shows
each class separately. There is no target score and no model chosen in advance.

## 2. Data Understanding

Use the three original prepared files unchanged:

| File | Rows | Role |
|---|---:|---|
| `train_task3.csv` | 585 | Fit candidates and run internal CV |
| `validation_task3.csv` | 196 | Compare candidates |
| `final_test_task3.csv` | 500 | Evaluate the fixed configuration |

Each file has **47 features plus the target**: eleven transformed numeric features and 36
one-hot columns. Negative transformed values are valid. The notebook displays training
summaries and development class counts; it does not compare test model outcomes at this stage.

## 3. Data Preparation

Preserve the existing split. Separate the target from the feature columns and check identical
schemas, finite numeric values and binary labels. Do not repeat imputation, encoding, scaling
or outlier removal on these already prepared values.

Training and test each contain one duplicate feature row; validation contains none. Retain
them to preserve the supplied data basis. No identical feature rows are found across the three
files. This check cannot establish that different rows represent different people.

The earlier preprocessing parameters and person identifiers are not present in these exports.
Consequently, this repository cannot verify whether earlier transformations were learned only
from training data. Its results are conditional on the supplied preparation.

## 4. Modeling

Compare the semester project's eight model families: decision tree, random forest, gradient
boosting, bagging, logistic regression, Gaussian Naive Bayes, kNN and MLP. A majority classifier
provides an additional reference.

Run baseline models on all 47 features. Then rank features with a training-only random forest
and keep the smallest leading set reaching 80% cumulative importance. Compare all eight
families again on this selection.

Tune all eight families with compact parameter grids and five stratified training folds
(seed 42). Feature selection is part of the pipeline and is relearned within each fold;
validation and test are not used to fit these stages. The full grids and model parameters
are visible in the notebook. The MLP iteration limit is increased to 2,000. Some fits still
reach that limit; convergence-warning counts are reported in the comparison and tuning tables.
These MLP results therefore reflect the chosen optimization budget. Other warning types remain visible.

Select across **baseline, feature-selection and tuned** candidates using the same validation
rows. Tuning does not automatically displace a stronger baseline. Reusing validation for
several comparisons makes these development scores optimistic; small ranking differences
are not evidence of general superiority.

## 5. Evaluation

Refit the selected pipeline on all **781 training-plus-validation rows**. If it contains a
selector, relearn that selector too; the final feature set can therefore differ from the
training-only selection. Evaluate the resulting model on all **500 final-test rows**.

Report accuracy, weighted F1, the per-class classification report and a confusion matrix.
The notebook states the selected candidate and how many latest-model users it misses.
The [README](README.md#data-and-results) summarizes the current run.

This test file has been evaluated in earlier project versions. These are reproducible course
results, not a fresh untouched benchmark. Unverified earlier preprocessing, absent person
identifiers and unclear feature timing limit conclusions about new customers.

## 6. Deployment

Export `model.pkl` with the fitted pipeline, its input schema, environment versions, file
hashes and three original test feature rows without target labels. Reload it and check all
500 predictions against the evaluated model. Verify that all three input files remain unchanged.

The local Gradio app loads this artifact without retraining. It offers an example selector,
a prediction and an optional prepared-feature editor. Numeric entries use the existing
transformed scale; one-hot values remain binary with at most one active category per group.
Edits explore model responses and do not imply causal effects.

The notebook contains all analysis and training code. `app.py` only provides the interface
and input checks. `pyproject.toml` and `uv.lock` define the reproducible environment.

![Technical data flow](docs/architecture.svg)

[D2 source](docs/architecture.d2)
