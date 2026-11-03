# CRISP-DM Process Overview

This project adapts the semester **Task 3: Modeling & Validation** using three supplied,
prepared CSV files. The [notebook](data_science.ipynb) contains the executable analysis;
this document explains the purpose, decisions and handover between its six CRISP-DM phases.
Gradio adds an interactive view of the experiment results.

![CRISP-DM process](docs/process.svg)

[D2 source](docs/process.d2)

Read each phase as **goal → steps → reason → output**. The diagrams show sequence and data
flow. Dashed return arrows represent possible methodological reviews; they are not automatic
loops in the code. The notebook executes from top to bottom.

## 1. Business Understanding

**Goal.** Distinguish customers using the latest sports/triathlon watch model from those using
an older model. `User of latest model` is the target: **0 = older, 1 = latest**. It describes
observed usage; the data does not establish future purchase intent or causal effects.

**Steps.** Define the target and selection criteria before comparing candidates. Use
validation **weighted F1**, then accuracy and the smaller absolute train–validation F1 gap
to break ties. A false positive labels an older-model user as a latest-model user; a false
negative misses a latest-model user. No unsupported business cost is assigned to either error.

**Reason.** Weighted F1 preserves the semester workflow's metric and weights class-specific
F1 by class frequency. Accuracy adds the proportion classified correctly. Per-class results
later reveal errors hidden by averages. No classifier or target score is prescribed.

**Output → Phase 2.** A defined classification task and an explicit comparison rule. Next,
check what information the supplied files actually contain.

## 2. Data Understanding

**Goal.** Understand the prepared inputs, their limitations and the distinct roles of the
three files before fitting models.

**Steps.** Inspect dimensions, a training preview, descriptive statistics and development
class counts. Read the common schema: **47 features plus the target**, comprising eleven
transformed numeric features (`num__`) and 36 one-hot category columns (`cat__`).

| Supplied file | Rows | Role in this project |
|---|---:|---|
| `train_task3.csv` | 585 | Learn candidates and tune parameters with internal CV |
| `validation_task3.csv` | 196 | Compare candidates after fitting |
| `final_test_task3.csv` | 500 | Evaluate the configuration after selection |

**Reason.** These files are already prepared Task-3 inputs. A negative transformed age value,
for example, is a value on a transformed scale; it is not evidence of a negative age in years.
Repeating raw-data cleaning on these numbers would misinterpret them.

**Output → Phase 3.** A known feature schema and a fixed outer split. Next, validate the
inputs and prepare their use without replacing the original semester data.

## 3. Data Preparation

**Goal.** Turn the supplied tables into checked model inputs while retaining their existing
rows and split.

**Steps.** Separate the target from the 47 features. Check matching columns, finite numeric
values, binary labels and identical feature rows. All three CSV files remain unchanged;
there is no new outer split, imputation, scaling or encoding.

Training and test each contain one duplicate feature row; validation contains none. There
are no identical feature rows across files. Assign identical training feature rows the same
group signature for internal CV. Retain both copies in the original data and in final fitting.

**Reason.** Grouping prevents a tuning fold from learning from one copy while evaluating the
other. It preserves the supplied files. A feature signature only detects identical inputs;
it does not identify different observations of the same person. Person identifiers are absent.

**Inherited preparation.** A local comparison links the training and validation features to
the archived Task-2 exports, up to CSV rounding. The finite test features also match that
export; two missing age values in the Task-2 export are filled in the supplied Task-3 file.
The archived Task-2 notebook learns some imputation steps, including a date median,
regression-based imputation and category modes, before its outer split. Its scaler and encoder
are fitted afterward on training. The exact historical execution and all export steps are
not fully documented in the supplied files.

This is a concrete upstream leakage concern. The present CV cannot undo preparation already
learned before its folds were created. Evaluation is conditional on these provided features;
we do not claim an independently validated pipeline starting from raw data.

**Output → Phase 4.** Checked `X` and `y` tables for the three supplied splits, plus training
group signatures. Next, compare models using a consistent evaluation procedure.

## 4. Modeling

**Goal.** Compare the semester model families and determine which configuration to carry
forward under the stated selection rule.

**Steps.** Use decision tree, random forest, gradient boosting, bagging, logistic regression,
Gaussian Naive Bayes, kNN and MLP. A majority classifier supplies a simple reference. The
eight families each pass through three stages, yielding **24 candidates for validation comparison**:

1. **Baseline:** fit on all 47 prepared features and evaluate on the supplied validation file.
2. **Feature selection:** rank features with a training-only random forest; keep the smallest
   leading set reaching 80% cumulative importance. Fit each family again using this selection.
3. **Tuning:** search compact parameter grids for each family with the feature selector
   inside the pipeline. Choose each family's parameters by mean CV weighted F1, then evaluate
   that fitted configuration on the supplied validation file.

**Reason.** The stages make the effect of the original experiment steps visible. The 80%
threshold follows the semester approach; it is not claimed to be an optimal universal value.
Small grids keep the analysis practical. A tuned candidate can lose to its baseline, so all
three stages remain eligible for final selection.

The two uses of validation have different jobs:

| Comparison | Data used | Decision |
|---|---|---|
| Internal five-fold CV | Only the 585 training rows | Hyperparameters within a family |
| Supplied validation file | The same 196 rows for all candidates | Final family and stage |

In five-fold cross-validation, each parameter setting is fitted on four parts and scored on
the held-out fifth part. Repeat this for all five parts and compare the mean scores. These
held-out parts belong to the training file; they are separate from the supplied validation file.

`StratifiedGroupKFold` creates five folds with seed 42, approximately preserving class
proportions while keeping identical feature groups together. Each search uses the same
folds. The notebook checks that no feature group crosses a fold boundary. The selector and
classifier are relearned inside each fold. This protects the newly fitted stages; inherited
preparation remains subject to the limitations in Phase 3.

MLP has an iteration limit of 2,000. Convergence-warning counts are displayed for candidate
fits and searches, so an exhausted optimization budget is visible. Other warnings remain visible.
Reusing the supplied validation file for 24 comparisons makes its scores development results;
small ranking differences do not establish general superiority.

**Output → Phase 5.** One chosen family, stage and parameter configuration. The test score
does not participate in that choice. Next, evaluate the fixed configuration after refitting.

## 5. Evaluation

**Goal.** Assess the selected configuration on the supplied final test and explain its errors.

**Steps.** Refit the selected pipeline on all **781 training-plus-validation rows**. If it
includes a selector, relearn that selector too. Its chosen features can change with the larger
fitting set. Apply the refitted pipeline to all **500 final-test rows**.

Report accuracy, weighted F1, the per-class classification report and a confusion matrix.
Read its rows as actual classes and columns as predictions: the off-diagonal entries show
false positives and false negatives. The notebook also counts missed latest-model users.

**Reason.** Once configuration selection is complete, validation rows can contribute to the
final fit. The comparison table distinguishes the earlier validation score from the final
test score after this refit. The test is used for assessment in this run, not another round
of parameter selection. It has been evaluated in earlier project versions, so it is not a
fresh untouched benchmark. Missing person identifiers, unclear feature timing and inherited
preparation limit conclusions about new customers.

**Output → Phase 6.** A fitted model, measured errors and stated limitations. The
[notebook's evaluation](data_science.ipynb#phase-5) records the current run. Next, export the
recorded results for the interactive explorer.

## 6. Deployment

**Goal.** Make the modeling process understandable through a small interactive results
explorer. Deployment here means presenting the analysis and its findings.

**Steps.** Export `results.json` with the 24 comparison records, ordered model families and
stages, convergence-warning counts, the selected model's test summary, environment versions
and input-file hashes. Reload the JSON and check that its comparison records and final scores
match the notebook. Verify that the three CSV checksums are unchanged.

The Gradio app offers one model-family selector. For that family, a chart shows training and
validation weighted F1 across baseline, feature selection and tuning. A short reading guide
reports the validation-score changes and largest training–validation gap. A larger positive
gap can indicate overfitting; small validation differences do not establish superiority.
Convergence diagnostics remain visible for model families with warnings during tuning.

The optional **How the three stages work** explanation connects the display to Phase 4.
The explorer uses only saved training and validation comparisons. Its model selector is a
display filter: it does not change the final configuration selected in the notebook, retrain
models or evaluate additional families on the final test.

**Reason.** The experiment stages are central to the original semester task. Exploring their
recorded results is more useful for understanding that task than manually editing transformed
customer features. A small, readable JSON file is sufficient; the app needs no serialized
model or prediction-input processing. The notebook remains the single place for fitting and
evaluation. `app.py` filters, plots and explains the saved comparisons.

**Output.** A locally runnable model-comparison explorer and a reproducible notebook.
`pyproject.toml` and `uv.lock` define the environment. If the analysis changes later, rerun the
notebook and inspect the regenerated `results.json`. Restart the app to load the new results;
restarting alone does not run the analysis.

![Technical data flow](docs/architecture.svg)

[D2 source](docs/architecture.d2)
