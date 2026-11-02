# Data Science — Semester 4

A compact portfolio adaptation of the semester project's **Task 3: Modeling & Validation**.
Classify whether a customer uses the latest sports/triathlon watch model using the original
three prepared CSV files and their existing training, validation and test split.

- [Notebook](data_science.ipynb): all analysis and code in six short CRISP-DM phases.
- [Process overview](CRISP-DM-Process-Overview.md): decisions, limitations and D2 diagrams.
- [Gradio explorer](app.py): compare baseline, feature selection and tuning for each model family.

## Approach

Check the supplied data → compare the original eight model families → apply random-forest
feature selection → run compact hyperparameter searches → select using validation weighted F1
→ refit on training plus validation → evaluate on the final test file.
Internal cross-validation keeps identical training observations in the same group.

Both baseline and optimized candidates remain eligible. The goal is a clear, reproducible
semester analysis; no particular score or model is required.

## Data and results

| File in `data/` | Rows | Purpose |
|---|---:|---|
| `train_task3.csv` | 585 | Training and internal cross-validation |
| `validation_task3.csv` | 196 | Candidate comparison and selection |
| `final_test_task3.csv` | 500 | Evaluation after model selection |

All three CSV files are unchanged copies of the semester files. Each contains 47 prepared
features and the target `User of latest model`. The supplied outer split is retained; no raw-data preparation is added.

Selected in the current run: **kNN / Tuned**.

| Final test · 500 rows | Value |
|---|---:|
| Accuracy | 0.738 |
| Weighted F1 | 0.731 |

These are course-data results. Archived Task-2 preparation includes imputation before splitting;
its full historical execution cannot be reconstructed from the exports alone.
Person identifiers are absent, and the test file has been evaluated in earlier project versions.
The results do not establish independent performance on new customers or forecast purchases.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/crnly/data-science-module-s4.git
cd data-science-module-s4
uv sync --locked
uv run python app.py
```

Open the local URL printed in the terminal, usually `http://127.0.0.1:7860`.
Choose a **model family** to compare its training and validation weighted F1 across
baseline, feature selection and tuning. The chart and short reading guide make score changes
and possible overfitting visible. Expand **How the three stages work** for the method.

The app reads `results.json`, exported by the notebook. It does not train models or make
customer predictions. Changing the display does not change the selected final model.
GitHub hosts the source; the explorer runs locally.

To reproduce the analysis, open `data_science.ipynb` in VS Code or Jupyter, select `.venv`
as the Python kernel, use the repository root as the working directory and run all cells.
The notebook also regenerates `results.json`. For a notebook editor:
`uv run --with jupyterlab jupyter lab`.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Verified with Python
3.14.4; use the locked Python 3.14 environment to reproduce the analysis.
A license has not yet been selected.
