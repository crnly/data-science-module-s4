# Data Science — Semester 4

A portfolio adaptation of the summer 2026 semester project at HdM Stuttgart. The task is to
classify whether a customer uses the latest model of a sports/triathlon watch.

- [Notebook](data_science.ipynb): the full analysis in six CRISP-DM chapters, with short
  Markdown explanations and all results.
- [Gradio demo](app.py): try the fixed model with three original example customers and editable inputs.
- [CRISP-DM process overview](CRISP-DM-Process-Overview.md): methodological decisions,
  process diagram and technical data flow.

## Approach

Inspect the original data → split training and validation by ID → compare an initial logistic
baseline and four preparation variants → evaluate three models, feature selection and limited
tuning → assess the fixed model on the provided test file.

## Results

The selected configuration is **logistic regression with feature selection** and **Cleaned**
preparation. This choice applies to the fixed validation split. An additional sensitivity
check shows how the ranking changes across other splits.

| Provided course test · 500 rows | Value |
|---|---:|
| Accuracy | 0.790 |
| Macro-F1 | 0.780 |

465 identifiers occur in both original files; their meaning across files is unresolved.
These results therefore do not establish independent performance or predict future purchases.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/crnly/data-science-module-s4.git
cd data-science-module-s4
uv sync --locked
uv run python app.py
```

Open the local URL printed in the terminal, usually `http://127.0.0.1:7860`.
Select an example customer, edit any values and click **Predict watch usage**.
The app loads the included `model.pkl`; it does not retrain the model. Input data is not saved.
The demo is local; GitHub hosts the code, not a running Python service.

### Reproduce the analysis

Open `data_science.ipynb` in VS Code or Jupyter, select the Python interpreter from `.venv`
as the kernel and run all cells in order. Use the repository root as the working directory.
If you need a notebook editor, run `uv run --with jupyterlab jupyter lab`.

The notebook regenerates `model.pkl`, including feature preparation and three original
example rows without their target labels. Before export, all 500 predictions are checked
against the evaluated model. Both original CSV files remain unchanged in `data/`.

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`. Verified with
Python 3.14.4; use Python 3.14 and the locked environment when loading the saved model.

## Scope

The analysis classifies observed model usage using the available data. All analysis and training
code and results are contained in the notebook. The app provides a small prediction interface;
the process overview explains the methodological decisions.
A license has not yet been selected.
