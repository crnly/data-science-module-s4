# Data Science — Semester 4

A portfolio adaptation of the summer 2026 semester project at HdM Stuttgart. The task is to
classify whether a customer uses the latest model of a sports/triathlon watch.

- [Notebook](data_science.ipynb): the full analysis in six CRISP-DM chapters, with short
  Markdown explanations and all results.
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

Clone the repository and create an environment; verified with Python 3.14.4:

```bash
git clone https://github.com/crnly/data-science-module-s4.git
cd data-science-module-s4
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open `data_science.ipynb` in Jupyter or VS Code, select the Python interpreter from `.venv`
as the kernel and use “Restart Kernel and Run All”. The working directory must be the
repository root. If you need a notebook editor, run `python -m pip install jupyterlab`, then
`jupyter lab`. Both original CSV files are included unchanged in `data/`.

## Scope

The analysis classifies observed model usage using the available data. All code and results
are contained in the notebook; the process overview explains the methodological decisions.
A license has not yet been selected.
