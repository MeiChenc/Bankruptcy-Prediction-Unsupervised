# Repository Guidelines

## Project Structure & Module Organization
- `Bankruptcy_Prediction.ipynb` holds the entire workflow: ingestion, feature engineering, modeling, and reporting. Keep exploratory, modeling, and export sections separated by top-level Markdown headings so collaborators can re-run selected blocks.
- Place training/holdout CSVs (e.g., `train.csv`, `test.csv`) at the repo root; reference them with relative paths to keep notebooks portable. Store generated artifacts such as `rf_bankruptcy_predictions.csv` or interim parquet files inside a dedicated `artifacts/` subfolder (create it if missing) to avoid polluting the root.
- Images intended for the README should live under `assets/` and be referenced with relative Markdown links.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment before installing dependencies.
- `pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib jupyter`: baseline stack for reproducing the notebook.
- `jupyter lab Bankruptcy_Prediction.ipynb` (or `jupyter notebook ...`): open the primary analysis. Run cells sequentially; restart the kernel before re-executing modeling blocks to ensure deterministic input pipelines.
- `python -m black src/` (if you add utility modules) and `nbqa black Bankruptcy_Prediction.ipynb`: keep scripts/notebooks formatted consistently.

## Coding Style & Naming Conventions
- Prefer Python 3.10+, 4-space indentation, and `black` defaults (line length 88). Use snake_case for variables/functions, PascalCase for classes, and ALL_CAPS for constants such as random seeds or column lists.
- When adding helper modules, type-hint public functions and document expected DataFrame schemas inline. Notebook cells performing non-trivial work should start with a brief Markdown explanation.

## Testing Guidelines
- There is no standalone test suite today; validate model work by re-running the notebook end-to-end and comparing summary metrics (F1, ROC AUC, confusion matrices) against the README table.
- For new reusable utilities, add lightweight unit tests under `tests/` (create if absent) using `pytest`, and run them with `pytest -q`.
- Log random seeds (`np.random.seed`, `RandomState`) near resampling/model blocks to make cross-checks reproducible.

## Commit & Pull Request Guidelines
- Recent history favors short, present-tense summaries (`Update README.md`). Follow `<scope>: <imperative action>` when expanding beyond documentation, e.g., `model: tune xgb depth`.
- Bundle data or artifact additions separately from notebook/code logic. Reference related issues or analysis goals in the PR description, include screenshots of new plots, and call out any dependency, data-path, or threshold updates reviewers must replicate.

## Data & Security Tips
- Never commit proprietary CSVs; add them to `.gitignore` and document acquisition steps in the PR instead.
- When working with confidential data, encrypt artifacts at rest and scrub notebooks of absolute paths before publishing.
