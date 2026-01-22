# Bankruptcy Prediction (Imbalanced Classification)

This project predicts corporate bankruptcy (`Bankrupt?`) using financial ratios. The analysis focuses on handling severe class imbalance, tuning multiple models, and selecting an optimal decision threshold. The full workflow and outputs live in `Bankruptcy_Prediction.ipynb`.

## Highlights

- Dataset size: 5,455 companies with 97 columns (including the target).
- Class imbalance: 170 bankrupt vs. 5,285 non-bankrupt (≈96.88% non-bankrupt).
- Imbalance handling: `RandomOverSampler` to balance training data.
- Models compared: Logistic Regression, Random Forest, XGBoost.
- Evaluation: 5-fold cross validation, threshold optimization on F1, ROC AUC, Accuracy, Confusion Matrix.

## Model Comparison (Notebook Results)

| Model | Best Threshold | F1 | ROC AUC | Accuracy |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.202 | 0.420 | 0.903 | 0.961 |
| Random Forest | 0.192 | 0.460 | 0.937 | 0.955 |
| XGBoost | 0.172 | **0.468** | 0.932 | 0.957 |

**Notes**
- The notebook uses Random Forest for the final test-set predictions.
- If you prioritize F1, XGBoost is slightly better; if you prioritize ROC AUC, Random Forest is marginally stronger.

## Figures from the Research Notebook

### Exploratory Analysis

![Spearman correlation heatmap](assets/notebook-figure-2.png)

![Feature correlation coefficients](assets/notebook-figure-3.png)

### Model Evaluation

![ROC curve comparison](assets/notebook-figure-10.png)

![Confusion matrices](assets/notebook-figure-14.png)

![F1 score comparison](assets/notebook-figure-11.png)

## Workflow Summary

1. Load and inspect training data (features + `Bankrupt?`).
2. Exploratory analysis (distributions, Spearman correlation heatmap).
3. Address class imbalance via random oversampling.
4. Train and tune:
   - Logistic Regression (`C=1`, `penalty=l1`, `solver=liblinear`).
   - Random Forest (`n_estimators=100`, `max_depth=10`, `min_samples_split=5`).
   - XGBoost (`n_estimators=200`, `learning_rate=0.01`, `max_depth=3`).
5. Evaluate with CV probabilities, select best F1 threshold, and compare metrics.
6. Generate predictions for the test set and export to CSV.

## Data Inputs and Outputs

- **Training data**: `train.csv` (must include `Bankrupt?`).
- **Test data**: `test.csv` (no `Bankrupt?`).
- **Output**: `rf_bankruptcy_predictions.csv` (ID + prediction).

> The notebook currently references `/content/train (1).csv` and `/content/test (2).csv`. Update these paths for your environment.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib
   ```
2. Open `Bankruptcy_Prediction.ipynb`.
3. Update file paths and run all cells.

## Project Files

- `Bankruptcy_Prediction.ipynb`: complete analysis, training, and evaluation.
- `README.md`: project overview and summary results.
- `assets/`: images exported from the notebook outputs.
