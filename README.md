 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 658bcb89108b0ddc646e868daf1271ab4390f84d..72d8951202d9329ec75e7f89dac5128dc2a73618 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,62 @@
-# Bankruptcy-Prediction-Unsupervised
\ No newline at end of file
+# Bankruptcy Prediction (Imbalanced Classification)
+
+This project predicts corporate bankruptcy (`Bankrupt?`) using financial ratios. The analysis focuses on handling severe class imbalance, tuning multiple models, and selecting an optimal decision threshold. The full workflow and outputs live in `Bankruptcy_Prediction.ipynb`.
+
+![Workflow overview](assets/workflow.svg)
+
+## Highlights
+
+- Dataset size: 5,455 companies with 97 columns (including the target).
+- Class imbalance: 170 bankrupt vs. 5,285 non-bankrupt (≈96.88% non-bankrupt).
+- Imbalance handling: `RandomOverSampler` to balance training data.
+- Models compared: Logistic Regression, Random Forest, XGBoost.
+- Evaluation: 5-fold cross validation, threshold optimization on F1, ROC AUC, Accuracy, Confusion Matrix.
+
+## Model Comparison (Notebook Results)
+
+| Model | Best Threshold | F1 | ROC AUC | Accuracy |
+| --- | --- | --- | --- | --- |
+| Logistic Regression | 0.202 | 0.420 | 0.903 | 0.961 |
+| Random Forest | 0.192 | 0.460 | 0.937 | 0.955 |
+| XGBoost | 0.172 | **0.468** | 0.932 | 0.957 |
+
+![Model comparison chart](assets/model-comparison.svg)
+
+**Notes**
+- The notebook uses Random Forest for the final test-set predictions.
+- If you prioritize F1, XGBoost is slightly better; if you prioritize ROC AUC, Random Forest is marginally stronger.
+
+## Workflow Summary
+
+1. Load and inspect training data (features + `Bankrupt?`).
+2. Exploratory analysis (distributions, Spearman correlation heatmap).
+3. Address class imbalance via random oversampling.
+4. Train and tune:
+   - Logistic Regression (`C=1`, `penalty=l1`, `solver=liblinear`).
+   - Random Forest (`n_estimators=100`, `max_depth=10`, `min_samples_split=5`).
+   - XGBoost (`n_estimators=200`, `learning_rate=0.01`, `max_depth=3`).
+5. Evaluate with CV probabilities, select best F1 threshold, and compare metrics.
+6. Generate predictions for the test set and export to CSV.
+
+## Data Inputs and Outputs
+
+- **Training data**: `train.csv` (must include `Bankrupt?`).
+- **Test data**: `test.csv` (no `Bankrupt?`).
+- **Output**: `rf_bankruptcy_predictions.csv` (ID + prediction).
+
+> The notebook currently references `/content/train (1).csv` and `/content/test (2).csv`. Update these paths for your environment.
+
+## How to Run
+
+1. Install dependencies:
+   ```bash
+   pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib
+   ```
+2. Open `Bankruptcy_Prediction.ipynb`.
+3. Update file paths and run all cells.
+
+## Project Files
+
+- `Bankruptcy_Prediction.ipynb`: complete analysis, training, and evaluation.
+- `README.md`: project overview and summary results.
+- `assets/`: static diagrams used in this README.
diff --git a/assets/model-comparison.svg b/assets/model-comparison.svg
new file mode 100644
index 0000000000000000000000000000000000000000..94e3d2d73a7e49c40bfe0e2e958ad3d8255d96b6
--- /dev/null
+++ b/assets/model-comparison.svg
@@ -0,0 +1,39 @@
+<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360">
+  <style>
+    .title { font-family: Arial, sans-serif; font-size: 20px; fill: #1b4965; font-weight: 600; }
+    .label { font-family: Arial, sans-serif; font-size: 14px; fill: #1b4965; }
+    .axis { stroke: #1b4965; stroke-width: 2; }
+  </style>
+  <text class="title" x="480" y="30" text-anchor="middle">Model Comparison (Notebook Results)</text>
+
+  <line class="axis" x1="90" y1="300" x2="900" y2="300" />
+  <line class="axis" x1="90" y1="70" x2="90" y2="300" />
+
+  <!-- Logistic Regression -->
+  <rect x="140" y="170" width="60" height="130" fill="#8ecae6" />
+  <rect x="220" y="100" width="60" height="200" fill="#219ebc" />
+  <rect x="300" y="90" width="60" height="210" fill="#ffb703" />
+  <text class="label" x="200" y="330" text-anchor="middle">Logistic Regression</text>
+
+  <!-- Random Forest -->
+  <rect x="420" y="160" width="60" height="140" fill="#8ecae6" />
+  <rect x="500" y="80" width="60" height="220" fill="#219ebc" />
+  <rect x="580" y="100" width="60" height="200" fill="#ffb703" />
+  <text class="label" x="520" y="330" text-anchor="middle">Random Forest</text>
+
+  <!-- XGBoost -->
+  <rect x="700" y="150" width="60" height="150" fill="#8ecae6" />
+  <rect x="780" y="90" width="60" height="210" fill="#219ebc" />
+  <rect x="860" y="95" width="60" height="205" fill="#ffb703" />
+  <text class="label" x="820" y="330" text-anchor="middle">XGBoost</text>
+
+  <text class="label" x="110" y="85">1.0</text>
+  <text class="label" x="110" y="300">0.0</text>
+
+  <rect x="90" y="320" width="14" height="14" fill="#8ecae6" />
+  <text class="label" x="110" y="332">F1</text>
+  <rect x="150" y="320" width="14" height="14" fill="#219ebc" />
+  <text class="label" x="170" y="332">ROC AUC</text>
+  <rect x="240" y="320" width="14" height="14" fill="#ffb703" />
+  <text class="label" x="260" y="332">Accuracy</text>
+</svg>
diff --git a/assets/workflow.svg b/assets/workflow.svg
new file mode 100644
index 0000000000000000000000000000000000000000..bc8405d0974205542de40d29b67841f396c4cdeb
--- /dev/null
+++ b/assets/workflow.svg
@@ -0,0 +1,31 @@
+<svg xmlns="http://www.w3.org/2000/svg" width="960" height="220" viewBox="0 0 960 220">
+  <style>
+    .box { fill: #f0f4f8; stroke: #1b4965; stroke-width: 2; rx: 10; }
+    .label { font-family: Arial, sans-serif; font-size: 16px; fill: #1b4965; }
+    .arrow { stroke: #1b4965; stroke-width: 2; marker-end: url(#arrowhead); }
+  </style>
+  <defs>
+    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
+      <polygon points="0 0, 10 3.5, 0 7" fill="#1b4965" />
+    </marker>
+  </defs>
+  <rect class="box" x="30" y="70" width="170" height="80" />
+  <text class="label" x="115" y="105" text-anchor="middle">Data Ingestion</text>
+  <text class="label" x="115" y="128" text-anchor="middle">(train/test)</text>
+
+  <rect class="box" x="250" y="70" width="190" height="80" />
+  <text class="label" x="345" y="105" text-anchor="middle">EDA &amp; Cleaning</text>
+  <text class="label" x="345" y="128" text-anchor="middle">(correlations)</text>
+
+  <rect class="box" x="490" y="70" width="190" height="80" />
+  <text class="label" x="585" y="105" text-anchor="middle">Resampling</text>
+  <text class="label" x="585" y="128" text-anchor="middle">(RandomOverSampler)</text>
+
+  <rect class="box" x="730" y="70" width="200" height="80" />
+  <text class="label" x="830" y="105" text-anchor="middle">Modeling &amp; Evaluation</text>
+  <text class="label" x="830" y="128" text-anchor="middle">(LR/RF/XGB)</text>
+
+  <line class="arrow" x1="200" y1="110" x2="250" y2="110" />
+  <line class="arrow" x1="440" y1="110" x2="490" y2="110" />
+  <line class="arrow" x1="680" y1="110" x2="730" y2="110" />
+</svg>
 
EOF
)

