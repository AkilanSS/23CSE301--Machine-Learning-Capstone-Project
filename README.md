# 23CSE301 Machine Learning Capstone Project

**Academic Year:** 2026–2027  
**Course:** 23CSE301 — Machine Learning  

## Overview

This repository contains three Jupyter notebooks for the Machine Learning capstone project. Each notebook follows a machine-learning workflow that includes dataset inspection, exploratory data analysis (EDA), preprocessing, feature extraction or engineering, model training, evaluation, and visualisation.

The project has three tracks:

1. **Regression:** Predicting true core temperature from infrared thermography measurements and related attributes.
2. **Classification:** Classifying EEG motor-imagery trials into four classes.
3. **Clustering:** Exploring structure in EEG motor-imagery data using unsupervised clustering.



## Repository Structure

Use the following layout, and adjust paths if your current repository differs:

```text
.
├── README.md
├── requirements.txt
├── datasets/
│   ├── ICI.xlsx
│   ├── A01T.gdf
│   ├── A01E.gdf
│   └── A01E.mat
└── notebooks/
    ├── regression.ipynb
    ├── classification.ipynb
    └── clustering.ipynb
```


## Track 1 — Regression

**Notebook:** `notebooks/regression.ipynb`  
**Dataset referenced:** `datasets/ICI.xlsx`  
**Target:** `True_Core_Temp_Slow`

The regression notebook loads data from four experimental rounds and builds models to estimate true core temperature from infrared thermography and associated attributes.

### Workflow

- Inspect dataset dimensions, column types, missing values, and target distribution.
- Explore features and relationships using visualisations.
- Clean and preprocess numerical and categorical features.
- Create and evaluate engineered features.
- Train and compare regression models.
- Tune selected model hyperparameters.
- Evaluate predictions and inspect model behaviour with plots.

### Algorithms

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

### Evaluation

The notebook evaluates models using R², root mean squared error (RMSE), and mean absolute error (MAE). It also includes cross-validation and prediction/residual visualisations.

## Track 2 — Classification

**Notebook:** `notebooks/classification.ipynb`  
**Dataset referenced:** `datasets/A01T.gdf` and `datasets/A01E.gdf`  
**Additional file, if used by the notebook:** `datasets/A01E.mat`

This track analyses EEG motor-imagery data for subject A01. The four classes are left hand, right hand, feet, and tongue.

### Workflow

- Load and inspect the EEG recordings.
- Preprocess the signal and create epochs.
- Extract features using Common Spatial Patterns (CSP).
- Train and compare classifiers.
- Evaluate predictions with classification metrics and visualisations.
- Tune selected models and examine the final model choice.

### Required classifiers

**Part A**

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVM)

**Part B**

6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. Multilayer Perceptron (MLP)

### Evaluation

The course rubric requires Accuracy, Precision, Recall, weighted F1-score, confusion matrices, and ROC-AUC. For multiclass classification, ROC-AUC should use a one-vs-rest (OvR) strategy. The final report should contain one consolidated comparison table covering all ten classifiers.

## Track 3 — Clustering

**Notebook:** `notebooks/clustering.ipynb`  
**Dataset referenced:** `datasets/A01T.gdf` and `datasets/A01E.gdf`

This track explores structure in EEG motor-imagery features using unsupervised learning. Ground-truth labels should be used only after clustering for interpretation or validation, not as model-fitting inputs.

### Workflow and algorithms

- Preprocess EEG data and extract features.
- Scale features where appropriate.
- Apply PCA for dimensionality reduction and visualisation.
- Fit **K-Means** clustering and inspect an elbow curve.
- Fit **Agglomerative Hierarchical Clustering** and compare linkage strategies.
- Visualise clusters with PCA and t-SNE.
- Compare cluster quality using evaluation metrics.

### Evaluation

The notebook reports or should report the following metrics for each evaluated clustering configuration:

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index

The course rubric also requires an elbow curve for K-Means, a dendrogram for hierarchical clustering, and PCA-based two-dimensional cluster visualisations. A t-SNE visualisation is also included in the project workflow.

## Results Summary

The tables below use the saved outputs in the submitted notebooks. Results are specific to the current notebook run and should be regenerated if the data or preprocessing changes.

### Regression model comparison

| Model | Test R² | Test RMSE | Test MAE | Mean 5-fold CV R² |
|---|---:|---:|---:|---:|
| Support Vector Regressor (SVR) | 0.6326 | 0.2949 | 0.2191 | 0.6641 |
| Gradient Boosting Regressor | 0.6173 | 0.3010 | 0.2211 | 0.7012 |
| Random Forest Regressor | 0.6132 | 0.3026 | 0.2257 | 0.6846 |
| Polynomial Regression (degree 2) | 0.5966 | 0.3090 | 0.2310 | 0.6415 |
| K-Nearest Neighbors Regressor | 0.5749 | 0.3172 | 0.2308 | 0.6595 |
| ElasticNet Regression | 0.5428 | 0.3290 | 0.2501 | 0.6526 |
| Ridge Regression | 0.5425 | 0.3291 | 0.2502 | 0.6527 |
| Lasso Regression | 0.5420 | 0.3293 | 0.2505 | 0.6518 |
| Linear Regression | 0.5395 | 0.3301 | 0.2516 | Not reported |
| Decision Tree Regressor | 0.5266 | 0.3348 | 0.2399 | 0.6003 |

**Interpretation:** SVR has the highest held-out test R² (0.6326) and the lowest RMSE (0.2949) in the saved baseline comparison. Gradient Boosting has a higher mean five-fold CV R² (0.7012) than SVR (0.6641), so its generalisation estimate differs from the ranking on this held-out test split. The final model choice should consider both results and the evaluation strategy.

### Classification model comparison

Metrics below are from the saved evaluation outputs. Precision, Recall and F1 are weighted averages. ROC-AUC is one-vs-rest (OvR) where it was reported in the notebook. The saved outputs do **not** report ROC-AUC for the first five Part A models in the same evaluation block; those cells are marked “Not reported” rather than estimated.

| Classifier | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.753 | 0.80 | 0.753 | 0.75 | Not reported in saved evaluation output |
| K-Nearest Neighbors | 0.708 | 0.75 | 0.708 | 0.70 | Not reported in saved evaluation output |
| Gaussian Naive Bayes | 0.688 | 0.71 | 0.688 | 0.68 | Not reported in saved evaluation output |
| Decision Tree Classifier | 0.628 | 0.64 | 0.628 | 0.63 | Not reported in saved evaluation output |
| Support Vector Machine (SVM) | 0.767 | 0.80 | 0.767 | 0.76 | Not reported in saved evaluation output |
| Random Forest Classifier | 0.684 | 0.751 | 0.684 | 0.673 | 0.935 |
| AdaBoost Classifier | 0.677 | 0.696 | 0.677 | 0.677 | 0.899 |
| Gradient Boosting Classifier | 0.719 | 0.741 | 0.719 | 0.712 | 0.930 |
| Bagging Classifier | 0.663 | 0.714 | 0.663 | 0.646 | 0.927 |
| Multilayer Perceptron (MLP) | 0.747 | 0.797 | 0.747 | 0.737 | 0.954 |

**Interpretation:** SVM has the highest reported accuracy among the ten classifiers in these saved outputs (0.767). MLP has the highest reported OvR ROC-AUC (0.954) among the models for which this metric is shown. To satisfy the final course rubric, calculate and add OvR ROC-AUC for Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree and SVM, then rerun and save the consolidated ten-model table. Confusion matrices are provided in the classification notebook.

### Clustering comparison

| Algorithm / configuration | Silhouette (higher is better) | Davies–Bouldin (lower is better) | Calinski–Harabasz (higher is better) | Adjusted Rand Index (post-hoc) |
|---|---:|---:|---:|---:|
| K-Means | 0.1653 | 2.1169 | 21.3555 | 0.2244 |
| Agglomerative — Ward | 0.1632 | 2.0198 | 18.5350 | 0.1170 |
| Agglomerative — Complete | 0.4280 | 0.4240 | 5.0330 | 0.0000 |
| Agglomerative — Average | 0.4280 | 0.4240 | 5.0330 | 0.0000 |
| Agglomerative — Single | 0.4280 | 0.4240 | 5.0330 | 0.0000 |

**Interpretation:** K-Means has the highest Adjusted Rand Index against the available reference labels (0.2244), although its silhouette score is low (0.1653). Complete, average and single linkage show higher silhouette values but an ARI of 0.0000 and a much lower Calinski–Harabasz index. A high silhouette score alone therefore does not establish agreement with the reference labels or a useful clustering solution.

## Environment Setup

### Requirements

- Python version supported by the installed dependency versions (Python 3.11 is a suggested starting point; confirm compatibility for MNE and the other packages).
- NumPy, Pandas, SciPy
- scikit-learn
- Matplotlib, Seaborn
- MNE for EEG signal processing
- OpenPyXL for Excel workbook loading
- JupyterLab and IPykernel
- Joblib

The `requirements.txt` file should contain the **exact package versions tested with these notebooks**. After setting up and successfully running the notebooks, you can record installed versions with:

```bash
python --version
python -m pip freeze
```

Review the generated package list and keep the dependencies needed by this project in `requirements.txt`. Avoid claiming a dependency set is verified until a clean installation and full notebook run have succeeded.

### 1. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If Python 3.11 is not installed, replace `3.11` with a version supported by the package versions you choose.

### 2. Install dependencies

From the repository root, run:

```bash
pip install -r requirements.txt
```

If MNE or another dependency fails to install, check its supported Python versions and resolve package compatibility before recording the final dependency versions.

### 3. Launch JupyterLab

```bash
jupyter lab
```

Open the notebooks under `notebooks/`. Confirm that dataset paths in the notebook code match the repository structure described above.

### 4. Verify the environment and notebooks

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, mne, openpyxl; print('Imports successful')"
```

Then restart each notebook kernel and run all cells from top to bottom. Verify that the results tables, metrics, and visualisations are displayed and that no cell fails.

## Reproducibility and Good Practice

- Use a consistent train/test split for models compared within a track.
- Set random seeds, such as `random_state=42`, wherever supported and appropriate.
- Fit scalers, encoders, and learned preprocessing steps on training data only.
- When using cross-validation, place learned preprocessing and feature extraction inside the cross-validation pipeline where appropriate to avoid data leakage.
- Display model comparison tables instead of relying only on separate print statements.
- Add titles, axis labels, and legends where relevant to plots.
- Keep the notebooks organised, with Markdown explanations for major steps.
- Save the final notebooks with outputs visible and verify a clean top-to-bottom execution before submission.

## Team Information


- **Member 1:** `[Akilan S S]`
- **Member 2:** `[Naveen Velan S N]`
- **Member 3:** `[Shreyas N]`


## Limitations

- Results depend on the dataset version, preprocessing, split strategy, hyperparameters, and installed package versions.
- The EEG analysis references subject A01; results should not be generalised to other subjects without additional evaluation.
- A GUI or publicly deployed application is not claimed here. If you implement one for the course bonus, document its setup and provide the actual deployment URL.
- The dataset paths in this README must match the paths used by the notebooks in the final repository.

## References and Academic Integrity

- Cite any external code, documentation, or other resources used in the notebooks.
- If generative AI was used for code scaffolding or documentation, disclose and cite that assistance in accordance with the course guidelines. Project-specific analysis, interpretation, and feature-engineering decisions must be reviewed and owned by the team.
