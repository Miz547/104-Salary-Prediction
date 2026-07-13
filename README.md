# 104 Salary Prediction Model Research System

This project provides a local Windows web app for salary prediction research using CatBoost, LightGBM, and XGBoost.

## Quick Start

On Windows, double-click:

```bat
Start.bat
```

Or run:

```bat
Start.bat
```

Then open:

```text
http://127.0.0.1:5001
```

## Main Pages

- `/index.html` home page
- `/predict.html` salary prediction
- `/salary-url.html` 104 job URL salary prediction
- `/salary_dashboard.html` model comparison dashboard
- `/report.html` research report
- `/data_cleaning.html` data cleaning notes
- `/catboost_training.html` CatBoost training analysis
- `/validation/salary_validation_admin_dashboard.html` validation dashboard

## Data and Models

Training CSV files and trained model artifacts are intentionally not included in this GitHub-ready package.

Expected local paths:

```text
data/104_0605.csv
outputs/trained_models/artifacts_min.joblib
outputs/trained_models/artifacts_max.joblib
```

If these files are not present, the static pages can still be opened, but prediction endpoints will not work until you add trained model artifacts or prepare the training data locally.

## Repository Notes

This upload package excludes:

- `.venv/`
- `__pycache__/`
- logs
- CSV training data
- `.joblib` model artifacts
- old backup HTML files
