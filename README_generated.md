# ML-Based Class 5 Construction Cost Estimator

This repository contains a Flask web application that provides **Class 5** construction cost estimates using a machine learning model trained on historical project data.

The app was developed as part of:

- **Course:** CSE6748 – Applied Analytics Practicum  
- **Institution:** Georgia Institute of Technology  
- **Semester:** Fall 2025  

---

## Overview

The goal of the project is to build a **data-driven Class 5 estimator** that can provide early-stage cost estimates based on:

- Project location (state, area type)
- Project type and category
- Complexity metrics (divisions, item codes, CSI groups, CIQS category)
- Economic factors (Area Cost Factor, CPI, inflation factor)

The application:

- Preprocesses and normalizes historical cost data to 2025 dollars
- Trains a machine learning model (Random Forest with preprocessing)
- Serves predictions through a user-friendly web interface and a simple JSON API

---

## Main Features

- **Home Page**
  - Brief description of the project and context.

- **Exploratory Data Analysis (EDA)**
  - Summary statistics (total projects, cost ranges, number of states/cities).
  - Visualizations:
    - Cost distribution
    - Cost by project type/category
    - Feature correlations

- **Model Comparison**
  - Trains and evaluates multiple models (e.g., Linear Regression, Random Forest, etc.).
  - Compares performance using MAPE, RMSE, and R².
  - Selects the best model for deployment.

- **Dashboard**
  - High-level charts (e.g., cost by state, category, or project type).
  - Helps explore patterns in the dataset.

- **Cost Estimator**
  - Interactive form to enter:
    - Location and area type
    - Project type and category
    - Complexity metrics
    - Economic factors (ACF, CPI, inflation factor)
  - Returns:
    - Estimated project cost
    - Confidence interval
    - Model performance metrics for transparency

- **Data Overview**
  - Description of key columns and their ranges/roles.
  - High-level notes on preprocessing.

- **Documentation**
  - Short user guide for the estimator.
  - Simple API documentation for `/api/estimate`.
  - Link to final project report (`static/reports/final_report.pdf`).
  - Team members and course information.

---

## Project Structure (Simplified)

```text
construction_cost_estimator/
├── app.py                    # Main Flask application
├── train_model.py            # Training script for the ML model (optional)
├── requirements.txt          # Pinned Python dependencies
├── INSTRUCTIONS.md           # Detailed setup & run instructions
├── data/
│   └── base_data_for_model.csv      # Processed dataset
├── models/
│   ├── construction_cost_model.pkl  # Trained model (from Jupyter notebook)
│   └── model_metrics.pkl            # Model performance metrics
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── eda.html
│   ├── model_comparison.html
│   ├── dashboard.html
│   ├── cost_estimator.html
│   ├── data_overview.html
│   └── documentation.html
└── static/
    ├── css/
    ├── images/
    └── reports/
        └── final_report.pdf
```

> Note: Exact files may differ slightly based on final app structure, but this reflects the intended layout.

---

## Quick Start

For full details, see **`INSTRUCTIONS.md`**.

### 1. Clone the repo

```bash
git clone https://github.com/kraryal/construction_cost_estimator.git
cd construction_cost_estimator
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
# or
python3 app.py
```

Then open a browser and go to:

- `http://localhost:5000/`

---

## Model Training Notes

- The current model (`models/construction_cost_model.pkl`) was originally trained in a **Jupyter notebook** using `data/base_data_for_model.csv`.
- `requirements.txt` pins package versions (especially `scikit-learn`) to ensure this pickle can be loaded reliably on other machines.

### Optional Retraining via Script

If you want to retrain the model programmatically instead of using the original notebook:

```bash
python train_model.py
```

This will:

1. Load `data/base_data_for_model.csv`
2. Engineer features and split data
3. Train the selected model (e.g., Random Forest)
4. Overwrite:
   - `models/construction_cost_model.pkl`
   - `models/model_metrics.pkl`

The Flask app will automatically use the updated model the next time you run `app.py`.

---

## API Endpoint (Optional Use)

The app exposes a simple JSON API for programmatic cost estimates:

- **Endpoint:** `POST /api/estimate`
- **Content-Type:** `application/json`

Example:

```bash
curl -X POST http://localhost:5000/api/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "project_state": "GA",
    "project_type": "Commercial",
    "construction_category": "Office",
    "cnt_division": 10,
    "cnt_item_code": 100,
    "cnt_csi_grp_unq": 20,
    "acf": 1.0
  }'
```

Response includes:

- `estimated_cost`
- `confidence_interval.low`
- `confidence_interval.high`
- `model_metrics` (MAPE, RMSE, R², model type)

---

## Reproducibility

- All Python dependencies are pinned in `requirements.txt`.
- The trained model was generated under these versions.
- As long as users:

  1. Create a virtual environment  
  2. Run `pip install -r requirements.txt`  
  3. Run `python app.py`  

  they should see consistent behaviour on Windows, macOS, or Linux.

---

## Disclaimer

This project is intended for **academic and conceptual** use as a Class 5 estimator.  
Predictions are approximate and should not be used as final construction bids in real projects.
