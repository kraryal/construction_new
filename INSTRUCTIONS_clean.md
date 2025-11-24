# Setup & Run Instructions

This document describes how to set up and run the **ML-Based Class 5 Construction Cost Estimator** Flask app on any machine (Windows, macOS, or Linux).

The environment is controlled via `requirements.txt`, so as long as that file is used, the app should behave the same across different devices and operating systems.

---

## 1. Prerequisites

- Python 3.10+ (project tested with Python 3.12)
- Git (optional but recommended if cloning from GitHub)

Check your Python version:

```bash
python --version
# or
python3 --version
```

---

## 2. Get the Project

### Option A – Clone from GitHub (recommended)

```bash
git clone https://github.com/kraryal/construction_cost_estimator.git
cd construction_cost_estimator
```

### Option B – Download ZIP

1. Go to the GitHub repository page.
2. Click **Code → Download ZIP**.
3. Extract the ZIP.
4. Open a terminal in the extracted project folder.

---

## 3. Create and Activate a Virtual Environment

Using a virtual environment avoids package conflicts with other projects.

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt should start with `(venv)`.

---

## 4. Install Dependencies

All required packages (with pinned versions) are listed in `requirements.txt`.

From the project root with the virtual environment active:

```bash
pip install -r requirements.txt
```

This will install (among others):

- Flask  
- numpy, pandas  
- scikit-learn (version compatible with the trained model)  
- matplotlib, seaborn  
- joblib  
- colorama  

---

## 5. Data and Model Files

The repository should already include:

- `data/base_data_for_model.csv`  
  Processed dataset used by the app and for training.  

- `models/construction_cost_model.pkl`  
  Trained ML model (originally trained in a Jupyter notebook).  

- `models/model_metrics.pkl`  
  Stored performance metrics (e.g., MAPE, RMSE, R²).

If these files are missing or you want to regenerate them:

```bash
python train_model.py
```

This script will:

1. Load `data/base_data_for_model.csv`
2. Engineer features
3. Train the model
4. Save updated `.pkl` files into `models/`

> Note: Retraining is **optional**. For most users (including graders), just installing requirements and running `app.py` is enough.

---

## 6. Run the Flask App

With the virtual environment active and dependencies installed:

```bash
python app.py
# or
python3 app.py
```

You should see console output like:

```text
Dataset shape: (17025, 38)
Created 4 geographic regions.
Loaded model from .../models/construction_cost_model.pkl
Starting Construction Cost Estimator...
 * Running on http://localhost:5000/
```

---

## 7. Access the Application

Open a web browser and go to:

- `http://localhost:5000/` – Home (overview)
- `http://localhost:5000/eda` – Exploratory Data Analysis
- `http://localhost:5000/model_comparison` – Model comparison
- `http://localhost:5000/dashboard` – Dashboard
- `http://localhost:5000/cost_estimator` – **Cost estimator form**
- `http://localhost:5000/data_overview` – Data overview
- `http://localhost:5000/documentation` – Documentation & final report

On the **Cost Estimator** page:

1. Select project location (state, area type).
2. Select project type and project category.
3. Enter complexity metrics:
   - Number of divisions
   - Number of item codes
   - Number of CSI groups
   - CIQS complexity category
4. Enter economic factors:
   - ACF (Area Cost Factor)
   - CPI (optional)
   - Inflation factor (optional)
5. Click **Generate Estimate**.

You will see:

- Predicted project cost (normalized to 2025 dollars)
- Confidence interval
- Model performance metrics (MAPE, RMSE, R²)

---

## 8. Optional: API Usage

The app also exposes a simple JSON API for programmatic estimates.

- **Endpoint:** `POST /api/estimate`
- **Content-Type:** `application/json`

Example request:

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

The JSON response includes:

- `estimated_cost`
- `confidence_interval.low`
- `confidence_interval.high`
- `model_metrics` (MAPE, RMSE, R², model type)

---

## 9. Notes for Graders / Reviewers

- Environment is fully specified by `requirements.txt`.
- The model pickle files in `models/` are compatible with the pinned `scikit-learn` version.
- The app is intended for **Class 5 conceptual estimates** in an academic setting, not for production bidding.
