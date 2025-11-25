```markdown
# ML-Based Class 5 Construction Cost Estimator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-Academic-yellow.svg)

**CSE6748 - Applied Analytics Practicum**  
Georgia Institute of Technology | Fall 2025

</div>

---

## 🎯 Project Overview

A machine learning web application that predicts early-stage construction costs (Class 5 estimates) with **21.97% MAPE**, successfully exceeding the target requirement of 25%.

### Key Achievement
✅ **Target Met:** MAPE < 25%  
✅ **Actual Result:** 21.97% MAPE  
✅ **Improvement:** 3.03% better than target

---

## 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Test MAPE** | **21.97%** | ✅ Meets Target |
| **R² Score** | **0.9463** | Excellent |
| **Test MAE** | $271,543.90 | Strong |
| **Test RMSE** | $412,583.00 | Reliable |
| **Dataset Size** | 17,025 projects | Large |

**Training Split:** 80% training (13,620) / 20% testing (3,405)

---

## 🚀 Features

### 📈 Analytics & Visualization
- **Exploratory Data Analysis** - Interactive charts and statistical insights
- **Performance Dashboard** - Real-time model metrics and visualizations
- **Model Comparison** - Compare Random Forest, XGBoost, Gradient Boosting, LightGBM
- **Data Overview** - Complete database schema and statistics

### 💰 Cost Estimation
- **Real-time Predictions** - Instant cost estimates using ML model
- **Confidence Intervals** - ±25% range for reliability
- **Similar Projects** - Find comparable projects in database
- **Input Validation** - Smart form with range checks

### 📚 Documentation
- **API Documentation** - Complete REST API reference
- **User Guide** - Step-by-step usage instructions
- **Model Details** - Algorithm explanation and training process
- **Team Information** - Contact and contribution details

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **HTML5/CSS3** - Modern responsive design
- **Bootstrap** - UI components
- **Chart.js** - Interactive visualizations
- **Vanilla JavaScript** - Dynamic interactions

### Machine Learning
- **Algorithm:** Random Forest Regressor
- **Features:** 13 engineered features
- **Preprocessing:** StandardScaler + OneHotEncoder
- **Validation:** 5-fold cross-validation

---

## 📁 Project Structure

```
construction_cost_estimator/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── INSTRUCTIONS.md                 # Setup guide
│
├── data/
│   └── base_data_for_model.csv    # Training dataset (17,025 projects)
│
├── models/
│   ├── construction_cost_model.pkl # Trained Random Forest model
│   └── model_metrics.json          # Performance metrics
│
├── templates/
│   ├── base.html                   # Base template
│   ├── home.html                   # Landing page
│   ├── eda.html                    # Exploratory analysis
│   ├── model_comparison.html       # Model comparison
│   ├── dashboard.html              # Performance dashboard
│   ├── cost_estimator.html         # Cost estimation form
│   ├── data_overview.html          # Dataset overview
│   ├── documentation.html          # API docs
│   └── error.html                  # Error page
│
└── static/
    ├── css/
    │   └── styles.css              # Custom styles
    └── images/                     # Generated plots
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

```bash
# 1. Clone repository
git clone https://github.com/kraryal/construction_cost_estimator.git
cd construction_cost_estimator

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place dataset
# Copy base_data_for_model.csv to data/ folder

# 6. Run application
python app.py

# 7. Open browser
# Navigate to http://localhost:5000
```

---

## 📋 Model Details

### Features Used (13 features)

**Economic Factors (2)**
- Inflation Factor (1.00 - 1.34)
- Area Cost Factor / ACF (0.80 - 1.19)

**Project Classification (4)**
- Project Type (categorical)
- Project Category (categorical)
- CIQS Complexity Category (1-4)
- Official Budget Range (categorical)

**Geographic Location (4)**
- Project State (50 states)
- County Name (varies by state)
- Area Type (Urban/Rural)
- Region (4 clusters from K-Means)

**Construction Details (2)**
- CNT Division Code (1.00 - 29.00)
- CNT Item Code (1.00 - 61.00)

### Training Process

1. **Data Loading:** 17,025 historical projects (2010-2025)
2. **Preprocessing:** Missing value imputation, normalization
3. **Feature Engineering:** Geographic clustering, economic factors
4. **Model Training:** Random Forest with 100 trees
5. **Validation:** 5-fold cross-validation
6. **Testing:** 20% holdout test set

---

## 🎓 Model Comparison Results

| Model | CV MAPE | Test MAPE | Test R² | Status |
|-------|---------|-----------|---------|--------|
| **Random Forest** | 23.30% | **21.97%** | **0.9463** | ✅ **Deployed** |
| XGBoost | 36.16% | 36.37% | 0.9258 | Above Target |
| LightGBM | 36.93% | 37.62% | 0.9232 | Above Target |
| Gradient Boosting | 42.48% | 43.75% | 0.9015 | Above Target |

**Winner:** Random Forest selected for best MAPE performance and model interpretability.

---

## 💻 Usage Examples

### Web Interface

1. Navigate to **Cost Estimator** page
2. Fill in project details:
   - Project Type: "Pavement Markers"
   - Budget Range: "$3M-$6M"
   - Complexity: "Category 4"
   - Location: State, County, Area Type
   - Economic Factors: Inflation, ACF
3. Click **"Calculate Cost Estimate"**
4. View prediction with confidence interval

### API Usage

```python
import requests

# Prepare project data
data = {
    'inflation_factor': 1.05,
    'official_budget_range': '$3M-$6M',
    'ciqs_complexity_category': 'Category 4',
    'cnt_division': 6,
    'cnt_item_code': 6,
    'county_name': 'Alcona County',
    'area_type': 'Rural',
    'acf': 1.01,
    'project_type': 'Pavement Markers',
    'project_category': 'Civil',
    'project_state': 'MI',
    'region': 'Region_3'
}

# Make prediction request
response = requests.post('http://localhost:5000/estimate_cost', data=data)
result = response.json()

# Display results
if result['success']:
    print(f"Estimated Cost: {result['estimated_cost_formatted']}")
    print(f"Confidence Range: {result['confidence_interval']['lower_formatted']} - {result['confidence_interval']['upper_formatted']}")
    print(f"Similar Projects: {result['similar_projects']['count']}")
```

**Sample Output:**
```
Estimated Cost: $4,358,432.11
Confidence Range: $3,268,824.08 - $5,448,040.14
Similar Projects: 6147
```

---

## 📊 Dataset Information

- **Source:** PCS (Project Cost System) Database
- **Time Period:** 2010-2025 (15 years)
- **Total Projects:** 17,025
- **Cost Normalization:** All costs adjusted to 2025 dollars
- **Geographic Coverage:** All 50 US states
- **Project Types:** Various construction categories

**Cost Statistics:**
- **Average:** $1,142,356
- **Median:** $856,470
- **Range:** $10,500 - $15,200,000

---

## 👥 Project Team

<table>
<tr>
<td align="center">
<img src="https://via.placeholder.com/100/7c3aed/ffffff?text=KA" width="100" style="border-radius:50%"><br>
<b>Krishna Aryal</b><br>
Data Engineering & Model Development<br>
<a href="mailto:karyal@gatech.edu">📧 Email</a> | 
<a href="https://github.com/kraryal">💻 GitHub</a>
</td>
<td align="center">
<img src="https://via.placeholder.com/100/ec4899/ffffff?text=KS" width="100" style="border-radius:50%"><br>
<b>Kumar Sawan</b><br>
Feature Engineering & Optimization<br>
<a href="mailto:ksawan@gatech.edu">📧 Email</a>
</td>
<td align="center">
<img src="https://via.placeholder.com/100/06b6d4/ffffff?text=NK" width="100" style="border-radius:50%"><br>
<b>Neema Kafwimi</b><br>
Model Evaluation & Deployment<br>
<a href="mailto:nkafwimi@gatech.edu">📧 Email</a>
</td>
</tr>
</table>

---

## 📖 Documentation

- **User Guide:** See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed setup
- **API Documentation:** Available at `/documentation` route
- **Notebook:** Original analysis in `organised_construction_notebook.ipynb`

---

## 🔒 Limitations

⚠️ **Important Notes:**

- **Class 5 Estimates Only:** Provides conceptual estimates (±25% accuracy)
- **Not for Bidding:** Not suitable for detailed bidding or final estimates
- **Historical Data:** Based on 2010-2025 projects; may not capture future trends
- **Normalized Costs:** All predictions in 2025 dollars
- **Geographic Bias:** Best performance on well-represented states
- **Requires Validation:** Always consult with construction professionals

---

## 📝 License

This project is part of an academic practicum for Georgia Institute of Technology.  
**Course:** CSE6748 - Applied Analytics Practicum  
**Semester:** Fall 2025

---

## 🙏 Acknowledgments

- **Construction Cost Database LLC** - Dataset provider and client
- **Georgia Tech CSE6748** - Course staff and instructors
- **Project Advisors** - Technical guidance and support
- **scikit-learn Community** - ML algorithms and tools

---

## 📞 Contact & Support

**Primary Contact:** Krishna Aryal  
**Email:** karyal@gatech.edu  
**GitHub:** [kraryal](https://github.com/kraryal)  
**Institution:** Georgia Institute of Technology

For bugs, issues, or questions, please open an issue on GitHub.

---

## 📈 Future Improvements

- [ ] Add more ML models (Neural Networks, Ensemble methods)
- [ ] Real-time data updates
- [ ] User authentication and project history
- [ ] Export predictions to PDF/Excel
- [ ] Integration with external cost databases
- [ ] Mobile application
- [ ] API rate limiting and authentication

---

<div align="center">

**Built with ❤️ for Construction Cost Database LLC**

*Accurate Early-Stage Cost Estimation Powered by Machine Learning*

[🏠 Home](http://localhost:5000) | [📊 Dashboard](http://localhost:5000/dashboard) | [💰 Estimate](http://localhost:5000/cost_estimator) | [📚 Docs](http://localhost:5000/documentation)

</div>
```

## INSTRUCTIONS.md

```markdown
# Construction Cost Estimator - Complete Setup Guide

**Step-by-step instructions to install, configure, and run the ML-Based Construction Cost Estimator application.**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Using the Application](#using-the-application)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)
8. [Advanced Topics](#advanced-topics)

---

## 1. System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10+, macOS 10.14+, Ubuntu 20.04+ |
| **Python** | 3.11 or higher |
| **RAM** | 4 GB (8 GB recommended) |
| **Disk Space** | 500 MB free |
| **Browser** | Chrome, Firefox, Safari, or Edge (latest) |

### Check Your Python Version

```bash
python --version
# Expected output: Python 3.11.x or higher
```

If Python is not installed or version is lower:
- **Windows/Mac:** Download from [python.org](https://www.python.org/downloads/)
- **Linux:** `sudo apt-get install python3.11`

---

## 2. Installation

### Step 1: Download the Project

**Option A: Using Git (Recommended)**

```bash
git clone https://github.com/kraryal/construction_cost_estimator.git
cd construction_cost_estimator
```

**Option B: Download ZIP**

1. Go to GitHub repository
2. Click **"Code"** → **"Download ZIP"**
3. Extract to your preferred location
4. Open terminal/command prompt in extracted folder

### Step 2: Create Virtual Environment

**Why?** Keeps dependencies isolated and avoids conflicts with other Python projects.

**Windows PowerShell:**

```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If execution policy error occurs:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again
```

**Windows Command Prompt:**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**✅ Success Indicator:** You should see `(venv)` at the start of your command prompt.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Collecting Flask==3.0.0
  Downloading Flask-3.0.0-py3-none-any.whl (...)
...
Successfully installed Flask-3.0.0 pandas-2.1.4 numpy-1.26.2 ...
```

**If installation fails:**

```bash
# Install packages individually
pip install Flask
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib
pip install scipy
```

### Step 5: Verify Installation

```bash
python -c "import flask, pandas, sklearn, matplotlib; print('✅ All packages installed successfully!')"
```

---

## 3. Configuration

### Step 1: Prepare Data Directory

```bash
# Create data folder
mkdir data

# Windows alternative:
md data
```

### Step 2: Add Dataset

1. Locate your `base_data_for_model.csv` file
2. Copy it to the `data/` folder
3. Final path should be: `data/base_data_for_model.csv`

**Verify file location:**

```bash
# Mac/Linux
ls -lh data/base_data_for_model.csv

# Windows
dir data\base_data_for_model.csv
```

**Expected output:** File size should be approximately 5-10 MB

### Step 3: Create Models Directory

```bash
mkdir models

# Windows:
md models
```

This folder will store the trained machine learning model.

---

## 4. Running the Application

### First-Time Launch

```bash
# Make sure you're in project directory and venv is activated
python app.py
```

### What Happens on First Run?

**Console Output Walkthrough:**

```
================================================================================
🏗️  ML-BASED CLASS 5 CONSTRUCTION COST ESTIMATOR
================================================================================

Loading and exploring the dataset...
✅ Dataset loaded successfully: 17025 projects with 38 features.
```
*The app loads your CSV and counts rows/columns.*

```
✅ Using 13 available columns (out of 14 possible):
Available columns: inflation_factor, official_budget_range, ...
```
*Shows which features will be used for prediction.*

```
================================================================================
DATA PREPROCESSING AND FEATURE ENGINEERING
================================================================================
Handled 1234 missing values. Remaining missing: 0
```
*Cleans data by filling missing values with median/mode.*

```
Feature Engineering:
1. Creating regional clusters from states and geographic coordinates
✅ Created 4 geographic regions based on clustering state coordinates.
```
*Uses K-Means to group states into regions.*

```
✅ Loaded 7 categorical features
✅ Loaded 4 numerical features
```
*Separates features for proper preprocessing.*

```
🔄 Training new model...
================================================================================
MODEL DEVELOPMENT
================================================================================
Training set: 13620 samples
Test set: 3405 samples

🚀 Training Random Forest...
```
*This takes 2-5 minutes on first run.*

```
✅ Training completed

================================================================================
📈 Model Performance:
================================================================================
  Train MAPE:   8.43%
  Test MAPE:    21.97%
  Test R²:      0.9463
  Test RMSE:    $412,583.00
  Test MAE:     $271,543.90
================================================================================
```
*Shows how well the model performs.*

```
✅ Model trained and saved

================================================================================
✅ System Ready!
================================================================================
🌐 Access at: http://localhost:5000
================================================================================

 * Running on http://0.0.0.0:5000
 * Debug mode: on
```
*Server is now running!*

### Accessing the Application

1. Open your web browser
2. Navigate to: **http://localhost:5000**
3. You should see the home page with project overview

### Subsequent Runs

After the first run, the model is saved. Future launches are faster:

```bash
python app.py
```

Output will show:
```
✅ Model loaded successfully
✅ System Ready!
```

Loads in seconds instead of minutes!

### Stopping the Application

Press `Ctrl + C` in the terminal to stop the server.

---

## 5. Using the Application

### 5.1 Home Page

**URL:** `http://localhost:5000`

**Features:**
- Overview of the project
- Quick statistics (total projects, avg/median cost)
- Cards with links to all features
- Team information

**Navigation:** Use the blue navigation bar at top

### 5.2 Exploratory Data Analysis (EDA)

**URL:** `http://localhost:5000/eda`

**What You'll See:**
- Cost distribution histogram
- Projects by type bar chart
- Geographic heat map
- Budget range analysis
- Interactive data tables

### 5.3 Model Comparison

**URL:** `http://localhost:5000/model_comparison`

**Content:**
- Comparison table of 4 ML models
- MAPE comparison chart (lower is better)
- R² Score comparison chart (higher is better)
- Model development journey
- Performance metrics

**Models Compared:**
1. ✅ Random Forest (Deployed) - 21.97% MAPE
2. XGBoost - 36.37% MAPE
3. LightGBM - 37.62% MAPE
4. Gradient Boosting - 43.75% MAPE

### 5.4 Performance Dashboard

**URL:** `http://localhost:5000/dashboard`

**Displays:**
- Current model R² score
- Mean Absolute Error (MAE)
- Learning curves
- Residual analysis
- Feature importance chart

### 5.5 Cost Estimator (Main Feature!)

**URL:** `http://localhost:5000/cost_estimator`

#### Step-by-Step Guide

**Step 1: Fill Project Type**
- Select from dropdown (e.g., "Pavement Markers")

**Step 2: Select Budget Range**
- Choose appropriate range (e.g., "$3M-$6M")

**Step 3: Choose Complexity**
- Select Category 1-4 based on project complexity

**Step 4: Enter Location**
- **State:** Select from dropdown (e.g., "MI")
- **County:** Type county name (e.g., "Alcona County")
- **Area Type:** Urban or Rural

**Step 5: Economic Factors**
- **Inflation Factor:** Range 1.00-1.34 (default shows median)
- **ACF (Area Cost Factor):** Range 0.80-1.19

**Step 6: Construction Codes**
- **CNT Division:** 1-29
- **CNT Item Code:** 1-61

**Step 7: Region**
- Auto-populated based on state (Region_0 to Region_3)

**Step 8: Click "Calculate Cost Estimate"**

#### Understanding Results

**Estimated Project Cost:** `$4,358,432.11`
- This is the ML model's prediction

**Confidence Range (±25%):** `$3,268,824.08 - $5,448,040.14`
- Upper and lower bounds for reliability

**Similar Projects in Database:**
- **Count:** 6,147 matching projects
- **Average:** $1,310,815.37
- **Median:** $856,470.56
- **Range:** $106,090.18 - $5,753,308.45

**Your Input Summary:**
- Review all entered values

#### Actions Available

- **New Estimate** - Start fresh prediction
- **Back to Dashboard** - View model performance
- **Print Results** - Print or save as PDF

### 5.6 Data Overview

**URL:** `http://localhost:5000/data_overview`

**Content:**
- Database schema diagram
- Table descriptions
- Feature explanations
- Data quality metrics

### 5.7 Documentation

**URL:** `http://localhost:5000/documentation`

**Includes:**
- Model performance summary
- Feature list and descriptions
- API documentation
- Usage examples
- Team information
- Contact details

---

## 6. Troubleshooting

### Problem 1: "Module not found" Error

**Error Message:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```bash
# Check if virtual environment is activated
# You should see (venv) in prompt

# If not activated:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Problem 2: "Dataset not found" Error

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/base_data_for_model.csv'
```

**Solution:**
```bash
# Check if file exists
ls data/  # Mac/Linux
dir data\ # Windows

# File must be named exactly: base_data_for_model.csv
# Check for typos or extra spaces
```

### Problem 3: Port 5000 Already in Use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**

**Windows:**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

**Mac/Linux:**
```bash
lsof -ti:5000 | xargs kill -9
```

**Alternative:** Change port in `app.py` (last line):
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Problem 4: Permission Denied (Windows)

**Error Message:**
```
cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem 5: Slow Performance / Hanging

**Symptoms:**
- App takes forever to load
- "Training model..." hangs

**Solutions:**

1. **First run takes time** - Model training needs 2-5 minutes
2. **Check RAM** - Close other applications
3. **Reduce dataset size** (if needed) - Use subset for testing
4. **Check disk space** - Ensure 500+ MB free

### Problem 6: Browser Shows Blank Page

**Solutions:**

1. **Check if app is running** - Look at terminal for errors
2. **Try different URL:**
   - `http://127.0.0.1:5000` instead of `localhost`
3. **Clear browser cache** - Ctrl+Shift+R (hard refresh)
4. **Try different browser** - Chrome, Firefox, Edge
5. **Check firewall** - May be blocking port 5000

### Problem 7: Prediction Returns Error

**Error on form submission**

**Checklist:**
- ✅ All required fields filled?
- ✅ Numerical values within specified ranges?
- ✅ Correct format for inputs?
- ✅ Model loaded successfully (check terminal)?

**Debug:**
```bash
# Check terminal output when clicking "Calculate"
# Look for Python error messages
```

---

## 7. API Reference

### Endpoint: POST /estimate_cost

**Description:** Get construction cost prediction

**Content-Type:** `application/x-www-form-urlencoded` or `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Example | Range/Options |
|-----------|------|----------|---------|---------------|
| inflation_factor | float | Yes | 1.05 | 1.00 - 1.34 |
| official_budget_range | string | Yes | "$3M-$6M" | Predefined ranges |
| ciqs_complexity_category | string | Yes | "Category 4" | Category 1-4 |
| cnt_division | int | Yes | 6 | 1 - 29 |
| cnt_item_code | int | Yes | 6 | 1 - 61 |
| county_name | string | Yes | "Alcona County" | Any US county |
| area_type | string | Yes | "Rural" | Urban/Rural |
| acf | float | Yes | 1.01 | 0.80 - 1.19 |
| project_type | string | Yes | "Pavement Markers" | Varies |
| project_category | string | Yes | "Civil" | Varies |
| project_state | string | Yes | "MI" | US state codes |
| region | string | Yes | "Region_3" | Region_0 to Region_3 |

**Example Request (Python):**

```python
import requests

url = 'http://localhost:5000/estimate_cost'

data = {
    'inflation_factor': 1.05,
    'official_budget_range': '$3M-$6M',
    'ciqs_complexity_category': 'Category 4',
    'cnt_division': 6,
    'cnt_item_code': 6,
    'county_name': 'Alcona County',
    'area_type': 'Rural',
    'acf': 1.01,
    'project_type': 'Pavement Markers',
    'project_category': 'Civil',
    'project_state': 'MI',
    'region': 'Region_3'
}

response = requests.post(url, data=data)
result = response.json()

print(result)
```

**Success Response (200):**

```json
{
  "success": true,
  "estimated_cost": 4358432.11,
  "estimated_cost_formatted": "$4,358,432.11",
  "confidence_interval": {
    "lower": 3268824.08,
    "upper": 5448040.14,
    "lower_formatted": "$3,268,824.08",
    "upper_formatted": "$5,448,040.14"
  },
  "similar_projects": {
    "count": 6147,
    "avg_cost": 1310815.37,
    "avg_cost_formatted": "$1,310,815.37",
    "median_cost": 856470.56,
    "median_cost_formatted": "$856,470.56",
    "min_cost": 106090.18,
    "min_cost_formatted": "$106,090.18",
    "max_cost": 5753308.45,
    "max_cost_formatted": "$5,753,308.45",
    "std_cost": 1245678.90,
    "std_cost_formatted": "$1,245,678.90",
    "match_type": "exact"
  },
  "input_data": { ... },
  "timestamp": "2025-11-25 14:30:22"
}
```

**Error Response (400/500):**

```json
{
  "success": false,
  "error": "Error message here"
}
```

---

## 8. Advanced Topics

### 8.1 Retrain Model with New Data

```bash
# 1. Stop the application (Ctrl+C)

# 2. Delete old model
rm models/construction_cost_model.pkl  # Mac/Linux
del models\construction_cost_model.pkl  # Windows

# 3. Update dataset
# Replace data/base_data_for_model.csv with new data

# 4. Restart application
python app.py
# Model will automatically retrain
```

### 8.2 Change Model Parameters

Edit `app.py`, find this section:

```python
RandomForestRegressor(
    n_estimators=100,      # Increase for better accuracy
    max_depth=20,          # Adjust tree depth
    random_state=42,
    n_jobs=-1
)
```

Modify values, delete model file, and retrain.

### 8.3 Deploy to Production

**Not recommended for production without:**
- ✅ User authentication
- ✅ HTTPS/SSL
- ✅ Rate limiting
- ✅ Error logging
- ✅ Database for storing estimates
- ✅ Input sanitization
- ✅ Security hardening

**For production deployment:** Consider using Gunicorn, Docker, and cloud hosting (AWS, Azure, GCP).

### 8.4 Export Predictions

Add this route to `app.py`:

```python
@app.route('/export_prediction/<format>')
def export_prediction(format):
    # Implement CSV/PDF export logic
    pass
```

### 8.5 Batch Predictions

Create a script for bulk predictions:

```python
# batch_predict.py
import pandas as pd
import joblib

model = joblib.load('models/construction_cost_model.pkl')
input_df = pd.read_csv('batch_input.csv')
predictions = model.predict(input_df)

output_df = input_df.copy()
output_df['predicted_cost'] = predictions
output_df.to_csv('batch_output.csv', index=False)
```

---

## 9. Support & Contact

### Getting Help

**Issues with Installation:**
- Check [Troubleshooting](#troubleshooting) section
- Verify all requirements are met
- Ensure virtual environment is activated

**Questions about Model:**
- Review [Documentation page](http://localhost:5000/documentation)
- Check notebook: `organised_construction_notebook.ipynb`

**Bug Reports:**
- Email: karyal@gatech.edu
- Include: Error message, OS, Python version

### Project Information

- **Course:** CSE6748 - Applied Analytics Practicum
- **Institution:** Georgia Institute of Technology
- **Semester:** Fall 2025
- **Client:** Construction Cost Database LLC

---

## 10. Checklist

### Installation Complete When:

- ✅ Python 3.11+ installed
- ✅ Virtual environment created and activated
- ✅ All packages installed successfully
- ✅ Dataset placed in `data/` folder
- ✅ Application starts without errors
- ✅ Home page loads in browser
- ✅ Can make predictions successfully

### Success Indicators:

```bash
(venv) $ python app.py
✅ Dataset loaded successfully: 17025 projects
✅ Model loaded successfully
✅ System Ready!
🌐 Access at: http://localhost:5000
```

---

**🎉 You're all set! Start estimating construction costs with machine learning!**

For any questions, contact Krishna Aryal at karyal@gatech.edu
```

---

Save both files in your project root directory:
- `README.md` - Project overview and quick start
- `INSTRUCTIONS.md` - Detailed setup guide

These documents provide comprehensive documentation for your project! 📚✨