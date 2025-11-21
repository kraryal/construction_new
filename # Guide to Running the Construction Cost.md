# Guide to Running the Construction Cost Estimator

## ML-Based Class 5 Construction Cost Estimator
**CSE6748 - Applied Analytics Practicum**  
**Team 04**

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation & Setup](#installation--setup)
3. [Running the Application](#running-the-application)
4. [Using the API](#using-the-api)
5. [Troubleshooting](#troubleshooting)
6. [Project Structure](#project-structure)
7. [Application Features](#application-features)

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

---

## Installation & Setup

### 1. Get the Code

```bash
# Clone repository (if using Git)
git clone https://github.com/[your-username]/construction-cost-estimator.git
cd construction-cost-estimator

# Or download and extract the ZIP from GitHub
```

### 2. Create a Virtual Environment

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
# If requirements.txt exists
pip install -r requirements.txt

# Otherwise install packages manually
pip install flask pandas numpy scikit-learn matplotlib seaborn colorama
```

### 4. Prepare the Dataset

- Ensure `base_data_for_model.csv` is in the root directory or a `data/` folder
- If using a different dataset location, update the path in `app.py`

---

## Running the Application

### 1. Start the Flask Server

```bash
python app.py
```

You should see output similar to this:
```
Starting Construction Cost Estimator...
STATE_FACTORS contains 50 states
TYPE_FACTORS contains 13 project types
CATEGORY_FACTORS contains 8 categories

=============================================================
Construction Cost Estimator
CSE6748 - Applied Analytics Practicum
Georgia Institute of Technology
=============================================================

Server starting...
Access the application at:
http://localhost:5000/
http://192.168.1.100:5000/

=============================================================
Useful URLs:
Main page:        http://localhost:5000/
Cost Estimator:   http://localhost:5000/cost_estimator
API Documentation:http://localhost:5000/documentation
Debug Factors:    http://localhost:5000/debug/factors
=============================================================
```

### 2. Access the Application

- Open your browser and go to: http://localhost:5000
- For access from other devices, use your computer's IP address shown in terminal

---

## Using the API

### Endpoint Information

- **URL**: `/api/estimate`
- **Method**: POST
- **Content-Type**: application/json

### Request Format

```json
{
  "project_state": "GA",
  "project_type": "Commercial",
  "construction_category": "Office",
  "cnt_division": 10,
  "cnt_item_code": 100,
  "cnt_csi_grp_unq": 20,
  "acf": 1.0
}
```

### Response Format

```json
{
  "estimated_cost": 5207578.25,
  "confidence_interval": {
    "low": 4063473.58,
    "high": 6351683.00
  },
  "model_metrics": {
    "mape": 21.97,
    "rmse": 271543.90,
    "r2": 0.9463,
    "model_type": "Random Forest"
  }
}
```

### Example API Call (Using cURL)

```bash
curl -X POST http://localhost:5000/api/estimate \
  -H "Content-Type: application/json" \
  -d '{"project_state":"GA","project_type":"Commercial","construction_category":"Office","cnt_division":10,"cnt_item_code":100,"cnt_csi_grp_unq":20,"acf":1.0}'
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Port already in use** | Change the port in `app.py` by modifying: `app.run(debug=True, host='0.0.0.0', port=5000)` to use port 8080 or 3000 instead |
| **Missing dependencies** | Install any missing packages with: `pip install [package-name]` |
| **Data file not found** | Check that your CSV file path matches what's in the code and update paths in `app.py` if necessary |
| **Web page styling issues** | Ensure that Bootstrap CSS is being loaded correctly and check the browser console for resource loading errors |
| **Model loading errors** | Verify that scikit-learn versions match between training and deployment |

### Network Access

- The application binds to all interfaces (0.0.0.0) by default, making it accessible from other devices on your network
- If you cannot access from other devices, check that your firewall allows connections to the port you're using

---

## Project Structure

```
construction-cost-estimator/
├── app.py                  # Main Flask application
├── data/                   # Data directory
│   └── base_data_for_model.csv  # Dataset file
├── static/                 # Static files
│   ├── css/                # CSS files
│   ├── js/                 # JavaScript files
│   ├── images/             # Image files
│   └── reports/            # Report files (PDF)
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Home page
│   ├── cost_estimator.html # Estimator form
│   └── ...                 # Other templates
└── requirements.txt        # Project dependencies
```

---

## Application Features

### Interactive Web Interface

- **Home Page**: Overview of the application and navigation
- **Cost Estimator**: Form for entering project details and obtaining cost estimates
- **Model Comparison**: Visualization of different model performances
- **EDA (Exploratory Data Analysis)**: Interactive data visualizations
- **Documentation**: API documentation and user guide
- **Dashboard**: Performance metrics of the selected model

### Machine Learning Capabilities

- Random Forest model with 21.97% MAPE (Mean Absolute Percentage Error)
- Confidence interval generation for cost estimates
- Feature importance visualization
- Residual analysis

### API Integration

- RESTful API for programmatic access
- JSON request/response format
- Comprehensive error handling

---

## Production Considerations

For deploying to production:

1. Set `debug=False` in `app.py`
2. Consider using a production WSGI server such as Gunicorn or uWSGI
3. Implement proper authentication for the API
4. Set up HTTPS using a reverse proxy like Nginx
5. Implement rate limiting for the API endpoints

---

*For more detailed information, refer to the full documentation or contact the development team.*