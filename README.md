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
