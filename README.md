<div align="center">

# 🏗️ ML-Based Class 5 Construction Cost Estimator

### Predicting Early-Stage Construction Costs with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-yellow?style=for-the-badge)](LICENSE)

**CSE6748 - Applied Analytics Practicum**  
**Georgia Institute of Technology | Fall 2025**

[🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [📖 Documentation](#-documentation) • [👥 Team](#-team)

---

### 🎯 Achievement: **21.97% MAPE** - Exceeds Target by 3.03%

<img src="https://via.placeholder.com/800x400/667eea/ffffff?text=ML+Cost+Estimator+Dashboard" alt="Dashboard Preview" width="100%">

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Installation Guide](#-installation-guide)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [API Documentation](#-api-documentation)
- [Screenshots](#-screenshots)
- [Team](#-team)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

A **production-ready web application** that leverages Random Forest machine learning to predict early-stage construction costs with exceptional accuracy. Built for Construction Cost Database LLC, this tool provides **Class 5 estimates** (±25% accuracy) for infrastructure projects.

### 🏆 Key Achievement

Target Requirement: MAPE < 25%
Actual Performance: MAPE = 21.97%
Result: ✅ Exceeded target by 3.03%

### 🎓 Academic Context

- **Course:** CSE6748 - Applied Analytics Practicum
- **Institution:** Georgia Institute of Technology
- **Semester:** Fall 2025
- **Client:** Construction Cost Database LLC
- **Dataset:** 17,025 historical projects (2010-2025)

---

## ✨ Key Features

<table>
<tr>
<td width="33%" align="center">

### 📊 Analytics

Real-time exploratory data analysis with interactive visualizations

**Features:**
- Cost distribution charts
- Geographic heatmaps
- Project type analysis
- Statistical insights

</td>
<td width="33%" align="center">

### 💰 Cost Prediction

Instant ML-powered cost estimates with confidence intervals

**Features:**
- 13-feature input form
- ±25% confidence range
- Similar project matching
- Detailed breakdowns

</td>
<td width="33%" align="center">

### 📈 Model Insights

Comprehensive model performance tracking and comparison

**Features:**
- 4 algorithm comparison
- Feature importance
- Learning curves
- Residual analysis

</td>
</tr>
</table>

---

## 🎯 Model Performance

### Current Production Model: Random Forest

<div align="center">

| Metric | Value | Status |
|:------:|:-----:|:------:|
| **Test MAPE** | **21.97%** | ✅ **Target Met** |
| **R² Score** | **0.9463** | 🎯 Excellent |
| **Test MAE** | **$271,543** | 📊 Strong |
| **Test RMSE** | **$412,583** | 📈 Reliable |
| **Dataset Size** | **17,025 projects** | 📦 Large Scale |

</div>

### Model Comparison Results

| Model | CV MAPE | Test MAPE | R² Score | Status |
|-------|:-------:|:---------:|:--------:|:------:|
| **🌲 Random Forest** | 23.30% | **21.97%** ✅ | **0.9463** | 🚀 **Deployed** |
| ⚡ XGBoost | 36.16% | 36.37% | 0.9258 | 📋 Alternative |
| 💡 LightGBM | 36.93% | 37.62% | 0.9232 | 📋 Alternative |
| 📊 Gradient Boosting | 42.48% | 43.75% | 0.9015 | ⚠️ Above Target |

**Why Random Forest?**
- ✅ Best MAPE performance (21.97%)
- ✅ Highest R² score (0.9463)
- ✅ Excellent interpretability
- ✅ Robust to outliers
- ✅ Fast training and prediction

---

## 🛠️ Technology Stack

### Backend

```text
🐍 Python 3.11+         - Core programming language
🌶️ Flask 3.0.0          - Web framework
🤖 scikit-learn 1.3.2   - Machine learning
🐼 Pandas 2.1.4         - Data manipulation
🔢 NumPy 1.26.2         - Numerical computing
📊 Matplotlib 3.8.2     - Visualizations
🎨 Seaborn 0.13.0       - Statistical plots
```

