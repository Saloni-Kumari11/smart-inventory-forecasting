<div align="center">

# 📦 Smart Inventory Forecasting

### AI-Powered Demand Prediction & Inventory Optimization for Small Business

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006600?style=for-the-badge)](https://xgboost.readthedocs.io)

*Helping small businesses predict demand, prevent stockouts, and optimize inventory using machine learning.*

</div>

---
## Live Link
https://smart-inventory-forecasting-8gz88ldjao29hgkztuddhv.streamlit.app/

## 🎯 Problem Statement

Small businesses lose **$1.1 trillion annually** due to overstocking and stockouts. Most cannot afford enterprise inventory systems. This project provides an **AI-powered, free, open-source** solution that:

- 📈 **Predicts future demand** using 7 machine learning models
- 📦 **Optimizes inventory levels** with safety stock and EOQ calculations
- ⚠️ **Prevents stockouts** through proactive reorder alerts
- 🎯 **Analyzes scenarios** (optimistic / normal / pessimistic)
- 🔍 **Detects anomalies** in sales patterns automatically

---

## ✨ Key Features

### 🤖 Machine Learning Pipeline

| Feature | Description |
|---------|-------------|
| **7 ML Models** | Linear, Ridge, Lasso, Random Forest, Extra Trees, Gradient Boosting, XGBoost |
| **55+ Features** | Calendar, cyclical, lag, rolling, interaction, and acceleration features |
| **Auto Model Selection** | Automatically picks the best performing model |
| **Cross Validation** | Time-series aware cross-validation for reliable metrics |

### 📊 Interactive Dashboard (6 Tabs)

| Tab | What It Shows |
|-----|---------------|
| 📊 **Overview** | Total sales, revenue trends, category analysis, product rankings |
| 🔍 **Product** | Individual product analysis, promotions, anomaly detection |
| 🤖 **Models** | All 7 models compared, predictions vs actual, feature importance |
| 📈 **Forecast** | Future demand prediction, confidence bands, downloadable CSV |
| 📦 **Inventory** | Stock projection, reorder alerts, holding and stockout costs |
| 🎯 **Scenarios** | Optimistic, Normal, Pessimistic demand scenarios |

### 📦 Inventory Intelligence

| Feature | Description |
|---------|-------------|
| **Safety Stock** | Calculated using service level approach with demand variability |
| **Reorder Point** | Dynamic reorder point based on lead time and safety stock |
| **EOQ** | Economic Order Quantity to minimize total inventory costs |
| **Stockout Projection** | Predicts exact day when stock will run out |
| **Cost Analysis** | Estimates holding costs and potential stockout costs |
| **Risk Assessment** | 4 levels: 🟢 LOW, 🟡 MEDIUM, 🟠 HIGH, 🔴 CRITICAL |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/risin-work/smart-inventory-forecasting.git
cd smart-inventory-forecasting

# 2. Install dependencies
pip install -r requirements.txt

