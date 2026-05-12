# 📈 Sales Prediction Using Python

**Data Science Project 4**

A machine learning project to predict product sales based on advertising spend across TV, Radio, and Newspaper channels — with actionable business insights on which channel delivers the best ROI.

---

## 📁 Dataset

- **File:** `Advertising.csv`
- **Rows:** 200
- **Features:** TV, Radio, Newspaper (ad spend in $K)
- **Target:** Sales (units)
- **Missing values:** None

---

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| `TV_Radio` | Interaction term (TV × Radio synergy) |
| `Total_Ad_Spend` | TV + Radio + Newspaper combined |
| `TV_Share` | TV spend as proportion of total budget |

---

## 🤖 Model Results

| Model | R² Score | MAE | CV R² (5-fold) |
|-------|----------|-----|----------------|
| **Linear Regression** | **0.9937** | **0.3548** | **0.9881** |
| Random Forest | 0.9904 | 0.4390 | 0.9883 |
| Gradient Boosting | 0.9887 | 0.4600 | 0.9884 |

🏆 **Best Model: Linear Regression** — explains **99.4%** of sales variance

---

## 📢 Business Insights

| Channel | Correlation with Sales | ROI (units per $1K) |
|---------|----------------------|----------------------|
| TV | r = 0.78 (Strong) | +0.046 units |
| Radio | r = 0.58 (Moderate) | **+0.189 units** ⭐ |
| Newspaper | r = 0.23 (Weak) | ~0.000 units ❌ |

- **Radio has the best ROI** — each $1K spent yields +0.189 sales units
- **TV has the highest absolute impact** on sales volume
- **Newspaper advertising has near-zero impact** — budget should be reallocated

---

## 📈 Visualizations

| Plot | Description |
|------|-------------|
| `01_overview.png` | Sales distribution + average ad spend per channel |
| `02_adspend_vs_sales.png` | Scatter plots with regression lines for all 3 channels |
| `03_correlation_heatmap.png` | Feature correlation heatmap |
| `04_model_comparison.png` | R², MAE and CV R² comparison across all models |
| `05_actual_vs_predicted.png` | Actual vs Predicted + Residuals plot |
| `06_importance_and_roi.png` | Feature importance + channel ROI bar chart |

---

## 🚀 How to Run

```bash
git clone https://github.com/sanjaikv2255/SalesPrediction
cd SalesPrediction

pip install pandas numpy matplotlib seaborn scikit-learn

jupyter notebook sales_prediction.ipynb
# or
python sales_prediction.py
```

---

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

---

## 📌 Key Conclusions

- All 3 models scored **above 98% accuracy** — the dataset has very clean linear relationships
- **Linear Regression is the best model** here because the data is inherently linear
- Businesses should **increase Radio budget** for best return on investment
- **Cut Newspaper spending** — it has no measurable impact on sales
- **TV + Radio combination** is the most effective advertising strategy

---

*Made by [Sanjai KV](https://github.com/sanjaikv2255)
