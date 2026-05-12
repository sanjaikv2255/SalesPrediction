
# =============================================================
# SALES PREDICTION USING PYTHON
# Author: Sanjai KV | GitHub: sanjaikv2255
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.alpha": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
TV_COLOR    = "#4C72B0"
RADIO_COLOR = "#DD8452"
NEWS_COLOR  = "#55A868"
COLORS      = [TV_COLOR, RADIO_COLOR, NEWS_COLOR]

print("=" * 58)
print("  SALES PREDICTION — CodeAlpha Internship")
print("=" * 58)

# ── 1. Load & Explore ─────────────────────────────────────────
df = pd.read_csv("Advertising.csv", index_col=0)
print(f"\n📌 Shape: {df.shape}")
print(f"📌 Columns: {list(df.columns)}")
print(f"\n📌 First 5 rows:")
print(df.head())
print(f"\n📌 Missing values: {df.isnull().sum().sum()}")
print(f"\n📌 Statistical Summary:")
print(df.describe().round(2))

# ── 2. Feature Engineering ────────────────────────────────────
# Interaction features — TV*Radio often has synergy effect
df["TV_Radio"]       = df["TV"] * df["Radio"]
df["TV_Newspaper"]   = df["TV"] * df["Newspaper"]
df["Total_Ad_Spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
df["TV_Share"]       = df["TV"] / df["Total_Ad_Spend"]

features_base = ["TV", "Radio", "Newspaper"]
features_full = ["TV", "Radio", "Newspaper", "TV_Radio", "Total_Ad_Spend", "TV_Share"]

X_base = df[features_base]
X_full = df[features_full]
y      = df["Sales"]

# ── 3. Train-Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42)
print(f"\n📌 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 58)
print("  MODEL EVALUATION RESULTS")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_full, y, cv=5, scoring="r2").mean()
    results[name] = {"model": model, "y_pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2, "cv_r2": cv}
    print(f"\n  {name}")
    print(f"  {'─'*40}")
    print(f"  MAE     : {mae:.4f}  (Sales Units)")
    print(f"  RMSE    : {rmse:.4f}  (Sales Units)")
    print(f"  R²      : {r2:.4f}  ({r2*100:.1f}% variance explained)")
    print(f"  CV R²   : {cv:.4f}  (5-fold cross-validation)")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['r2']:.4f})")

# ── 5. Plot 1 — Sales Distribution & Ad Spend Overview ────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Sales distribution
ax = axes[0]
ax.hist(df["Sales"], bins=20, color=TV_COLOR, edgecolor="white", alpha=0.85)
ax.axvline(df["Sales"].mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean: {df['Sales'].mean():.1f}")
ax.set_title("Sales Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Sales (units)"); ax.set_ylabel("Count")
ax.legend(fontsize=9)

# Ad spend comparison
ax2 = axes[1]
means = [df["TV"].mean(), df["Radio"].mean(), df["Newspaper"].mean()]
bars = ax2.bar(["TV", "Radio", "Newspaper"], means, color=COLORS, edgecolor="white", width=0.5)
for bar, val in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"${val:.1f}K", ha="center", fontsize=11, fontweight="bold")
ax2.set_title("Average Ad Spend by Channel", fontsize=12, fontweight="bold")
ax2.set_ylabel("Average Spend ($K)")

fig.suptitle("Sales & Advertising Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/01_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Plot 1 saved: Overview")

# ── 6. Plot 2 — Ad Spend vs Sales (3 scatter plots) ──────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
channels = [("TV", TV_COLOR), ("Radio", RADIO_COLOR), ("Newspaper", NEWS_COLOR)]

for ax, (ch, color) in zip(axes, channels):
    ax.scatter(df[ch], df["Sales"], alpha=0.6, color=color, edgecolors="white", s=45)
    # Regression line
    z = np.polyfit(df[ch], df["Sales"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[ch].min(), df[ch].max(), 100)
    ax.plot(x_line, p(x_line), color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    corr = df[ch].corr(df["Sales"])
    ax.set_title(f"{ch} vs Sales\n(r = {corr:.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{ch} Ad Spend ($K)"); ax.set_ylabel("Sales (units)")

fig.suptitle("Advertising Spend vs Sales by Channel", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_adspend_vs_sales.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot 2 saved: Ad Spend vs Sales")

# ── 7. Plot 3 — Correlation Heatmap ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[["TV", "Radio", "Newspaper", "Sales", "Total_Ad_Spend", "TV_Radio"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, ax=ax,
            linewidths=0.5, annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot 3 saved: Correlation Heatmap")

# ── 8. Plot 4 — Model Performance Comparison ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
names = list(results.keys())
r2s   = [results[n]["r2"]   for n in names]
maes  = [results[n]["mae"]  for n in names]
cvs   = [results[n]["cv_r2"] for n in names]

for ax, vals, label, fmt in zip(axes,
        [r2s, maes, cvs],
        ["R² Score", "MAE (Sales Units)", "CV R² (5-fold)"],
        [".3f", ".3f", ".3f"]):
    bars = ax.bar(names, vals, color=COLORS, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:{fmt}}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(label)
    ax.set_title(f"{label} Comparison", fontweight="bold", fontsize=11)
    ax.set_xticklabels(names, fontsize=8.5, rotation=10)

fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot 4 saved: Model Comparison")

# ── 9. Plot 5 — Actual vs Predicted (Best Model) ─────────────
best_pred = results[best_name]["y_pred"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Actual vs Predicted scatter
ax = axes[0]
ax.scatter(y_test, best_pred, alpha=0.7, color=TV_COLOR,
           edgecolors="white", linewidths=0.4, s=55)
min_val = min(y_test.min(), best_pred.min()) - 0.5
max_val = max(y_test.max(), best_pred.max()) + 0.5
ax.plot([min_val, max_val], [min_val, max_val],
        color="red", linestyle="--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Sales", fontsize=11)
ax.set_ylabel("Predicted Sales", fontsize=11)
ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"R² = {results[best_name]['r2']:.4f}",
        transform=ax.transAxes, fontsize=11, color=TV_COLOR, fontweight="bold")

# Residuals plot
ax2 = axes[1]
residuals = y_test.values - best_pred
ax2.scatter(best_pred, residuals, alpha=0.65, color=RADIO_COLOR,
            edgecolors="white", linewidths=0.4, s=55)
ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Predicted Sales", fontsize=11)
ax2.set_ylabel("Residuals", fontsize=11)
ax2.set_title("Residual Plot", fontsize=11, fontweight="bold")

fig.suptitle(f"Prediction Analysis — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/05_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot 5 saved: Actual vs Predicted")

# ── 10. Plot 6 — Feature Importance + Ad ROI Analysis ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature Importance (Random Forest)
rf = results["Random Forest"]["model"]
imp = pd.Series(rf.feature_importances_, index=features_full).sort_values()
imp_colors = ["#4C72B0" if v == imp.max() else "#a8c4e0" for v in imp.values]
axes[0].barh(imp.index, imp.values, color=imp_colors, edgecolor="white", height=0.55)
for i, val in enumerate(imp.values):
    axes[0].text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
axes[0].set_xlabel("Importance Score", fontsize=10)
axes[0].set_title("Feature Importance — Random Forest", fontsize=11, fontweight="bold")

# ROI: Sales per $1K spent on each channel (Linear Regression coefficients)
lr = results["Linear Regression"]["model"]
lr_base = LinearRegression().fit(df[features_base], y)
coefs = pd.Series(lr_base.coef_, index=features_base)
bar_colors = [TV_COLOR, RADIO_COLOR, NEWS_COLOR]
bars = axes[1].bar(coefs.index, coefs.values, color=bar_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, coefs.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"+{val:.2f}", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Sales Increase per $1K Spend", fontsize=10)
axes[1].set_title("Ad Channel ROI (Linear Regression)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Advertising Channel")

fig.suptitle("Feature Importance & Channel ROI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/06_importance_and_roi.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot 6 saved: Feature Importance & ROI")

# ── 11. Summary ───────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FINAL RESULTS SUMMARY")
print("=" * 58)
print(f"\n  {'Model':<25} {'R²':>8} {'MAE':>8} {'CV R²':>8}")
print(f"  {'─'*52}")
for name in results:
    r = results[name]
    print(f"  {name:<25} {r['r2']:>8.4f} {r['mae']:>7.4f} {r['cv_r2']:>8.4f}")

lr_base2 = LinearRegression().fit(df[features_base], y)
coefs2   = pd.Series(lr_base2.coef_, index=features_base)
print(f"\n  📢 Ad Channel ROI (sales units per $1K spent):")
for ch, val in coefs2.items():
    print(f"     {ch:<12}: +{val:.3f} units")

print(f"\n🏆 Best Model : {best_name}")
print(f"   R²         = {results[best_name]['r2']:.4f} ({results[best_name]['r2']*100:.1f}%)")
print(f"   MAE        = {results[best_name]['mae']:.4f} sales units")
print(f"\n✅ All 6 plots saved to /plots/")
print("✅ Task 4 Complete! Ready for GitHub.\n")
