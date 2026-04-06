"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Task 1 — Summary Statistics
# -----------------------------
def compute_summary(df):

     numeric_df = df.select_dtypes(include="number")

     summary = pd.DataFrame({
        "count": numeric_df.count(),
        "mean": numeric_df.mean(),
        "median": numeric_df.median(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max()
     })

     os.makedirs("output", exist_ok=True)
     summary.to_csv("output/summary.csv")

     return summary


# -----------------------------
# Task 2 — Distribution Plots
# -----------------------------
def plot_distributions(df, columns, output_path):
    plt.figure(figsize=(10, 8))

    for i, col in enumerate(columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -----------------------------
# Task 3 — Correlation Heatmap
# -----------------------------
def plot_correlation(df, output_path):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/sample_sales.csv")

    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # Task 1
    compute_summary(df)

    # Task 2 (choose 4 numeric columns)
    numeric_cols = df.select_dtypes(include="number").columns[:4]
    plot_distributions(df, numeric_cols, "output/distributions.png")

    # Task 3
    plot_correlation(df, "output/correlation.png")


