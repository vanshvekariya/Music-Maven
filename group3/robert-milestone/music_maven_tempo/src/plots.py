from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_onset_plot(onset_env: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 4.8))
    plt.plot(onset_env)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Onset Strength")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_histogram(df: pd.DataFrame, out_path: Path, column: str, title: str, bins: int = 30) -> None:
    plt.figure(figsize=(9, 5.5))
    values = df[column].dropna().astype(float)
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_scatter(df: pd.DataFrame, out_path: Path, x: str, y: str, title: str) -> None:
    plt.figure(figsize=(7.2, 7.2))
    plt.scatter(df[x], df[y], alpha=0.45)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_genre_boxplot(df: pd.DataFrame, out_path: Path, column: str, title: str) -> None:
    grouped = [group[column].dropna().astype(float).values for _, group in df.groupby("genre")]
    labels = [name for name, _ in df.groupby("genre")]
    plt.figure(figsize=(12, 6))
    plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(column.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
