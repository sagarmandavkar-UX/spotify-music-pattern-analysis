from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from spotify_analysis.analysis import AUDIO_FEATURES, high_popularity_flag, load_tracks, unique_tracks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/dataset.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BLUE, GOLD, PURPLE, OLIVE, ORANGE = "#2463A9", "#C58B24", "#8A5A9B", "#6F7C3E", "#D26A3A"


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def main(quick: bool = False) -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rows = load_tracks(DATA)
    tracks = unique_tracks(rows)
    high_flag, threshold = high_popularity_flag(tracks)

    correlations = tracks[["popularity", *AUDIO_FEATURES]].corr(method="spearman")["popularity"].drop("popularity")
    correlation_table = correlations.rename("spearman_correlation").sort_values(key=abs, ascending=False).reset_index()
    correlation_table = correlation_table.rename(columns={"index": "feature"})
    correlation_table.to_csv(REPORTS / "popularity_correlations.csv", index=False)

    genre_profile = rows.groupby("track_genre", as_index=False).agg(
        genre_tag_rows=("track_id", "size"),
        unique_tracks=("track_id", "nunique"),
        mean_popularity=("popularity", "mean"),
        median_popularity=("popularity", "median"),
        mean_danceability=("danceability", "mean"),
        mean_energy=("energy", "mean"),
        mean_acousticness=("acousticness", "mean"),
        mean_instrumentalness=("instrumentalness", "mean"),
        mean_valence=("valence", "mean"),
    )
    genre_profile.to_csv(REPORTS / "genre_profile.csv", index=False)

    comparison = tracks.assign(high_popularity=high_flag).groupby("high_popularity")[AUDIO_FEATURES].mean().T
    comparison.columns = ["other_tracks", "high_popularity_tracks"]
    comparison["difference"] = comparison["high_popularity_tracks"] - comparison["other_tracks"]
    comparison.reset_index().rename(columns={"index": "feature"}).to_csv(
        REPORTS / "high_popularity_audio_profile.csv", index=False
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(tracks[AUDIO_FEATURES])
    rng = np.random.default_rng(42)
    silhouette_n = min(5_000 if quick else 12_000, len(tracks))
    silhouette_idx = rng.choice(len(tracks), size=silhouette_n, replace=False)
    k_values = range(3, 6 if quick else 8)
    silhouette_rows = []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled[silhouette_idx])
        silhouette_rows.append({"k": k, "silhouette_score": silhouette_score(scaled[silhouette_idx], labels, sample_size=min(5_000, silhouette_n), random_state=42)})
    silhouette = pd.DataFrame(silhouette_rows)
    best_k = int(silhouette.loc[silhouette["silhouette_score"].idxmax(), "k"])
    silhouette.to_csv(REPORTS / "cluster_selection.csv", index=False)

    cluster_model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    tracks = tracks.copy()
    tracks["cluster"] = cluster_model.fit_predict(scaled)
    cluster_profile = tracks.groupby("cluster").agg(
        tracks=("track_id", "size"),
        mean_popularity=("popularity", "mean"),
        **{f"mean_{feature}": (feature, "mean") for feature in AUDIO_FEATURES},
    ).reset_index()
    cluster_profile.to_csv(REPORTS / "cluster_profile.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    plot_n = min(6_000, len(tracks))
    plot_idx = rng.choice(len(tracks), size=plot_n, replace=False)
    pca_output = pd.DataFrame({"pc1": coordinates[plot_idx, 0], "pc2": coordinates[plot_idx, 1], "cluster": tracks.loc[plot_idx, "cluster"].to_numpy(), "popularity": tracks.loc[plot_idx, "popularity"].to_numpy()})
    pca_output.to_csv(REPORTS / "pca_sample.csv", index=False)

    summary = {
        "genre_tag_rows": int(len(rows)),
        "unique_tracks": int(len(tracks)),
        "genre_labels": int(rows["track_genre"].nunique()),
        "duplicate_track_rows": int(rows.duplicated("track_id").sum()),
        "missing_metadata_cells": int(rows[["artists", "album_name", "track_name"]].isna().sum().sum()),
        "mean_popularity": float(tracks["popularity"].mean()),
        "high_popularity_threshold": threshold,
        "high_popularity_track_count": int(high_flag.sum()),
        "strongest_audio_correlation_feature": str(correlation_table.iloc[0]["feature"]),
        "strongest_audio_correlation": float(correlation_table.iloc[0]["spearman_correlation"]),
        "selected_clusters": best_k,
        "selected_silhouette": float(silhouette.loc[silhouette["k"].eq(best_k), "silhouette_score"].iloc[0]),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "caveat": "Popularity is a snapshot score; this dataset does not contain historical trend observations or causal evidence.",
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(tracks["popularity"], bins=np.arange(0, 102, 4), color=BLUE, edgecolor="white")
    ax.axvline(threshold, color=ORANGE, linestyle="--", lw=2, label=f"90th percentile threshold = {threshold:.0f}")
    ax.set(title="Spotify popularity scores across unique tracks", xlabel="Popularity score (0–100)", ylabel="Unique tracks")
    ax.legend(frameon=False)
    save_figure(FIGURES / "popularity_distribution.png")

    top_genres = genre_profile.nlargest(12, "mean_popularity").sort_values("mean_popularity")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_genres["track_genre"], top_genres["mean_popularity"], color=BLUE)
    ax.set(title="Genres with the highest mean popularity", xlabel="Mean popularity across genre-tag rows", ylabel="", xlim=(0, 100))
    for i, value in enumerate(top_genres["mean_popularity"]):
        ax.text(value + 1, i, f"{value:.1f}", va="center")
    save_figure(FIGURES / "top_genres.png")

    corr_plot = correlation_table.sort_values("spearman_correlation")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [ORANGE if value < 0 else BLUE for value in corr_plot["spearman_correlation"]]
    ax.barh(corr_plot["feature"], corr_plot["spearman_correlation"], color=colors)
    ax.axvline(0, color="#333333", lw=1)
    ax.set(title="Audio-feature association with track popularity", xlabel="Spearman correlation with popularity", ylabel="")
    save_figure(FIGURES / "popularity_correlations.png")

    bounded = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
    profile_plot = comparison.loc[bounded, ["other_tracks", "high_popularity_tracks"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(profile_plot))
    width = 0.38
    ax.bar(x - width / 2, profile_plot["other_tracks"], width, color="#B9C3CF", label="Other unique tracks")
    ax.bar(x + width / 2, profile_plot["high_popularity_tracks"], width, color=GOLD, label=f"Popularity ≥ {threshold:.0f}")
    ax.set_xticks(x, [label.replace("ness", "ness\n") if label == "instrumentalness" else label for label in profile_plot.index], rotation=20, ha="right")
    ax.set(title="Average audio profile: high-popularity vs other tracks", ylabel="Mean feature value (0–1)", xlabel="", ylim=(0, 1))
    ax.legend(frameon=False)
    save_figure(FIGURES / "high_popularity_profile.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = [BLUE, GOLD, PURPLE, OLIVE, ORANGE, "#4C9F8A", "#7C6DAA"]
    for cluster in sorted(pca_output["cluster"].unique()):
        group = pca_output[pca_output["cluster"].eq(cluster)]
        ax.scatter(group["pc1"], group["pc2"], s=9, alpha=0.35, color=palette[int(cluster) % len(palette)], label=f"Cluster {cluster}")
    ax.set(title=f"Audio-feature clusters projected into two dimensions (k={best_k})", xlabel="Principal component 1", ylabel="Principal component 2")
    ax.legend(frameon=False, markerscale=2, ncols=2)
    save_figure(FIGURES / "audio_clusters_pca.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
