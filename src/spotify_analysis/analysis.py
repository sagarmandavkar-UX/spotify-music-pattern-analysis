from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

REQUIRED = {"track_id", "artists", "track_name", "popularity", "track_genre", *AUDIO_FEATURES}


def load_tracks(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not data["popularity"].between(0, 100).all():
        raise ValueError("Popularity must be between 0 and 100")
    if data["track_id"].isna().any():
        raise ValueError("Track IDs cannot be missing")
    return data


def unique_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic record per Spotify track ID for track-level analysis."""
    return data.sort_values(["track_id", "track_genre"]).drop_duplicates("track_id", keep="first").reset_index(drop=True)


def high_popularity_flag(data: pd.DataFrame, quantile: float = 0.90) -> tuple[pd.Series, float]:
    threshold = float(data["popularity"].quantile(quantile))
    return data["popularity"].ge(threshold), threshold

