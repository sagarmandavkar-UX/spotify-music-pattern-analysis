from pathlib import Path

import pandas as pd

from spotify_analysis.analysis import AUDIO_FEATURES, high_popularity_flag, load_tracks, unique_tracks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/dataset.csv"


def test_dataset_contract() -> None:
    rows = load_tracks(DATA)
    assert rows.shape == (114000, 20)
    assert rows["track_id"].nunique() == 89741
    assert rows["track_genre"].nunique() == 114
    assert rows["popularity"].between(0, 100).all()


def test_unique_tracks_are_deterministic() -> None:
    tracks = unique_tracks(load_tracks(DATA))
    assert len(tracks) == 89741
    assert not tracks["track_id"].duplicated().any()
    assert tracks[AUDIO_FEATURES].notna().all().all()


def test_high_popularity_flag() -> None:
    sample = pd.DataFrame({"popularity": list(range(10))})
    flag, threshold = high_popularity_flag(sample, 0.8)
    assert threshold == 7.2
    assert flag.sum() == 2

