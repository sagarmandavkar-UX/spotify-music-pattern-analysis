# Data

This repository uses Maharshi Pandya's **Spotify Tracks Dataset**.

- Source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- File: `dataset.csv`
- Snapshot downloaded: 2026-10-08
- Size: 114,000 genre-tag rows, 21 columns, 89,741 unique track IDs
- Coverage: 114 genre labels in the downloaded file
- Kaggle license label: **Database: Open Database, Contents: © Original Authors**

The raw CSV is downloaded unchanged by `python scripts/download_data.py` and is not committed because of its size. It contains 24,259 repeated `track_id` rows because tracks may appear under multiple genre labels. The analysis uses unique track IDs for overall popularity relationships and clustering, while genre comparisons retain genre-tag rows and state that unit explicitly.

`popularity` is a snapshot score, not a historical chart position. This dataset cannot measure how a song changed over time or prove which audio traits caused popularity.

Refresh the source file with `python scripts/download_data.py`.
