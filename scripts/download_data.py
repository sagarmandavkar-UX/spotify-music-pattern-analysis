from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset"


def main() -> None:
    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as bundle:
        bundle.extract("dataset.csv", destination)
    print(f"Downloaded dataset to {destination}")


if __name__ == "__main__":
    main()

