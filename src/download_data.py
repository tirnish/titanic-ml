import kagglehub
import shutil
from pathlib import Path

path = kagglehub.competition_download("titanic")

sourсe = Path(path)
destination = Path("data/raw")

destination.mkdir(parents=True, exist_ok=True)

for file in sourсe.iterdir():
    shutil.copy2(file, destination / file.name)

print(f"Data downloaded to: {destination}")

