import kagglehub
import shutil
from pathlib import Path

path = kagglehub.competition_download("titanic")

sourse = Path(path)
destination = Path("data/raw")

destination.mkdir(parents=True, exist_ok=True)

for file in sourse.iterdir():
    shutil.copy2(file, destination / file.name)

print(f"Data downloaded to: {destination}")

