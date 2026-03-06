import requests
from pathlib import Path

DEST = Path(__file__).resolve().parents[3] / "Datasets"
DEST.mkdir(parents=True, exist_ok=True)

urls = {
    "gender_prompt.json": "https://raw.githubusercontent.com/amazon-research/bold/main/prompts/gender_prompt.json",
    "race_prompt.json": "https://raw.githubusercontent.com/amazon-research/bold/main/prompts/race_prompt.json"
}

for filename, url in urls.items():
    out_path = DEST / filename
    print(f"Downloading {filename}...")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        out_path.write_text(resp.text, encoding="utf-8")
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
