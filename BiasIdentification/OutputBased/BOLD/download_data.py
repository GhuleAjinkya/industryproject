import requests

urls = {
    "gender_prompt.json": "https://raw.githubusercontent.com/amazon-research/bold/main/prompts/gender_prompt.json",
    "race_prompt.json": "https://raw.githubusercontent.com/amazon-research/bold/main/prompts/race_prompt.json"
}

for filename, url in urls.items():
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}: {response.status_code}")
