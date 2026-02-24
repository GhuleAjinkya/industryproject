import json
import random

def sample_prompts(file_path, num_samples=20):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_prompts = []
    for category, subjects in data.items():
        for subject, prompts in subjects.items():
            for p in prompts:
                all_prompts.append({
                    "category": category,
                    "subject": subject,
                    "prompt": p
                })
    
    return random.sample(all_prompts, min(num_samples, len(all_prompts)))

if __name__ == "__main__":
    gender_samples = sample_prompts('gender_prompt.json', 20)
    race_samples = sample_prompts('race_prompt.json', 20)
    
    samples = {
        "gender": gender_samples,
        "race": race_samples
    }
    
    with open('sampled_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=4)
    
    print(f"Sampled {len(gender_samples)} gender prompts and {len(race_samples)} race prompts.")
