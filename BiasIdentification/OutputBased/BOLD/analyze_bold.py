import json
import pandas as pd
from transformers import pipeline, set_seed
from detoxify import Detoxify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

def main():
    try:
        with open('sampled_prompts.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print("sampled_prompts.json not found. Please run sample_prompts.py first.")
        return

    print("Loading models...")
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline('text-generation', model='gpt2', device=device)
    detoxifier = Detoxify('original')
    analyzer = SentimentIntensityAnalyzer()
    
    set_seed(42)
    results = []

    for category, category_samples in samples.items():
        print(f"Analyzing {category}...")
        for item in category_samples:
            prompt = item['prompt']
            
            completions = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
            generated_text = completions[0]['generated_text']

            completion_only = generated_text[len(prompt):].strip()
            if not completion_only:
                completion_only = generated_text
            
            tox_scores = detoxifier.predict(completion_only)
            
            sent_scores = analyzer.polarity_scores(completion_only)
            
            results.append({
                "category": category,
                "subject": item['subject'],
                "prompt": prompt,
                "completion": completion_only,
                "toxicity": tox_scores['toxicity'],
                "severe_toxicity": tox_scores['severe_toxicity'],
                "obscene": tox_scores['obscene'],
                "threat": tox_scores['threat'],
                "insult": tox_scores['insult'],
                "identity_attack": tox_scores['identity_attack'],
                "sentiment_compound": sent_scores['compound'],
                "sentiment_pos": sent_scores['pos'],
                "sentiment_neu": sent_scores['neu'],
                "sentiment_neg": sent_scores['neg']
            })

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("Analysis complete. Results saved to results.csv")

if __name__ == "__main__":
    main()
