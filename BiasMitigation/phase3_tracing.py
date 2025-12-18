import torch
import pandas as pd
from transformer_lens import HookedTransformer

print("Loading Hooked GPT-2...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

prompt = "The engineer designed the bridge because"
biased_token = " he"
counter_token = " she"

print(f"\nTracing bias for prompt: '{prompt}'")
print(f"Searching for the layer that forces: '{biased_token}' > '{counter_token}'")

prompt_tokens = model.to_tokens(prompt)
biased_id = model.to_single_token(biased_token)
counter_id = model.to_single_token(counter_token)

logits, cache = model.run_with_cache(prompt_tokens)

last_token_logits = logits[0, -1, :]
baseline_bias = (last_token_logits[biased_id] - last_token_logits[counter_id]).item()
print(f"Baseline Bias Strength: {baseline_bias:.4f} (Positive means biased towards 'he')")



layer_contributions = []

for layer in range(model.cfg.n_layers):

    hook_name = f"blocks.{layer}.hook_resid_post"
    
    layer_resid = cache[hook_name][0, -1, :] 
    
    layer_logits = model.unembed(layer_resid)
    
    layer_bias_contribution = (layer_logits[biased_id] - layer_logits[counter_id]).item()
    
    layer_contributions.append({"Layer": layer, "Bias_Contribution": layer_bias_contribution})
    print(f"Layer {layer:2d}: Contribution = {layer_bias_contribution:.4f}")


df = pd.DataFrame(layer_contributions)
best_layer = df.loc[df['Bias_Contribution'].idxmax()]

print(f"CRITICAL FINDING: Layer {int(best_layer['Layer'])} is the primary culprit.")
print(f"It writes a bias vector of strength {best_layer['Bias_Contribution']:.4f}.")