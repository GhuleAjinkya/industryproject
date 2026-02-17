"""
==============================================================================
 MODEL LOADER & INTEGRATION SCRIPT
 Loads GPT-2, Gemma 3 (1B), and DeepSeek-R1-Distill (1.5B)
 and provides guidance on passing them to the project's analysis scripts.
==============================================================================

 Models loaded (lowest parameter versions):
   - GPT-2           : "gpt2"                                     (~124M params)
   - Gemma 3         : "google/gemma-3-1b-pt"                     (~1B params)
   - DeepSeek        : "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" (~1.5B params)

 Repository: https://github.com/GhuleAjinkya/industryproject
==============================================================================
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
)


# ============================================================================
#  1. LOAD GPT-2  (124M parameters — the smallest GPT-2 checkpoint)
# ============================================================================
print("=" * 60)
print(" Loading GPT-2 (124M) ...")
print("=" * 60)

gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

# Also create a HuggingFace text-generation pipeline (used by analyze_bold.py)
device = 0 if torch.cuda.is_available() else -1
gpt2_pipeline = pipeline("text-generation", model=gpt2_model_name, device=device)

print(f"  GPT-2 loaded successfully  |  Device: {'CUDA' if device == 0 else 'CPU'}")


# ============================================================================
#  2. LOAD GEMMA 3  (1B parameters — the smallest Gemma 3 checkpoint)
# ============================================================================
print("\n" + "=" * 60)
print(" Loading Gemma 3 1B ...")
print("=" * 60)

gemma_model_name = "google/gemma-3-1b-pt"
gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
gemma_model.eval()

gemma_pipeline = pipeline(
    "text-generation",
    model=gemma_model,
    tokenizer=gemma_tokenizer,
    device_map="auto" if torch.cuda.is_available() else None,
)

print("  Gemma 3 1B loaded successfully")


# ============================================================================
#  3. LOAD DEEPSEEK  (1.5B parameters — smallest DeepSeek-R1 distilled model)
# ============================================================================
print("\n" + "=" * 60)
print(" Loading DeepSeek-R1-Distill-Qwen 1.5B ...")
print("=" * 60)

deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
deepseek_model.eval()

deepseek_pipeline = pipeline(
    "text-generation",
    model=deepseek_model,
    tokenizer=deepseek_tokenizer,
    device_map="auto" if torch.cuda.is_available() else None,
)

print("  DeepSeek-R1-Distill-Qwen 1.5B loaded successfully")


# ============================================================================
#  IMPORTS FROM THE INDUSTRY PROJECT REPOSITORY
# ============================================================================
#
#  Before running these imports, clone the repository into this directory:
#
#      git clone https://github.com/GhuleAjinkya/industryproject.git
#
#  Then make sure the folder structure looks like:
#
#      ath/
#      ├── load_models.py            <-- this file
#      └── industryproject/
#          ├── BOLDTests/
#          │   └── analyze_bold.py
#          ├── BiasMitigation/
#          │   ├── phase1_baseline.py
#          │   ├── phase2_tracing.py
#          │   └── phase3_mitigation.py
#          └── probability_based.py
#
# ============================================================================

import sys
import os

# Add the repo paths so Python can find the modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industryproject")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "BOLDTests"))
sys.path.insert(0, os.path.join(REPO_ROOT, "BiasMitigation"))

# ── Import the 5 project files ──────────────────────────────────────────────
# NOTE: These files are written as standalone scripts (they execute on import).
#       To use them as modules you MUST refactor them first.  The comments below
#       explain exactly what to change in each file so you can pass model objects
#       as parameters instead of having the models hardcoded inside each script.
# ─────────────────────────────────────────────────────────────────────────────

# import analyze_bold          # from BOLDTests/analyze_bold.py
# import phase1_baseline       # from BiasMitigation/phase1_baseline.py
# import phase2_tracing        # from BiasMitigation/phase2_tracing.py
# import phase3_mitigation     # from BiasMitigation/phase3_mitigation.py
# import probability_based     # from probability_based.py


# ============================================================================
# ============================================================================
#
#                   HOW TO PASS MODELS AS PARAMETERS
#                     TO EACH PROJECT FILE (DETAILED)
#
# ============================================================================
# ============================================================================
#
# All five scripts in the repository currently **hardcode** their model loading.
# To reuse them with GPT-2 / Gemma 3 / DeepSeek you need to make two changes
# per file:
#
#   A) Wrap the script logic in a function that accepts model & tokenizer args.
#   B) Call that function from this loader, passing the objects created above.
#
# Below is a file-by-file guide.
#
# ============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FILE 1:  BOLDTests/analyze_bold.py                                    │
# │  Currently loads:  pipeline('text-generation', model='gpt2')           │
# │  Uses: HuggingFace text-generation pipeline                            │
# │                                                                        │
# │  STEP 1 — Refactor analyze_bold.py                                    │
# │  Change:                                                               │
# │      def main():                                                       │
# │          ...                                                           │
# │          generator = pipeline('text-generation', model='gpt2', ...)    │
# │                                                                        │
# │  To:                                                                   │
# │      def main(generator=None):                                         │
# │          ...                                                           │
# │          if generator is None:                                         │
# │              generator = pipeline('text-generation', model='gpt2',     │
# │                                   device=device)                       │
# │                                                                        │
# │  STEP 2 — Call from this file                                         │
# │      import analyze_bold                                               │
# │                                                                        │
# │      # Run with GPT-2 (default):                                      │
# │      analyze_bold.main()                                               │
# │                                                                        │
# │      # Run with Gemma 3:                                               │
# │      analyze_bold.main(generator=gemma_pipeline)                       │
# │                                                                        │
# │      # Run with DeepSeek:                                              │
# │      analyze_bold.main(generator=deepseek_pipeline)                    │
# │                                                                        │
# │  The function uses  generator(prompt, max_length=50,                   │
# │  num_return_sequences=1, truncation=True)  — any HuggingFace           │
# │  text-generation pipeline is compatible.                               │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FILE 2:  BiasMitigation/phase1_baseline.py                           │
# │  Currently loads:                                                      │
# │      tokenizer = GPT2Tokenizer.from_pretrained("gpt2")                │
# │      model     = GPT2LMHeadModel.from_pretrained("gpt2")              │
# │  Uses: raw model.generate() with tokenizer encode/decode               │
# │                                                                        │
# │  STEP 1 — Refactor phase1_baseline.py                                 │
# │  Wrap everything after the class definition in:                        │
# │                                                                        │
# │      def run_baseline(model=None, tokenizer=None):                     │
# │          if model is None:                                             │
# │              tokenizer = GPT2Tokenizer.from_pretrained("gpt2")         │
# │              model = GPT2LMHeadModel.from_pretrained("gpt2")           │
# │          model.eval()                                                  │
# │          ...  # rest of original script                                │
# │                                                                        │
# │  STEP 2 — Call from this file                                         │
# │      import phase1_baseline                                            │
# │                                                                        │
# │      # Run with GPT-2:                                                 │
# │      phase1_baseline.run_baseline(gpt2_model, gpt2_tokenizer)         │
# │                                                                        │
# │      # Run with Gemma 3:                                               │
# │      phase1_baseline.run_baseline(gemma_model, gemma_tokenizer)        │
# │                                                                        │
# │      # Run with DeepSeek:                                              │
# │      phase1_baseline.run_baseline(deepseek_model, deepseek_tokenizer)  │
# │                                                                        │
# │  NOTE: The script calls model.generate(**inputs, max_new_tokens=15,    │
# │  pad_token_id=tokenizer.eos_token_id).  All three models loaded above  │
# │  support .generate(), so this works directly.  For Gemma/DeepSeek you  │
# │  must also pass the MATCHING tokenizer so that encoding/decoding is    │
# │  correct.                                                              │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FILE 3:  BiasMitigation/phase2_tracing.py                            │
# │  Currently loads:                                                      │
# │      model = HookedTransformer.from_pretrained("gpt2-small")           │
# │  Uses: TransformerLens HookedTransformer for activation tracing        │
# │                                                                        │
# │  ⚠ IMPORTANT: TransformerLens HookedTransformer is NOT a standard      │
# │  HuggingFace model.  It has its own internal architecture and methods  │
# │  like run_with_cache(), W_E, unembed(), to_tokens(), etc.             │
# │                                                                        │
# │  You CANNOT directly pass a raw HuggingFace model here.               │
# │                                                                        │
# │  OPTIONS:                                                              │
# │   A) Use TransformerLens's own model loading (only supports some       │
# │      models — GPT-2, some LLaMA variants, Gemma may work via          │
# │      HookedTransformer.from_pretrained("gemma-2b")):                   │
# │                                                                        │
# │      from transformer_lens import HookedTransformer                    │
# │                                                                        │
# │      def run_tracing(model_name="gpt2-small"):                         │
# │          model = HookedTransformer.from_pretrained(model_name)         │
# │          model.eval()                                                  │
# │          ...                                                           │
# │                                                                        │
# │      # Then call:                                                      │
# │      run_tracing("gpt2-small")     # GPT-2                            │
# │      run_tracing("gemma-2b")       # Gemma (if TransformerLens         │
# │                                    # supports it)                      │
# │                                                                        │
# │   B) Convert a HuggingFace model to a HookedTransformer:              │
# │      hooked = HookedTransformer.from_pretrained(                       │
# │          "gpt2", hf_model=gpt2_model                                  │
# │      )                                                                 │
# │      — Check TransformerLens docs for supported conversions.           │
# │                                                                        │
# │   C) For DeepSeek specifically, TransformerLens likely does NOT        │
# │      support it.  You would need to rewrite the tracing logic using    │
# │      standard PyTorch hooks (register_forward_hook) on the raw         │
# │      HuggingFace model layers instead.                                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FILE 4:  BiasMitigation/phase3_mitigation.py                         │
# │  Currently loads:                                                      │
# │      model = HookedTransformer.from_pretrained("gpt2-small")           │
# │  Uses: TransformerLens hooks to apply a steering vector on layer 11    │
# │                                                                        │
# │  Same constraints as phase2_tracing.py — uses TransformerLens.         │
# │                                                                        │
# │  STEP 1 — Refactor phase3_mitigation.py                               │
# │  Wrap the logic in a function:                                         │
# │                                                                        │
# │      def run_mitigation(model_name="gpt2-small",                       │
# │                         steering_layer=11,                              │
# │                         steering_strength=4.0):                         │
# │          model = HookedTransformer.from_pretrained(model_name)         │
# │          model.eval()                                                  │
# │          ...                                                           │
# │                                                                        │
# │  STEP 2 — Call from this file                                         │
# │      import phase3_mitigation                                          │
# │                                                                        │
# │      phase3_mitigation.run_mitigation("gpt2-small")                    │
# │      phase3_mitigation.run_mitigation("gemma-2b")  # if supported     │
# │                                                                        │
# │  NOTE: The steering hook hardcodes layer 11 and a gender direction     │
# │  vector derived from " he"/" she" embeddings.  For non-GPT-2 models   │
# │  you must:                                                             │
# │    1) Verify the token IDs for " he"/" she" in the new tokenizer.     │
# │    2) Adjust the target layer number (layer 11 is specific to GPT-2's │
# │       12-layer architecture; Gemma 1B has 18 layers, DeepSeek 1.5B    │
# │       has 28 layers — re-run phase2_tracing first to find the most    │
# │       biased layer for each model).                                    │
# │    3) Tune steering_strength (4.0 may be too strong or weak for       │
# │       different model scales).                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FILE 5:  probability_based.py                                         │
# │  Currently loads:                                                      │
# │      from deepeval.models import GeminiModel                           │
# │      model = GeminiModel(...)     (hardcoded — not visible in snippet) │
# │  Uses: deepeval BiasMetric with model.generate()                       │
# │                                                                        │
# │  This file uses the DeepEval framework.  DeepEval's BiasMetric and    │
# │  test_stereotypical_preference() expect an object with a .generate()   │
# │  method that takes a string prompt and returns a string response.      │
# │                                                                        │
# │  STEP 1 — Create a wrapper class that implements DeepEval's            │
# │  interface using any HuggingFace model:                                │
# │                                                                        │
# │      from deepeval.models import DeepEvalBaseLLM                       │
# │                                                                        │
# │      class HFModelWrapper(DeepEvalBaseLLM):                            │
# │          def __init__(self, model, tokenizer, model_name):             │
# │              self.model = model                                        │
# │              self.tokenizer = tokenizer                                │
# │              self._model_name = model_name                             │
# │                                                                        │
# │          def load_model(self):                                         │
# │              return self.model                                         │
# │                                                                        │
# │          def generate(self, prompt: str, **kwargs) -> str:             │
# │              inputs = self.tokenizer(prompt, return_tensors="pt")      │
# │              inputs = {k: v.to(self.model.device) for k,v             │
# │                        in inputs.items()}                              │
# │              with torch.no_grad():                                     │
# │                  output_ids = self.model.generate(                     │
# │                      **inputs, max_new_tokens=100,                     │
# │                      pad_token_id=self.tokenizer.eos_token_id         │
# │                  )                                                     │
# │              return self.tokenizer.decode(                             │
# │                  output_ids[0][inputs['input_ids'].shape[-1]:],        │
# │                  skip_special_tokens=True                              │
# │              )                                                         │
# │                                                                        │
# │          async def a_generate(self, prompt: str, **kwargs) -> str:     │
# │              return self.generate(prompt, **kwargs)                    │
# │                                                                        │
# │          def get_model_name(self) -> str:                              │
# │              return self._model_name                                   │
# │                                                                        │
# │  STEP 2 — Refactor probability_based.py                               │
# │  Change:                                                               │
# │      model = GeminiModel(...)                                          │
# │  To:                                                                   │
# │      def run_probability_test(model):                                  │
# │          bias_metric = BiasMetric(model=model, threshold=0.5)          │
# │          ...  # rest of the script                                     │
# │                                                                        │
# │  STEP 3 — Call from this file                                         │
# │      import probability_based                                          │
# │                                                                        │
# │      gpt2_eval = HFModelWrapper(gpt2_model, gpt2_tokenizer, "gpt2")   │
# │      probability_based.run_probability_test(gpt2_eval)                 │
# │                                                                        │
# │      gemma_eval = HFModelWrapper(gemma_model, gemma_tokenizer,         │
# │                                  "gemma-3-1b")                         │
# │      probability_based.run_probability_test(gemma_eval)                │
# │                                                                        │
# │      ds_eval = HFModelWrapper(deepseek_model, deepseek_tokenizer,      │
# │                               "deepseek-r1-1.5b")                     │
# │      probability_based.run_probability_test(ds_eval)                   │
# │                                                                        │
# │  The key function test_stereotypical_preference(model, stereo, anti)   │
# │  calls model.generate(prompt).  The wrapper above makes this work     │
# │  for any HuggingFace model.                                           │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ============================================================================
#  QUICK-START EXAMPLE  —  Run all 5 scripts with Gemma 3
# ============================================================================
#
#   1.  git clone https://github.com/GhuleAjinkya/industryproject.git
#   2.  pip install transformers torch deepeval detoxify vaderSentiment
#              transformer-lens pandas
#   3.  Refactor each script as described above (wrap in functions).
#   4.  Uncomment the imports at the top of this file.
#   5.  Then:
#
#       # BOLD analysis with Gemma 3
#       analyze_bold.main(generator=gemma_pipeline)
#
#       # Baseline bias audit with Gemma 3
#       phase1_baseline.run_baseline(gemma_model, gemma_tokenizer)
#
#       # Probability-based CrowS-Pairs test with Gemma 3
#       gemma_eval = HFModelWrapper(gemma_model, gemma_tokenizer, "gemma-3-1b")
#       probability_based.run_probability_test(gemma_eval)
#
#       # For tracing & mitigation (phase2 & phase3), check TransformerLens
#       # compatibility or rewrite with PyTorch hooks.
#
# ============================================================================


# ============================================================================
#  VERIFICATION — quick sanity check that all three models can generate text
# ============================================================================
if __name__ == "__main__":
    test_prompt = "The scientist discovered that"

    print("\n" + "=" * 60)
    print(" VERIFICATION: Generating text with each model")
    print("=" * 60)

    # --- GPT-2 ---
    print("\n[GPT-2]")
    gpt2_out = gpt2_pipeline(test_prompt, max_length=40, num_return_sequences=1, truncation=True)
    print(f"  {gpt2_out[0]['generated_text']}")

    # --- Gemma 3 ---
    print("\n[Gemma 3 1B]")
    gemma_out = gemma_pipeline(test_prompt, max_length=40, num_return_sequences=1, truncation=True)
    print(f"  {gemma_out[0]['generated_text']}")

    # --- DeepSeek ---
    print("\n[DeepSeek-R1-Distill 1.5B]")
    ds_out = deepseek_pipeline(test_prompt, max_length=40, num_return_sequences=1, truncation=True)
    print(f"  {ds_out[0]['generated_text']}")

    print("\n" + "=" * 60)
    print(" All models loaded and verified successfully!")
    print("=" * 60)
