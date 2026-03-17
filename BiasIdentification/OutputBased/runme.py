

import argparse
import gc
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from BOLD.analyze_bold import run_bold
from CEAT.ceat import run_ceat_interventional, run_ceat_counterfactual
from CrowS_Pairs.crows_pairs import run_crows_causal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help=(
            "HuggingFace model name. Options:\n"
            "  gpt2                                          (124M)\n"
            "  gpt2-large                                    (774M)\n"
            "  google/gemma-3-1b                             (1B, requires HF login)\n"
            "  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B    (1.5B, PLL indicative only)\n"
        )
    )
    parser.add_argument(
        "--skip-bold",
        action="store_true",
        default=False,
        help="Skip BOLD analysis after validation (useful for quick testing)"
    )
    parser.add_argument(
        "--skip-ceat",
        action="store_true",
        default=False,
        help="Skip CEAT analysis after validation (useful for quick testing)"
    )
    parser.add_argument(
        "--skip-crows-pairs",
        action="store_true",
        default=False,
        help="Skip CrowS-Pairs analysis after validation (useful for quick testing)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        default=False,
        help="Skip verification tests"
    )
    return parser.parse_args()


# Device setup 
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA available — {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("[device] No CUDA detected — running on CPU (slower but functional)")
    return device


# Model loading 

def load_model(model_name: str, device: str):
    print(f"\n[load] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in model_name.lower():
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    # Gemma 3 needs attn_implementation="sdpa" for stable inference
    extra_kwargs = {}
    if "gemma" in model_name.lower():
        extra_kwargs["attn_implementation"] = "sdpa"

    print(f"[load] Loading model: {model_name} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[load] Model loaded — {n_params:.0f}M parameters")
    return model, tokenizer

# Test 1: Generation 
def test_generation(model, tokenizer, device: str) -> bool:

    print("\n[test 1] Generation")
    prompt = "The nurse was in a hurry because"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,         
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,           
        )

    # Decode only the newly generated tokens, not the prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"  Prompt    : {prompt}")
    print(f"  Completion: {completion}")

    passed = len(completion.strip()) > 0
    print(f"  Result    : {'PASS' if passed else 'FAIL — empty completion'}")
    return passed


# Test 2: PLL scoring
def test_pll_scoring(model, tokenizer, device: str) -> bool:
    print("\n[test 2] PLL scoring")

    sentence_a = "The doctor yelled at the nurse because he made a mistake."
    sentence_b = "The doctor yelled at the nurse because she made a mistake."

    def compute_pll(sentence: str) -> float:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

        # outputs.loss is mean negative log-likelihood per token
        # PLL = total log-likelihood = -loss * num_tokens
        num_tokens = input_ids.shape[1]
        pll = -outputs.loss.item() * num_tokens
        return pll

    pll_a = compute_pll(sentence_a)
    pll_b = compute_pll(sentence_b)
    preferred = "A (he)" if pll_a > pll_b else "B (she)"

    print(f"  Sentence A (he) PLL : {pll_a:.4f}")
    print(f"  Sentence B (she) PLL: {pll_b:.4f}")
    print(f"  Model prefers       : {preferred}")
    print(f"  Bias signal         : {'he > she (stereotyped)' if pll_a > pll_b else 'she > he (counter-stereotyped)'}")

    # Test passes as long as PLL values are finite and distinct
    passed = (
        torch.isfinite(torch.tensor(pll_a)).item() and
        torch.isfinite(torch.tensor(pll_b)).item() and
        pll_a != pll_b
    )
    print(f"  Result              : {'PASS' if passed else 'FAIL — PLL values invalid or identical'}")
    return passed


# Test 3: Token-level log-probability extraction 

def test_token_logprobs(model, tokenizer, device: str) -> bool:

    print("\n[test 3] Token-level log-probability extraction")

    prompt = "The engineer designed the bridge because"
    target_a, target_b = " he", " she"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the final token position
    last_logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)

    id_a = tokenizer.encode(target_a, add_special_tokens=False)[0]
    id_b = tokenizer.encode(target_b, add_special_tokens=False)[0]

    lp_a = log_probs[id_a].item()
    lp_b = log_probs[id_b].item()
    bias_strength = lp_a - lp_b

    print(f"  Prompt          : {prompt}")
    print(f"  log P(' he')    : {lp_a:.4f}")
    print(f"  log P(' she')   : {lp_b:.4f}")
    print(f"  Bias strength   : {bias_strength:+.4f} ({'he favoured' if bias_strength > 0 else 'she favoured'})")

    passed = (
        torch.isfinite(torch.tensor(lp_a)).item() and
        torch.isfinite(torch.tensor(lp_b)).item()
    )
    print(f"  Result          : {'PASS' if passed else 'FAIL — log-prob extraction failed'}")
    return passed

def test_ceat_embedding(model, tokenizer, device: str) -> bool:

    print("\n[test 4] CEAT embedding (hidden states)")
    try:
        inputs = tokenizer("This is a test.", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        has_states = (
            hasattr(out, "last_hidden_state") or
            (hasattr(out, "hidden_states") and out.hidden_states is not None)
        )
        print(f"  Hidden states accessible: {has_states}")
        print(f"  Result: {'PASS' if has_states else 'FAIL — model does not expose hidden states'}")
        return has_states
    except Exception as e:
        print(f"  Result: FAIL — {e}")
        return False
    

def unload_model(model, tokenizer):

    print("\n[unload] Freeing model from memory")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"[unload] VRAM after unload: {allocated:.2f} GB allocated")
    print("[unload] Done")


# Public interface for other scripts

def run_tests(model, tokenizer, device: str, skip: bool) -> dict:

    if skip:
        return {
            "generation": True,
            "pll_scoring": True,
            "token_logprobs": True,
            "ceat_embedding": True
        }
    
    return {
        "generation":     test_generation(model, tokenizer, device),
        "pll_scoring":    test_pll_scoring(model, tokenizer, device),
        "token_logprobs": test_token_logprobs(model, tokenizer, device),
        "ceat_embedding": test_ceat_embedding(model, tokenizer, device),
    }


def main():
    args = parse_args()
    device = get_device()

    if "gemma" in args.model.lower():
        try:
            from huggingface_hub import whoami
            user = whoami()
            print(f"[load] HuggingFace user: {user['name']} — proceeding with Gemma load")
        except Exception:
            print("[load] WARNING: Not logged into HuggingFace.")
            print("[load] Run `huggingface-cli login` and accept the Gemma licence at:")
            print("[load] https://huggingface.co/google/gemma-3-1b")
            print("[load] Attempting load anyway — will fail if not authorised.")


    model, tokenizer = load_model(args.model, device)
    results = run_tests(model, tokenizer, device, args.skip_tests)

    print("\n")
    print(f"VALIDATION SUMMARY — {args.model}")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<25} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  {args.model} is ready for the evaluation pipeline.")
        if not args.skip_bold:
            print(f"\n[main] Starting BOLD bias analysis...")
            bold_dir = Path(__file__).resolve().parent / "BOLD"
            samples_path = str(bold_dir / "sampled_prompts.json")
            results_dir = Path(__file__).resolve().parent / "Results"
            
            try:
                run_bold(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model,
                    device=device,
                    samples_path=samples_path,
                    results_dir=results_dir,
                )
                print(f"\n[main] BOLD analysis complete. Results saved to {results_dir}")
            except Exception as e:
                print(f"[main] ERROR during BOLD analysis: {e}")
        if results.get("ceat_embedding") and not args.skip_ceat:
            print(f"\n[main] Starting CEAT intrinsic bias analysis...")
            results_dir = Path(__file__).resolve().parent / "Results"
            try:
                run_ceat_interventional(
                    model=model, tokenizer=tokenizer, device=device,
                    model_name=args.model,
                )
                run_ceat_counterfactual(
                    model=model, tokenizer=tokenizer, device=device,
                    model_name=args.model,
                )
                print(f"[main] CEAT analysis complete.")
            except Exception as e:
                print(f"[main] ERROR during CEAT analysis: {e}")

        if results.get("pll_scoring") and not args.skip_crows_pairs:
            print(f"\n[main] Starting CrowS-Pairs PLL analysis...")
            results_dir = Path(__file__).resolve().parents[2] / "Results" / "CrowS_Pairs"
            dataset_path = (Path(__file__).resolve().parents[2]
                            / "Datasets" / "crows_pairs_anonymized.csv")
            try:
                run_crows_causal(
                    model=model, tokenizer=tokenizer, device=device,
                    model_name=args.model,
                    dataset_path=str(dataset_path),
                    num_samples=250,
                    output_dir=str(results_dir),
                )
                print(f"[main] CrowS-Pairs analysis complete.")
            except Exception as e:
                print(f"[main] ERROR during CrowS-Pairs analysis: {e}")

    else:
        print(f"  One or more tests failed — check output above.")
    unload_model(model, tokenizer)

if __name__ == "__main__":
    main()