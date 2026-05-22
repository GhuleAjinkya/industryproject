# Causal Inference for Bias Identification and Mitigation in LLMs

A research pipeline that applies **causal inference** (Pearl's Ladder of Causation) to identify, locate, and mitigate bias in Large Language Models. Evaluated on **GPT-2 Small** and **Gemma 3 1B** across three standardised benchmarks: BOLD, CEAT, and CrowS-Pairs.

> **Please See:** This readme.md file only describes the project in brief and provides instructions on how to install and run this project. Please refer to [Sections 4, 5 and 6 of the Report](https://docs.google.com/document/d/1FI_s8hveRjqKKgkyrwiW6yLqDPfJhDIFCDayR14tczQ/edit?usp=sharing) for an in-depth discussion on the design decisions made for this project.

---

## Repository Structure

```
industryproject/
├── BiasIdentification/
│   ├── LayerBased/               # Mechanistic layer-level analysis (GPT-2)
│   │   ├── bias_dataset.json
│   │   ├── phase1_baseline.py
│   │   ├── phase2_tracing.py
│   │   ├── phase3_mitigation.py
│   │   ├── phase5_causal_tracing.py
│   │   └── phase5_nli_metric.py
│   └── OutputBased/
│       ├── BOLD/                 # Sentiment & toxicity analysis
│       │   ├── download_data.py
│       │   ├── sample_prompts.py
│       │   ├── analyze_bold.py
│       │   └── load_model.py
│       ├── CEAT/                 # Intrinsic embedding bias (WEAT)
│       │   └── ceat.py
│       └── CrowS_Pairs/         # Stereotype preference (PLL scoring)
│           └── crows_pairs.py
├── Datasets/                     # Downloaded data (auto-populated)
├── Results/                      # Output CSVs from all benchmarks
└── runme.py                      # Main entry point
```

---

## What It Does

The pipeline evaluates LLM bias at three levels of Pearl's Ladder of Causation:

| Level | Method | What it measures |
|---|---|---|
| L1 — Association | CEAT (WEAT effect size), CrowS-Pairs PLL, BOLD sentiment | Correlational bias in outputs and embeddings |
| L2 — Intervention | Activation steering, INLP, prompt mitigation | Effect of `do(demographic=neutral)` on model behaviour |
| L3 — Counterfactual | Activation patching, counterfactual token swap | Causal proof of bias origin; individual treatment effects |

Three mitigation strategies are compared before/after across all benchmarks:
- **Prompt mitigation** — debiasing instruction prepended at inference time
- **Activation steering** — subtracts a gender/race direction vector at the identified bias layer
- **INLP** — projects out the bias subspace from the input embedding matrix

---

## Requirements

- Python 3.9+
- PyTorch (CUDA recommended; CPU works but is slow)
- A HuggingFace account with Gemma licence accepted (only needed for `google/gemma-3-1b`)

Install dependencies:

```bash
pip install torch transformers accelerate requests vader-sentiment detoxify transformer-lens scikit-learn numpy pandas
```

For Gemma, log in to HuggingFace and accept the model licence at
https://huggingface.co/google/gemma-3-1b, then run:

```bash
huggingface-cli login
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/GhuleAjinkya/industryproject.git
cd industryproject
```

**2. Download datasets**

```bash
# BOLD prompts (gender + race)
python BiasIdentification/OutputBased/BOLD/download_data.py

# CrowS-Pairs — download manually from:
# https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv
# Place it at: Datasets/crows_pairs_anonymized.csv
```

**3. Sample BOLD prompts**

```bash
python BiasIdentification/OutputBased/BOLD/sample_prompts.py
```

This writes `sampled_prompts.json` into the BOLD directory (10 gender + 10 race prompts by default).

---

## Running the Pipeline

All benchmarks are run through the central entry point `runme.py`.

**Run everything on GPT-2 (default):**

```bash
python runme.py
```

**Run on a different model:**

```bash
python runme.py --model gpt2-large
python runme.py --model google/gemma-3-1b       # requires HF login
python runme.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Skip individual benchmarks:**

```bash
python runme.py --skip-bold          # skip BOLD analysis
python runme.py --skip-ceat          # skip CEAT intrinsic bias
python runme.py --skip-crows-pairs   # skip CrowS-Pairs scoring
python runme.py --skip-tests         # skip validation checks (faster re-runs)
```

**Example — run only CrowS-Pairs on Gemma:**

```bash
python runme.py --model google/gemma-3-1b --skip-bold --skip-ceat
```

---

## Validation

Before any benchmark runs, `runme.py` validates the loaded model against four checks:

| Test | What it checks |
|---|---|
| Generation | Model produces non-empty completions |
| PLL scoring | Pseudo-log-likelihood values are finite and distinct |
| Token log-probs | Per-token log-probability extraction works |
| CEAT embedding | Hidden states are accessible |

All four must pass before the evaluation pipeline starts. Use `--skip-tests` to bypass this on subsequent runs of the same model.

---

## Results

All output is written to `Results/`. Each benchmark produces:

- **BOLD** — per-run CSVs, summary CSVs, a mitigation comparison CSV (with deltas vs baseline), and a counterfactual CSV with Average Treatment Effect estimates by demographic category.
- **CEAT** — effect size tables across L1/L2/L3 analysis and INLP before/after comparison.
- **CrowS-Pairs** — stereotype preference rates, interventional differentials, and counterfactual causal effects per bias category.

---

## Layer-Based Analysis (GPT-2 only)

For a deeper mechanistic investigation of *where* in GPT-2 bias originates, run the phase scripts in order:

```bash
cd BiasIdentification/LayerBased

python phase1_baseline.py        # establish keyword stereotype rate
python phase2_tracing.py         # logit lens — find the bias layer
python phase3_mitigation.py      # apply activation steering at Layer 11
python phase5_causal_tracing.py  # activation patching — causal proof
python phase5_nli_metric.py      # NLI-based semantic stereotype rate
```

These scripts use TransformerLens and require it to be installed separately:

```bash
pip install transformer-lens
```

---

## Authors

Ajinkya Ghule, Archit Boraste, Atharva Dhamdhere, Ayush Andure,
Guide: Dr. Snehal Rathi