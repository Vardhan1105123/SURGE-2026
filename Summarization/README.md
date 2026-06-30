# Abstractive Legal Summarization

**Project:** SURGE-2026  
**Advisor:** Prof. Ashutosh Modi  

This repository contains an optimized pipeline for fine-tuning the `allenai/led-base-16384` (Longformer Encoder-Decoder) model to perform abstractive summarization on complex Indian Legal Judgments. The project utilizes the **IN-Abs dataset** (Download link: [IN-Abs Dataset on Zenodo](https://zenodo.org/record/7152317#.Yz6mJ9JByC0)) for training and evaluation.

## Architectural Overview and Hardware Constraints

The end-to-end pipeline encompasses semantic tagging, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO). To accommodate hardware limitations (a single 8GB VRAM GPU), the pipeline is modularized into distinct execution phases. 

To maintain rigorous scientific consistency, the following training hyperparameters were held strictly constant across all model iterations (Phase 2 onwards) to respect memory limits and ensure fair benchmarking:
- **LoRA Rank:** `16`
- **LoRA Alpha:** `32`
- **Target Matrices:** `q`, `k`, `v`
- **Batch Size:** `1`
- **Gradient Accumulation:** `8 steps`
- **Max Encoder Tokens:** `6,144` (Hardware bottleneck limitation)

---

## The Pipeline

The architecture is divided into five major phases:

1. **Phase 1: Baseline SFT (`src/baseline`)** 
   Initial Supervised Fine-Tuning (SFT) using a lightweight LoRA configuration (targeting `q` and `v` matrices).
   
2. **Phase 2: Optimized Baseline (`src/optimized_baseline`)** 
   Enhanced SFT using expanded LoRA target matrices (`q`, `k`, `v`), Cosine Annealing learning rate schedulers, and an expanded generation limit of 1024 tokens. 

3. **Phase 3: Rhetorical Role Tagging (`src/rhetorical_role`)** 
   Integration of the OpenNYAI pipeline to perform zero-shot semantic parsing on legal logic (tagging sections such as `(preamble)`, `(fact)`, and `(argument by petitioner)`). These tags are natively injected into the prompts to provide structural guidance to the LLM.

4. **Phase 4: Direct Preference Optimization (`src/dpo_baseline` & `src/dpo_rhetorical`)** 
   Application of DPO to contrast positive, human-written summaries against negative, model-generated hallucinations. This penalizes repetition and enforces strict factual adherence.

5. **Phase 5: Dynamic Map-Reduce Chunking (`src/dynamic_chunking`)** 
   To address the 6,144 token limitation on documents extending up to 36,000+ tokens, a Dynamic Map-Reduce algorithm was implemented. The system segments the text into 5,000-token windows, processes the chunks in parallel, and recursively concatenates them to produce a coherent, full-context summary.

---

## Empirical Results

The primary models were evaluated against an unseen test dataset using ROUGE (exact overlap) and BERTScore (semantic similarity). 

| Model Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Optimized Baseline** | **0.4624** | 0.2222 | 0.2298 | 0.8125 |
| **Rhetorical Model** | 0.4454 | 0.2266 | 0.2314 | **0.8144** |
| **DPO Baseline** | 0.4278 | 0.2055 | 0.2162 | 0.8093 |
| **DPO Rhetorical** | 0.4618 | **0.2267** | **0.2345** | 0.8117 |

### Extreme-Length Stress Test (Dynamic Chunking)

While standard ROUGE metrics apply to documents below the 6,144-token threshold, many Indian Legal Judgments exceed this limit, previously causing truncation of judicial analysis.

To definitively prove the necessity of the Dynamic Chunking Engine, we stress-tested all four models against the **Top 15 longest documents** in the test set (ranging from 8,500 to over 47,000 tokens).

#### Empirical Results on Long Documents

| Model Variant | Mode | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Optimized Baseline** | Standard (Truncated) | 0.3125 | 0.1740 | 0.1595 | 0.8401 |
| **Optimized Baseline** | **Chunking (Lossless)** | 0.5283 | 0.2900 | 0.2157 | **0.8403** |
| **Rhetorical Model** | Standard (Truncated) | 0.2471 | 0.1249 | 0.1250 | 0.8330 |
| **Rhetorical Model** | **Chunking (Lossless)** | 0.5180 | 0.2855 | 0.2165 | 0.8368 |
| **DPO Baseline** | Standard (Truncated) | 0.3173 | 0.1612 | 0.1548 | 0.8372 |
| **DPO Baseline** | **Chunking (Lossless)** | 0.5274 | 0.2870 | 0.2203 | 0.8369 |
| **DPO Rhetorical** | Standard (Truncated) | 0.2818 | 0.1489 | 0.1440 | 0.8333 |
| **DPO Rhetorical** | **Chunking (Lossless)** | **0.5769** | **0.3185** | **0.2370** | 0.8379 |

As demonstrated, the standard inference mechanism fails on massive documents, leading to truncation and loss of critical information. The Dynamic Map-Reduce algorithm restores comprehensive data retention, propelling the **DPO Rhetorical** model to a 0.5769 ROUGE-1 score. This validates that the combination of structural tagging, preference optimization, and lossless context chunking creates a highly effective legal summarization engine.

**[➡️ View the raw Empirical Results on Long Documents (JSON)](analysis/chunking_empirical_results.json)**

### Chunking Constraints

While the Dynamic Chunking mechanism effectively resolves context truncation, it currently relies on a hardcoded, static **500-token overlap** between adjacent chunks to maintain semantic continuity. It is theoretically possible that systematically fine-tuning this overlap parameter—or introducing a dynamic overlap boundary based on rhetorical section breaks—could further reduce border-hallucinations and improve the cohesiveness of the final merged summary, though this remains unexplored in the current pipeline.



---

## Environment Setup

This repository requires **two separate Conda environments**. The environments are decoupled to prevent dependency conflicts, as the OpenNYAI legal tagger relies on older NLP dependencies (e.g., specific `spaCy` versions) that clash with the modern PyTorch and `transformers` packages required for training the LED model.

1. **`legalsum` Environment**: The primary environment for all model training, Map-Reduce chunking, DPO, and ROUGE evaluations. Install via `requirements_legalsum.txt`.
2. **`opennyai_env` Environment**: A modularized environment strictly for executing the zero-shot OpenNYAI extraction pipeline in Phase 3. Install via `requirements_opennyai.txt`.

### Installation Commands

You can use either Conda (Recommended) or standard Python virtual environments (`venv`).

#### Option A: Using Conda (Recommended)
```bash
# 1. Create and activate the primary ML environment
conda create -n legalsum python=3.10 -y
conda activate legalsum
pip install -r requirements_legalsum.txt

# 2. Create and activate the OpenNYAI NLP environment
conda create -n opennyai_env python=3.10 -y
conda activate opennyai_env
pip install -r requirements_opennyai.txt
```

#### Option B: Using Standard Python `venv`
```bash
# 1. Create and activate the primary ML environment
python3.10 -m venv legalsum_venv
source legalsum_venv/bin/activate  # On Windows use: legalsum_venv\Scripts\activate
pip install -r requirements_legalsum.txt
deactivate

# 2. Create and activate the OpenNYAI NLP environment
python3.10 -m venv opennyai_venv
source opennyai_venv/bin/activate  # On Windows use: opennyai_venv\Scripts\activate
pip install -r requirements_opennyai.txt
deactivate
```

### Running the End-to-End Pipeline

Due to the conflicting dependencies between the OpenNYAI parser and modern HuggingFace architectures, this project requires dynamic environment switching. To automate this, we provide a unified orchestrator script that will seamlessly tag the dataset, execute supervised fine-tuning, and align the model via Direct Preference Optimization.

#### For Conda Users (Default)
If you named your Conda environments `opennyai_env` and `legalsum`, you can run the pipeline directly:
```bash
python src/rhetorical_role/run_rhetorical_pipeline.py
```
*(You can specify custom Conda names using `--opennyai-env custom_name --legalsum-env custom_name`)*

#### For Virtual Environment (`venv`) Users
If you used standard Python virtual environments, specify the `--env-type venv` flag and pass the relative or absolute paths to your environment directories. The script automatically handles Windows/Linux executable path routing.
```bash
python src/rhetorical_role/run_rhetorical_pipeline.py \
    --env-type venv \
    --opennyai-env path/to/opennyai_venv \
    --legalsum-env path/to/legalsum_venv
```

---

## Repository Structure

```text
Summarization/
├── README.md                                   # Main documentation
├── requirements_legalsum.txt                   # Primary ML dependencies (Torch, Transformers, TRL)
├── requirements_opennyai.txt                   # Modular NLP dependencies (OpenNYAI, spaCy)
├── analysis/                                   # Exploratory data analysis scripts
├── data/
│   ├── raw/                                    # Original unstructured legal documents
│   │   └── IN-Abs/                             # IN-Abs raw text files
│   └── processed/                              # Pre-tokenized, tagged, and preference datasets
├── models/                                     # Cached HuggingFace base models
├── src/
│   ├── baseline/                               # Phase 1: Basic SFT
│   │   └── train_v1_baseline.py                # Trains initial LED-base model on raw data
│   ├── common/                                 # Shared evaluation metrics and dataloaders
│   │   ├── dataset_loader.py                   # Common IO for loading train/test splits
│   │   ├── check_lengths.py                    # Token length distribution analysis
│   │   ├── evaluate_models.py                  # ROUGE/BERTScore evaluation suite
│   │   └── evaluate_dpo_models.py              # DPO-specific evaluation wrappers
│   ├── data_processing/                        # Raw dataset extraction and cleaning
│   ├── dpo_baseline/                           # Phase 4: DPO applied to Optimized Baseline
│   │   └── train_dpo_baseline.py               # Runs DPO alignment on the baseline model
│   ├── dpo_dataset_generation/                 # Phase 4: Constructing positive/negative pairs
│   │   ├── create_dpo_dataset.py               # Generates hallucinated negative samples
│   │   └── train_dpo_dataset.py                # Compiles the final DPO JSONL pairs
│   ├── dpo_rhetorical/                         # Phase 4: DPO applied to Rhetorical model
│   │   └── train_dpo_rhetorical.py             # Runs DPO alignment on the tagged model
│   ├── dynamic_chunking/                       # Phase 5: Map-Reduce inference engine
│   │   ├── dynamic_chunking_inference.py       # Core recursive summarizer algorithm
│   │   ├── evaluate_chunking_benchmark.py      # Stress-tests the 15 longest documents
│   │   └── run_chunking_on_all.py              # Bulk inference generator for datasets
│   ├── optimized_baseline/                     # Phase 2: Enhanced SFT
│   │   ├── train_optimized_baseline.py         # SFT with expanded Q,K,V targeting
│   │   └── batch_orchestrator.py               # VRAM-safe batch generation tool
│   ├── rhetorical_role/                        # Phase 2: OpenNYAI integration & Rhetorical SFT
│   │   ├── run_opennyai_local.py               # Generates rhetorical tags for the dataset
│   │   ├── train_rhetorical_role.py            # Trains LED on the tagged dataset
│   │   └── run_rhetorical_pipeline.py          # Unified orchestrator for Phase 1-3
│   └── scripts/                                # End-to-end testing and orchestration
│       ├── run_dual_dpo.py                     # Queues DPO for both Baseline and Rhetorical
│       └── run_final_tests.py                  # Full automated ROUGE/BERTScore evaluation
└── weights/                                    # LoRA adapter checkpoints
```

---

## Running the End-to-End Orchestrators

To run the complete Rhetorical + DPO pipeline from start to finish:
```bash
python src/rhetorical_role/run_rhetorical_pipeline.py
```

To run a full inference benchmark on the resulting models (ROUGE, BERTScore) and execute a stress test of the Dynamic Chunking algorithm on a massive legal document:
```bash
python src/scripts/run_final_tests.py
```
