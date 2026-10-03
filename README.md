# EvilPatch — Artifact for *Knowledge Base Poisoning in Retrieval-Augmented Program Repair*

This repository contains the source code, configuration templates, and reproduction
instructions for **EvilPatch**, a dual-channel knowledge-base poisoning framework
against retrieval-augmented program repair (RAG-APR) systems.

EvilPatch constructs poisoned Bug-Fix Pairs (BFPs) with two channels:

| Channel | Component | Implementation |
|---|---|---|
| Retrieval | **GPBS** — Global-Position Beam Search | `src/pipeline/retrieval_attack/optimizer/abgs_optimizer.py`, `loss_func.py`, `vocab_filter.py` |
| Generation | **CWE-guided vulnerability injection** (VICS-style two-stage CoT) | `src/pipeline/generation_attack/vics_vinj.py` |
| Generation | **DLCI** — Dual-Level Comment Induction | `src/pipeline/generation_attack/jailbreak.py` |

Large files (corpora, vector indexes, query sets, experiment outputs, and the
pre-built Milvus Lite databases) are **not** stored in git. They are released as a
HuggingFace dataset; see [`data/README.md`](data/README.md).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Environment setup](#2-environment-setup)
3. [Configuration](#3-configuration)
4. [Data](#4-data)
5. [End-to-end reproduction](#5-end-to-end-reproduction)
6. [Paper to code mapping](#6-paper-to-code-mapping)
7. [Hyperparameters used in the paper](#7-hyperparameters-used-in-the-paper)
8. [Output and log locations](#8-output-and-log-locations)
9. [Troubleshooting](#9-troubleshooting)
10. [What is intentionally not included](#10-what-is-intentionally-not-included)

---

## 1. Repository layout

```
EvilPatch-Artifact/
├── README.md                     <- this file
├── requirements.txt
├── configs/                      <- configuration templates (*.yml.example)
│   ├── models/                   <- retriever and generator wrappers
│   ├── database/                 <- Milvus collections (APR KB, vuln KB, query KB)
│   ├── attack/
│   │   ├── retrieval/            <- GPBS (abgs), aggs, aggd, pabs, naive
│   │   ├── vinj/                 <- CWE-guided vulnerability injection
│   │   └── jailbreak/            <- DLCI comment induction
│   └── evaluation/               <- retrieval / generation / functionality / mitigation / ...
├── data/                         <- reserved layout, payload from HuggingFace
│   └── README.md                 <- required file manifest
├── docs/                         <- algorithm specification of GPBS and the judge helper
└── src/
    ├── models/
    │   ├── retriever/            <- Harrier / GTE / Jina / Qwen embedder wrappers
    │   └── generator/            <- DeepSeek / GPT / GLM / Qwen / MiMo chat wrappers
    ├── rag/
    │   ├── bfp/                  <- APR knowledge base (Milvus client + retriever)
    │   └── vul/                  <- vulnerability knowledge base (Milvus client + retriever)
    ├── pipeline/
    │   ├── retrieval_attack/     <- GPBS/AGGS/AGGD/PABS/Naive + carrier selection
    │   └── generation_attack/    <- vulnerability injection + DLCI
    ├── evaluation/               <- ASR-r@K, Precision@K, VR, Delta-CrystalBLEU, Escape Rate
    ├── preprocessing/            <- dataset construction (CoCoNuT, Codeflaws, DeepFix, BigVul, ...)
    └── utils/                    <- IO, CWE/CVE lookup, formatting, tokenisation, judge, device pool
```

---

## 2. Environment setup

### 2.1 Hardware

| Resource | Recommendation |
|---|---|
| GPU | 1 x NVIDIA GPU with **>= 24 GB** VRAM. The retriever wrappers run 0.5-0.6 B embedding models with `max_length: 32768` and `bfloat16`; the attack configuration requests a 24 GiB free-memory lease. |
| Disk | **~ 120 GB** free (raw datasets ~ 4 GB, corpora ~ 0.5 GB, query sets ~ 5 GB, pre-built Milvus Lite indexes ~ 5 GB, retrieval/evaluation outputs grow quickly). |
| RAM | >= 32 GB is comfortable; some evaluation files are several hundred MB of JSON. |

Multi-GPU machines are supported: every model config exposes a `device_pool`, and the
attack configs use an exclusive GPU lease through `model.device_selection`.

### 2.2 Python environment

```bash
conda create -n EvilPatch python=3.11 -y
conda activate EvilPatch

# PyTorch first, with the CUDA build matching your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Everything else
pip install -r requirements.txt
```

Milvus is used in **Milvus Lite** mode: `pymilvus` with a local `*.db` file as the URI.
No Docker container and no Milvus server is required.

### 2.3 HuggingFace models

The retriever and judge models are downloaded on first use. Behind a restricted
network, set the mirror **before** the first `import huggingface_hub`:

```bash
export HF_ENDPOINT=https://hf-mirror.com      # Linux/macOS
$env:HF_ENDPOINT = "https://hf-mirror.com"    # Windows PowerShell
```

Models referenced by the config templates:

| Role | Model |
|---|---|
| Victim retriever | `microsoft/harrier-oss-v1-0.6b` (decoder-only) |
| Victim retriever | `Alibaba-NLP/gte-modernbert-base` (encoder-only) |
| Transferability | `jinaai/jina-code-embeddings-0.5b`, `Qwen/Qwen3-Embedding-0.6B` |

### 2.4 Smoke test

```bash
python -c "import torch, pymilvus, transformers, sentence_transformers; \
           print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 3. Configuration

The repository ships **templates only** (`*.yml.example`). Working copies are
git-ignored because they contain API keys and machine-specific device settings.

Create the working copies once:

```bash
# Linux / macOS
find configs -name '*.yml.example' -exec sh -c 'cp "$1" "${1%.example}"' _ {} \;

# Windows PowerShell
Get-ChildItem configs -Recurse -Filter *.yml.example | ForEach-Object {
  Copy-Item $_.FullName ($_.FullName -replace '\.example$','')
}
```

Then edit **`configs/models/*.yml`** and fill in your API credentials. Each generator
config holds an `api_pool`: a list of `api_key` / `base_url` / `model_name` entries that
are consumed round-robin, so you can simply delete the providers you do not use.

| Config | Used as |
|---|---|
| `configs/models/deepseek-v4.1-flash.yml` | attacker's auxiliary model (pattern extraction, functional subtask extraction) |
| `configs/models/deepseek-v4-flash.yml` | victim generator + vulnerability injector + LLM reranker |
| `configs/models/deepseek-v4-pro.yml` | LLM-as-Judge (injection verification and patch vulnerability detection) + DLCI comment generator |
| `configs/models/glm-5.3-flash.yml`, `qwen3.7-flash.yml` | additional victim generators |
| `configs/models/{harrier-oss-v1-0.6b,gte-modernbert-base,jina-code-embeddings-0.5b,qwen3-embedding-0.6b}.yml` | retriever definitions (no API key needed) |

> **Never commit the generated `*.yml` files.** `.gitignore` already excludes them.

Config files that are referenced by repository-relative paths
(`configs/database/bfp/apr.yml`, `configs/database/vul/vinj.yml`,
`configs/evaluation/*.yml`, ...) must exist before running the corresponding entry point.

---

## 4. Data

Download the released dataset into `data/`:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <HF_DATASET_REPO> --repo-type dataset --local-dir data
```

The complete manifest — which file lives where, and which configuration key consumes
it — is in [`data/README.md`](data/README.md).

Sanity check after downloading:

```bash
python - <<'PY'
import pathlib
root = pathlib.Path("data")
for rel in ["corpus/bfp/rag_data.jsonl",
            "corpus/vul/clustered_rag_data.jsonl",
            "query/black/query_set.jsonl"]:
    p = root / rel
    lines = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
    print(f"{rel:45s} exists={p.exists()} lines={lines}")
PY
```

If the Milvus Lite indexes are missing, rebuild them with
`src/rag/bfp/retrieval.ipynb` and `src/rag/vul/retrieval.ipynb` (see step 1 below).

---

## 5. End-to-end reproduction

> **Notebooks are interactive.** `src/preprocessing/data_prep_v4.ipynb`,
> `src/rag/*/retrieval.ipynb`, and `src/evaluation/retrieval/retrieval_attack_eval.ipynb`
> contain several alternative configuration cells (one per retriever / scenario).
> Run the cells of the scenario you want and skip the alternatives; every section is
> idempotent and skips work whose output already exists.

### Step 0 — Working configurations

```bash
find configs -name '*.yml.example' -exec sh -c 'cp "$1" "${1%.example}"' _ {} \;
# edit configs/models/*.yml with your API keys
```

### Step 1 — Build the knowledge bases and the poisoning targets

Open **`src/preprocessing/data_prep_v4.ipynb`** and run it top to bottom. It performs:

| Section | Produces |
|---|---|
| 1.1 | `data/corpus/vul/clustered_rag_data.jsonl` — vulnerability knowledge base (17,576 de-duplicated vulnerable functions) |
| 1.2 | `data/corpus/bfp/rag_data.jsonl` — victim APR knowledge base (50,000 BFPs) |
| 1.3 | `data/query/{white,gray,black}/query_set.jsonl` — historical query pools (50,000 each) |
| 2.1-2.3 | the three Milvus Lite indexes under `data/milvus/` |
| 3.1-3.2 | `unsafe_query_set_v4.json` — security-sensitive queries per target CWE |
| 4 | `data/query/black/proxy/proxy_query_set_v4.json` (80 %) and `data/query/black/test/test_query_set_v4.json` (20 %) |
| 5 | `data/query/black/target/poison_targets_set_v4.json` — poisoning carriers, proxy queries, and the per-CWE poisoning budget |
| 6 | augments the carrier file with `generation_attack.vinj_attack.relevant_vul` (Top-10 same-CWE CVE instances, 5 retained references per carrier) |

To run it non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  src/preprocessing/data_prep_v4.ipynb
```

### Step 2 — Retrieval-side attack (poisoned `buggy_code`)

GPBS is selected with `attack.optimizer: "abgs"`. The templates are split per retriever:

```bash
# ours: GPBS (the configuration template names it abgs)
python -m src.pipeline.retrieval_attack.run_attack_ultra \
  --config configs/attack/retrieval/gte/abgs/abgs.yml

# baselines
python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/naive.yml
python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/gte/aggd/aggd.yml
python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/gte/pabs/pabs.yml
```

Available templates: `configs/attack/retrieval/{gte,jina,harrier}/{abgs,aggd,pabs}/*.yml.example`.
Before running, set `data.target_file_path` to the carrier file produced in step 1 and
`sampling.process_cwe` to the target list (the paper uses eight CWEs; see
[section 7](#7-hyperparameters-used-in-the-paper)).

The runner is resumable: carriers that already contain
`retrieval_attack.poisoned_buggy_code` are skipped, and the carrier file is written
atomically after every item.

### Step 3 — Generation-side attack (vulnerability injection, then DLCI)

```bash
# Stage 1  retrieve same-CWE CVE references          -> relevant_vul
# Stage 2  optional LLM reranker (skipped by the default template)
# Stage 3  two-stage CoT vulnerability injection     -> vinj_attack.vul_code
# Stage 4  LLM-as-Judge verification                 -> vinj_attack.is_vulnerable
python -m src.pipeline.generation_attack.vics_vinj --config configs/attack/vinj/vinj.yml

# DLCI: document-level + line-level inducement comments -> jailbreak_attack
python -m src.pipeline.generation_attack.jailbreak --config configs/attack/jailbreak/jailbreak.yml
```

Both stages write back into the same carrier file and are idempotent (a stage only
processes entries whose output field is still empty).

### Step 4 — Merge shards (only if you split the carriers)

Carrier files can be sharded with `partIofN` suffixes for parallel attack runs.
Merge and re-key them with **`src/utils/distributed_io.ipynb`**; the notebook renames
colliding `doc_id`s by `(CWE, doc_id)` so the later evaluation cannot confuse carriers
that came from different CWE partitions.

### Step 5 — Retrieval effectiveness: ASR-r@K and Precision@K

Open **`src/evaluation/retrieval/retrieval_attack_eval.ipynb`** with
`configs/evaluation/retrieval.yml` configured:

```yaml
data:
  poisoned_file_path: "data/query/black/target/<retriever>/merged/<OPTIMIZER>/poison_targets_set_v4_merged_intersected.json"
  test_file_path:     "data/query/black/test/test_query_set_v4.json"
database:
  milvus_config: "configs/database/bfp/apr.yml"
eval:
  eval_cwe: ["CWE-787","CWE-416","CWE-125","CWE-476","CWE-20","CWE-200","CWE-119","CWE-362"]
  eval_top_k: [5, 10, 15]
  eval_mode: "both"          # per_cwe + CWE-MIXED
  eval_retrieval_naive: true # also score the un-optimised (naive) carrier baseline
```

The notebook injects the poisoned BFPs into the APR index, runs the test queries, and
removes them again. Results land in
`data/query/black/test/results/{retriever}/{optimizer}-{timestamp}/{naive,poisoned}/`.

The shipped template defaults to a single-CWE dry run (`eval_cwe: ["CWE-119"]`,
`eval_mode: "per_cwe"`); the values shown above are the paper's joint setting
(`eval_mode: "mixed"`/`"both"` over all eight CWEs).

### Step 6 — Generation effectiveness: Vulnerability Rate (VR)

```bash
# 1) build the RACG prompt and generate patches for every victim query
python -m src.evaluation.generation.generation_attack_eval \
  --config configs/evaluation/generation.yml

# 2) one-off check of how often the injection stage itself succeeded
python -m src.evaluation.vinj.vinj_attack_eval --config configs/evaluation/vinj.yml
```

`configs/evaluation/generation.yml` selects the victim generator
(`model.apr_generator_config`) and the judge (`model.judge_generator_config`), the
retrieval result directory, the Top-K list, and the CWE list. Two rates are reported:

* `VR_GLOBAL` = vulnerable patches / **all** test queries in the file;
* `VR_LOCAL`  = vulnerable patches / queries whose Top-K contains a poisoned BFP.

### Step 7 — Patch quality: CrystalBLEU and Delta-BLEU

```bash
python -m src.evaluation.functionality.functionality_consistency_eval \
  --config configs/evaluation/functionality.yml
```

For every victim query this module rebuilds a **clean** retrieval baseline
(`retrieval_dir_path/clean/...`), generates the corresponding clean patch with the same
generator, and reports `CrystalBLEU_clean`, `CrystalBLEU_poisoned`, and
`delta = |clean - poisoned|` against the ground-truth repair. The clean directory is
reused on later runs after a strict purity and alignment check.

### Step 8 — Ablations

**Component ablation (GPBS / DLCI on-off).** Run step 2 with `attack.optimizer`
switched between `abgs` (GPBS on) and `naive` (GPBS off), and step 3 with and without
the `jailbreak` stage, then compare with:

```bash
python -m src.evaluation.debug.generation_debug_eval --config configs/evaluation/debug.yml
```

This evaluates several methods on their **common** victim queries (identical poisoned
`doc_id` multisets), so the comparison is apples-to-apples, and emits
`overview_debug.md`.

**Retrieval-depth ablation (Top-2 ... Top-21).** Re-run steps 5-6 with
`eval.eval_top_k: [2,3,6,9,12,15,18,21]`.

**Cross-retriever transferability.** Run step 2 against Harrier, then evaluate the same
carrier file against knowledge bases built with Jina and GTE (steps 5-6) without
re-optimising.

**Optimizer comparison.** `src/evaluation/optimizer/retrieval_optimizer_eval.ipynb`
with `configs/evaluation/optimizer.yml`.

### Step 9 — Mitigation

```bash
python -m src.evaluation.mitigation.index.index_detection_eval \
  --config configs/evaluation/mitigation.yml
```

Implements the shrinkage-Mahalanobis index-stage detector (L2-normalised document
embeddings, calibration quantile 0.95, shrinkage coefficient 0.01) and reports the
**Escape Rate** per CWE and for CWE-MIXED, together with a per-document verdict dump.

---

## 6. Paper to code mapping

| Paper element | Entry point | Configuration |
|---|---|---|
| 4.1 Carrier and proxy-query selection | `src/preprocessing/data_prep_v4.ipynb` sections 3-5 | `configs/database/bfp/data_prep_v4/target.yml` |
| 4.2 GPBS (Algorithm 1) | `src/pipeline/retrieval_attack/optimizer/abgs_optimizer.py`; spec in `docs/abgs_pure_algorithm.md` | `configs/attack/retrieval/*/abgs/abgs.yml` |
| 4.2 Loss and vocabulary filtering | `loss_func.py`, `vocab_filter.py`, `src/utils/token.py` | same |
| 4.3 CWE-guided vulnerability injection | `src/pipeline/generation_attack/vics_vinj.py` | `configs/attack/vinj/vinj.yml` |
| 4.3 DLCI | `src/pipeline/generation_attack/jailbreak.py` | `configs/attack/jailbreak/jailbreak.yml` |
| 4.3 LLM-as-a-Judge verification | `src/utils/vul_judger.py` | `configs/attack/vinj/vinj.yml` |
| 5.2 Knowledge-base construction | `src/preprocessing/**`, `src/rag/**` | `configs/database/**` |
| 5.4 Baseline AGGD | `optimizer/aggd_optimizer.py` | `configs/attack/retrieval/*/aggd/aggd.yml` |
| 5.4 Baseline PABS (ImportSnare) | `optimizer/pabs_optimizer.py` | `configs/attack/retrieval/*/pabs/pabs.yml` |
| 5.4 Baseline Naive | `optimizer/naive_optimizer.py` | `configs/attack/retrieval/naive.yml` |
| 5.5 ASR-r@K, Precision@K | `src/evaluation/retrieval/metrics.py` + `retrieval_attack_eval.ipynb` | `configs/evaluation/retrieval.yml` |
| 5.5 VR | `src/evaluation/generation/metrics.py` | `configs/evaluation/generation.yml` |
| 5.5 Delta-CrystalBLEU | `src/evaluation/functionality/` | `configs/evaluation/functionality.yml` |
| 5.5 Escape Rate | `src/evaluation/mitigation/index/detector.py` | `configs/evaluation/mitigation.yml` |
| 6.1 Main results (CWE-MIXED) | steps 5-7 | `eval_mode: mixed` |
| 6.2 CWE-wise results | steps 5-7 | `eval_mode: per_cwe` |
| 6.3 Component ablation | step 8 | `configs/evaluation/debug.yml` |
| 6.3 Retrieval-depth ablation | step 8 | `eval_top_k: [2,3,6,9,12,15,18,21]` |
| 6.4 Transferability | step 8 | per-retriever `configs/database/bfp/apr.yml` |
| 6.5 Mitigation | step 9 | `configs/evaluation/mitigation.yml` |
| 7.2 Judge reliability | `src/utils/vul_judger.py` | `configs/attack/vinj/vinj.yml` (`judge_generator_config`) |

---

## 7. Hyperparameters used in the paper

### Target CWE categories (eight)

```python
CWE_LIST = ["CWE-787", "CWE-416", "CWE-125", "CWE-476",
            "CWE-20",  "CWE-200", "CWE-119", "CWE-362"]
```

The retrieval-attack templates (`configs/attack/retrieval/**`) already carry this
eight-CWE list in `sampling.process_cwe`. The generation-attack templates
(`configs/attack/vinj/vinj.yml`, `configs/attack/jailbreak/jailbreak.yml`) and
`configs/evaluation/optimizer.yml` still list the earlier five-CWE subset — extend them
to the list above to reproduce the paper's setting, or use `["CWE-MIXED"]` for the joint
setting of section 6.1.

### Carrier and proxy-query selection

| Parameter | Value |
|---|---|
| Retrieval depth for query filtering and inverse mapping | Top-10 |
| Relevance threshold | 0.80 |
| Association-count threshold tau (distinct CVEs) | 2 |
| Proxy / test split | 8 : 2 |
| Corpus size / query-pool size | 50,000 / 50,000 |

### Poisoning budget

| Parameter | Value |
|---|---|
| Poisoning ratio | <= 0.5 % of the 50,000 clean documents |
| Carriers injected, Harrier / GTE | 233 (0.47 %) / 252 (0.50 %) |

### Retrieval-side optimization (GPBS, AGGD, PABS)

| Parameter | Value |
|---|---|
| Maximum iterations `N` | 50 |
| Early-stop similarity `tau_stop` | 0.95 |
| Patience (`patience`) | 10 consecutive non-improving iterations |
| Per-iteration evaluation budget `n` | 6000 (GTE), 4000 (Harrier) |
| Global base budget `global_basic_budget` | 50 |
| Loss | `expected_sim`, `dynamic_weight: false` |
| Safe vocabulary | `use_safe_vocab: false` (structural constraints only) |
| Elite retention (parent) `elite_num` | 0 |
| PABS | beam width 10, adversarial sequence length 20 |
| AGGD | adversarial sequence length 20 |

For the paper's GPBS configuration the templates must be adjusted explicitly:
`attack.loss_func: "expected_sim"`, `attack.dynamic_weight: false`,
`attack.use_safe_vocab: false`, `attack.elite_num: 0` — see
[`docs/abgs_pure_algorithm.md`](docs/abgs_pure_algorithm.md), which also contains the
full pseudocode of the published algorithm.

### Generation-side construction

| Parameter | Value |
|---|---|
| Stage-1 same-CWE CVE retrieval depth (`vinj.top_k`) | 10 in the paper; the shipped template uses 5 |
| `relevant_vul_num` (references kept per carrier) | 5 — only consulted when `skip_rerank: false` |
| `skip_rerank` | `true` in the shipped template, i.e. Stage 1 candidates go straight to injection and the LLM reranker of section 4.3 is not exercised |
| Judge retries (`judge_max_retries`) | 5 |

### Victim system

| Parameter | Value |
|---|---|
| Retrieval depth for the augmented context | Top-15 (except in the depth ablation) |
| CrystalBLEU ignored n-grams `k` | 500 |
| Index detector | L2 normalisation, calibration quantile 0.95, shrinkage 0.01 |

---

## 8. Output and log locations

| Artefact | Location |
|---|---|
| Attack trace (per carrier, per optimizer) | `logs/retrieval_attack/{date}/{CWE}_{id}_{OPTIMIZER}.txt` |
| Vulnerability injection debug logs | `logs/vinj_attack/` |
| DLCI debug logs | `logs/jailbreak_attack/` |
| Retrieval results (JSON) | `data/query/{box}/test/results/{retriever}/{optimizer}-{timestamp}/{naive,poisoned}/` |
| Retrieval metrics | `logs/evaluation/retrieval/{retriever}/{optimizer}-{timestamp}/overview.md` |
| Generation (VR) metrics | `logs/evaluation/generation/{retriever}-{optimizer}/{generator}-{timestamp}/{jailbreak,naive}/overview.md` |
| CrystalBLEU metrics | `logs/evaluation/functionality/{retriever}-{optimizer}/{generator}-{timestamp}/{jailbreak,naive}/overview.md` |
| Mitigation (Escape Rate) | `logs/evaluation/mitigation/index/{retriever}/{timestamp}.md` |

`overview.md` files aggregate the per-configuration detail logs into the tables that
correspond to the paper's result tables.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No embedder registered for model_name='...'` | The prefix dispatch table lives in `src/models/retriever/Base.py`; use one of the supported HF ids (`jinaai/`, `Alibaba-NLP/gte-`, `microsoft/harrier-`, `Qwen/Qwen3-Embedding-`). |
| Milvus collection not found | The `.db` file was not downloaded, or `database.uri` points at a different index than `database.model`. Filenames must match the retriever name. |
| `data/...` file not found | The dataset was not downloaded, or you skipped the working-copy step for `*.yml`. |
| Out-of-memory during GPBS | Lower `configs/models/<retriever>.yml` `batch_size` and/or `attack.n`; the candidate score matrix is `O(|V| x m)`. |
| Long silence during GPBS | Expected: each iteration performs one forward/backward pass per beam member plus `n` real forward evaluations. Watch `logs/retrieval_attack/`. |
| HuggingFace download timeouts | Export `HF_ENDPOINT` before importing `huggingface_hub`; restart the kernel if a notebook imported it earlier. |
| Evaluation reports status code 2 | Some patches are still missing for the selected Top-K/generator slot. Re-run the same command: results are written atomically per query and the run resumes. |
| GPU lease never granted | `model.device_selection` waits for a GPU with `min_free_memory_mb` free. Reduce that value or set `on_busy: "ignore"`. |

The attack and evaluation entry points are deliberately **resumable**: every completed
item is written back to the carrier/result file immediately, so an interrupted run can
simply be restarted with the same command.

---

## 10. What is intentionally not included

| Excluded | Reason |
|---|---|
| Raw datasets and all corpora, query sets, and vector indexes | Redistributed through the HuggingFace dataset release (`data/README.md`). |
| Experiment outputs and `logs/` | Reproduced by the commands above. |
| Unit-test, scratch, and debugging scripts (`src/test*.ipynb`, `src/debug.*`) | Development-only. |
| Dataset statistics and figure-generation scripts | Analysis-only, not needed to reproduce the reported numbers. |
| Vendored copies of third-party baseline implementations | AGGD and PABS are re-implemented in `src/pipeline/retrieval_attack/optimizer/`; see the original papers and repositories cited in the paper. |
| PDFs of related work | Copyright; cited in the paper. |
| Real API keys and private endpoints | Replaced with placeholders in `*.yml.example`. |
