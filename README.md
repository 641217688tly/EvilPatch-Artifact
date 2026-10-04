# EvilPatch

Core implementation for **EvilPatch: Knowledge Base Poisoning in
Retrieval-Augmented Program Repair**. The artifact includes GPBS, adapted
AGGD/PABS baselines, CWE-guided vulnerability injection, DLCI, and retrieval,
generation, patch-similarity, and defense evaluation. Tests, plotting/analysis
scripts, old preprocessing versions, and unrelated vendored repositories are
excluded. `abgs` is the implementation/configuration name for GPBS.

## 1. Environment and configuration

Run commands from the repository root on **Linux with Python 3.11**. Retrieval
optimization requires a CUDA GPU; the paper used RTX 4090 GPUs with one
optimization task per GPU. Generation and judging use remote APIs, not local
LLM weights. Install a PyTorch build compatible with your CUDA driver before
installing the remaining dependencies:

```bash
conda create -n EvilPatch python=3.11 -y
conda activate EvilPatch
# Install the CUDA-compatible torch build for your machine.
pip install -r requirements.txt
python -m ipykernel install --user --name EvilPatch --display-name "Python 3 (EvilPatch)"
python scripts/setup_configs.py
```

The setup script copies `.yml.example` templates to `.yml` without overwriting
existing configurations. Working YAML files, data, and logs are Git-ignored.
Dependencies are minimum requirements, not a frozen environment lockfile.
Milvus Lite local databases are intended for Linux; native Windows is not the
reproduction environment.

Edit the four generator configurations in `configs/models/`: DeepSeek-V4-Flash,
GLM-5.3-Flash, Qwen3.7-Flash, and the auxiliary DeepSeek-V4.1-Flash. Set each
`api_pool` entry's `api_key`, `base_url`, and provider-specific `model_name`.
Keep the top-level model identifier for wrapper selection. Check provider
support for the model, context window, and `max_tokens` before a batch run;
no credentials or model access are supplied. API model revisions can change results.

Retriever examples use `cuda:0`. Adjust `device_pool` for your machine. The
attack examples use exclusive GPU leases, a 20,000-MB free-memory admission
threshold, and a 300-second wait limit; these are launch safeguards, not new
algorithm parameters. Adjust them to available hardware. Embedding weights
are downloaded on first use; `HF_ENDPOINT` may be set by the caller.

## 2. Data and indexes

The experiment data are hosted on Hugging Face:
[Artifact-E4E0/evilpatch-artifact-data](https://huggingface.co/datasets/Artifact-E4E0/evilpatch-artifact-data).
Download the files from the dataset page, extract any archives, and place the
data under this repository's `data/` directory following the paths in
[data/README.md](data/README.md). If the download already contains a top-level
`data/` folder, merge its contents into the local `data/` folder rather than
creating `data/data/`. Preserve any existing outputs you wish to retain.

The code repository does not bundle datasets. To reproduce the published
experiments, use the released corpora, query splits, and retriever-specific
carriers rather than resampling them; new API runs may still yield different results.

For rebuilding, open `src/preprocessing/data_prep_v4.ipynb` with the EvilPatch
kernel and run it in order. It builds the corpora and Milvus indexes, filters
queries, splits proxy/test sets, selects carriers, and assigns CVE references.
Defaults follow the paper: 50,000 clean BFPs, non-overlapping CoCoNuT historical
queries, eight CWEs, Top-10 filtering/inverse retrieval, approximately 0.5%
carrier budget, and five references selected from Top-10 same-CWE CVEs.

GTE is the default victim retriever. `configs/database/bfp/apr.yml` and
`configs/database/bfp/data_prep_v4/target.yml` must use the same model and index
URI. For Harrier, change both and rebuild/select its carriers separately.
Historical-query and vulnerability indexes use Harrier. If loading prepared
corpora/splits instead of rebuilding them, run notebook initialization and the
Section 2 index-building cells. Later cells skip existing outputs; use a fresh
data tree when changing preprocessing parameters.

## 3. Construct poisoned BFPs

The following is the default GTE/GPBS run. Start from its matching initial
carrier file and make a **working copy**; attack stages modify that copy in place:

```bash
cp data/query/black/target/poison_targets_set_v4.json data/query/black/target/gte/gpbs.json
python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/gte/abgs/abgs.yml
python -m src.pipeline.generation_attack.vics_vinj --config configs/attack/vinj/vinj.yml
python -m src.pipeline.generation_attack.jailbreak --config configs/attack/jailbreak/jailbreak.yml
```

The injection stage expects the five `relevant_vul` references prepared by v4.
Its `skip_rerank: true` reuses that selection. If references are missing, rebuild
Section 6 rather than relying on this fallback setting to select five CVEs.
GPBS uses 50 iterations, patience 10, similarity stopping threshold 0.95, and
candidate budgets of 6,000 for GTE and 4,000 for Harrier. Each paper optimization
task had a 12 GPU-hour resource allocation; the launcher does not enforce this
wall-time budget automatically.

Before evaluation, open `src/utils/distributed_io.ipynb` and run **Section 0
and Section 3 only** for a single-file run. The default Section 3 retains
completed attacks, applies the approximately 0.5% budget, and resolves cross-CWE
ID collisions, writing `data/query/black/target/gte/gpbs_final.json` without
altering the working input. Sections 1/2 are optional sharding/merging utilities.
For baseline comparisons, list all method outputs in Section 3 to retain common
carriers with consistent IDs. Do not reuse finalized negative IDs for new
carrier construction or overwrite prior result files with a new carrier set.

## 4. Evaluate

Open `src/evaluation/retrieval/retrieval_attack_eval.ipynb` and run it in order
using `configs/evaluation/retrieval.yml`. It inserts finalized poisoned BFPs
into the local APR index and reports ASR-r@15 and unnormalized Precision@15.
The default evaluates individual CWE groups and CWE-MIXED (`eval_mode: both`).
Do not run multiple retrieval evaluations against the same index concurrently.

Copy the emitted run directory, **without its trailing `/poisoned`**, into
`data.retrieval_dir_path` in `generation.yml`, `functionality.yml`, and
`mitigation2.yml`; replace the `<RUN_ID>` placeholder. Keep their
`data.poisoned_file_path` pointed at the same finalized carrier file.

```bash
python -m src.evaluation.generation.generation_attack_eval --config configs/evaluation/generation.yml
python -m src.evaluation.functionality.functionality_consistency_eval --config configs/evaluation/functionality.yml
```

Run generation before functionality evaluation, sequentially. Repeat for each
victim generator by changing `model.apr_generator_config` in both configs.
The generation evaluator completes patch generation before judging and resumes
completed work by default. Keep `clear_*` flags false unless intentionally
discarding saved results. The paper's VR is **VR_GLOBAL** (all test queries);
VR_LOCAL is conditional on retrieving poison. CrystalBLEU ignores 500 frequent
shared n-grams and measures similarity, not functional correctness. Metrics and
patches are persisted under the retrieval run directory; logs are under `logs/`.

| Experiment | Configuration changes |
| --- | --- |
| AGGD / PABS | Use the corresponding retrieval template and a fresh matching carrier copy; point injection/DLCI and evaluation configs at that method's files. |
| Naive / without GPBS | Use `configs/attack/retrieval/naive.yml` with a fresh carrier copy. Finalization/retrieval require complete generation fields: either complete injection and DLCI, or reuse the same completed generation file via Section 3. Evaluate without GPBS with DLCI enabled; evaluate Naive with the no-DLCI switches below. |
| Without DLCI | Set `eval_jailbreak_ablation: true` in generation **and functionality** configs on the completed attack; set it consistently in defense evaluation if used. This replaces retrieved patches with their injected, no-DLCI versions. Naive uses these switches too: DLCI fields may exist for staging, but its comments are not passed to the victim. |
| Retrieval depth | Set retrieval `eval_top_k` to `[3, 6, 9, 12, 15, 18, 21]` and APR `retrieval.top_k` to at least 21; match downstream depths. These require new runs, not inferred results. |
| Transferability | Keep the Harrier-constructed finalized BFPs; change the victim APR model/index to GTE, Jina, or Qwen and rebuild the clean index. Do not re-optimize those BFPs. |

The automatic `eval_retrieval_naive` option uses original clean patches and is
**not** the paper's vulnerability-injected Naive baseline; leave it false.

## 5. Defenses

Index-stage detection uses normalized embeddings, quantile 0.95, and covariance
shrinkage 0.01:

```bash
python -m src.evaluation.mitigation.index.index_detection_eval --config configs/evaluation/mitigation.yml
```

CodeGuarder additionally requires its official `Root_Causes.json` knowledge
file at the path documented in `data/README.md` and a **separate** environment
for Jina-v3/FAISS. Do not install these pinned dependencies in EvilPatch:

```bash
conda env create -f configs/evaluation/codeguarder-worker-env.yml
conda activate EvilPatch
python -m src.evaluation.mitigation.generation.generation_guard_eval --config configs/evaluation/mitigation2.yml
```

`codeguarder.env_name` locates the worker environment; alternatively set
`codeguarder.python_executable` to its Python binary. Defense outputs are saved
under the retrieval run's `defense/` directory. Repeat for each victim generator.

## Code map and use restrictions

- Core retrieval algorithms: `src/pipeline/retrieval_attack/optimizer/`.
- Vulnerability injection and DLCI: `src/pipeline/generation_attack/`.
- Model wrappers, indexes, and shared utilities: `src/models/`, `src/rag/`, `src/utils/`.
- Metrics and defense runners: `src/evaluation/`.

Small legacy implementations imported by runtime factories remain to keep those
factories usable; they are not additional paper baselines. AGGD/PABS are adapted
implementations of the methods cited in the paper; CodeGuarder uses external
knowledge records. Retain applicable upstream attribution/data-use conditions.
Use this artifact only for authorized research in researcher-controlled indexes;
do not publish poisoned samples into third-party knowledge bases or repositories.
