# Data directory

Everything under `data/` is **ignored by git**. This directory only reserves the
layout that the source code and the configuration templates expect. All payloads
are distributed separately as a HuggingFace dataset.

```bash
# from the repository root
pip install -U "huggingface_hub[cli]"
huggingface-cli download <HF_DATASET_REPO> --repo-type dataset --local-dir data
```

Replace `<HF_DATASET_REPO>` with the dataset repository id listed in the paper's
Artifact Description (the anonymous mirror URL).

---

## Required layout

| Path | Contents | Consumed by |
|---|---|---|
| `data/raw/bfp/CoCoNut.jsonl` | CoCoNuT C/C++ bug-fix pairs (raw pool) | `src/preprocessing/data_prep_v*.ipynb` |
| `data/raw/bfp/Codeflaws.json` | Codeflaws buggy/accepted submissions | `src/preprocessing/data_prep_v*.ipynb` |
| `data/raw/bfp/Deepfix.json` | DeepFix student programs | `src/preprocessing/data_prep_v*.ipynb` |
| `data/raw/vul/BigVul.jsonl` | BigVul vulnerable functions | `src/preprocessing/vul/**` |
| `data/raw/vul/CVEfixes.jsonl` | CVEfixes vulnerable functions | `src/preprocessing/vul/**` |
| `data/raw/vul/ReposVul_v2.jsonl` | ReposVul vulnerable functions | `src/preprocessing/vul/**` |
| `data/raw/cwe/cwec_v4.19.1.xml` | MITRE CWE dictionary | `configs/attack/vinj/vinj.yml` → `data.cwe_file_path` |
| `data/raw/cwe/CVE_v2026.03.18.json` | NVD CVE dictionary | `configs/attack/vinj/vinj.yml` → `data.cve_file_path` |
| `data/corpus/bfp/rag_data.jsonl` | **Victim APR knowledge base**: 50,000 bug-fix pairs sampled from the three repair sources | `configs/database/bfp/apr.yml` → `data.corpus` |
| `data/corpus/vul/clustered_rag_data.jsonl` | **Vulnerability knowledge base**: 17,576 de-duplicated vulnerable functions from BigVul + CVEfixes + ReposVul, with `cwe_desc` / `cve_desc` / `cluster_id` | `configs/database/vul/vinj.yml` → `data.corpus` |
| `data/query/black/query_set.jsonl` | 50,000 historical queries used as the black-box proxy-query pool | `configs/database/bfp/query.yml` → `data.corpus` |
| `data/query/black/num_17576_alpha_1_target_vul_code_retriever_harrier-oss-v1-0.6b_retrieval_results.json` | CVE → historical-query retrieval results (security-sensitive query filtering) | poison-carrier selection |
| `data/query/black/unsafe_query_set_v4.json` | Security-sensitive queries grouped by CWE | poison-carrier selection |
| `data/query/black/proxy/proxy_query_set_v4.json` | Proxy query set `Q⁺` (80 % of the security-sensitive queries) | `configs/database/bfp/data_prep_v4/target.yml` → `data.query` |
| `data/query/black/test/test_query_set_v4.json` | Held-out test query set (20 %) | `configs/evaluation/retrieval.yml` → `data.test_file_path` |
| `data/query/black/target/poison_targets_set_v4.json` | Poisoning carriers + proxy queries + retrieved CVE references | `configs/attack/retrieval/*/**.yml` → `data.target_file_path` |
| `data/milvus/bfp/apr/gte-modernbert-base.db` | Pre-built APR vector index (GTE victim retriever) | `configs/database/bfp/apr.yml` → `database.uri` |
| `data/milvus/bfp/apr/jina-code-embeddings-0.5b.db` | Pre-built APR vector index (Jina victim retriever) | `configs/database/bfp/apr.yml` |
| `data/milvus/bfp/query/black/harrier-oss-v1-0.6b.db` | Historical-query vector index | `configs/database/bfp/query.yml` → `database.uri` |
| `data/milvus/vul/harrier-oss-v1-0.6b.db` | Vulnerable-function vector index | `configs/database/vul/vinj.yml` → `database.uri` |
| `data/cache/` | Runtime caches (CrystalBLEU n-grams, detector references, injection cache) | created automatically |

Additional `data/query/{white,gray}/` subtrees are reserved for the white-box and
grey-box variants of the query sets. The paper's main results use the black-box
setting (`data/query/black/`).

---

## Regenerating the data instead of downloading it

The preprocessing notebooks under `src/preprocessing/` rebuild
`data/corpus/bfp/rag_data.jsonl` and `data/corpus/vul/clustered_rag_data.jsonl`
from the raw datasets:

```bash
jupyter nbconvert --to notebook --execute src/preprocessing/data_prep_v1.ipynb
jupyter nbconvert --to notebook --execute src/preprocessing/data_prep_v4.ipynb
```

This is optional — the released dataset already contains the built corpora.

The Milvus Lite indexes under `data/milvus/` can always be rebuilt from the
corpora with `src/rag/bfp/retrieval.ipynb` (APR and query indexes) and
`src/rag/vul/retrieval.ipynb` (vulnerability index).
