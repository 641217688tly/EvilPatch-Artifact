# Data layout

Data are hosted at
[Artifact-E4E0/evilpatch-artifact-data](https://huggingface.co/datasets/Artifact-E4E0/evilpatch-artifact-data)
on Hugging Face and are not bundled in this code repository. Download the files,
extract any archives, and place them under `data/` according to the layout below.
Avoid an extra `data/data/` level if the download already contains a `data/` folder.
`.gitkeep` files only reserve directories; they are not data payloads.

For the published experiments, use the released frozen corpora, query splits,
and retriever-specific carrier files rather than resampling them. New runs of
preprocessing may select different carriers and do not guarantee identical metrics.

| Path under `data/` | Contents |
| --- | --- |
| `corpus/bfp/rag_data.jsonl` | 50,000 clean BFPs |
| `corpus/vul/clustered_rag_data.jsonl` | Deduplicated BigVul, CVEfixes, and ReposVul vulnerability corpus |
| `query/black/query_set.jsonl` | 50,000 historical CoCoNuT BFPs disjoint from the clean corpus |
| `query/black/unsafe_query_set_v4.json` | Security-sensitive query groups used for detector calibration |
| `query/black/proxy/proxy_query_set_v4.json` | CWE-keyed proxy query groups |
| `query/black/test/test_query_set_v4.json` | Disjoint CWE-keyed test query groups |
| `query/black/target/poison_targets_set_v4.json` | Initial carriers with proxy anchors and five vulnerability references |
| `query/black/target/{gte,harrier}/` | Working and finalized attack files for each retriever |
| `raw/cwe/cwec_v4.19.1.xml` | CWE definitions required by vulnerability injection |
| `defense/codeguarder/Root_Causes.json` | Official CodeGuarder security knowledge records |
| `milvus/`, `cache/`, `query/black/test/results/` | Locally generated indexes, caches, and results |

To rebuild data with `src/preprocessing/data_prep_v4.ipynb`, supply these
already formatted source files (raw-dataset-specific conversion scripts are
not included in this minimal artifact):

- `raw/bfp/CoCoNut.jsonl`, `raw/bfp/Codeflaws.json`, `raw/bfp/Deepfix.json`.
  Each record requires `language`, `buggy_code`, and `fixed_code`.
- `raw/vul/BigVul.jsonl`, `raw/vul/CVEfixes.jsonl`, `raw/vul/ReposVul_v2.jsonl`.
  Each record requires `id`, `source`, `language`, `cwe_id`, `cve_id`, `cwe_desc`,
  `cve_desc`, `vul_code`, and `fixed_code`. Use scalar CWE identifiers such as
  `CWE-787`; records must have consistent identifiers across the combined corpus.

The notebook skips existing outputs. Run changed preprocessing settings in a
fresh data tree (or archive the corresponding outputs and indexes first);
loading old outputs does not update their sampling parameters or data sources.
Keep query IDs, carrier IDs, and their associated metadata together when moving
released files. Finalization handles carrier ID collisions across CWE groups.
