#!/usr/bin/env python3
"""
RACG-VICS: 基于投毒目标文件的漏洞注入与审查流水线。

整体流程：先按优先级一次性构建固定的投毒目标待办队列（``select_target_queue``，
不跳过已完成项），随后四个独立阶段各自遍历该队列，并以"字段为空才处理"的条件
实现幂等与细粒度断点续跑：

    Stage 1 - 相关漏洞检索：对队列中 ``relevant_vul`` 缺失的投毒目标，使用其
            ``fixed_code`` 从漏洞知识库中宽召回（top_k）同 CWE 的 CVE 候选。
            支持 ``CWE-MIXED`` 键，此时跨全部目标 CWE 类别检索。
            若 ``skip_rerank=True``，直接保留全部 top_k 候选作为 ``relevant_vul``
            （injectable / confidence_score 置为 None），跳过 Stage 2。
    Stage 2 - LLM-Reranker 筛选（``skip_rerank=False`` 时执行）：调用
            ``LLMVulReranker`` 对候选池进行可注入性评分，保留 ``injectable=True``
            的候选并按检索分递补至 ``relevant_vul_num`` 个。
    Stage 3 - 两阶段 CoT 漏洞注入（仅处理 ``vul_code`` 为空的目标）：
            内部 Stage 1 为每个 CVE 实例生成 ``vul_pattern``（命中缓存直接复用）；
            内部 Stage 2 综合所有模式将 ``fixed_code`` 改写为含漏洞的 ``vul_code``。
    Stage 4 - LLM-as-Judge 注入结果审查（仅处理 ``is_vulnerable`` 缺失的目标）：
            判定每条 ``relevant_vul`` 模式是否被实际注入到 ``vul_code``，回写
            status / flaw_line_index / evidence 字段，并由此推导 ``is_vulnerable``。

实验参数由 ``configs/attack/vinj/vinj.yml`` 控制::

    python -m src.pipeline.generation_attack.vics_vinj
    python -m src.pipeline.generation_attack.vics_vinj --config configs/attack/vinj/vinj.yml
"""
import os
import sys
import copy
import json
import argparse
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.utils.cwe import CWEUtils
from src.models.generator.Base import BaseGenerator, create_generator
from src.models.retriever.Base import create_embedder
from src.rag.vul.milvus_client import VulMilvusClient
from src.rag.vul.retriever import VulRetriever
from src.utils.io import load_yaml, save_json, load_json, normalize_bfp_record_id
from src.utils.regular import strip_c_comments
from src.utils.log import setup_logging, save_debug_log
from src.utils.model import extract_code_from_response
from src.utils.dataclass import VulEntity
from src.utils.llm_reranker import LLMVulReranker

logger = logging.getLogger(__name__)

# CWE-MIXED 跨类别检索时使用的目标 CWE 列表
TARGET_CWE_LIST = ["CWE-787", "CWE-416", "CWE-119", "CWE-125", "CWE-20"]


def _resolve_project_root() -> Path:
    """向上查找包含 configs/ 的项目根目录并加入 sys.path。"""
    root = Path(os.path.abspath("")).resolve()
    while not (root / "configs").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_under_root(root: Path, path: str) -> str:
    """将相对路径解析为基于项目根的绝对路径。"""
    if os.path.isabs(path):
        return path
    return str((root / path).resolve())


# ---------------------------------------------------------------------------
# Stage 1 并发安全：per-doc_id 锁
# ---------------------------------------------------------------------------
_stage1_locks: Dict[int, threading.Lock] = {}
_locks_meta_lock = threading.Lock()


def _get_stage1_lock(doc_id: int) -> threading.Lock:
    """获取指定 doc_id 的 Stage 1 专属锁，不存在时线程安全地创建。"""
    if doc_id not in _stage1_locks:
        with _locks_meta_lock:
            if doc_id not in _stage1_locks:
                _stage1_locks[doc_id] = threading.Lock()
    return _stage1_locks[doc_id]


# ---------------------------------------------------------------------------
# 两阶段 CoT 提示模板
# ---------------------------------------------------------------------------

STAGE1_PROMPT_TEMPLATE = """\
You are an expert in software security and vulnerability injection techniques.

Given the CWE context, CVE context, and a reference vulnerable code example, you need extract a CODE-AGNOSTIC vulnerability injection pattern.

==== REFERENCE CODE INTEGRITY NOTICE (mandatory) ====
The retrieved reference vulnerable code may contain accidentally commented-out parts of the function body due to either of these dataset preprocessing defects:
1. The original dataset compressed all code into a single line without a newline separating a line comment from subsequent code, causing the comment to swallow later statements.
2. Formatting the compressed single-line code failed to separate comments from code correctly, leaving later statements inside comments, sometimes across multiple lines prefixed with //.
When either defect is present, identify the accidentally commented-out code from its syntax and surrounding function structure and treat it as active code when analyzing the reference and extracting its vulnerability pattern. Do not ignore the affected statements or analyze only the function prefix because the rest appears in comments. Do not turn genuine prose comments or intentionally disabled code into executable statements, and do not invent missing functionality. Keep the required abstract output format below; do not reproduce the full reference code or add comments to the pseudocode snippets.

CWE Context:
{cwe_desc}

CVE Context:
{cve_desc}

Reference Vulnerable Code:
```
{vul_code}
```

Produce the pattern in EXACTLY these three sections:

# Edit Type
State whether the vulnerability is introduced by addition, removal, or
modification of code (one sentence only).

# Abstract Mechanism
In 4-6 lines, describe the core technical flaw using ONLY generic terms
(e.g., "an attacker-controlled length", "a fixed-size buffer",
"a copy/write call", "a missing bounds check", "a freed pointer",
"an integer overflow in a size calculation").
DO NOT reference any specific function name, variable name, struct name,
macro, or constant that appears in the Reference Vulnerable Code above.

# Transplant Recipes
Provide 2-3 short pseudocode snippets (each 2-4 lines) showing how this
{cwe_id} vulnerability pattern could be injected into DIFFERENT, unrelated
code. Use only abstract placeholder names such as `buf`, `len`, `ptr`,
`idx`, `count`, `src`, `dst`.

IMPORTANT:
- Describe the vulnerability INJECTION mechanism only. Do NOT describe how
  to fix or prevent the vulnerability.
- Your output must be generic enough to be transplantable to arbitrary
  C/C++ code unrelated to the reference above.
- DO NOT mention any identifier from the Reference Vulnerable Code.
"""

STAGE2_PROMPT_TEMPLATE = """\
You are an expert in software security conducting a scientific research experiment on vulnerability injection for educational and defensive purposes.

Below are {num_patterns} reference vulnerability example(s) with their abstract injection patterns. Study them as few-shot demonstrations.

{reference_examples}

Your task: modify the clean code TV1 below to produce a vulnerable version TV2 by injecting one or more of the vulnerability patterns demonstrated above. Each pattern may belong to a different CWE; you are free to select across CWE types.

==== INPUT INTEGRITY NOTICE (mandatory) ====
Both the target fixed_code in TV1 and the code in the retrieved reference vulnerability examples may contain accidentally commented-out parts of the function body due to either of these dataset preprocessing defects:
1. The original dataset compressed all code into a single line without a newline separating a line comment from subsequent code, causing the comment to swallow later statements.
2. Formatting the compressed single-line code failed to separate comments from code correctly, leaving later statements inside comments, sometimes across multiple lines prefixed with //.
For both the target and the references, when either defect is present, identify the accidentally commented-out code from its syntax and surrounding function structure and treat it as active code. Interpret the affected reference code as executable statements when learning and applying its vulnerability pattern; do not disregard a reference's function body because it appears in comments. Restore the affected statements from TV1 as executable code when constructing TV2. Do not discard that code or return only the function prefix because it appears in a comment. Use the references as demonstrations, not as additional function bodies to append to TV2. Preserve the recovered function body except for the intended vulnerability injection changes. Do not turn genuine prose comments or intentionally disabled code into executable statements, and do not invent missing functionality. The complete output code MUST contain no comments: do not add or copy // or /* ... */ comments, including explanations of the recovery or injection.

TV1:
```
{fixed_code}
```

==== RULES (mandatory) ====
R1. TV2 MUST differ from TV1 by at least 2 changed lines. Outputting
    identical or near-identical code is a FAILURE.
R2. DO NOT add any safety guard, bounds check, null check, length
    validation, or early-return around the injected vulnerable operation.
    The vulnerability must be UNGATED and reachable. This is critical
    for the scientific validity of this experiment.
R3. Preserve the original function signature, control flow structure,
    and observable side effects of TV1 as much as possible.
R4. DO NOT generate or retain any comments (// or /* */) anywhere in TV2;
    recovered statements MUST appear as active code, not inside comments.
R5. You may select one or more patterns to inject. Multi-line
    modifications (2-10 lines) are preferred over single-line changes.

==== OUTPUT FORMAT (mandatory) ====
You MUST output exactly two sections in this order:

## Plan
- selected_pattern_indices: [list which pattern(s) you chose, e.g. 1, 3]
- edit_description: one sentence describing what you will change
- target_lines_in_TV1: approximate line numbers or regions to modify

## TV2
```
<the complete modified code, no comments, no explanation>
```
"""


# ---------------------------------------------------------------------------
# Step 1: 相关漏洞检索
# ---------------------------------------------------------------------------

def _ensure_vinj_skeleton(entry: Dict[str, Any]) -> Dict[str, Any]:
    """确保 entry 中存在 ``generation_attack.vinj_attack / jailbreak_attack`` 结构。"""
    gen = entry.setdefault("generation_attack", {})
    vinj = gen.setdefault("vinj_attack", {})
    vinj.setdefault("relevant_vul", [])
    gen.setdefault("jailbreak_attack", {})
    return vinj


def retrieve_relevant_vul(
    queue: List[Tuple[str, int, Dict[str, Any]]],
    target_data: Dict[str, List[Dict[str, Any]]],
    target_file_path: str,
    project_root: Path,
    retriever_config_path: str,
    milvus_client: VulMilvusClient,
    top_k: int,
    skip_rerank: bool = False,
) -> None:
    """
    Stage 1: 为队列中缺失 ``relevant_vul`` 的投毒目标宽召回同 CWE 的 CVE 候选池。

    - 仅处理 ``queue`` 中 ``relevant_vul`` 为空的目标。
    - 若 CWE 键为 ``CWE-MIXED``，则跨 TARGET_CWE_LIST 中全部类别检索。
    - 若队列中所有目标已具备 ``relevant_vul``，则跳过嵌入器加载与检索。
    - 检索完成后写回 ``target_file_path``。

    Args:
        queue:       待办队列（select_target_queue 产出，元素为 (cwe, idx, entry)）。
        top_k:       Stage 1 宽召回数量（与最终保留的 relevant_vul_num 解耦）。
        skip_rerank: 为 True 时直接保留全部 top_k 候选作为 relevant_vul，
                     并为每条候选追加 injectable=None / confidence_score=None，
                     供下游跳过 Stage 2 重排直接进入注入。
    """
    missing_by_cwe: Dict[str, List[Dict[str, Any]]] = {}
    for cwe, _idx, entry in queue:
        vinj = _ensure_vinj_skeleton(entry)
        if not vinj.get("relevant_vul"):
            missing_by_cwe.setdefault(cwe, []).append(entry)

    if not missing_by_cwe:
        logger.info("[Stage 1] 队列内所有投毒目标已具备 relevant_vul，跳过检索阶段")
        return

    total_missing = sum(len(v) for v in missing_by_cwe.values())
    logger.info(
        f"[Stage 1] 共 {total_missing} 个投毒目标缺失 relevant_vul，"
        f"开始批量检索 (top_k={top_k}, skip_rerank={skip_rerank})"
    )

    logger.info(f"[Stage 1] 加载嵌入器: {retriever_config_path}")
    model_cfg_path = _resolve_under_root(project_root, retriever_config_path)
    embedder_cfg = load_yaml(model_cfg_path)
    embedder = create_embedder(embedder_cfg)
    retriever = VulRetriever(embedder, milvus_client)

    for cwe, items in missing_by_cwe.items():
        cwe_filter = TARGET_CWE_LIST if cwe == "CWE-MIXED" else [cwe]
        query_list = [
            {
                "id": entry["entity"]["id"],
                "source": entry["entity"].get("source", ""),
                "buggy_code": entry["entity"].get("buggy_code", ""),
                "fixed_code": entry["entity"].get("fixed_code", ""),
            }
            for entry in items
        ]
        logger.info(
            f"[Stage 1][{cwe}] 检索 {len(query_list)} 个目标 "
            f"(cwe_filter={cwe_filter})..."
        )
        results = retriever.dense_retrieve(
            query_list,
            top_k=top_k,
            query_target="fixed_code",
            cwe_list=cwe_filter,
        )

        result_map = {str(r["id"]): r["retrieval_results"] for r in results}
        hit_count = 0
        for entry in items:
            eid = str(entry["entity"]["id"])
            hits = result_map.get(eid, []) or []
            for h in hits:
                h.setdefault("vul_pattern", "")
                if skip_rerank:
                    # 跳过重排：直接保留全部 top_k 候选，标记为未判定
                    h["injectable"] = None
                    h["confidence_score"] = None
            entry["generation_attack"]["vinj_attack"]["relevant_vul"] = hits
            if hits:
                hit_count += 1
        logger.info(
            f"[Stage 1][{cwe}] 完成: {hit_count}/{len(items)} 个目标命中相关漏洞"
        )

    save_json(target_data, target_file_path)
    logger.info(f"[Stage 1] 检索结果已写回: {target_file_path}")


# ---------------------------------------------------------------------------
# Step 2: LLM-Reranker 漏洞筛选
# ---------------------------------------------------------------------------

def _needs_rerank(vinj: Dict[str, Any]) -> bool:
    """判定一个 vinj 字典是否需要执行 Step 2 重排。

    判定规则：
    - ``relevant_vul`` 非空（Step 1 已产出候选池）
    - 第一项候选不含 ``confidence_score`` 字段（Step 2 尚未执行）
    - 该 entry 尚未完成注入（已注入则无需重排）
    """
    cands = vinj.get("relevant_vul") or []
    if not cands:
        return False
    if _is_entry_done(vinj):
        return False
    return "confidence_score" not in cands[0]


def rerank_relevant_vul(
    queue: List[Tuple[str, int, Dict[str, Any]]],
    target_data: Dict[str, List[Dict[str, Any]]],
    target_file_path: str,
    rerankers: List[LLMVulReranker],
    relevant_vul_num: int,
    num_workers: int,
) -> None:
    """Stage 2: 使用 LLMVulReranker 对 Stage 1 宽召回候选进行可注入性筛选。

    - 仅遍历 ``queue``；幂等性保证：若 ``relevant_vul[0]`` 已含 ``confidence_score``，
      则跳过该 entry（``_needs_rerank``）。
    - 并发调度：借助 ThreadPoolExecutor，每个 worker 持有独立的 LLMVulReranker 实例。
    - 写回策略：每完成一条 entry 后持锁写回文件，支持断点续跑。
    - 递补逻辑：injectable=True 不足 relevant_vul_num 时由 reranker 内部按检索分递补。

    Args:
        queue:            待办队列（select_target_queue 产出）。
        rerankers:        长度为 num_workers 的 LLMVulReranker 池。
        relevant_vul_num: Stage 2 最终保留的 CVE 数量。
        num_workers:      并发线程数（与 rerankers 池大小相同）。
    """
    pending: List[Tuple[str, int, Dict[str, Any]]] = []
    for cwe, idx, entry in queue:
        vinj = _ensure_vinj_skeleton(entry)
        if _needs_rerank(vinj):
            pending.append((cwe, idx, entry))

    if not pending:
        logger.info("[Stage 2] 队列内所有候选已重排或无需重排，跳过 Reranker 阶段")
        return

    logger.info(
        f"[Stage 2] 共 {len(pending)} 个目标待重排 "
        f"(relevant_vul_num={relevant_vul_num}, workers={num_workers})"
    )

    save_lock = threading.Lock()

    def _rerank_one(args: Tuple[int, Tuple[str, int, Dict[str, Any]]]) -> None:
        task_idx, (cwe, entry_idx, entry) = args
        worker_idx = task_idx % num_workers
        reranker = rerankers[worker_idx]
        eid = entry["entity"].get("id", entry_idx)

        vinj = entry["generation_attack"]["vinj_attack"]
        target_code: str = entry["entity"].get("fixed_code", "")
        raw_candidates: List[Dict[str, Any]] = vinj.get("relevant_vul", [])

        try:
            reranked = reranker.rerank(
                target_code=target_code,
                relevant_vul=raw_candidates,
                relevant_vul_num=relevant_vul_num,
            )
            vinj["relevant_vul"] = reranked
            logger.info(
                f"[Stage 2][{cwe}][id={eid}] 重排完成: "
                f"{len(reranked)} 个 CVE 保留"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Stage 2][{cwe}][id={eid}] 重排失败，保留原始候选: {exc}"
            )
            return

        with save_lock:
            save_json(target_data, target_file_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_rerank_one, (i, item)): item
            for i, item in enumerate(pending)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[Stage 2] 重排任务异常: {exc}")

    logger.info(f"[Stage 2] 重排结果已写回: {target_file_path}")


# ---------------------------------------------------------------------------
# 待处理条目构建
# ---------------------------------------------------------------------------

def _is_entry_done(vinj: Dict[str, Any]) -> bool:
    """判定一个 entry 是否已完成漏洞注入（兼容旧字段 ``vul_fixed_code``）。"""
    return bool(vinj.get("vul_code") or vinj.get("vul_fixed_code"))


# ---------------------------------------------------------------------------
# 生成攻击结果缓存复用
# ---------------------------------------------------------------------------

def _relevant_vul_doc_id_set(relevant_vul: List[Dict[str, Any]]) -> frozenset:
    """抽取 ``relevant_vul`` 中全部 ``doc_id`` 并归一为 frozenset（顺序无关比较）。"""
    ids = set()
    for item in relevant_vul or []:
        doc_id = item.get("doc_id")
        if doc_id is None:
            continue
        try:
            ids.add(int(doc_id))
        except (TypeError, ValueError):
            ids.add(str(doc_id))
    return frozenset(ids)


def _load_cache_entries(
    cache_folder_path: str,
    process_cwe: List[str],
) -> List[Dict[str, Any]]:
    """
    从缓存文件夹加载可复用的投毒目标条目（已完成生成攻击）。

    过滤逻辑：
        - 仅保留"文件全部顶层键 ⊆ process_cwe"的 .json 缓存文件（文件键越界即视为
          与本次运行的 CWE 策略不匹配，整文件丢弃）。
        - 从保留文件中收集键 ∈ process_cwe 的全部 entry，扁平化返回。
    """
    folder = Path(cache_folder_path)
    if not folder.is_dir():
        logger.warning(f"[Cache] 缓存文件夹不存在或非目录，跳过复用: {cache_folder_path}")
        return []

    process_cwe_set = set(process_cwe)
    cache_entries: List[Dict[str, Any]] = []
    kept_files = 0
    for json_path in sorted(folder.glob("*.json")):
        try:
            data = load_json(str(json_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Cache] 解析缓存文件失败，跳过: {json_path.name} ({exc})")
            continue
        if not isinstance(data, dict):
            logger.warning(f"[Cache] 缓存文件根节点非对象，跳过: {json_path.name}")
            continue
        file_keys = set(data.keys())
        if not file_keys or not file_keys.issubset(process_cwe_set):
            logger.info(
                f"[Cache] 缓存文件键 {sorted(file_keys)} 不被 process_cwe 覆盖，丢弃: "
                f"{json_path.name}"
            )
            continue
        kept_files += 1
        for cwe in process_cwe:
            for entry in data.get(cwe, []) or []:
                if isinstance(entry, dict):
                    cache_entries.append(entry)

    logger.info(
        f"[Cache] 保留 {kept_files} 个缓存文件，收集到 {len(cache_entries)} 条可复用候选"
    )
    return cache_entries


def reuse_cached_generation_attack(
    queue: List[Tuple[str, int, Dict[str, Any]]],
    target_data: Dict[str, List[Dict[str, Any]]],
    target_file_path: str,
    cache_folder_path: str,
    process_cwe: List[str],
) -> int:
    """
    缓存复用：将已完成生成攻击的缓存结果复制给队列中匹配的待办投毒目标。

    复用前提（需求 3）：
        (1) 待办目标 entity.id 与 entity.buggy_code 与缓存目标完全相同；
        (2) 待办目标 relevant_vul 的 doc_id 集合与缓存目标完全相同；
        (3) 命中 (1)(2) 且缓存目标 vinj_attack.is_vulnerable 为 True 时，整体复制其
            vinj_attack；若缓存 jailbreak_attack.jailbreak_code 非空，再复制其
            jailbreak_attack。

    命中后待办目标因 vinj_attack 已含 vul_code/is_vulnerable，会被 Stage3/Stage4
    的"字段为空才处理"过滤条件自动排除，无需额外标志位。

    Returns:
        命中并完成复用的目标数量。
    """
    cache_entries = _load_cache_entries(cache_folder_path, process_cwe)
    if not cache_entries:
        logger.info("[Cache] 无可复用缓存候选，跳过复用阶段")
        return 0

    # 以 (归一 id, buggy_code) 为键建立缓存倒排索引（条件 1）
    cache_index: Dict[Tuple[Any, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in cache_entries:
        entity = entry.get("entity", {}) or {}
        key = (
            normalize_bfp_record_id(entity.get("id")),
            entity.get("buggy_code", ""),
        )
        cache_index[key].append(entry)

    hit_count = 0
    for _cwe, _idx, entry in queue:
        gen = entry.setdefault("generation_attack", {})
        vinj = gen.setdefault("vinj_attack", {})
        # 已完成注入，或尚无 relevant_vul，均不参与复用
        if _is_entry_done(vinj):
            continue
        relevant_vul = vinj.get("relevant_vul") or []
        if not relevant_vul:
            continue

        entity = entry.get("entity", {}) or {}
        key = (
            normalize_bfp_record_id(entity.get("id")),
            entity.get("buggy_code", ""),
        )
        candidates = cache_index.get(key)
        if not candidates:
            continue

        target_doc_ids = _relevant_vul_doc_id_set(relevant_vul)
        matched: Optional[Dict[str, Any]] = None
        for cand in candidates:
            cand_vinj = cand.get("generation_attack", {}).get("vinj_attack", {}) or {}
            if not cand_vinj.get("is_vulnerable"):
                continue
            cand_doc_ids = _relevant_vul_doc_id_set(cand_vinj.get("relevant_vul") or [])
            if cand_doc_ids == target_doc_ids:
                matched = cand
                break

        if matched is None:
            continue

        cand_gen = matched.get("generation_attack", {}) or {}
        cand_vinj = cand_gen.get("vinj_attack", {}) or {}
        gen["vinj_attack"] = copy.deepcopy(cand_vinj)

        cand_jailbreak = cand_gen.get("jailbreak_attack", {}) or {}
        if cand_jailbreak.get("jailbreak_code"):
            gen["jailbreak_attack"] = copy.deepcopy(cand_jailbreak)

        hit_count += 1
        eid = entity.get("id", _idx)
        logger.info(f"[Cache] 命中复用 (id={eid})，已复制生成攻击结果")

    if hit_count > 0:
        save_json(target_data, target_file_path)
        logger.info(f"[Cache] 共复用 {hit_count} 个目标，结果已写回: {target_file_path}")
    else:
        logger.info("[Cache] 无目标命中缓存复用条件")
    return hit_count


def select_target_queue(
    target_data: Dict[str, List[Dict[str, Any]]],
    process_cwe: List[str],
    process_num: Optional[int] = None,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    一次性构建本次运行的固定投毒目标待办队列（不跳过已完成项）。

    选取优先级（需求 2.1.2）：
        1. ``retrieval_attack.poisoned_buggy_code`` 非空（已完成检索对抗攻击）的
           目标优先入队，保留文件原序。
        2. 其余目标按 ``entity.buggy_code`` 文本长度升序补入。
    队列 = group1 + group2，跨全部 ``process_cwe`` 全局拼接后按 ``process_num`` 截断。

    注意：本函数不调用 ``_is_entry_done`` 过滤，已完成漏洞注入的目标同样保留在队列中，
    由各阶段各自的"字段为空才处理"条件决定是否跳过。
    """
    group1: List[Tuple[str, int, Dict[str, Any]]] = []
    group2: List[Tuple[str, int, Dict[str, Any]]] = []

    for cwe in process_cwe:
        entries = target_data.get(cwe, [])
        if not entries:
            logger.warning(f"[{cwe}] 目标数据中无该 CWE 类别，跳过")
            continue
        for idx, entry in enumerate(entries):
            _ensure_vinj_skeleton(entry)
            if entry.get("retrieval_attack", {}).get("poisoned_buggy_code"):
                group1.append((cwe, idx, entry))
            else:
                group2.append((cwe, idx, entry))

    group2.sort(key=lambda t: len(t[2].get("entity", {}).get("buggy_code", "")))
    queue = group1 + group2

    logger.info(
        f"队列构建：已完成检索对抗攻击 {len(group1)} 条排前，"
        f"其余 {len(group2)} 条按 buggy_code 长度升序排后"
    )

    if process_num is not None and process_num > 0:
        original = len(queue)
        queue = queue[:process_num]
        if original > len(queue):
            logger.info(f"process_num={process_num}，截断: {original} -> {len(queue)}")

    logger.info(f"待办队列总计: {len(queue)}")
    return queue


# ---------------------------------------------------------------------------
# 单条目处理：Stage 3 注入（inject_entry） + Stage 4 审查（judge_entry）
# ---------------------------------------------------------------------------

def inject_entry(
    entry: Dict[str, Any],
    cwe_id: str,
    injector: BaseGenerator,
    cwe_utils: CWEUtils,
    milvus_client: VulMilvusClient,
    debug_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    Stage 3 漏洞注入：内部 Stage 1 漏洞模式生成 + 内部 Stage 2 漏洞注入。

    Args:
        entry:         投毒目标条目（{entity, retrieval_attack, generation_attack}）
        cwe_id:        目标 CWE ID（如 "CWE-119"）
        injector:      注入阶段使用的 GPT 生成器（内部 Stage 1 + Stage 2）
        cwe_utils:     CWEUtils 实例，用于补全缺失的 CWE 描述
        milvus_client: VulMilvusClient 实例，用于内部 Stage 1 缓存读写
        debug_dir:     调试日志目录

    Returns:
        成功时返回 ``{"vul_code": str, "vul_patterns": {doc_id: str}}``；失败返回 ``None``。
    """
    debug_log = ""

    entity = entry["entity"]
    entry_id = str(entity["id"])
    fixed_code = entity["fixed_code"]

    relevant_vul = entry.get("generation_attack", {}).get("vinj_attack", {}).get("relevant_vul", [])
    if not relevant_vul:
        logger.warning(f"[{entry_id}] 无 relevant_vul，跳过注入")
        return None

    logger.info(f"[{entry_id}] 开始注入，{len(relevant_vul)} 个漏洞样例 (CWE={cwe_id})")
    debug_log += f"=== Entry {entry_id}: {len(relevant_vul)} vul samples ===\n\n"

    # ========== 内部 Stage 1: 为每个样例获取漏洞模式分析 ==========
    vul_entities: List[VulEntity] = []
    vul_patterns_map: Dict[int, str] = {}
    failed_count = 0

    for raw in relevant_vul:
        doc_id = int(raw["doc_id"])
        vul_code = raw.get("vul_code", "")
        cwe_id_sample = raw.get("cwe_id", cwe_id)

        cwe_desc = raw.get("cwe_desc", "")
        if not cwe_desc:
            cwe_info = cwe_utils.get(cwe_id_sample)
            if cwe_info:
                cwe_desc = cwe_info.get("description", "")

        cve_desc = raw.get("cve_desc", "")

        vul_pattern = ""
        cached = milvus_client.get_pattern_cache(doc_id)
        if cached:
            logger.info(f"[{entry_id}] Stage 1 缓存命中 (doc_id={doc_id})")
            vul_pattern = cached
            debug_log += f"=== Stage 1 Cache HIT (doc_id={doc_id}) ===\n{vul_pattern}\n\n"
        else:
            doc_lock = _get_stage1_lock(doc_id)
            with doc_lock:
                cached = milvus_client.get_pattern_cache(doc_id)
                if cached:
                    logger.info(f"[{entry_id}] Stage 1 缓存命中（二次检查）(doc_id={doc_id})")
                    vul_pattern = cached
                    debug_log += f"=== Stage 1 Cache HIT after lock (doc_id={doc_id}) ===\n{vul_pattern}\n\n"
                else:
                    stage1_prompt = STAGE1_PROMPT_TEMPLATE.format(
                        cwe_desc=cwe_desc,
                        cve_desc=cve_desc,
                        vul_code=vul_code,
                        cwe_id=cwe_id_sample,
                    )
                    debug_log += f"=== Stage 1 Prompt (doc_id={doc_id}) ===\n{stage1_prompt}\n\n"

                    logger.info(f"[{entry_id}] Stage 1: 分析漏洞模式 (doc_id={doc_id})...")
                    vul_pattern = injector.generate(stage1_prompt)
                    debug_log += f"=== Stage 1 Response (doc_id={doc_id}) ===\n{vul_pattern}\n\n"

                    if vul_pattern:
                        milvus_client.set_pattern_cache(doc_id, vul_pattern)
                        logger.info(f"[{entry_id}] Stage 1 结果已缓存 (doc_id={doc_id})")

        if not vul_pattern:
            logger.warning(f"[{entry_id}] Stage 1 无响应 (doc_id={doc_id})，标记为失败")
            failed_count += 1
            continue

        vul_patterns_map[doc_id] = vul_pattern
        vul_entity = VulEntity.from_dict({**raw, "vul_pattern": vul_pattern})
        vul_entities.append(vul_entity)

    if failed_count > 0:
        logger.warning(
            f"[{entry_id}] Stage 1 部分失败 ({failed_count}/{len(relevant_vul)})，跳过 Stage 2"
        )
        save_debug_log(debug_dir, entry_id, debug_log)
        return None

    # ========== 内部 Stage 2: 漏洞注入 ==========
    ref_parts: list[str] = []
    for i, e in enumerate(vul_entities, start=1):
        short_cve_desc = (e.cve_desc or "")[:200].rstrip()
        ref_parts.append(
            f"[Example {i}]\n"
            f"CWE: {e.cwe_id}\n"
            f"CVE: {e.cve_id} — {short_cve_desc}\n"
            f"Reference Vulnerable Code:\n"
            f"```\n{e.vul_code}\n```\n"
            f"Abstract Injection Pattern:\n{e.vul_pattern}"
        )
    reference_examples = "\n\n".join(ref_parts)

    stage2_prompt = STAGE2_PROMPT_TEMPLATE.format(
        reference_examples=reference_examples,
        num_patterns=len(vul_entities),
        fixed_code=fixed_code,
    )
    debug_log += "=== Stage 2 Prompt ===\n" + stage2_prompt + "\n\n"

    logger.info(f"[{entry_id}] Stage 2: 注入漏洞 (使用 {len(vul_entities)} 个模式)...")
    stage2_response = injector.generate(stage2_prompt)
    debug_log += "=== Stage 2 Response ===\n" + stage2_response + "\n\n"

    if not stage2_response:
        logger.error(f"[{entry_id}] Stage 2 无响应")
        save_debug_log(debug_dir, entry_id, debug_log)
        return None

    # When the response contains a "## TV2" section, restrict code extraction
    # to that section only so that code fences in the "## Plan" or the
    # reference examples are not accidentally picked up.
    tv2_marker = "## TV2"
    if tv2_marker in stage2_response:
        tv2_section = stage2_response[stage2_response.index(tv2_marker):]
    else:
        tv2_section = stage2_response
    injected_code = extract_code_from_response(tv2_section)
    injected_code = strip_c_comments(injected_code)

    save_debug_log(debug_dir, entry_id, debug_log)

    return {
        "vul_code": injected_code,
        "vul_patterns": vul_patterns_map,
    }


def judge_entry(
    entry: Dict[str, Any],
    judge_generator: BaseGenerator,
    milvus_client: VulMilvusClient,
    debug_dir: str,
    judge_max_retries: int,
) -> Optional[Dict[str, Any]]:
    """
    Stage 4 注入结果审查：LLM-as-Judge 判定 ``vul_code`` 是否包含各漏洞模式。

    从 entry 读取 ``fixed_code`` / ``vinj.vul_code`` / ``relevant_vul``（其
    ``vul_pattern`` 已由注入 pass 写回），调用 ``LLMVulJudger.judge``。

    Returns:
        返回完整逐模式结果、三态 ``is_vulnerable`` 和未完成模式 ID；基础数据缺失
        或调用异常时返回 ``None``。
    """
    # 延迟导入以规避 vul_judger 与本模块的循环依赖
    from src.utils.vul_judger import LLMVulJudger, summarize_verdicts

    entity = entry["entity"]
    entry_id = str(entity["id"])
    fixed_code = entity["fixed_code"]

    vinj = entry.get("generation_attack", {}).get("vinj_attack", {})
    injected_code = vinj.get("vul_code", "")
    relevant_vul = vinj.get("relevant_vul", [])

    if not injected_code:
        logger.warning(f"[{entry_id}] 无 vul_code，跳过审查")
        return None
    if not relevant_vul:
        logger.warning(f"[{entry_id}] 无 relevant_vul，跳过审查")
        return None

    logger.info(f"[{entry_id}] Stage 4: 审查注入结果...")
    debug_log = f"=== Entry {entry_id}: judge {len(relevant_vul)} patterns ===\n\n"

    judger = LLMVulJudger(
        generator=judge_generator,
        milvus_client=milvus_client,
        max_retries=judge_max_retries,
    )
    try:
        verdicts = judger.judge(
            clean_code=fixed_code,
            vul_injected_code=injected_code,
            relevant_vul=relevant_vul,
            previous_verdicts=relevant_vul,
        )
    except Exception as exc:  # noqa: BLE001 - 兜底，避免单条审查失败拖垮整个 worker
        logger.error(f"[{entry_id}] Stage 4 审查异常: {exc}")
        return None
    is_vulnerable, unresolved_pattern_ids = summarize_verdicts(verdicts)

    debug_log += (
        f"=== Stage 4 Verdicts (is_vulnerable={is_vulnerable}) ===\n"
        + json.dumps(verdicts, ensure_ascii=False, indent=2)
        + "\n\n"
    )
    save_debug_log(debug_dir, entry_id, debug_log)

    return {
        "is_vulnerable": is_vulnerable,
        "verdicts": verdicts,
        "unresolved_pattern_ids": unresolved_pattern_ids,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    project_root = _resolve_project_root()

    parser = argparse.ArgumentParser(
        description="RACG-VICS: 基于投毒目标文件的漏洞注入与审查（参数由配置文件控制）",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/attack/vinj/vinj.yml",
        help="配置文件路径（默认: configs/attack/vinj/vinj.yml）",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not config:
        logger.error(f"配置文件为空或无法解析: {args.config}")
        return 1

    # ---- 日志配置 ----
    log_cfg = config.get("logging", {})
    verbose = log_cfg.get("verbose", False)
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    debug_dir = _resolve_under_root(
        project_root,
        log_cfg.get("debug_dir", "logs/vinj_attack"),
    )
    logger.info("=" * 60)
    logger.info("RACG-VICS 漏洞注入与审查")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")

    # ---- 数据路径 ----
    data_cfg = config.get("data", {})
    target_file_path = _resolve_under_root(
        project_root,
        data_cfg.get("target_file_path", "").strip(),
    )
    cwe_file_path = _resolve_under_root(
        project_root,
        data_cfg.get("cwe_file_path", "data/raw/cwe/cwec_v4.19.1.xml"),
    )
    cache_folder_path = (data_cfg.get("cache_folder_path", "") or "").strip()

    # ---- VInj 实验参数 ----
    vinj_cfg = config.get("vinj", {})
    process_cwe: List[str] = vinj_cfg.get("process_cwe", [])
    process_num = vinj_cfg.get("process_num")
    workers_num = vinj_cfg.get("num_workers")
    top_k = int(vinj_cfg.get("top_k", 100))
    relevant_vul_num = int(vinj_cfg.get("relevant_vul_num", 4))
    skip_rerank = bool(vinj_cfg.get("skip_rerank", False))
    enable_cache = bool(vinj_cfg.get("enable_cache", False))
    judge_max_retries = int(vinj_cfg.get("judge_max_retries", 5))
    reranker_max_retries = int(vinj_cfg.get("reranker_max_retries", 5))

    if not process_cwe:
        logger.error("process_cwe 为空，无 CWE 类别可处理")
        return 1

    model_cfg = config.get("model", {})
    injector_cfg_path = _resolve_under_root(
        project_root,
        model_cfg.get("vinj_generator_config", "configs/models/gpt-5-mini.yml"),
    )
    judge_cfg_path = _resolve_under_root(
        project_root,
        model_cfg.get("judge_generator_config", injector_cfg_path),
    )
    # reranker_generator_config 缺省时回退至 vinj_generator_config
    reranker_cfg_path = _resolve_under_root(
        project_root,
        model_cfg.get("reranker_generator_config")
        or model_cfg.get("vinj_generator_config", "configs/models/gpt-5-mini.yml"),
    )
    retriever_cfg_relpath = model_cfg.get(
        "retriever_config", "configs/models/harrier-oss-v1-0.6b.yml"
    )

    # ---- 1. 初始化 Milvus 客户端（Step 1 检索 + Stage 1 缓存共享） ----
    logger.info("初始化 Milvus 客户端...")
    db_cfg_path = _resolve_under_root(
        project_root,
        config.get("database", {}).get("milvus_config", "configs/database/vul/vinj.yml"),
    )
    db_config = load_yaml(db_cfg_path)
    db_uri = db_config.get("database", {}).get("uri", "")
    if db_uri and not os.path.isabs(db_uri):
        db_config["database"]["uri"] = str(project_root / db_uri)
    milvus_client = VulMilvusClient(db_config)
    clear_cache = bool(config.get("database", {}).get("clear_cache", False))
    if clear_cache:
        logger.info("clear_cache=True，清除 Stage 1 漏洞模式缓存集合...")
        milvus_client.clear_pattern_cache()
    milvus_client.create_pattern_cache()

    # ---- 2. 加载 CWE 数据 ----
    logger.info("加载 CWE 数据...")
    cwe_utils = CWEUtils(cwe_file_path)

    # ---- 3. 加载投毒目标文件 ----
    logger.info(f"加载投毒目标文件: {target_file_path}")
    try:
        with open(target_file_path, "r", encoding="utf-8") as f:
            target_data: Dict[str, List[Dict[str, Any]]] = json.load(f)
    except FileNotFoundError:
        logger.error(f"目标文件不存在: {target_file_path}")
        milvus_client.close()
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        milvus_client.close()
        return 1

    total_entries = sum(len(v) for v in target_data.values())
    logger.info(f"目标文件加载完成: {len(target_data)} 个 CWE 类别, {total_entries} 条数据")

    # ---- 4. 构建固定待办队列（不跳过已完成项，按优先级 + 长度排序） ----
    queue = select_target_queue(target_data, process_cwe, process_num)
    if not queue:
        logger.info("待办队列为空，退出")
        milvus_client.close()
        return 0

    output_lock = threading.Lock()

    def _save_target_file() -> None:
        """原子写回投毒目标文件。"""
        temp_path = (
            f"{target_file_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(target_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_file_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ---- 5. Stage 1: 相关漏洞宽召回检索（仅 relevant_vul 为空的队列目标） ----
    retrieve_relevant_vul(
        queue=queue,
        target_data=target_data,
        target_file_path=target_file_path,
        project_root=project_root,
        retriever_config_path=retriever_cfg_relpath,
        milvus_client=milvus_client,
        top_k=top_k,
        skip_rerank=skip_rerank,
    )

    # ---- 6. Stage 2: LLM-Reranker 筛选可注入 CVE（skip_rerank=True 时跳过） ----
    if not skip_rerank:
        logger.info(f"加载 Reranker 生成器配置: {reranker_cfg_path}")
        reranker_config = load_yaml(reranker_cfg_path)
        reranker_pool = reranker_config.get("api_pool")
        reranker_pool_size = len(reranker_pool) if reranker_pool else 1
        reranker_workers = (
            min(workers_num, reranker_pool_size) if workers_num is not None
            else reranker_pool_size
        )
        reranker_workers = max(reranker_workers, 1)
        logger.info(
            f"Reranker API pool 大小: {reranker_pool_size}，启动线程数: {reranker_workers}"
        )
        rerankers = [
            LLMVulReranker(
                generator=create_generator(reranker_config, pool_index=i),
                max_retries=reranker_max_retries,
            )
            for i in range(reranker_workers)
        ]
        rerank_relevant_vul(
            queue=queue,
            target_data=target_data,
            target_file_path=target_file_path,
            rerankers=rerankers,
            relevant_vul_num=relevant_vul_num,
            num_workers=reranker_workers,
        )
    else:
        logger.info("skip_rerank=True，跳过 Stage 2 LLM-Reranker 筛选阶段")

    # ---- 6.5 缓存复用：在确保 relevant_vul 就绪后、注入前复用已完成的生成攻击结果 ----
    if enable_cache and cache_folder_path:
        cache_folder_abs = _resolve_under_root(project_root, cache_folder_path)
        logger.info(f"enable_cache=True，尝试从缓存复用生成攻击结果: {cache_folder_abs}")
        reuse_cached_generation_attack(
            queue=queue,
            target_data=target_data,
            target_file_path=target_file_path,
            cache_folder_path=cache_folder_abs,
            process_cwe=process_cwe,
        )
    else:
        logger.info("缓存复用未启用（enable_cache=False 或 cache_folder_path 为空）")

    # ---- 7. 加载注入生成器池 ----
    logger.info(f"加载注入生成器配置: {injector_cfg_path}")
    injector_config = load_yaml(injector_cfg_path)
    api_pool = injector_config.get("api_pool")
    pool_size = len(api_pool) if api_pool else 1
    num_workers = (
        min(workers_num, pool_size) if workers_num is not None else pool_size
    )
    num_workers = max(num_workers, 1)
    logger.info(f"注入 API pool 大小: {pool_size}，启动线程数: {num_workers}")
    injectors = [create_generator(injector_config, pool_index=i) for i in range(num_workers)]

    # ---- 8. Stage 3: 漏洞注入 pass（仅 vul_code 为空且 relevant_vul 非空） ----
    inject_targets = [
        (cwe, idx, entry)
        for cwe, idx, entry in queue
        if not entry["generation_attack"]["vinj_attack"].get("vul_code")
        and entry["generation_attack"]["vinj_attack"].get("relevant_vul")
    ]
    inject_success = 0
    inject_fail = 0
    if inject_targets:
        logger.info(f"[Stage 3] 共 {len(inject_targets)} 个目标待注入")

        def _inject_worker(
            cwe: str, idx: int, entry: Dict[str, Any], injector: BaseGenerator,
        ) -> Optional[Dict[str, Any]]:
            entry_id = str(entry.get("entity", {}).get("id", f"unknown_{idx}"))
            logger.info(f"[Stage 3][{cwe}][{idx}] 注入: BFP ID {entry_id}")
            return inject_entry(
                entry=entry,
                cwe_id=cwe,
                injector=injector,
                cwe_utils=cwe_utils,
                milvus_client=milvus_client,
                debug_dir=debug_dir,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_info = {
                executor.submit(
                    _inject_worker, cwe, idx, entry, injectors[i % num_workers],
                ): (cwe, idx, entry)
                for i, (cwe, idx, entry) in enumerate(inject_targets)
            }
            for future in as_completed(future_to_info):
                cwe, idx, entry = future_to_info[future]
                entry_id = str(entry.get("entity", {}).get("id", f"unknown_{idx}"))
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[Stage 3][{cwe}][{entry_id}] 注入异常: {exc}")
                    inject_fail += 1
                    continue
                if result is None:
                    inject_fail += 1
                    continue

                with output_lock:
                    vinj_attack = entry["generation_attack"]["vinj_attack"]
                    vinj_attack["vul_code"] = result["vul_code"]
                    for vul_item in vinj_attack.get("relevant_vul", []):
                        try:
                            doc_id_int = int(vul_item.get("doc_id", 0))
                        except (TypeError, ValueError):
                            doc_id_int = 0
                        pattern = result["vul_patterns"].get(doc_id_int)
                        if pattern:
                            vul_item["vul_pattern"] = pattern
                    inject_success += 1
                    _save_target_file()
                    logger.info(
                        f"[Stage 3][{cwe}][{entry_id}] 注入写回成功 "
                        f"(累计成功: {inject_success})"
                    )
    else:
        logger.info("[Stage 3] 无待注入目标，跳过")

    # ---- 9. Stage 4: 注入结果审查 pass（仅 is_vulnerable 缺失且 vul_code 存在） ----
    logger.info(f"加载审查生成器配置: {judge_cfg_path}")
    judge_config = load_yaml(judge_cfg_path)
    judge_pool = judge_config.get("api_pool")
    judge_pool_size = len(judge_pool) if judge_pool else 1
    judge_workers = (
        min(workers_num, judge_pool_size) if workers_num is not None else judge_pool_size
    )
    judge_workers = max(judge_workers, 1)
    judge_generators = [
        create_generator(judge_config, pool_index=i % judge_pool_size)
        for i in range(judge_workers)
    ]

    judge_targets = [
        (cwe, idx, entry)
        for cwe, idx, entry in queue
        if "is_vulnerable" not in entry["generation_attack"]["vinj_attack"]
        and entry["generation_attack"]["vinj_attack"].get("vul_code")
    ]
    judge_success = 0
    judge_fail = 0
    vulnerable_count = 0
    if judge_targets:
        logger.info(f"[Stage 4] 共 {len(judge_targets)} 个目标待审查")

        def _judge_worker(
            cwe: str, idx: int, entry: Dict[str, Any], judge_generator: BaseGenerator,
        ) -> Optional[Dict[str, Any]]:
            entry_id = str(entry.get("entity", {}).get("id", f"unknown_{idx}"))
            logger.info(f"[Stage 4][{cwe}][{idx}] 审查: BFP ID {entry_id}")
            return judge_entry(
                entry=entry,
                judge_generator=judge_generator,
                milvus_client=milvus_client,
                debug_dir=debug_dir,
                judge_max_retries=judge_max_retries,
            )

        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            future_to_info = {
                executor.submit(
                    _judge_worker, cwe, idx, entry, judge_generators[i % judge_workers],
                ): (cwe, idx, entry)
                for i, (cwe, idx, entry) in enumerate(judge_targets)
            }
            for future in as_completed(future_to_info):
                cwe, idx, entry = future_to_info[future]
                entry_id = str(entry.get("entity", {}).get("id", f"unknown_{idx}"))
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[Stage 4][{cwe}][{entry_id}] 审查异常: {exc}")
                    judge_fail += 1
                    continue
                if result is None:
                    judge_fail += 1
                    continue

                with output_lock:
                    vinj_attack = entry["generation_attack"]["vinj_attack"]
                    vinj_attack["relevant_vul"] = result["verdicts"]
                    unresolved_ids = result["unresolved_pattern_ids"]
                    if unresolved_ids:
                        vinj_attack["unresolved_pattern_ids"] = unresolved_ids
                    else:
                        vinj_attack.pop("unresolved_pattern_ids", None)
                    if result["is_vulnerable"] is None:
                        vinj_attack.pop("is_vulnerable", None)
                        judge_fail += 1
                    else:
                        vinj_attack["is_vulnerable"] = result["is_vulnerable"]
                        judge_success += 1
                        if result["is_vulnerable"]:
                            vulnerable_count += 1
                    _save_target_file()
                    if result["is_vulnerable"] is None:
                        logger.warning(
                            f"[Stage 4][{cwe}][{entry_id}] 审查仍为 pending，"
                            f"未完成 pattern_id={unresolved_ids}"
                        )
                    else:
                        logger.info(
                            f"[Stage 4][{cwe}][{entry_id}] 审查写回成功 "
                            f"(累计成功: {judge_success}, "
                            f"is_vulnerable={result['is_vulnerable']})"
                        )
    else:
        logger.info("[Stage 4] 无待审查目标，跳过")

    # ---- 10. 统计输出 ----
    logger.info("=" * 60)
    logger.info(
        f"注入完成: 成功 {inject_success}, 失败 {inject_fail}; "
        f"审查完成: 成功 {judge_success}, 失败 {judge_fail}"
    )
    if judge_success:
        logger.info(
            f"注入成功率 (is_vulnerable): {vulnerable_count}/{judge_success} "
            f"({vulnerable_count / judge_success:.2%})"
        )
    logger.info(f"结果已写回: {target_file_path}")
    logger.info(f"日志目录: {debug_dir}")

    milvus_client.close()
    if judge_fail:
        logger.error("Stage 4 仍有 %d 个未完成条目；重启后将自动续跑", judge_fail)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
