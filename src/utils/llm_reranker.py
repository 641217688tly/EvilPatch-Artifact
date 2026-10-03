"""
LLM-based Vulnerability Exemplar Selection (Reranker) 工具模块。

实现论文《Exploring and Improving Real-World Vulnerability Data Generation via
Prompting Large Language Models》(ICSE '26) §2.5.2 "Exemplar Selection" 中的
自适应样例选择逻辑：

    给定一段待注入漏洞的目标代码（target_code）以及从漏洞知识库中语义召回的
    多个 CVE 候选（relevant_vul），使用 LLM 一次性评估每个候选的漏洞注入机制
    能否被迁移到目标代码中，并给出 0-10 的置信度分数；最终按分数降序返回前
    relevant_vul_num 个 CVE。

使用场景：
    漏洞注入流水线（vics_vinj）Step 1 之后、Stage 1/2 之前：
    先以较大的 top_k（如 100）从漏洞知识库召回 CVE，再调用
    LLMVulReranker.rerank() 从中筛选出最易迁移的 N 个 CVE，
    写回 entry["generation_attack"]["vinj_attack"]["relevant_vul"]。
"""

import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

RERANK_PROMPT_TEMPLATE = """\
You are an expert in software security and vulnerability injection techniques.

For the purposes of scientific research and education, your task is to assess whether the vulnerability INTRODUCTION MECHANISM demonstrated in each CVE example can be transplanted into the given target code.

Focus exclusively on the ABSTRACT INJECTION MECHANISM — the generic pattern of unsafe operations (e.g., removing a bounds check, relaxing a size constraint, introducing a dangling pointer, adding an unchecked write). Do NOT require function names, variable names, struct names, or code structure to match.

For each CVE candidate listed below, evaluate:
1. Whether the core unsafe transformation (V1 → V2) is structurally feasible to replicate in the Target Code — i.e., the Target Code contains analogous constructs (buffers, pointers, loops, resource lifetimes, arithmetic, etc.) that the same class of edit could target.
2. A confidence score in [0, 10] reflecting how directly the pattern can be transplanted:
     0  = completely incompatible (e.g., target has no relevant constructs)
     1-3 = plausible CWE type but weak structural fit
     4-6 = moderate fit; requires some adaptation
     7-9 = strong fit; mechanism is directly applicable with minor adjustments
     10 = near-direct transplant possible (target has essentially the same pattern)
3. A one-sentence rationale citing the specific construct in the Target Code that makes the injection feasible or infeasible.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY a single JSON array (no prose, no explanation outside the JSON).
- Wrap the JSON array in a ```json fenced code block.
- Produce EXACTLY one element per CVE candidate listed, preserving the input order.
- Every element MUST include all of: doc_id, cwe_id, cve_id, injectable,
  confidence_score, rationale.
- "injectable" MUST be a JSON boolean (true or false).
- "confidence_score" MUST be an integer in [0, 10].
- "rationale" MUST be a single concise English sentence.

Output format (example — do NOT copy these values):
```json
[
    {{
        "doc_id": 1,
        "cwe_id": "CWE-787",
        "cve_id": "CVE-2021-1234",
        "injectable": true,
        "confidence_score": 8,
        "rationale": "Target code performs an unchecked memcpy into a fixed-size stack buffer, directly matching the out-of-bounds-write pattern."
    }},
    {{
        "doc_id": 2,
        "cwe_id": "CWE-416",
        "cve_id": "CVE-2020-5678",
        "injectable": false,
        "confidence_score": 1,
        "rationale": "Target code contains no dynamic memory allocation or pointer lifecycle management that a use-after-free pattern could target."
    }}
]
```

==================== Target Code (to be injected) ====================
```
{target_code}
```

==================== CVE Candidates ====================
{formatted_candidates}

Now produce the JSON assessment array.
"""


# ---------------------------------------------------------------------------
# LLMVulReranker 主类
# ---------------------------------------------------------------------------

class LLMVulReranker:
    """
    基于 LLM 的 CVE 样例重排序器（Exemplar Selection）。

    对从漏洞知识库语义召回的 CVE 候选进行可注入性判定，按置信度分数筛选
    出最适合被迁移到目标代码中的前 N 个 CVE。

    Attributes:
        generator:   BaseGenerator 子类实例，用于调用 LLM 评估 CVE 可注入性。
        max_retries: LLM 调用及解析失败时的最大重试次数，默认 5。
    """

    def __init__(
        self,
        generator: BaseGenerator,
        max_retries: int = 5,
    ) -> None:
        self.generator = generator
        self.max_retries = max(1, int(max_retries))

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def rerank(
        self,
        target_code: str,
        relevant_vul: List[Dict[str, Any]],
        relevant_vul_num: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        对召回的 CVE 候选进行可注入性评分，返回最相关的前 N 个结果。

        评分由 LLM 一次性完成：所有候选在同一 prompt 中被全局比较，
        模型输出每个候选是否可注入（injectable）及置信度分数（0-10）。

        Args:
            target_code:       待注入漏洞的目标代码片段（通常为 BFP.fixed_code）。
            relevant_vul:      从漏洞知识库语义召回的 CVE 候选列表（最多 100 条）。
            relevant_vul_num:  最终返回的 CVE 数量，默认 4。

        Returns:
            原始 CVE 字典的子集（数量为 relevant_vul_num），每项追加
            ``injectable`` 与 ``confidence_score`` 字段。优先返回
            ``injectable=True`` 的条目（按 ``confidence_score`` 降序）；
            若不足 relevant_vul_num，则从剩余候选中按原检索分 (score 字段)
            降序递补，递补项的 ``injectable`` / ``confidence_score`` 置为 None。
            LLM 全部重试失败时走 fallback，同样保证非空返回。
        """
        if not relevant_vul:
            logger.warning("[LLMVulReranker] relevant_vul is empty, returning [].")
            return []

        if relevant_vul_num <= 0:
            logger.warning("[LLMVulReranker] relevant_vul_num <= 0, returning [].")
            return []

        prompt = self._build_prompt(target_code, relevant_vul)

        parsed: Optional[List[Dict[str, Any]]] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.generator.generate(prompt)
            except Exception as exc:
                logger.warning(
                    f"[LLMVulReranker] generate() raised on attempt "
                    f"{attempt}/{self.max_retries}: {exc}"
                )
                continue

            if not raw:
                logger.warning(
                    f"[LLMVulReranker] empty response on attempt "
                    f"{attempt}/{self.max_retries}"
                )
                continue

            try:
                parsed = self._parse_response(raw)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    f"[LLMVulReranker] parse failed on attempt "
                    f"{attempt}/{self.max_retries}: {exc}"
                )
                continue

        if parsed is None:
            logger.error(
                f"[LLMVulReranker] all {self.max_retries} attempts failed; "
                "using fallback (score-sorted, injectable=False)."
            )
            return self._build_fallback(relevant_vul, relevant_vul_num)

        merged = self._merge_with_relevant_vul(relevant_vul, parsed)
        return self._select_top_n(merged, relevant_vul_num)

    # ------------------------------------------------------------------
    # 辅助方法：Prompt 构建
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff(v1: str, v2: str) -> str:
        """使用 difflib 计算 V1 → V2 的统一格式 diff。"""
        lines = difflib.unified_diff(
            v1.splitlines(),
            v2.splitlines(),
            fromfile="V1_fixed",
            tofile="V2_vulnerable",
            lineterm="",
        )
        result = "\n".join(lines)
        return result if result else "(no diff available)"

    def _build_prompt(
        self,
        target_code: str,
        relevant_vul: List[Dict[str, Any]],
    ) -> str:
        """组装发送给 LLM 的完整提示词。"""
        formatted = self._format_candidates(relevant_vul)
        return RERANK_PROMPT_TEMPLATE.format(
            target_code=target_code,
            formatted_candidates=formatted,
        )

    @classmethod
    def _format_candidates(cls, relevant_vul: List[Dict[str, Any]]) -> str:
        """
        将每个 CVE 候选格式化为提示词中的可读文本块。

        每个块包含：序号、doc_id、cwe_id/desc、cve_id/desc、V1(fixed_code)、
        V2(vul_code)、以及 diff（优先使用字段值，缺失时现场计算）。
        """
        blocks: List[str] = []
        for i, cve in enumerate(relevant_vul, start=1):
            doc_id = cve.get("doc_id", "N/A")
            cwe_id = cve.get("cwe_id", "")
            cve_id = cve.get("cve_id", "")
            cwe_desc = cve.get("cwe_desc", "")
            cve_desc = cve.get("cve_desc", "")
            v1 = cve.get("fixed_code", "")
            v2 = cve.get("vul_code", "")
            diff = (cve.get("diff") or "").strip()
            if not diff:
                diff = cls._compute_diff(v1, v2)

            block = (
                f"---- Candidate {i} ----\n"
                f"doc_id: {doc_id}\n"
                f"cwe_id: {cwe_id}  |  cwe_desc: {cwe_desc}\n"
                f"cve_id: {cve_id}  |  cve_desc: {cve_desc}\n"
                f"V1 (non-vulnerable / fixed):\n```\n{v1}\n```\n"
                f"V2 (vulnerable version):\n```\n{v2}\n```\n"
                f"Diff (V1 → V2):\n```diff\n{diff}\n```"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # 辅助方法：响应解析
    # ------------------------------------------------------------------

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _BARE_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\})(?:\s*,\s*\{.*?\})*\s*\]", re.DOTALL)

    _INJECTABLE_TRUE_VALS = {"true", "yes", "1"}
    _INJECTABLE_FALSE_VALS = {"false", "no", "0"}

    @classmethod
    def _parse_response(cls, raw: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 响应，提取 JSON 数组并做字段类型校验。

        Raises:
            ValueError: 必填字段缺失、类型不符或 injectable 取值非法
            json.JSONDecodeError: JSON 解析失败
        """
        text = raw.strip()
        json_str: Optional[str] = None

        fence_match = cls._JSON_FENCE_RE.search(text)
        if fence_match:
            json_str = fence_match.group(1).strip()
        else:
            arr_match = cls._BARE_ARRAY_RE.search(text)
            if arr_match:
                json_str = arr_match.group(0)
            elif text.startswith("[") and text.endswith("]"):
                json_str = text

        if not json_str:
            raise ValueError("no JSON array found in LLM response")

        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError(f"expected JSON array, got {type(data).__name__}")

        results: List[Dict[str, Any]] = []
        required = {"doc_id", "cwe_id", "cve_id", "injectable", "confidence_score"}
        for i, elem in enumerate(data):
            if not isinstance(elem, dict):
                raise ValueError(f"element[{i}] is not a JSON object")
            missing = required - elem.keys()
            if missing:
                raise ValueError(f"element[{i}] missing fields: {sorted(missing)}")

            # --- normalize injectable ---
            raw_inj = elem["injectable"]
            if isinstance(raw_inj, bool):
                injectable = raw_inj
            elif isinstance(raw_inj, (int, float)):
                injectable = bool(raw_inj)
            elif isinstance(raw_inj, str):
                low = raw_inj.strip().lower()
                if low in cls._INJECTABLE_TRUE_VALS:
                    injectable = True
                elif low in cls._INJECTABLE_FALSE_VALS:
                    injectable = False
                else:
                    raise ValueError(
                        f"element[{i}].injectable has unrecognized value {raw_inj!r}"
                    )
            else:
                raise ValueError(
                    f"element[{i}].injectable has unexpected type "
                    f"{type(raw_inj).__name__}"
                )

            # --- normalize confidence_score ---
            try:
                score = int(elem["confidence_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"element[{i}].confidence_score cannot be coerced to int: "
                    f"{elem['confidence_score']!r}"
                ) from exc
            score = max(0, min(10, score))

            results.append({
                "doc_id": elem["doc_id"],
                "cwe_id": str(elem["cwe_id"]),
                "cve_id": str(elem["cve_id"]),
                "injectable": injectable,
                "confidence_score": score,
                "rationale": str(elem.get("rationale", "")),
            })

        return results

    # ------------------------------------------------------------------
    # 辅助方法：合并与选择
    # ------------------------------------------------------------------

    @staticmethod
    def _alignment_key(doc_id: Any, cve_id: Any) -> Tuple[str, str]:
        """对齐键：(str(doc_id), str(cve_id))，规避 int/str 不一致。"""
        return (str(doc_id), str(cve_id))

    @classmethod
    def _merge_with_relevant_vul(
        cls,
        relevant_vul: List[Dict[str, Any]],
        judged: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        将 LLM 评分结果按 (doc_id, cve_id) 回填到原始 CVE 字典中。

        对于 LLM 未返回结果的候选，记 warning 并赋默认值
        (injectable=False, confidence_score=0)。
        """
        judged_index: Dict[Tuple[str, str], Dict[str, Any]] = {
            cls._alignment_key(j["doc_id"], j["cve_id"]): j for j in judged
        }

        merged: List[Dict[str, Any]] = []
        for item in relevant_vul:
            key = cls._alignment_key(item.get("doc_id"), item.get("cve_id"))
            verdict = judged_index.get(key)
            out = dict(item)
            if verdict is None:
                logger.warning(
                    f"[LLMVulReranker] no assessment for doc_id={item.get('doc_id')} "
                    f"cve_id={item.get('cve_id')}, defaulting injectable=False score=0."
                )
                out["injectable"] = False
                out["confidence_score"] = 0
                out["rationale"] = "missing_in_response"
            else:
                out["injectable"] = verdict["injectable"]
                out["confidence_score"] = verdict["confidence_score"]
                out["rationale"] = verdict["rationale"]
            merged.append(out)
        return merged

    @staticmethod
    def _select_top_n(
        merged: List[Dict[str, Any]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """
        从评分后的完整列表中选出前 n 个结果。

        优先返回 injectable=True 的条目（按 confidence_score 降序）至多 n 个。
        若 True 组不足 n，则从剩余候选中按检索相似度分数（score 字段）降序递补
        （排除已选项），补足到 n 个。递补项的 injectable / confidence_score 置为
        None（语义为"未判定"），rationale 标记为 "score_backfill"。
        """
        true_group = sorted(
            (x for x in merged if x.get("injectable")),
            key=lambda x: x.get("confidence_score", 0),
            reverse=True,
        )
        top = true_group[:n]

        if len(top) >= n:
            return top

        selected_keys = {
            LLMVulReranker._alignment_key(x.get("doc_id"), x.get("cve_id"))
            for x in top
        }
        remaining = [
            x for x in merged
            if LLMVulReranker._alignment_key(x.get("doc_id"), x.get("cve_id"))
            not in selected_keys
        ]
        remaining.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        backfill_needed = n - len(top)
        backfilled: List[Dict[str, Any]] = []
        for item in remaining[:backfill_needed]:
            out = dict(item)
            out["injectable"] = None
            out["confidence_score"] = None
            out["rationale"] = "score_backfill"
            backfilled.append(out)

        result = top + backfilled
        if len(result) < n:
            logger.warning(
                f"[LLMVulReranker] only {len(result)} candidates available "
                f"(requested {n}); pool exhausted."
            )
        else:
            logger.info(
                f"[LLMVulReranker] {len(top)} injectable=True + "
                f"{len(backfilled)} score-backfilled = {len(result)} returned."
            )
        return result

    # ------------------------------------------------------------------
    # 辅助方法：Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback(
        relevant_vul: List[Dict[str, Any]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """
        LLM 全部重试失败时构造兜底输出。

        所有候选标记为 injectable=None / confidence_score=None（语义为"未判定"），
        按原始检索相似度分数（score 字段）降序取前 n 个返回，
        保证调用方不收到空列表。
        """
        fallback: List[Dict[str, Any]] = []
        for item in relevant_vul:
            out = dict(item)
            out["injectable"] = None
            out["confidence_score"] = None
            out["rationale"] = "reranker_failed"
            fallback.append(out)

        fallback.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return fallback[:n]
