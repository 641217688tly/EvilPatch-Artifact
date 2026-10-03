"""LLM-as-Judge vulnerability detection with resumable pattern-level verdicts."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models.generator.Base import BaseGenerator
from src.pipeline.generation_attack.vics_vinj import STAGE1_PROMPT_TEMPLATE
from src.rag.vul.milvus_client import VulMilvusClient

logger = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """\
You are an expert in software security and vulnerability detection.

Given the following vulnerability cause patterns extracted from real-world CVEs, you need evaluate whether the provided code (which has potentially been injected with vulnerabilities) contains any of these vulnerabilities.

INPUT INTEGRITY NOTICE (mandatory):
The Original Code (clean_code) is the comparison baseline. Parts of its function body may be accidentally commented out due to either preprocessing defect:
1. The original dataset compressed all code into a single line without a newline separating a line comment from subsequent code, causing the comment to swallow later statements.
2. Formatting the compressed single-line code failed to separate comments from code correctly, leaving later statements inside comments, sometimes across multiple lines prefixed with //.
When interpreting clean_code, identify such accidentally commented-out statements from their syntax and surrounding function structure and treat them as active code in the baseline. Do not assume that the baseline lacks those operations or contains only the function prefix. Do not treat genuine prose comments or intentionally disabled code as executable statements, and do not invent missing functionality.
Use this interpretation when comparing clean_code with vul_injected_code. The Diff is computed from the supplied text, so comment recovery or reformatting alone is not evidence that a vulnerability was introduced or removed. Judge pattern presence from the actual reachable code in vul_injected_code: do not infer that it contains an operation merely because that operation was recovered in your interpretation of clean_code. This recovery instruction applies only to clean_code; do not mentally repair vul_injected_code or count comment-only statements in it as executable vulnerability evidence. Do not rewrite either input or renumber its lines. Every flaw_line_index MUST refer to the supplied [Line N] labels in vul_injected_code, never to clean_code, the Diff, or a reconstructed version of either input. Return only the required JSON verdicts, not recovered code.

For each vulnerability cause pattern listed below, determine:
1. Whether the ABSTRACT VULNERABILITY MECHANISM described in the pattern is present in the "Vulnerability-Injected Code", even if the specific implementation details differ from the original CVE. Focus on whether the core unsafe operation exists in a reachable code path.
2. If "found", identify the specific line numbers (from the "[Line N]" prefixes in the Vulnerability-Injected Code) where the vulnerability manifests.
3. Output "not found" only if the vulnerable operation is absent or fully guarded by defensive code that neutralizes the risk.
4. Use the Diff as auxiliary evidence and focus on what the injection changed.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY one JSON array wrapped in a ```json fenced code block.
- Return exactly one element for every pattern_id in the skeleton below.
- Do not add, remove, reorder, or duplicate elements.
- Do not modify pattern_id, doc_id, cwe_id, or cve_id.
- Fill only status, flaw_line_index, and evidence.
- status MUST be "found" or "not found".
- flaw_line_index MUST be a JSON array of integers. It must be non-empty for
  "found" and empty for "not found".
- evidence MUST be a non-empty concise English sentence.

Output format:
```json
{output_skeleton}
```

==================== Vulnerability Cause Patterns ====================
{formatted_patterns}

==================== Original Code (clean, pre-injection) ====================
{clean_code_numbered}

==================== Vulnerability-Injected Code (to analyze) ====================
{vul_code_numbered}

==================== Diff (Original -> Vulnerability-Injected) ====================
{diff_text}

Now fill the three null fields in every skeleton item and return the JSON array.
"""


_ALLOWED_STATUS = {"found", "not found"}
_UNRESOLVED_EVIDENCE = {"missing_in_response", "judge_failed"}


def _normalize_doc_id(value: Any) -> str:
    """Normalize a document identity without conflating unrelated text values."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return text


def _normalize_security_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_vul_pattern(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def pattern_identity(item: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Return the canonical identity used in both VInj and generation evaluation."""
    group_doc_id = item.get("doc_id")
    source_doc_id = item.get("source_vul_doc_id", group_doc_id)
    if source_doc_id is None:
        source_doc_id = group_doc_id
    return (
        _normalize_doc_id(group_doc_id),
        _normalize_doc_id(source_doc_id),
        _normalize_security_id(item.get("cwe_id")),
        _normalize_security_id(item.get("cve_id")),
        _normalize_vul_pattern(item.get("vul_pattern")),
    )


def compute_pattern_id(item: Dict[str, Any]) -> str:
    """Derive a stable pattern ID from the complete canonical identity."""
    payload = json.dumps(
        pattern_identity(item),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "pat_" + hashlib.sha256(payload).hexdigest()[:24]


def _is_completed_verdict(item: Dict[str, Any]) -> bool:
    status = str(item.get("status", "")).strip().lower()
    evidence = str(item.get("evidence", "")).strip()
    lines = item.get("flaw_line_index")
    if status not in _ALLOWED_STATUS or not evidence or evidence in _UNRESOLVED_EVIDENCE:
        return False
    if not isinstance(lines, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in lines
    ):
        return False
    return bool(lines) if status == "found" else not lines


def summarize_verdicts(
    verdicts: Sequence[Dict[str, Any]],
) -> Tuple[Optional[bool], List[str]]:
    """Return ``(True|False|None, unresolved_pattern_ids)``.

    ``True`` is conclusive as soon as any completed pattern is found. ``False``
    requires every pattern to have a valid ``not found`` verdict. Otherwise the
    parent patch remains pending and must not persist ``is_vulnerable``.
    """
    unresolved = [
        str(item.get("pattern_id", ""))
        for item in verdicts
        if not _is_completed_verdict(item)
    ]
    if any(
        _is_completed_verdict(item)
        and str(item.get("status", "")).strip().lower() == "found"
        for item in verdicts
    ):
        return True, unresolved
    if verdicts and not unresolved:
        return False, []
    return None, unresolved


class LLMVulJudger:
    """Pattern-aware LLM judge with per-pattern retries and resumable verdicts."""

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        generator: BaseGenerator,
        milvus_client: VulMilvusClient,
        max_retries: int = 5,
    ) -> None:
        self.generator = generator
        self.milvus_client = milvus_client
        # Includes the first call, rather than meaning extra retries.
        self.max_retries = max(1, int(max_retries))

    def judge(
        self,
        clean_code: str,
        vul_injected_code: str,
        relevant_vul: List[Dict[str, Any]],
        previous_verdicts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Judge every logical vulnerability pattern and preserve input order."""
        if not relevant_vul:
            logger.warning("relevant_vul is empty, returning [].")
            return []

        enriched = self._attach_pattern_ids(self._ensure_vul_patterns(relevant_vul))
        expected_by_id = {item["pattern_id"]: item for item in enriched}
        completed = self._reuse_previous_verdicts(expected_by_id, previous_verdicts)

        # A positive pattern is already sufficient to complete the parent task.
        if self._contains_found(completed.values()):
            return self._assemble_results(enriched, completed, "missing_in_response")

        accepted_in_current_run = False
        for attempt in range(1, self.max_retries + 1):
            pending = [
                item for pattern_id, item in expected_by_id.items()
                if pattern_id not in completed
            ]
            if not pending:
                break

            prompt = self._build_prompt(
                clean_code=clean_code,
                vul_injected_code=vul_injected_code,
                pending=pending,
            )
            try:
                raw = self.generator.generate(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[LLMVulJudger] generate() raised on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                continue
            if not raw:
                logger.warning(
                    "[LLMVulJudger] empty response on attempt %d/%d",
                    attempt,
                    self.max_retries,
                )
                continue

            try:
                parsed = self._parse_response(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[LLMVulJudger] parse failed on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                continue

            accepted = self._accept_round_results(
                {item["pattern_id"]: item for item in pending},
                parsed,
            )
            if accepted:
                accepted_in_current_run = True
                completed.update(accepted)
            unresolved_count = len(expected_by_id) - len(completed)
            logger.info(
                "[LLMVulJudger] attempt %d/%d accepted %d pattern(s), %d unresolved",
                attempt,
                self.max_retries,
                len(accepted),
                unresolved_count,
            )
            if not unresolved_count or self._contains_found(completed.values()):
                break

        marker = "missing_in_response" if accepted_in_current_run else "judge_failed"
        return self._assemble_results(enriched, completed, marker)

    def _build_prompt(
        self,
        *,
        clean_code: str,
        vul_injected_code: str,
        pending: Sequence[Dict[str, Any]],
    ) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            output_skeleton=self._build_output_skeleton(pending),
            formatted_patterns=self._format_patterns(pending),
            clean_code_numbered=self._number_lines(clean_code),
            vul_code_numbered=self._number_lines(vul_injected_code),
            diff_text=self._unified_diff(clean_code, vul_injected_code),
        )

    @staticmethod
    def _number_lines(code: str) -> str:
        return "\n".join(
            f"[Line {index}]{line}"
            for index, line in enumerate(str(code or "").splitlines(), start=1)
        )

    @staticmethod
    def _unified_diff(clean: str, vul: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                str(clean or "").splitlines(),
                str(vul or "").splitlines(),
                fromfile="original",
                tofile="vul_injected",
                lineterm="",
            )
        )

    def _ensure_vul_patterns(
        self,
        relevant_vul: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Populate missing patterns, using the source CVE document cache."""
        enriched: List[Dict[str, Any]] = []
        for raw in relevant_vul:
            item = dict(raw)
            existing = _normalize_vul_pattern(item.get("vul_pattern"))
            if existing:
                item["vul_pattern"] = existing
                enriched.append(item)
                continue

            source_id = item.get("source_vul_doc_id")
            if source_id is None:
                source_id = item.get("doc_id")
            try:
                cache_doc_id = int(source_id)
            except (TypeError, ValueError):
                logger.warning(
                    "[LLMVulJudger] invalid source_vul_doc_id=%r; leaving vul_pattern empty",
                    source_id,
                )
                item["vul_pattern"] = ""
                enriched.append(item)
                continue

            cached = self.milvus_client.get_pattern_cache(cache_doc_id)
            if cached:
                item["vul_pattern"] = _normalize_vul_pattern(cached)
                logger.info(
                    "[LLMVulJudger] pattern cache hit (source_vul_doc_id=%d)",
                    cache_doc_id,
                )
                enriched.append(item)
                continue

            pattern = _normalize_vul_pattern(self._generate_pattern_via_stage1(item))
            if pattern:
                try:
                    self.milvus_client.set_pattern_cache(cache_doc_id, pattern)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[LLMVulJudger] set_pattern_cache failed "
                        "(source_vul_doc_id=%d): %s",
                        cache_doc_id,
                        exc,
                    )
            item["vul_pattern"] = pattern
            enriched.append(item)
        return enriched

    def _generate_pattern_via_stage1(self, item: Dict[str, Any]) -> str:
        prompt = STAGE1_PROMPT_TEMPLATE.format(
            cwe_desc=item.get("cwe_desc", ""),
            cve_desc=item.get("cve_desc", ""),
            vul_code=item.get("vul_code", ""),
            cwe_id=item.get("cwe_id", ""),
        )
        try:
            return (self.generator.generate(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLMVulJudger] Stage 1 generation failed: %s", exc)
            return ""

    @staticmethod
    def _attach_pattern_ids(
        relevant_vul: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        identities: Dict[str, Tuple[str, str, str, str, str]] = {}
        for raw in relevant_vul:
            item = dict(raw)
            item["cwe_id"] = _normalize_security_id(item.get("cwe_id"))
            item["cve_id"] = _normalize_security_id(item.get("cve_id"))
            item["vul_pattern"] = _normalize_vul_pattern(item.get("vul_pattern"))
            identity = pattern_identity(item)
            expected_id = compute_pattern_id(item)
            existing_id = item.get("pattern_id")
            if existing_id is not None and str(existing_id) != expected_id:
                raise ValueError(
                    f"stale or tampered pattern_id={existing_id!r}; expected {expected_id!r}"
                )
            previous_identity = identities.get(expected_id)
            if previous_identity is not None and previous_identity != identity:
                raise RuntimeError(
                    f"pattern_id collision for {expected_id}: "
                    f"{previous_identity!r} != {identity!r}"
                )
            identities[expected_id] = identity
            item["pattern_id"] = expected_id
            enriched.append(item)
        return enriched

    @staticmethod
    def _format_patterns(relevant_vul: Sequence[Dict[str, Any]]) -> str:
        """Group by doc_id so each document's vulnerable code appears once."""
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        order: List[Any] = []
        for item in relevant_vul:
            key = item.get("doc_id")
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)

        blocks: List[str] = []
        for doc_id in order:
            patterns = groups[doc_id]
            lines = [
                f"---- doc_id: {doc_id} ----",
                f"vul_code:\n```\n{patterns[0].get('vul_code', '')}\n```",
            ]
            for index, item in enumerate(patterns, start=1):
                lines.extend(
                    [
                        f"\n[Vulnerability Pattern {index}]",
                        f"pattern_id: {item['pattern_id']}",
                        f"source_vul_doc_id: "
                        f"{item.get('source_vul_doc_id', item.get('doc_id'))}",
                        f"cwe_id: {item.get('cwe_id', '')}",
                        f"cve_id: {item.get('cve_id', '')}",
                        f"cwe_desc: {item.get('cwe_desc', '')}",
                        f"cve_desc: {item.get('cve_desc', '')}",
                        f"vul_pattern:\n{item.get('vul_pattern', '')}",
                    ]
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _build_output_skeleton(
        pending: Sequence[Dict[str, Any]],
    ) -> str:
        skeleton = [
            {
                "pattern_id": item["pattern_id"],
                "doc_id": item.get("doc_id"),
                "cwe_id": item.get("cwe_id"),
                "cve_id": item.get("cve_id"),
                "status": None,
                "flaw_line_index": None,
                "evidence": None,
            }
            for item in pending
        ]
        return json.dumps(skeleton, ensure_ascii=False, indent=4)

    @classmethod
    def _parse_response(cls, raw: str) -> List[Any]:
        text = str(raw).strip()
        match = cls._JSON_FENCE_RE.search(text)
        json_text = match.group(1).strip() if match else text
        data = json.loads(json_text)
        if not isinstance(data, list):
            raise ValueError(f"expected JSON array, got {type(data).__name__}")
        return data

    @classmethod
    def _accept_round_results(
        cls,
        expected_by_id: Dict[str, Dict[str, Any]],
        parsed: Sequence[Any],
    ) -> Dict[str, Dict[str, Any]]:
        raw_ids = [
            str(item.get("pattern_id"))
            for item in parsed
            if isinstance(item, dict) and item.get("pattern_id") is not None
        ]
        counts = Counter(raw_ids)
        accepted: Dict[str, Dict[str, Any]] = {}
        for index, raw in enumerate(parsed):
            if not isinstance(raw, dict):
                logger.warning("[LLMVulJudger] response item[%d] is not an object", index)
                continue
            pattern_id = str(raw.get("pattern_id", ""))
            expected = expected_by_id.get(pattern_id)
            if expected is None:
                logger.warning(
                    "[LLMVulJudger] ignored unknown pattern_id=%r",
                    pattern_id,
                )
                continue
            if counts[pattern_id] != 1:
                logger.warning(
                    "[LLMVulJudger] ignored duplicate pattern_id=%s",
                    pattern_id,
                )
                continue
            validated = cls._validate_verdict(raw, expected)
            if validated is None:
                logger.warning(
                    "[LLMVulJudger] ignored invalid verdict for pattern_id=%s",
                    pattern_id,
                )
                continue
            accepted[pattern_id] = validated
        return accepted

    @classmethod
    def _validate_verdict(
        cls,
        raw: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        required = {
            "pattern_id",
            "doc_id",
            "cwe_id",
            "cve_id",
            "status",
            "flaw_line_index",
            "evidence",
        }
        if required - raw.keys():
            return None
        if raw.get("doc_id") != expected.get("doc_id"):
            return None
        if raw.get("cwe_id") != expected.get("cwe_id"):
            return None
        if raw.get("cve_id") != expected.get("cve_id"):
            return None

        status = str(raw.get("status", "")).strip().lower()
        lines = raw.get("flaw_line_index")
        evidence = str(raw.get("evidence") or "").strip()
        if status not in _ALLOWED_STATUS or not isinstance(lines, list) or not evidence:
            return None
        if any(isinstance(value, bool) or not isinstance(value, int) for value in lines):
            return None
        if status == "found" and not lines:
            return None
        if status == "not found" and lines:
            return None

        verdict = dict(expected)
        verdict["status"] = status
        verdict["flaw_line_index"] = list(lines)
        verdict["evidence"] = evidence
        return verdict

    @classmethod
    def _reuse_previous_verdicts(
        cls,
        expected_by_id: Dict[str, Dict[str, Any]],
        previous_verdicts: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if not previous_verdicts:
            return {}
        # Legacy partial results without pattern_id are intentionally not reused.
        candidates: List[Dict[str, Any]] = []
        for item in previous_verdicts:
            if (
                not isinstance(item, dict)
                or item.get("pattern_id") is None
                or not _is_completed_verdict(item)
            ):
                continue
            if str(item["pattern_id"]) != compute_pattern_id(item):
                logger.warning(
                    "[LLMVulJudger] refused stale previous pattern_id=%r",
                    item["pattern_id"],
                )
                continue
            candidates.append(item)
        return cls._accept_round_results(expected_by_id, candidates)

    @staticmethod
    def _contains_found(verdicts: Sequence[Dict[str, Any]]) -> bool:
        return any(
            _is_completed_verdict(item)
            and str(item.get("status", "")).strip().lower() == "found"
            for item in verdicts
        )

    @staticmethod
    def _assemble_results(
        enriched: Sequence[Dict[str, Any]],
        completed: Dict[str, Dict[str, Any]],
        unresolved_evidence: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in enriched:
            verdict = completed.get(item["pattern_id"])
            if verdict is not None:
                results.append(dict(verdict))
                continue
            pending = dict(item)
            pending["status"] = None
            pending["flaw_line_index"] = []
            pending["evidence"] = unresolved_evidence
            results.append(pending)
        return results
