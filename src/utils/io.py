"""
提供文件读写等IO操作
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

_INT64_MAX = (1 << 63) - 1
_IdentityKey = Tuple[str, Any]


def normalize_bfp_record_id(value: Any) -> Any:
    """Normalize JSON int/string IDs to a stable, comparable value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (
            stripped.startswith("-") and len(stripped) > 1 and stripped[1:].isdigit()
        ):
            try:
                return int(stripped)
            except ValueError:
                pass
        return stripped
    return str(value)


def bfp_record_ids_equal(left: Any, right: Any) -> bool:
    """Compare BFP record IDs after normalizing JSON representation differences."""
    return normalize_bfp_record_id(left) == normalize_bfp_record_id(right)


def _cross_cwe_id_candidate(cwe: str, original_id: Any, nonce: int) -> int:
    payload = f"{cwe}\x1f{original_id!r}\x1f{nonce}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _INT64_MAX
    return -(value or 1)


def build_cross_cwe_id_map(
    records_by_cwe: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[_IdentityKey, int]:
    """Build stable negative INT64 IDs for identities duplicated across CWE groups."""
    cwes_by_id: Dict[Any, set[str]] = defaultdict(set)
    occupied: set[int] = set()
    for cwe, records in records_by_cwe.items():
        for record in records:
            raw = (record.get("entity") or {}).get("id")
            normalized = normalize_bfp_record_id(raw)
            if normalized is None:
                raise ValueError(f"missing entity.id in CWE {cwe}")
            cwes_by_id[normalized].add(str(cwe))
            if isinstance(normalized, int) and not isinstance(normalized, bool):
                occupied.add(normalized)

    identities = sorted(
        (
            (cwe, original_id)
            for original_id, cwes in cwes_by_id.items()
            if len(cwes) > 1
            for cwe in cwes
        ),
        key=lambda item: (item[0], repr(item[1])),
    )
    mapping: Dict[_IdentityKey, int] = {}
    for cwe, original_id in identities:
        nonce = 0
        while True:
            candidate = _cross_cwe_id_candidate(cwe, original_id, nonce)
            if candidate not in occupied:
                mapping[(cwe, original_id)] = candidate
                occupied.add(candidate)
                break
            nonce += 1
    return mapping


def apply_cross_cwe_id_map(
    records: Iterable[MutableMapping[str, Any]],
    cwe: str,
    mapping: Mapping[_IdentityKey, int],
) -> int:
    """Apply a cross-CWE mapping and preserve renamed values as ``original_id``."""
    changed = 0
    for record in records:
        entity = record.get("entity") or {}
        original = normalize_bfp_record_id(entity.get("id"))
        replacement = mapping.get((str(cwe), original))
        if replacement is None:
            continue
        entity.setdefault("original_id", entity.get("id"))
        entity["id"] = replacement
        record["entity"] = entity
        changed += 1
    return changed


def validate_global_entity_ids(
    records_by_cwe: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Raise if final poisoned BFP IDs are missing or not globally unique."""
    seen: Dict[Any, str] = {}
    for cwe, records in records_by_cwe.items():
        for record in records:
            value = normalize_bfp_record_id((record.get("entity") or {}).get("id"))
            if value is None:
                raise ValueError(f"missing entity.id in CWE {cwe}")
            previous = seen.get(value)
            if previous is not None:
                raise ValueError(
                    f"duplicate final entity.id={value!r} in CWE {previous} and {cwe}"
                )
            seen[value] = str(cwe)


def identity_snapshot(
    records_by_cwe: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[_IdentityKey, Any]:
    """Return ``(CWE, original-or-current ID) -> current ID`` for consistency checks."""
    snapshot: Dict[_IdentityKey, Any] = {}
    for cwe, records in records_by_cwe.items():
        for record in records:
            entity = record.get("entity") or {}
            original = normalize_bfp_record_id(entity.get("original_id", entity.get("id")))
            snapshot[(str(cwe), original)] = normalize_bfp_record_id(entity.get("id"))
    return snapshot


def load_json(filepath: str) -> Any:
    """
    从 JSON 文件加载数据。

    参数:
        filepath: JSON 文件路径

    返回:
        json.load 解析结果（根节点为对象时通常为 dict，亦可能为 list 等）

    异常:
        json.JSONDecodeError: 文件内容非合法 JSON 时抛出
        OSError: 文件无法打开时抛出
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        logger.info(f"Loaded {len(data)} records from {filepath}")
    else:
        logger.info(f"Loaded JSON from {filepath}")
    return data


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载数据。

    参数:
        filepath: JSONL 文件路径

    返回:
        包含字典的列表
    """
    data: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {filepath}: {e}")
    logger.info(f"Loaded {len(data)} records from {filepath}")
    return data

def load_json_stream(
    filepath: str,
    *,
    array_prefix: str = "item",
    use_float: bool = True,
) -> List[Dict[str, Any]]:
    """
    以流式方式解析 JSON 文件，根节点须为数组 ``[ {...}, ... ]``。

    使用 ``ijson`` 增量读取，适合体量很大的数组型 JSON，减轻解析阶段峰值内存。

    参数:
        filepath: JSON 文件路径
        array_prefix: ijson 前缀；根数组默认为 ``"item"``。若根为对象包裹数组，例如
            ``{"results": [...]}``，应传入 ``"results.item"``。
        use_float: 为 True（默认）时数字解析为 ``int``/``float``，与 ``json.load`` 一致，
            便于后续 ``json.dump``；为 False 时浮点可能为 ``decimal.Decimal``，标准库 JSON
            无法直接写出。

    返回:
        数组中每个元素组成的列表（通常每条为 dict）

    异常:
        FileNotFoundError: 文件不存在
        ijson/common.JSONError: 内容非合法 JSON 或与 prefix 不匹配时由 ijson 抛出
    """
    import ijson

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {filepath}")

    out: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for obj in ijson.items(f, array_prefix, use_float=use_float):
            if not isinstance(obj, dict):
                raise TypeError(
                    f"根数组元素须为 JSON 对象（dict），得到 {type(obj).__name__}"
                )
            out.append(obj)
    logger.info(f"Stream-loaded {len(out)} records from {filepath}")
    return out


def save_json(data: Any, filepath: str, ensure_ascii: bool = False) -> None:
    """
    保存数据到 JSON 文件。

    参数:
        data: 要保存的数据
        filepath: 输出文件路径
        ensure_ascii: 是否确保 ASCII 编码
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
    logger.info(f"Saved results to {filepath}")


def save_jsonl(data: List[Dict[str, Any]], filepath: str, ensure_ascii: bool = False) -> None:
    """
    保存数据到 JSONL 文件（每行一个 JSON 对象）。

    参数:
        data: 要保存的数据列表，每个元素是一个字典
        filepath: 输出文件路径（建议以 .jsonl 结尾）
        ensure_ascii: 是否确保 ASCII 编码
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
    logger.info(f"Saved {len(data)} records to {filepath}")


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置。

    参数:
        filepath: YAML 文件路径

    返回:
        配置字典
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
