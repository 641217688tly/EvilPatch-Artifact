# 存储正则表达式相关的工具函数
import re
from typing import Dict, Any
from pathlib import Path

def strip_c_comments(code: str) -> str:
    """移除 C/C++ 源代码中的注释（// 与 /* ... */）。"""
    # 删除块注释
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # 删除行注释
    no_line = re.sub(r"//.*?$", "", no_block, flags=re.MULTILINE)
    return no_line


def extract_code_from_response(response_text: str) -> str:
    """Extract the first fenced code block, or return the stripped response."""
    match = re.search(r"```(?:\w*)\n(.*?)```", response_text or "", flags=re.DOTALL)
    return match.group(1).strip() if match else (response_text or "").strip()


def parse_retrieval_result_stem(stem: str) -> Dict[str, Any]:
    """
    从文件名 stem（无扩展名）解析 CWE 检索结果参数。

    支持可选的 ``_TOP_{k}`` 后缀；若省略则 ``top_k`` 为 None。

    参数:
        stem: 如 ``CWE119_RELEVANCE_0.75_TOP_10`` 或 ``CWE119_RELEVANCE_0.75``

    返回:
        字典: cwe_id (如 CWE-119), relevance_threshold (float), top_k (Optional[int])

    异常:
        ValueError: stem 不符合 ``CWE<num>_RELEVANCE_<threshold>`` 模式时抛出
    """
    m = re.match(r"(?i)^CWE(\d+)_RELEVANCE_([\d.]+)(?:_TOP_(\d+))?$", stem.strip())
    if not m:
        raise ValueError(
            f"无法从 stem 解析 CWE 检索参数（期望 CWE<num>_RELEVANCE_<threshold>[_TOP_<k>]）: {stem!r}"
        )
    cwe_num, threshold, topk_opt = m.groups()
    return {
        "cwe_id": f"CWE-{cwe_num}",
        "relevance_threshold": float(threshold),
        "top_k": int(topk_opt) if topk_opt is not None else None,
    }


def parse_retrieval_filename(filepath: str) -> Dict[str, Any]:
    """
    从检索结果文件名中解析 CWE ID、相关性阈值、Top-K 参数。

    文件名格式: CWE{NUM}_RELEVANCE_{threshold}_TOP_{topk}.json
    示例: CWE119_RELEVANCE_0.75_TOP_10.json
            -> {"cwe_id": "CWE-119", "relevance_threshold": 0.75, "top_k": 10}

    参数:
        filepath: 检索结果文件路径（仅使用文件名部分进行解析）

    返回:
        包含 cwe_id、relevance_threshold、top_k 的字典

    异常:
        ValueError: 文件名不符合预期格式时抛出
    """
    stem = Path(filepath).stem
    try:
        info = parse_retrieval_result_stem(stem)
    except ValueError as e:
        raise ValueError(
            f"文件名不符合预期格式 'CWE{{NUM}}_RELEVANCE_{{threshold}}_TOP_{{topk}}'，"
            f"无法解析参数: {filepath}"
        ) from e
    if info["top_k"] is None:
        raise ValueError(
            f"文件名不符合预期格式 'CWE{{NUM}}_RELEVANCE_{{threshold}}_TOP_{{topk}}'，"
            f"无法解析参数: {filepath}"
        )
    return {
        "cwe_id": info["cwe_id"],
        "relevance_threshold": info["relevance_threshold"],
        "top_k": info["top_k"],
    }
