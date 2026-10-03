"""
检索对抗攻击优化器评估指标。

主要提供针对单个优化器配置在某一 CWE（或 CWE-MIXED）分组上的统计指标。
"""

from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List


def _parse_iso_datetime(ts: Any) -> datetime:
    """
    解析 ISO 格式时间字符串，兼容带 / 不带时区以及 None / 空字符串。

    Args:
        ts: 时间字符串或 None

    Returns:
        datetime 对象；无法解析时返回 None
    """
    if not ts:
        return None
    ts = str(ts).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _compute_elapsed_seconds(start_time: Any, end_time: Any) -> float:
    """
    根据起止时间字符串计算耗时（秒）。

    Args:
        start_time: 开始时间 ISO 字符串
        end_time: 结束时间 ISO 字符串

    Returns:
        耗时秒数；无法解析时返回 0.0
    """
    st = _parse_iso_datetime(start_time)
    et = _parse_iso_datetime(end_time)
    if st is None or et is None:
        return 0.0
    return (et - st).total_seconds()


def _compute_diff_char_count(text_before: str, text_after: str) -> int:
    """
    估算两段文本存在差异的字符个数。

    使用 SequenceMatcher 的 opcodes，对每一个非 equal 操作取两侧长度的最大值，
    作为该处差异所涉及的字符数（可覆盖替换、插入、删除三种编辑类型）。

    Args:
        text_before: 扰动前文本
        text_after: 扰动后文本

    Returns:
        差异字符数估算值
    """
    sm = SequenceMatcher(None, text_before, text_after)
    diff_chars = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            diff_chars += max(i2 - i1, j2 - j1)
    return diff_chars


def _extract_log_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    从单条投毒目标中抽取评估所需的日志字段。

    兼容两种存储位置：
      1. run_attack_ultra 写入的 retrieval_attack.configs.logs
      2. 直接挂在 retrieval_attack 下的字段

    Returns:
        字段字典，包含 init_avg_sim / final_avg_sim / total_iterations /
        start_time / end_time / elapsed_seconds
    """
    ra = entry.get("retrieval_attack", {}) or {}
    logs = (ra.get("configs") or {}).get("logs", {}) or {}

    init_avg_sim = logs.get("init_avg_sim")
    final_avg_sim = logs.get("final_avg_sim")
    total_iterations = logs.get("total_iterations")
    start_time = logs.get("start_time")
    end_time = logs.get("end_time")

    # 若 configs.logs 不存在，尝试直接从 retrieval_attack 取
    if init_avg_sim is None:
        init_avg_sim = ra.get("init_avg_sim")
    if final_avg_sim is None:
        final_avg_sim = ra.get("final_avg_sim")
    if total_iterations is None:
        total_iterations = ra.get("total_iterations")
    if start_time is None:
        start_time = ra.get("start_time")
    if end_time is None:
        end_time = ra.get("end_time")

    elapsed = _compute_elapsed_seconds(start_time, end_time)

    return {
        "init_avg_sim": init_avg_sim if init_avg_sim is not None else 0.0,
        "final_avg_sim": final_avg_sim if final_avg_sim is not None else 0.0,
        "total_iterations": int(total_iterations) if total_iterations is not None else 0,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed,
    }


def filter_optimized_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从条目列表中筛选出已完成检索对抗优化的样本。

    判定条件：
      - retrieval_attack 节点存在
      - 存在非空的 poisoned_buggy_code（优化后的对抗代码）
      - 存在优化日志字段（优先从 retrieval_attack.configs.logs 读取，否则尝试 retrieval_attack 直接字段）

    Args:
        entries: 投毒目标条目列表

    Returns:
        已完成优化的条目子集
    """
    optimized = []
    for entry in entries:
        ra = entry.get("retrieval_attack", {}) or {}
        if not ra.get("poisoned_buggy_code"):
            continue
        # 必须存在某种形式的优化日志；仅 poisoned_buggy_code 存在但日志缺失时视为未优化
        has_config_logs = bool(
            (ra.get("configs") or {}).get("logs")
        )
        has_direct_logs = any(
            ra.get(k) is not None
            for k in ("init_avg_sim", "final_avg_sim", "total_iterations")
        )
        if not (has_config_logs or has_direct_logs):
            continue
        optimized.append(entry)
    return optimized


def compute_optimizer_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算某一优化器配置在指定 CWE 分组上的评估指标。

    指标说明：
      - total_samples: 优化的总样本个数
      - total_time: 所有样本的总制作时间（秒）
      - avg_time: 单个样本的平均制作时间（秒）
      - avg_sim_improvement: 所有样本 (final_avg_sim - init_avg_sim) 的平均值
      - avg_per_iter_sim_improvement: 相似度提升均值 / 实际迭代轮数均值
      - avg_perturbation_chars: 扰动字符数的平均值
      - avg_perturbation_ratio: 扰动字符占 poisoned_buggy_code 长度的比例的平均值

    Args:
        entries: 已完成优化的投毒目标条目列表

    Returns:
        指标字典
    """
    entries = filter_optimized_entries(entries)
    total_samples = len(entries)

    if total_samples == 0:
        return {
            "total_samples": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "avg_sim_improvement": 0.0,
            "avg_per_iter_sim_improvement": 0.0,
            "avg_perturbation_chars": 0.0,
            "avg_perturbation_ratio": 0.0,
        }

    total_time = 0.0
    sim_improvements: List[float] = []
    iterations: List[int] = []
    perturbation_chars: List[int] = []
    perturbation_ratios: List[float] = []

    for entry in entries:
        entity = entry.get("entity", {}) or {}
        ra = entry.get("retrieval_attack", {}) or {}
        logs = _extract_log_fields(entry)

        # 1) 时间效率
        total_time += logs["elapsed_seconds"]

        # 2) 优化器有效性
        sim_improvements.append(logs["final_avg_sim"] - logs["init_avg_sim"])
        iterations.append(logs["total_iterations"])

        # 3) 样本隐蔽性
        buggy_code = entity.get("buggy_code", "")
        poisoned_buggy_code = ra.get("poisoned_buggy_code", "")
        diff_chars = _compute_diff_char_count(buggy_code, poisoned_buggy_code)
        perturbation_chars.append(diff_chars)
        denom = len(poisoned_buggy_code) if len(poisoned_buggy_code) > 0 else 1
        perturbation_ratios.append(diff_chars / denom)

    avg_sim_improvement = sum(sim_improvements) / total_samples
    avg_iterations = sum(iterations) / total_samples
    avg_per_iter = (
        avg_sim_improvement / avg_iterations if avg_iterations > 0 else 0.0
    )

    return {
        "total_samples": total_samples,
        "total_time": total_time,
        "avg_time": total_time / total_samples,
        "avg_sim_improvement": avg_sim_improvement,
        "avg_per_iter_sim_improvement": avg_per_iter,
        "avg_perturbation_chars": sum(perturbation_chars) / total_samples,
        "avg_perturbation_ratio": sum(perturbation_ratios) / total_samples,
    }


def format_metric_value(value: float, metric_name: str) -> str:
    """
    根据指标类型选择合适的小数位数。

    Args:
        value: 指标数值
        metric_name: 指标名称

    Returns:
        格式化后的字符串
    """
    if metric_name in ("total_samples",):
        return str(int(value))
    if metric_name in ("total_time", "avg_time"):
        return f"{value:.2f}"
    if metric_name in ("avg_perturbation_chars",):
        return f"{value:.2f}"
    return f"{value:.4f}"
