"""
该工具模块用于存储与日志相关的工具函数
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """
    设置日志配置。

    参数:
        level: 日志级别，默认为 INFO
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_debug_log(debug_dir: str, entry_id: str, log_text: str) -> bool:
    """
    保存调试日志到文件。

    用于记录完整的提示词和 LLM 响应，方便调试和分析。

    参数:
        debug_dir: 调试日志保存目录
        entry_id: 条目 ID (如 "633-A-bug-17175159-17175278")
        log_text: 日志内容

    返回:
        bool: 保存成功返回 True，失败返回 False
    """
    try:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        # 兼容 int/str 等 entry_id 输入，统一转字符串后再清洗
        safe_id = re.sub(r'[<>:"/\|?*]', "_", str(entry_id))
        log_path = debug_path / f"{safe_id}.txt"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        return True
    except Exception as e:
        logger.warning(f"日志保存失败 {debug_dir}/{entry_id}.txt: {e}")
        return False
