import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  原有工具函数（保留）
# ═══════════════════════════════════════════════════════════════════════════════

def parse_meta_info(meta_info: str, group_id: int, line_number: int) -> str:
    """
    解析元数据信息，生成描述字符串

    参数:
        meta_info: 原始元数据字符串（通常以 \t 分隔）
        group_id: 文件组编号
        line_number: 行号（1-indexed）

    返回:
        描述字符串
    """
    parts = meta_info.split('\t')

    if len(parts) >= 2:
        project_id = parts[0]
        file_path = parts[1] if len(parts) > 1 else ""
        description = f"Project ID: {project_id}"

        if file_path:
            if '/' in file_path:
                filename = file_path.split('/')[-1]
                description += f", File: {filename}"

            path_parts = file_path.split('/')
            for part in path_parts:
                if len(part) == 40 and all(c in '0123456789abcdef' for c in part.lower()):
                    description += f", Commit: {part[:8]}"
                    break
        return description
    else:
        return f"Group: {group_id}, Line: {line_number}"


def smart_replace_in_context(context: str, bug: str, fix: str) -> str:
    """
    智能地在 context 中匹配 bug 并替换为 fix（旧版兼容接口）。
    新代码请使用 apply_single_replacement。
    """
    new_ctx, count, _ = apply_single_replacement(context, bug, fix)
    return new_ctx if count == 1 else context


def save_entries_to_json(entries: List[Dict], output_file: Path) -> None:
    """将数据条目保存为 JSON 文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def get_available_groups(base_dir: str) -> List[int]:
    """获取所有可用的文件组编号"""
    base_path = Path(base_dir)
    group_ids = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            group_id = int(item.name)
            if (item / f"add_{group_id}.txt").exists():
                group_ids.append(group_id)
    return sorted(group_ids)


# ═══════════════════════════════════════════════════════════════════════════════
#  新增 / 重构工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_file_path(meta_line: str) -> str:
    """
    从 meta.txt 的一行中提取文件路径。
    格式: "project_id\\tfile_path\\n"
    """
    parts = meta_line.strip().split('\t')
    return parts[1] if len(parts) >= 2 else ""


def count_lines_fast(path: Path) -> int:
    """以二进制分块读取统计文件行数，避免全量载入内存。"""
    n = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            n += chunk.count(b'\n')
    return n


def compute_group_key(file_path: str, context_hash: str) -> str:
    """由 (file_path, context_hash) 生成确定性分组主键。"""
    raw = f"{file_path}||{context_hash}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def md5_hash(text: str) -> str:
    """返回字符串的 MD5 十六进制摘要。"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def truncate_incomplete_last_line(path: Path) -> int:
    """
    截断 JSONL 文件末尾可能存在的残缺行（未以 '\\n' 结尾的部分）。

    返回:
        截断后文件的有效行数
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0

    with open(path, 'r+b') as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return 0

        f.seek(file_size - 1)
        last_byte = f.read(1)
        if last_byte == b'\n':
            # 文件正常结束，统计行数
            f.seek(0)
            count = 0
            for _ in f:
                count += 1
            return count

        # 末尾不是换行符，说明最后一行不完整，需要截断
        pos = file_size - 1
        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == b'\n':
                break

        # pos == 0 且第一个字节也不是 '\n' → 整个文件就是一个残行
        if pos == 0:
            f.seek(0)
            if f.read(1) != b'\n':
                f.truncate(0)
                return 0

        truncate_at = pos + 1
        f.truncate(truncate_at)

    # 统计截断后的有效行数
    return count_lines_fast(path)


# ─── 匹配与替换 ──────────────────────────────────────────────────────────────

def _build_whitespace_pattern(text: str) -> str:
    """将 text 转义后，把连续空白替换为 \\s+ 正则模式。"""
    escaped = re.escape(text)
    return re.sub(r'(\\ )+', r'\\s+', escaped)


def count_matches_in_context(context: str, bug: str) -> Tuple[int, str]:
    """
    统计 bug 在 context 中的匹配次数，依次尝试三种策略。

    返回:
        (match_count, strategy_name)
        strategy_name ∈ {'exact', 'regex_whitespace', 'normalized', 'none'}
    """
    # 策略 1: 精确匹配
    count = context.count(bug)
    if count > 0:
        return (count, 'exact')

    # 策略 2: 正则空白匹配（bug 中的连续空格 → \s+）
    try:
        pattern = _build_whitespace_pattern(bug)
        matches = re.findall(pattern, context)
        if matches:
            return (len(matches), 'regex_whitespace')
    except re.error:
        pass

    # 策略 3: 标准化空白（双方连续空白压缩为单空格）
    try:
        norm_bug = re.sub(r'\s+', ' ', bug)
        norm_ctx = re.sub(r'\s+', ' ', context)
        count = norm_ctx.count(norm_bug)
        if count > 0:
            return (count, 'normalized')
    except Exception:
        pass

    return (0, 'none')


def apply_single_replacement(
    context: str, bug: str, fix: str
) -> Tuple[str, int, str]:
    """
    在 context 中将 bug 替换为 fix（仅替换第一次出现）。

    返回:
        (new_context, match_count, strategy)
        - match_count == 1 且替换成功时 new_context 是替换后的文本
        - match_count != 1 时 new_context 返回原 context（不做任何修改）
    """
    match_count, strategy = count_matches_in_context(context, bug)

    if match_count != 1:
        return (context, match_count, strategy)

    # 精确匹配 → str.replace
    if strategy == 'exact':
        return (context.replace(bug, fix, 1), 1, 'exact')

    # 正则空白 / 标准化 → 用 re.sub 做首次替换
    if strategy == 'regex_whitespace':
        pattern = _build_whitespace_pattern(bug)
    else:
        norm_bug = re.sub(r'\s+', ' ', bug)
        pattern = re.sub(r'\s+', r'\\s+', re.escape(norm_bug))

    try:
        new_context, n = re.subn(pattern, fix, context, count=1)
        if n == 1:
            return (new_context, 1, strategy)
    except re.error:
        pass

    return (context, 0, 'none')
