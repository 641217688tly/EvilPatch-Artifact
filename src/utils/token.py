"""
分词相关工具函数。

提供检索对抗攻击中所需的 token 级别操作，安全词表机制被解耦为两条独立职责：
  - ``is_safe_position``：判定输入代码序列中“哪些位置可被扰动”（保护代码结构）
  - ``is_safe_candidate``：判定检索器词表中“哪些词元可作为替换候选”（保证隐蔽性）
  - ``build_frozen_mask`` / ``get_mutable_positions``：基于 ``is_safe_position``
    构建可扰动位置 mask
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def _strip_subword_markers(token_text: str) -> str:
    """
    剥离子词边界标记，返回词元的“核心文本”。

    三个检索器（Harrier=Qwen3、Jina=Qwen2 的 BPE，GTE=ModernBERT 的 WordPiece）
    都用子词边界标记表示“词的开头/续接”，这些标记本身不是代码标点，不应参与冻结判定：
      - BPE（Harrier/Jina）：词首 token 解码后带前导空格，如 " memcpy"
      - WordPiece（GTE）：续接子词带 "##" 前缀，如 "##tion"

    Args:
        token_text: tokenizer.decode 得到的单个词元文本

    Returns:
        去除首尾空白与 "##" 前缀后的核心文本（可能为空字符串）
    """
    core = token_text.strip()          # 去除 BPE 前导空格及 \t \n 等边界空白
    if core.startswith("##"):          # 去除 WordPiece 续接前缀
        core = core[2:]
    return core


def is_safe_position(
    token_text: str,
    frozen_token_dict: dict,
) -> bool:
    """
    判断代码序列中某个位置的词元是否“可安全扰动”（可作为扰动位置）。

    仅设置一条冻结规则（松绑后）：当词元的核心文本命中 ``structure`` / ``keywords``
    关键字（精确匹配）或包含 ``punctuation_and_operators`` 中的标点字符（字符包含）
    时冻结该位置；其余一律放行——包括非英文字符（如非英文代码注释），使其也能参与扰动。

    关键词采用“精确匹配”而非“子串包含”，以避免 "if" 命中 "shift"、"int" 命中
    "print" 等误冻结。

    约定：``True`` = 该位置可扰动（安全），``False`` = 冻结。

    Args:
        token_text: tokenizer 解码后的单个词元文本
        frozen_token_dict: 结构化冻结集合，含 "structure" / "keywords" /
            "punctuation_and_operators" 三个键

    Returns:
        True 表示该位置可参与对抗扰动，False 表示应被冻结
    """
    core = _strip_subword_markers(token_text)
    if not core:
        # 纯空白/纯边界标记：不含任何结构关键字或标点，按单一规则放行
        return True

    # 结构关键字 + 类型/访问控制关键字：精确匹配才冻结
    if (
        core in frozen_token_dict.get("structure", [])
        or core in frozen_token_dict.get("keywords", [])
    ):
        return False

    # 标点与运算符：核心文本只要包含冻结标点字符即冻结
    punct_set = frozen_token_dict.get("punctuation_and_operators", [])
    for ch in core:
        if ch in punct_set:
            return False
    return True


def is_safe_candidate(
    token_text: str,
    frozen_token_dict: dict,
) -> bool:
    """
    判断检索器词表中的某个词元是否“可安全用作扰动替换候选”。

    设置两条冻结规则（松绑后）：
      1. 隐蔽性约束：词元必须属于 {英文单词、标点、运算符、数字，以及三者的混合}，
         即仅允许 ASCII 可打印字符；出现中文/韩文/法文重音/emoji/特殊符号等其他语言
         或不可打印控制符时冻结（保证扰动后代码在人工/LLM 审查下的隐蔽性）。
      2. 结构约束：仅用 ``structure``（精确匹配）与 ``punctuation_and_operators``
         （字符包含）冻结；**不使用 keywords**——因此把类型/访问控制关键字（如
         "int" / "char"）用作替换候选是允许的（它们看起来仍像正常代码）。

    约定：``True`` = 该词元可用作替换候选（安全），``False`` = 冻结。

    Args:
        token_text: tokenizer 解码后的单个词元文本
        frozen_token_dict: 结构化冻结集合，此处仅使用 "structure" 与
            "punctuation_and_operators" 两个键

    Returns:
        True 表示该词元可作为对抗替换候选，False 表示应被冻结
    """
    # 规则 1：仅允许英文/标点/运算符/数字及其混合（ASCII 且可打印）
    if not token_text or not token_text.isascii():
        return False
    # if any(not ch.isprintable() for ch in token_text):
    #     # 拒绝含控制符/换行/制表符的词元（空格是可打印字符，予以保留）
    #     return False

    core = _strip_subword_markers(token_text)
    # if not core:
    #     # 纯空白/纯边界标记不是有意义的替换候选
    #     return False

    # 规则 2：仅用 structure（精确）+ punctuation_and_operators（包含）冻结
    if core in frozen_token_dict.get("structure", []):
        return False
    punct_set = frozen_token_dict.get("punctuation_and_operators", [])
    for ch in core:
        if ch in punct_set:
            return False

    return True


def build_frozen_mask(
    token_ids: List[int],
    tokenizer,
    frozen_token_dict: dict,
) -> List[bool]:
    """
    为一段 token 序列构建冻结 mask。

    返回长度与 ``token_ids`` 相同的布尔列表：
      - ``True``  → 该位置被冻结（不可替换）
      - ``False`` → 该位置可扰动

    冻结条件（满足任一即冻结）：
      1. 是 special token（[CLS] / [SEP] / [PAD] 等）
      2. is_safe_position 返回 False（命中 structure/keywords 关键字或含冻结标点）

    Args:
        token_ids: 输入 token ID 列表
        tokenizer: HuggingFace tokenizer 实例
        frozen_token_dict: 需要冻结的token集合

    Returns:
        冻结 mask 列表
    """
    special_ids = set(tokenizer.all_special_ids)
    frozen = []

    for tid in token_ids:
        if tid in special_ids:
            frozen.append(True)
            continue

        text = tokenizer.decode([tid], skip_special_tokens=False)
        if is_safe_position(text, frozen_token_dict):
            frozen.append(False)
        else:
            frozen.append(True)
            
    return frozen


def get_mutable_positions(
    token_ids: List[int],
    tokenizer,
    frozen_token_dict: dict,
    offset: int = 0,
) -> List[int]:
    """
    返回可扰动的 token 位置索引列表。

    这是 ``build_frozen_mask`` 的便捷封装，直接返回 ``frozen[i] == False``
    且 ``i >= offset`` 的索引列表。

    ``offset`` 用于跳过序列开头的 prompt prefix 段：当嵌入模型需要
    instruction prefix（如 Jina 的 ``"code2code"``）时，prefix 的 token
    会拼接在代码 token 序列前端，但不应参与对抗优化。此时令
    ``offset = len(prefix_ids)`` 即可冻结整个 prefix 段。

    Args:
        token_ids: 输入 token ID 列表
        tokenizer: HuggingFace tokenizer 实例
        frozen_token_dict: 需要冻结的token集合
        offset: 起始偏移量，索引 < offset 的位置一律跳过（默认 0，即无偏移）

    Returns:
        可扰动位置索引列表（升序）
    """
    frozen = build_frozen_mask(token_ids, tokenizer, frozen_token_dict)
    return [i for i, f in enumerate(frozen) if not f and i >= offset]


def count_mutable_tokens(
    text_seq: str,
    tokenizer,
    frozen_token_dict: dict,
) -> int:
    """
    统计一段代码在 AGGD 规则下可扰动的 token 个数。

    与优化器一致：``encode(..., add_special_tokens=True)`` 后数
    ``len(get_mutable_positions(...))``。special token 会被自动冻结，且文档
    instruction prefix 未加入该代码序列，因此不会被误计为可扰动位置。

    Args:
        text_seq: 源代码文本（通常为 ``buggy_code``）
        tokenizer: HuggingFace tokenizer
        frozen_token_dict: 需要冻结的token集合

    Returns:
        可扰动位置数量
    """
    token_ids = tokenizer.encode(text_seq, add_special_tokens=True)
    return len(get_mutable_positions(token_ids, tokenizer, frozen_token_dict))
