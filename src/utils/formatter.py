import re
import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# ── 格式化相关 ──
# ═══════════════════════════════════════════════════════════════

_CLANG_FORMAT_BIN: str | None = None


def _get_clang_format_bin() -> str:
    global _CLANG_FORMAT_BIN
    if _CLANG_FORMAT_BIN is None:
        bin_path = shutil.which("clang-format")
        if bin_path is None:
            raise FileNotFoundError(
                "未找到 clang-format，请先执行: pip install clang-format"
            )
        _CLANG_FORMAT_BIN = bin_path
    return _CLANG_FORMAT_BIN


class ClangFormatStyle(Enum):
    """clang-format 支持的预设代码风格"""
    LLVM = "LLVM"
    GOOGLE = "Google"
    CHROMIUM = "Chromium"
    MOZILLA = "Mozilla"
    WEBKIT = "WebKit"
    MICROSOFT = "Microsoft"
    GNU = "GNU"


class ClangFormatError(Exception):
    """clang-format 格式化过程中产生的异常"""
    pass


def format_c_code(
    code: str,
    style: ClangFormatStyle = ClangFormatStyle.LLVM,
    language: str = "cpp",
) -> str:
    """
    将被压缩为一行的 C/C++ 代码片段进行格式化。

    Args:
        code:     待格式化的 C/C++ 代码片段。
        style:    格式化风格，默认为 LLVM。
        language: 源代码语言，"cpp" 或 "c"。

    Returns:
        格式化后的代码字符串。
    """
    clang_format_bin = _get_clang_format_bin()

    suffix = ".c" if language.lower() == "c" else ".cpp"
    cmd = [
        clang_format_bin,
        f"--style={style.value}",
        f"--assume-filename=snippet{suffix}",
    ]

    result = subprocess.run(
        cmd,
        input=code,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise ClangFormatError(
            f"clang-format 退出码 {result.returncode}，错误信息:\n{result.stderr}"
        )

    return result.stdout


# ═══════════════════════════════════════════════════════════════
# ── 语言检测相关 ──
# ═══════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """语言检测结果"""
    language: str          # "c" 或 "cpp"
    confidence: float      # 置信度 0.0 ~ 1.0
    c_score: int           # C 语言总得分
    cpp_score: int         # C++ 总得分
    matched_rules: list    # 命中的规则列表，用于调试

    def __repr__(self) -> str:
        return (
            f"DetectionResult(language='{self.language}', "
            f"confidence={self.confidence:.2f}, "
            f"c_score={self.c_score}, cpp_score={self.cpp_score})"
        )


# ── 强信号短路正则（命中即可百分百确定语言，互斥使用）──
# C++ 强信号：只要匹配到其中一项，代码绝对不是纯 C
_CPP_STRONG_RE = re.compile(
    r'std::'
    r'|template\s*<'
    r'|namespace\s+\w+'
    r'|namespace\s*\{'
    r'|class\s+\w+\s*[:{]'
    r'|\bvirtual\b'
    r'|\bcout\b'
    r'|\bcerr\b'
    r'|\bnullptr\b'
    r'|\bconstexpr\b'
    r'|\busing\s+namespace\b'
    r'|\boverride\b'
    r'|\bnew\s+\w'
    r'|\bdelete\b'
    r'|\bthrow\b'
    r'|\btry\s*\{'
    r'|\bcatch\s*\('
)
# C 强信号：只要匹配到其中一项，且同时不存在任何 C++ 强信号
_C_STRONG_RE = re.compile(
    r'\b_Bool\b'
    r'|\b_Generic\b'
    r'|\b_Atomic\b'
    r'|\b_Noreturn\b'
    r'|\b_Static_assert\b'
    r'|\b_Thread_local\b'
    r'|\brestrict\b'
)

# ── 用于检测的正则表达式及其权重 ──
# 设计原则：
#   - C++ 规则：匹配 C++ 独有或绝大多数情况下仅 C++ 使用的语法特征
#   - C 规则：匹配 C 风格惯用法和 C 独有的语法特征
#   - 权重越高表示该特征的区分度越强

_CPP_PATTERNS: list[tuple[re.Pattern, int, str]] = [
    # ── 头文件 / 模块 ──
    (re.compile(r'#include\s*<(iostream|fstream|sstream|iomanip)>'),
     8, "C++ I/O 头文件"),
    (re.compile(r'#include\s*<(vector|list|deque|array|forward_list)>'),
     8, "C++ STL 序列容器头文件"),
    (re.compile(r'#include\s*<(map|set|unordered_map|unordered_set|multimap|multiset)>'),
     8, "C++ STL 关联容器头文件"),
    (re.compile(r'#include\s*<(algorithm|numeric|functional|iterator|ranges)>'),
     8, "C++ STL 算法头文件"),
    (re.compile(r'#include\s*<(string|string_view|regex)>'),
     7, "C++ 字符串头文件"),
    (re.compile(r'#include\s*<(memory|shared_mutex|mutex|thread|future|atomic|condition_variable)>'),
     8, "C++ 内存/并发头文件"),
    (re.compile(r'#include\s*<(type_traits|concepts|variant|optional|any|tuple|utility)>'),
     8, "C++ 元编程/工具头文件"),
    (re.compile(r'#include\s*<(stdexcept|exception|cassert)>'),
     6, "C++ 异常头文件"),
    (re.compile(r'\b(import|export)\s+\w+[\.\w]*\s*;'),
     9, "C++20 模块语法"),

    # ── 命名空间 ──
    (re.compile(r'\busing\s+namespace\s+\w+'),
     8, "using namespace 声明"),
    (re.compile(r'\bnamespace\s+\w+\s*\{'),
     9, "namespace 定义"),
    (re.compile(r'\bnamespace\s*\{'),
     9, "匿名 namespace"),

    # ── std 命名空间 ──
    (re.compile(r'\bstd::\w+'),
     7, "std:: 命名空间限定符"),
    (re.compile(r'\b(cout|cin|cerr|clog|endl)\b'),
     7, "C++ 标准流对象"),

    # ── 类与继承 ──
    (re.compile(r'\bclass\s+\w+[\s:]+'),
     7, "class 定义"),
    (re.compile(r'\b(public|private|protected)\s*:'),
     7, "访问控制说明符"),
    (re.compile(r'\bvirtual\s+'),
     7, "virtual 关键字"),
    (re.compile(r'\boverride\b'),
     8, "override 说明符"),
    (re.compile(r'\bfinal\b'),
     5, "final 说明符"),
    (re.compile(r'\bfriend\s+(class|function)?\s*\w+'),
     6, "friend 声明"),
    (re.compile(r'\bexplicit\b'),
     7, "explicit 关键字"),
    (re.compile(r'\boperator\s*[+\-*/=%<>!&|^~\[\]()]+'),
     8, "运算符重载"),
    (re.compile(r'\b\w+::\w+'),
     3, ":: 作用域解析运算符"),
    (re.compile(r'\bthis\s*->'),
     5, "this 指针"),

    # ── 模板 ──
    (re.compile(r'\btemplate\s*<[^>]*>'),
     8, "template 模板声明"),
    (re.compile(r'\btypename\b'),
     8, "typename 关键字"),

    # ── C++ 类型转换 ──
    (re.compile(r'\b(static_cast|dynamic_cast|const_cast|reinterpret_cast)\s*<'),
     9, "C++ 风格类型转换"),
    (re.compile(r'\btypeid\s*\('),
     7, "typeid 运算符"),

    # ── 异常处理 ──
    (re.compile(r'\btry\s*\{'),
     6, "try 块"),
    (re.compile(r'\bcatch\s*\('),
     6, "catch 块"),
    (re.compile(r'\bthrow\b'),
     5, "throw 表达式"),
    (re.compile(r'\bnoexcept\b'),
     8, "noexcept 说明符"),

    # ── 现代 C++ 特性 (C++11/14/17/20) ──
    (re.compile(r'\bnullptr\b'),
     7, "nullptr 字面量"),
    (re.compile(r'\bconstexpr\b'),
     7, "constexpr 说明符"),
    (re.compile(r'\bconsteval\b'),
     9, "consteval (C++20)"),
    (re.compile(r'\bconstinit\b'),
     9, "constinit (C++20)"),
    (re.compile(r'\bdecltype\s*\('),
     8, "decltype 类型推导"),
    (re.compile(r'\bauto\s+\w+\s*='),
     3, "auto 类型推导 (C++11 风格)"),
    (re.compile(r'\bfor\s*\(\s*(const\s+)?(auto\s*&?)\s+\w+\s*:\s*'),
     9, "范围 for 循环"),

    # ── lambda 表达式 ──
    (re.compile(r'\[[\w\s,&=]*\]\s*\([^)]*\)\s*(mutable\s*)?(->[\w\s:&*<>]+)?\s*\{'),
     9, "lambda 表达式 (完整形式)"),
    (re.compile(r'\[[\w\s,&=]*\]\s*\{'),
     7, "lambda 表达式 (无参形式)"),

    # ── new / delete ──
    (re.compile(r'\bnew\s+\w+'),
     5, "new 运算符"),
    (re.compile(r'\bdelete\s*(\[\])?\s+\w+'),
     6, "delete 运算符"),

    # ── 智能指针 ──
    (re.compile(r'\b(shared_ptr|unique_ptr|weak_ptr|make_shared|make_unique)\b'),
     9, "智能指针"),

    # ── STL 容器/类型 ──
    (re.compile(r'\b(vector|list|deque|map|set|unordered_map|unordered_set|multimap|multiset)\s*<'),
     8, "STL 容器模板"),
    (re.compile(r'\b(string|wstring|u16string|u32string)\b(?!\s*\.h)'),
     4, "C++ string 类型"),
    (re.compile(r'\b(pair|tuple|optional|variant|any)\s*<'),
     8, "C++ 工具类模板"),

    # ── 引用类型 ──
    (re.compile(r'(const\s+)?\w+\s*&\s+\w+'),
     3, "左值引用"),
    (re.compile(r'\w+\s*&&\s+\w+'),
     6, "右值引用"),

    # ── 流操作符 ──
    (re.compile(r'<<\s*("|\'|\w+)'),
     3, "流插入运算符 <<"),
    (re.compile(r'>>\s*\w+'),
     2, "流提取运算符 >>"),

    # ── C++20 概念与协程 ──
    (re.compile(r'\bconcept\s+\w+'),
     9, "concept (C++20)"),
    (re.compile(r'\brequires\b'),
     7, "requires (C++20)"),
    (re.compile(r'\b(co_await|co_return|co_yield)\b'),
     9, "协程关键字 (C++20)"),

    # ── 结构化绑定 ──
    (re.compile(r'\bauto\s*\[\s*\w+(\s*,\s*\w+)*\s*\]'),
     9, "结构化绑定 (C++17)"),
]

_C_PATTERNS: list[tuple[re.Pattern, int, str]] = [
    # ── 头文件 ──
    (re.compile(r'#include\s*<(stdio|stdlib|string|math|ctype|signal|errno|assert)\.h>'),
     5, "C 标准头文件 (.h)"),
    (re.compile(r'#include\s*<(unistd|fcntl|sys/types|sys/stat|sys/socket|netinet|arpa)'),
     4, "POSIX/系统头文件"),

    # ── C 标准库函数 ──
    (re.compile(r'\b(printf|fprintf|sprintf|snprintf|puts|fputs)\s*\('),
     4, "C 标准 I/O 输出函数"),
    (re.compile(r'\b(scanf|fscanf|sscanf|fgets|gets)\s*\('),
     4, "C 标准 I/O 输入函数"),
    (re.compile(r'\b(malloc|calloc|realloc|free)\s*\('),
     5, "C 动态内存管理函数"),
    (re.compile(r'\b(memcpy|memset|memmove|memcmp)\s*\('),
     3, "C 内存操作函数"),
    (re.compile(r'\b(strlen|strcpy|strncpy|strcat|strcmp|strncmp|strstr|strchr)\s*\('),
     4, "C 字符串操作函数"),
    (re.compile(r'\b(fopen|fclose|fread|fwrite|fseek|ftell|rewind)\s*\('),
     4, "C 文件操作函数"),

    # ── C 特有关键字 ──
    (re.compile(r'\brestrict\b'),
     8, "restrict 限定符 (C99)"),
    (re.compile(r'\b_Bool\b'),
     9, "_Bool 类型 (C99)"),
    (re.compile(r'\b_Complex\b'),
     9, "_Complex 类型 (C99)"),
    (re.compile(r'\b_Imaginary\b'),
     9, "_Imaginary 类型 (C99)"),
    (re.compile(r'\b_Atomic\b'),
     8, "_Atomic 限定符 (C11)"),
    (re.compile(r'\b_Generic\b'),
     9, "_Generic 选择 (C11)"),
    (re.compile(r'\b_Noreturn\b'),
     8, "_Noreturn 说明符 (C11)"),
    (re.compile(r'\b_Static_assert\b'),
     8, "_Static_assert (C11)"),
    (re.compile(r'\b_Thread_local\b'),
     8, "_Thread_local (C11)"),
    (re.compile(r'\b_Alignas\b'),
     8, "_Alignas (C11)"),
    (re.compile(r'\b_Alignof\b'),
     8, "_Alignof (C11)"),

    # ── C 风格惯用法 ──
    (re.compile(r'\bNULL\b'),
     3, "NULL 宏"),
    (re.compile(r'\btypedef\s+struct\b'),
     5, "typedef struct 惯用法"),
    (re.compile(r'\btypedef\s+enum\b'),
     4, "typedef enum 惯用法"),
    (re.compile(r'\bstruct\s+\w+\s*\*'),
     2, "struct 指针声明"),
    (re.compile(r'->\s*\w+'),
     2, "-> 成员访问"),
    (re.compile(r'\(\s*(void|int|char|float|double|long|unsigned)\s*\*?\s*\)\s*\w+'),
     4, "C 风格强制类型转换"),
    (re.compile(r'\b(void|int|char)\s*\*\s*\w+\s*=\s*(malloc|calloc)\s*\('),
     6, "malloc 返回值赋值"),

    # ── 函数指针 ──
    (re.compile(r'\w+\s*\(\s*\*\s*\w+\s*\)\s*\('),
     4, "函数指针声明"),

    # ── #define 宏 ──
    (re.compile(r'#define\s+\w+\s*\('),
     2, "函数式宏定义"),
    (re.compile(r'#define\s+\w+\s+\w+'),
     1, "对象式宏定义"),
]


def detect_c_or_cpp(code: str, cpp_dominance_threshold: float = 0.6) -> DetectionResult:
    """
    基于启发式规则判断一段代码片段是 C 语言还是 C++ 语言。

    通过对代码片段进行多模式正则匹配，分别累计 C 和 C++ 的加权得分，
    并依据得分差异给出语言判断及置信度。

    该函数专为 C/C++ 混合数据集设计，且针对 C 语言占比较高的数据集做了偏置：
      - 当检测到 C++ 独有特征（如 class、template、std:: 等）时判定为 C++
      - 当仅检测到 C 风格特征（如 printf、malloc、typedef struct 等）时判定为 C
      - 当两者特征均不明显或得分相近时，优先判定为 C

    Args:
        code: 待检测的 C/C++ 代码片段（函数级别即可，不要求是完整文件）。
        cpp_dominance_threshold: C++ 得分占总分的比例阈值，C++ 占比严格超过
            该值时才判定为 C++。默认 0.6，即 C++ 得分必须占总分 60% 以上。
            取值范围 (0.0, 1.0)，值越大对 C++ 判定越严格（越倾向于判定为 C）。

    Returns:
        DetectionResult 对象，包含以下字段：
          - language:      "c" 或 "cpp"
          - confidence:    置信度 (0.0 ~ 1.0)
          - c_score:       C 规则匹配总分
          - cpp_score:     C++ 规则匹配总分
          - matched_rules: 命中的规则描述列表

    Example:
        >>> result = detect_c_or_cpp('void swap(int *a,int *b){int t=*a;*a=*b;*b=t;}')
        >>> result.language
        'c'
        >>> result = detect_c_or_cpp('std::vector<int> v;for(auto& x:v){std::cout<<x;}')
        >>> result.language
        'cpp'
    """
    if not code or not code.strip():
        return DetectionResult(
            language="c",
            confidence=0.0,
            c_score=0,
            cpp_score=0,
            matched_rules=[],
        )

    # ── 强信号短路：仅当两侧信号互斥时才提前返回 ──
    # 避免「C++ 代码中混用了 C 标准库」导致误判，只有在
    # "有 C++ 强信号 且 无 C 强信号" 或 "有 C 强信号 且 无 C++ 强信号" 时才触发
    _has_cpp_strong = bool(_CPP_STRONG_RE.search(code))
    _has_c_strong   = bool(_C_STRONG_RE.search(code))
    if _has_cpp_strong and not _has_c_strong:
        return DetectionResult(
            language="cpp",
            confidence=0.90,
            c_score=0,
            cpp_score=1,
            matched_rules=["[SHORTCUT] C++ 强信号命中，快速判定为 C++"],
        )
    if _has_c_strong and not _has_cpp_strong:
        return DetectionResult(
            language="c",
            confidence=0.90,
            c_score=1,
            cpp_score=0,
            matched_rules=["[SHORTCUT] C 强信号命中，快速判定为 C"],
        )
    # 两侧均命中或均未命中 → 回落到完整评分流程

    c_score = 0
    cpp_score = 0
    matched_rules: list[str] = []

    # 匹配 C++ 规则
    for pattern, weight, description in _CPP_PATTERNS:
        matches = pattern.findall(code)
        if matches:
            hit_score = weight * len(matches)
            cpp_score += hit_score
            matched_rules.append(f"[C++ +{hit_score:>3}] {description} (×{len(matches)})")

    # 匹配 C 规则
    for pattern, weight, description in _C_PATTERNS:
        matches = pattern.findall(code)
        if matches:
            hit_score = weight * len(matches)
            c_score += hit_score
            matched_rules.append(f"[C   +{hit_score:>3}] {description} (×{len(matches)})")

    # ── 判定逻辑（C 优先偏置）──
    #
    # 设计原理：
    #   1. C 是 C++ 的子集，纯 C 代码中不会出现 C++ 独有语法，
    #      但 C++ 代码中经常混用 C 风格函数（printf、malloc 等）
    #   2. 数据集中 C 占比更高 → 先验概率偏向 C
    #   3. 因此要求 C++ 得分必须「显著压过」C 得分才判定为 C++
    #
    # 判定规则：
    #   - 总分为 0          → 默认 C
    #   - 仅命中 C 规则     → C
    #   - 仅命中 C++ 规则   → C++
    #   - 两者均命中时：
    #       cpp_ratio = cpp_score / total
    #       cpp_ratio >  threshold  → C++
    #       cpp_ratio <= threshold  → C  （C 优先偏置生效区间）

    total = c_score + cpp_score

    if total == 0:
        # 无任何规则命中，默认 C，最低置信度
        language = "c"
        confidence = 0.5

    elif cpp_score == 0:
        # 仅命中 C 规则，确信为 C
        language = "c"
        confidence = min(0.6 + c_score / (c_score + 20), 0.95)

    elif c_score == 0:
        # 仅命中 C++ 规则，确信为 C++
        language = "cpp"
        confidence = min(0.6 + cpp_score / (cpp_score + 20), 0.99)

    else:
        # ── 两者均有命中：C 优先偏置判定 ──
        cpp_ratio = cpp_score / total

        if cpp_ratio > cpp_dominance_threshold:
            # C++ 得分占比显著超过阈值 → 判定 C++
            #
            # 置信度：ratio 越远超 threshold，置信度越高
            # 当 ratio = threshold 时 confidence ≈ 0.55（刚过界，不太确信）
            # 当 ratio → 1.0 时 confidence → 0.95
            overshoot = (cpp_ratio - cpp_dominance_threshold) / (1.0 - cpp_dominance_threshold)
            language = "cpp"
            confidence = min(0.55 + overshoot * 0.40, 0.95)

        else:
            # C++ 得分占比 ≤ 阈值 → 判定 C（偏置生效）
            #
            # 置信度：ratio 越低（C 信号越强），置信度越高
            # 当 ratio ≈ threshold 时 confidence ≈ 0.55（临界区域，不太确信）
            # 当 ratio → 0 时 confidence → 0.90
            undershoot = (cpp_dominance_threshold - cpp_ratio) / cpp_dominance_threshold
            language = "c"
            confidence = min(0.55 + undershoot * 0.35, 0.90)

    return DetectionResult(
        language=language,
        confidence=round(confidence, 4),
        c_score=c_score,
        cpp_score=cpp_score,
        matched_rules=matched_rules,
    )


# ═══════════════════════════════════════════════════════════════
# ── 整合工具：自动检测语言 + 格式化 ──
# ═══════════════════════════════════════════════════════════════

def auto_format_c_code(
    code: str,
    style: ClangFormatStyle = ClangFormatStyle.LLVM,
) -> tuple[str, DetectionResult]:
    """
    自动检测代码片段的语言类型（C / C++），然后进行格式化。

    Args:
        code:  待格式化的 C/C++ 代码片段。
        style: 格式化风格，默认为 LLVM。

    Returns:
        (formatted_code, detection_result) 二元组。
    """
    detection = detect_c_or_cpp(code)
    formatted = format_c_code(code, style=style, language=detection.language)
    return formatted, detection


# ═══════════════════════════════════════════════════════════════
# ── 批量并行格式化 ──
# ═══════════════════════════════════════════════════════════════

# 顶层函数（非闭包），使其可被 ProcessPoolExecutor 通过 pickle 跨进程传递
def _format_one_worker(args: tuple[str, str]) -> tuple[str, DetectionResult]:
    """
    工作进程入口：解包参数后调用 auto_format_c_code。

    Args:
        args: (code, style_value) 二元组，style_value 为 ClangFormatStyle.value 字符串。

    Returns:
        (formatted_code, DetectionResult) 二元组。
    """
    code, style_value = args
    style = ClangFormatStyle(style_value)
    formatted, detection = auto_format_c_code(code, style=style)
    return formatted, detection


def _format_one_worker_safe(args: tuple[str, str]) -> tuple[str, DetectionResult]:
    """
    _format_one_worker 的容错版本：捕获所有异常，失败时返回原始代码。
    必须为顶层函数，以保证 ProcessPoolExecutor 可跨进程 pickle。
    """
    code = args[0]
    try:
        return _format_one_worker(args)
    except Exception:
        return code, DetectionResult(
            language="c", confidence=0.0, c_score=0, cpp_score=0,
            matched_rules=["[ERROR] 格式化失败，返回原始代码"],
        )


def auto_format_c_code_batch(
    codes: list[str],
    style: ClangFormatStyle = ClangFormatStyle.LLVM,
    max_workers: int | None = None,
    chunksize: int = 16,
    on_error: str = "keep",
    show_progress: bool = True,
) -> list[tuple[str, DetectionResult]]:
    """
    并行格式化大批量 C/C++ 代码片段。

    内部使用 ProcessPoolExecutor 将多个 clang-format 子进程并发执行，
    加速比接近 CPU 核心数。输出列表与输入列表顺序严格对应。

    Args:
        codes:         待格式化的代码字符串列表。
        style:         格式化风格，默认为 LLVM。
        max_workers:   工作进程数，默认为 max(1, cpu_count() - 2)。
        chunksize:     每次向工作进程分发的任务块大小，较大值可减少 IPC 开销。
        on_error:      单条格式化失败时的处理策略：
                         "keep"  — 返回原始代码（不中断整批，推荐）；
                         "raise" — 直接抛出异常。
        show_progress: 是否显示 tqdm 进度条，默认为 True。

    Returns:
        与 codes 等长的 (formatted_code, DetectionResult) 列表。
    """
    if not codes:
        return []

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 2)

    style_value = style.value
    args_list = [(code, style_value) for code in codes]

    results: list[tuple[str, DetectionResult]] = []

    worker = _format_one_worker_safe if on_error == "keep" else _format_one_worker

    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(worker, args_list, chunksize=chunksize)
        if show_progress and _has_tqdm:
            mapped = _tqdm(mapped, total=len(codes), desc="Formatting", unit="snippet")
        for result in mapped:
            results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
# ── 使用示例 ──
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_code = ('template<typename T>T max_val(const std::vector<T>& v){'
                 'T m=v[0];for(const auto& x:v){if(x>m)m=x;}return m;}')
    print(f"\n原始代码:\n  {demo_code}\n")

    formatted, detection = auto_format_c_code(demo_code)
    print(f"检测结果: {detection}")
    print(f"\n格式化输出:\n{formatted}")