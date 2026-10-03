"""
工具模块：提供 CWE 数据查询等辅助功能。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def normalize_cwe_id(raw: str) -> str:
    """将 CWE 标识统一为大写并去掉 ``-`` 与 ``_``，便于字符串比较。"""
    return raw.upper().replace("-", "").replace("_", "")


# ---------------------------------------------------------------------------
# 命名空间常量
# ---------------------------------------------------------------------------

_CWE_NS = "http://cwe.mitre.org/cwe-7"
_XHTML_NS = "http://www.w3.org/1999/xhtml"

_NS = {
    "cwe": _CWE_NS,
    "xhtml": _XHTML_NS,
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RelatedWeakness:
    """
    表示 CWE 条目之间的关系（如父子、先后等）。

    示例：CWE-79 (XSS) 与 CWE-74 (注入类弱点) 的关系
        nature="ChildOf", cwe_id="74", view_id="1000", ordinal="Primary"
        表示：CWE-79 是 CWE-74 的子类，在研究视图(1000)中为主关系
    """
    nature: str       # 关系类型：ChildOf(子类)/ParentOf(父类)/CanPrecede(可导致)/PeerOf(同级)等
    cwe_id: str       # 关联的 CWE ID，如 "74"
    view_id: str      # 所属视图的 ID，如 "1000"(研究视图)
    ordinal: str      # 关系重要性：Primary(主要)/Other(次要)


@dataclass
class Consequence:
    """
    描述安全弱点的潜在后果和影响。

    示例：CWE-79 (XSS) 的后果
        scopes=["Confidentiality", "Integrity", "Availability"]
        impacts=["Execute Unauthorized Code", "Bypass Protection Mechanism"]
        note="攻击者可以在用户浏览器中执行任意脚本，窃取会话令牌或执行恶意操作"
    """
    scopes: List[str] = field(default_factory=list)
    # 影响范围，如：["Confidentiality"(机密性), "Integrity"(完整性), "Availability"(可用性)]

    impacts: List[str] = field(default_factory=list)
    # 具体影响，如：["Read Application Data"(读取应用数据),
    #            "Gain Privileges or Assume Identity"(获取权限或冒充身份),
    #            "Execute Unauthorized Code"(执行未授权代码)]

    note: str = ""
    # 补充说明，提供额外上下文或详细解释


@dataclass
class Mitigation:
    """
    缓解措施，描述如何减少或消除弱点的风险。

    示例：CWE-89 (SQL注入) 的缓解措施
        phases=["Architecture and Design", "Implementation"]
        strategy="Input Validation"
        description="使用参数化查询或预编译语句，避免直接拼接SQL字符串"
        effectiveness="High"
    """
    phases: List[str] = field(default_factory=list)
    # 适用的开发阶段，如：["Architecture and Design"(架构设计),
    #                  "Implementation"(实现),
    #                  "Build and Compilation"(构建编译),
    #                  "Operation"(运维)]

    strategy: str = ""
    # 缓解策略类型，如："Input Validation"(输入验证)、
    #                "Output Encoding"(输出编码)、
    #                "Parameterization"(参数化)、
    #                "Compilation or Build Hardening"(编译加固)

    description: str = ""
    # 具体缓解措施的详细描述

    effectiveness: str = ""
    # 有效性评估："High"(高)/"Medium"(中)/"Low"(低)/"Defense in Depth"(纵深防御)


@dataclass
class ObservedExample:
    """
    真实观测到的安全漏洞案例（CVE记录）。

    示例：CWE-1004 (Cookie缺少HttpOnly标志) 的观测案例
        reference="CVE-2024-47833"
        description="Python机器学习库的会话Cookie未使用HTTPOnly安全属性"
        link="https://www.cve.org/CVERecord?id=CVE-2024-47833"
    """
    reference: str = ""
    # CVE编号，如 "CVE-2024-47833"

    description: str = ""
    # 漏洞的简要描述

    link: str = ""
    # CVE详情页面的URL链接


@dataclass
class CWEEntry:
    """
    表示一条 CWE 条目（Weakness / Category / View）的结构化信息。

    示例1：CWE-79 (跨站脚本XSS) - 典型的Weakness条目
        id="79"
        name="Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"
        entry_type="Weakness"
        abstraction="Base"          # 基础级：具体但通用的弱点类型
        structure="Simple"
        status="Stable"             # 稳定状态，定义已成熟
        description="产品在生成网页时未能正确中和用户可控输入..."
        extended_description="有多种XSS变体，包括反射型、存储型、DOM型..."

    示例2：CWE-699 (软件开发视图) - 典型的View条目
        id="699"
        name="Software Development"
        entry_type="View"
        status="Draft"
        description="This view organizes weaknesses around concepts..."

    示例3：CWE-1000 (研究概念) - 研究用的顶级分类
        id="1000"
        name="Research Concepts"
        entry_type="View"
        status="Draft"
        description="用于促进弱点研究的视图，包含所有弱点..."
    """

    # ==================== 基本属性 ====================
    id: str
    # CWE 唯一标识符，如 "79"、"89"、"1004"（纯数字，不带 CWE- 前缀）

    name: str
    # CWE 名称/标题，如 "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"

    entry_type: str
    # 条目类型："Weakness"(弱点) | "Category"(分类) | "View"(视图)
    # - Weakness：具体的软件安全弱点
    # - Category：弱点的逻辑分组/分类
    # - View：特定角度组织弱点的视图

    abstraction: str = ""
    # 抽象级别（仅 Weakness 有）：
    # - "Class"(类级)：高抽象，如 "Input Validation and Representation"
    # - "Base"(基础级)：中等抽象，如 "SQL Injection"
    # - "Variant"(变体)：具体实现级，如 "Sensitive Cookie Without 'HttpOnly' Flag"
    # - "Pillar"(支柱)：核心概念分类，数量很少

    structure: str = ""
    # 结构类型（仅 Weakness 有）："Simple" / "Composite" / "Chain"

    status: str = ""
    # 状态："Stable"(稳定) / "Draft"(草案) / "Incomplete"(不完整) / "Deprecated"(已弃用)

    # ==================== 描述信息 ====================
    description: str = ""
    # 简要描述：一句话概括该弱点的核心问题
    # 示例："The product uses a cookie to store sensitive information, but the cookie is not marked with the HttpOnly flag."

    extended_description: str = ""
    # 详细描述：多段落详细解释该弱点的技术细节、攻击场景、成因等
    # 示例："The HttpOnly flag directs compatible browsers to prevent client-side script from accessing cookies..."

    summary: str = ""
    # 摘要说明（仅 Category 有）：对该分类的概括性描述

    # ==================== 关系信息 ====================
    related_weaknesses: List[RelatedWeakness] = field(default_factory=list)
    # 与其他 CWE 条目的关系列表
    # 示例：CWE-79 可能包含 [ChildOf CWE-74, CanPrecede CWE-494, PeerOf CWE-352]

    # ==================== 平台信息 ====================
    applicable_platforms: Dict[str, List[str]] = field(default_factory=dict)
    # 适用平台信息，按类型分类
    # 示例：{"Language": ["C", "C++"], "Technology": ["Web Server"], "Operating_System": ["Windows"]}

    # ==================== 引入方式 ====================
    modes_of_introduction: List[str] = field(default_factory=list)
    # 弱点通常被引入的开发阶段
    # 示例：["Architecture and Design", "Implementation", "Build and Compilation"]

    # ==================== 利用评估 ====================
    likelihood_of_exploit: str = ""
    # 被利用的可能性评估："High" / "Medium" / "Low" / "None"

    # ==================== 安全后果 ====================
    common_consequences: List[Consequence] = field(default_factory=list)
    # 该弱点可能导致的常见安全后果列表
    # 示例：XSS 可能导致机密性泄露、完整性破坏等

    # ==================== 缓解措施 ====================
    potential_mitigations: List[Mitigation] = field(default_factory=list)
    # 减少或消除该弱点风险的缓解措施列表
    # 示例：输入验证、输出编码、使用安全API等

    # ==================== 实际案例 ====================
    observed_examples: List[ObservedExample] = field(default_factory=list)
    # 该弱点的真实CVE观测案例列表，提供真实世界中的漏洞证据

    # ==================== 映射说明 ====================
    mapping_usage: str = ""
    # 映射使用规则："Allowed"(允许) / "Prohibited"(禁止) / "Discouraged"(不推荐)
    # 指导如何将该 CWE 映射到实际漏洞

    mapping_rationale: str = ""
    # 映射规则的详细理由说明

    def __str__(self) -> str:
        lines: List[str] = [
            f"CWE-{self.id}: {self.name}",
            f"  类型: {self.entry_type}" + (f" / {self.abstraction}" if self.abstraction else ""),
            f"  状态: {self.status}",
        ]
        if self.description:
            lines.append(f"  描述: {self.description}")
        if self.extended_description:
            lines.append(f"  详述: {self.extended_description[:200]}{'...' if len(self.extended_description) > 200 else ''}")
        if self.related_weaknesses:
            rels = ", ".join(f"{r.nature} CWE-{r.cwe_id}" for r in self.related_weaknesses[:5])
            lines.append(f"  关联弱点: {rels}")
        if self.likelihood_of_exploit:
            lines.append(f"  利用可能性: {self.likelihood_of_exploit}")
        if self.mapping_usage:
            lines.append(f"  映射用法: {self.mapping_usage}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _tag(local: str) -> str:
    """返回带命名空间的完整标签名。"""
    return f"{{{_CWE_NS}}}{local}"


def _elem_text(elem: Optional[ET.Element]) -> str:
    """提取元素的全部文本内容（包含 xhtml 子元素的文本），去除多余空白。"""
    if elem is None:
        return ""
    parts: List[str] = []
    for text in elem.itertext():
        parts.append(text)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _parse_weakness(elem: ET.Element) -> CWEEntry:
    """将 <Weakness> XML 元素解析为 CWEEntry。"""
    entry = CWEEntry(
        id=elem.get("ID", ""),
        name=elem.get("Name", ""),
        entry_type="Weakness",
        abstraction=elem.get("Abstraction", ""),
        structure=elem.get("Structure", ""),
        status=elem.get("Status", ""),
    )

    # 描述
    entry.description = _elem_text(elem.find(_tag("Description")))
    entry.extended_description = _elem_text(elem.find(_tag("Extended_Description")))

    # 关联弱点
    rw_container = elem.find(_tag("Related_Weaknesses"))
    if rw_container is not None:
        for rw in rw_container.findall(_tag("Related_Weakness")):
            entry.related_weaknesses.append(RelatedWeakness(
                nature=rw.get("Nature", ""),
                cwe_id=rw.get("CWE_ID", ""),
                view_id=rw.get("View_ID", ""),
                ordinal=rw.get("Ordinal", ""),
            ))

    # 适用平台
    ap = elem.find(_tag("Applicable_Platforms"))
    if ap is not None:
        for child in ap:
            local = child.tag.split("}")[-1]  # Language / Technology / Operating_System
            name_or_class = child.get("Name") or child.get("Class") or ""
            if name_or_class:
                entry.applicable_platforms.setdefault(local, []).append(name_or_class)

    # 引入方式
    moi = elem.find(_tag("Modes_Of_Introduction"))
    if moi is not None:
        for intro in moi.findall(_tag("Introduction")):
            phases = [_elem_text(p) for p in intro.findall(_tag("Phase"))]
            entry.modes_of_introduction.extend(phases)

    # 利用可能性
    entry.likelihood_of_exploit = _elem_text(elem.find(_tag("Likelihood_Of_Exploit")))

    # 后果
    cc = elem.find(_tag("Common_Consequences"))
    if cc is not None:
        for cons in cc.findall(_tag("Consequence")):
            c = Consequence(
                scopes=[_elem_text(s) for s in cons.findall(_tag("Scope"))],
                impacts=[_elem_text(i) for i in cons.findall(_tag("Impact"))],
                note=_elem_text(cons.find(_tag("Note"))),
            )
            entry.common_consequences.append(c)

    # 缓解措施
    pm = elem.find(_tag("Potential_Mitigations"))
    if pm is not None:
        for mit in pm.findall(_tag("Mitigation")):
            m = Mitigation(
                phases=[_elem_text(p) for p in mit.findall(_tag("Phase"))],
                strategy=_elem_text(mit.find(_tag("Strategy"))),
                description=_elem_text(mit.find(_tag("Description"))),
                effectiveness=_elem_text(mit.find(_tag("Effectiveness"))),
            )
            entry.potential_mitigations.append(m)

    # 观测示例（CVE）
    oe_container = elem.find(_tag("Observed_Examples"))
    if oe_container is not None:
        for oe in oe_container.findall(_tag("Observed_Example")):
            entry.observed_examples.append(ObservedExample(
                reference=_elem_text(oe.find(_tag("Reference"))),
                description=_elem_text(oe.find(_tag("Description"))),
                link=_elem_text(oe.find(_tag("Link"))),
            ))

    # 映射说明
    mn = elem.find(_tag("Mapping_Notes"))
    if mn is not None:
        entry.mapping_usage = _elem_text(mn.find(_tag("Usage")))
        entry.mapping_rationale = _elem_text(mn.find(_tag("Rationale")))

    return entry


def _parse_category(elem: ET.Element) -> CWEEntry:
    """将 <Category> XML 元素解析为 CWEEntry。"""
    entry = CWEEntry(
        id=elem.get("ID", ""),
        name=elem.get("Name", ""),
        entry_type="Category",
        status=elem.get("Status", ""),
    )
    entry.summary = _elem_text(elem.find(_tag("Summary")))
    entry.description = entry.summary  # 保持接口统一

    mn = elem.find(_tag("Mapping_Notes"))
    if mn is not None:
        entry.mapping_usage = _elem_text(mn.find(_tag("Usage")))
        entry.mapping_rationale = _elem_text(mn.find(_tag("Rationale")))

    # Category 下的 Relationships（成员列表）
    rels = elem.find(_tag("Relationships"))
    if rels is not None:
        for member in rels.findall(_tag("Has_Member")):
            entry.related_weaknesses.append(RelatedWeakness(
                nature="HasMember",
                cwe_id=member.get("CWE_ID", ""),
                view_id=member.get("View_ID", ""),
                ordinal="",
            ))

    return entry


def _parse_view(elem: ET.Element) -> CWEEntry:
    """将 <View> XML 元素解析为 CWEEntry。"""
    entry = CWEEntry(
        id=elem.get("ID", ""),
        name=elem.get("Name", ""),
        entry_type="View",
        status=elem.get("Status", ""),
    )
    entry.description = _elem_text(elem.find(_tag("Objective")))
    return entry


# ---------------------------------------------------------------------------
# 主工具类
# ---------------------------------------------------------------------------


class CWEUtils:
    """
    CWE 数据查询工具类。

    从 MITRE 提供的 cwec_*.xml 文件中解析所有 CWE 条目（Weakness / Category / View），
    并提供按 CWE ID 快速检索的接口。

    Usage::

        cwe = CWEUtils("data/raw/cwe/cwec_v4.19.1.xml")
        entry = cwe.get("79")          # 返回 CWEEntry 对象
        entry = cwe.get("CWE-79")      # 同上，忽略前缀大小写
        print(entry)

        info = cwe.get_info("79")      # 返回结构化字典（适合序列化）
        entries = cwe.search("injection")  # 按名称关键词搜索
    """

    def __init__(self, xml_path: str | Path = "../../data/raw/cwe/cwec_v4.19.1.xml") -> None:
        self._path = Path(xml_path)
        self._entries: Dict[str, CWEEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # 加载与解析
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """解析 XML 文件，构建 ID -> CWEEntry 的内存索引。"""
        if not self._path.exists():
            raise FileNotFoundError(f"CWE XML 文件不存在: {self._path}")

        tree = ET.parse(self._path)
        root = tree.getroot()

        section_parsers = {
            _tag("Weakness"):  _parse_weakness,
            _tag("Category"):  _parse_category,
            _tag("View"):      _parse_view,
        }

        container_tags = [
            _tag("Weaknesses"),
            _tag("Categories"),
            _tag("Views"),
        ]

        for container_tag in container_tags:
            container = root.find(container_tag)
            if container is None:
                continue
            for child in container:
                parser = section_parsers.get(child.tag)
                if parser is not None:
                    entry = parser(child)
                    if entry.id:
                        self._entries[entry.id] = entry

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_cwe_id(cwe_id: str | int) -> str:
        """将各种形式的 CWE ID 统一为纯数字字符串。

        支持: 79, "79", "CWE-79", "cwe-79", "CWE79"
        """
        s = str(cwe_id).strip()
        # 去掉 CWE- 或 CWE 前缀（大小写不敏感）
        s = re.sub(r"(?i)^cwe-?", "", s)
        return s

    def get(self, cwe_id: str | int) -> Optional[CWEEntry]:
        """
        按 CWE ID 查询条目，返回 :class:`CWEEntry` 对象，未找到时返回 ``None``。

        参数
        ----
        cwe_id:
            CWE 编号，支持多种格式：``79`` / ``"79"`` / ``"CWE-79"`` / ``"cwe-79"``

        返回
        ----
        CWEEntry | None
        """
        normalized = self._normalize_cwe_id(cwe_id)
        return self._entries.get(normalized)

    def get_info(self, cwe_id: str | int) -> Optional[Dict]:
        """
        按 CWE ID 查询，返回结构化字典（方便 JSON 序列化或打印）。

        字典包含以下键：
        ``id``, ``name``, ``entry_type``, ``abstraction``, ``status``,
        ``description``, ``extended_description``,
        ``related_weaknesses``, ``applicable_platforms``,
        ``modes_of_introduction``, ``likelihood_of_exploit``,
        ``common_consequences``, ``potential_mitigations``,
        ``observed_examples``, ``mapping_usage``, ``mapping_rationale``

        未找到时返回 ``None``。
        """
        entry = self.get(cwe_id)
        if entry is None:
            return None
        return {
            "id": entry.id,
            "name": entry.name,
            "entry_type": entry.entry_type,
            "abstraction": entry.abstraction,
            "structure": entry.structure,
            "status": entry.status,
            "description": entry.description,
            "extended_description": entry.extended_description,
            "summary": entry.summary,
            "related_weaknesses": [
                {
                    "nature": rw.nature,
                    "cwe_id": rw.cwe_id,
                    "view_id": rw.view_id,
                    "ordinal": rw.ordinal,
                }
                for rw in entry.related_weaknesses
            ],
            "applicable_platforms": entry.applicable_platforms,
            "modes_of_introduction": entry.modes_of_introduction,
            "likelihood_of_exploit": entry.likelihood_of_exploit,
            "common_consequences": [
                {
                    "scopes": c.scopes,
                    "impacts": c.impacts,
                    "note": c.note,
                }
                for c in entry.common_consequences
            ],
            "potential_mitigations": [
                {
                    "phases": m.phases,
                    "strategy": m.strategy,
                    "description": m.description,
                    "effectiveness": m.effectiveness,
                }
                for m in entry.potential_mitigations
            ],
            "observed_examples": [
                {
                    "reference": oe.reference,
                    "description": oe.description,
                    "link": oe.link,
                }
                for oe in entry.observed_examples
            ],
            "mapping_usage": entry.mapping_usage,
            "mapping_rationale": entry.mapping_rationale,
        }

    def search(self, keyword: str, *, case_sensitive: bool = False) -> List[CWEEntry]:
        """
        按关键词搜索 CWE 条目名称，返回匹配的 :class:`CWEEntry` 列表。

        参数
        ----
        keyword:
            搜索关键词（支持正则表达式）。
        case_sensitive:
            是否区分大小写，默认不区分。
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(keyword, flags)
        return [
            entry for entry in self._entries.values()
            if pattern.search(entry.name) or pattern.search(entry.description)
        ]

    def get_prompt(self, cwe_id: str | int) -> str:
        """
        构建面向 LLM 的 CWE 文本上下文（不包含 CVE 信息）。

        参数
        ----
        cwe_id:
            必填，CWE 编号，支持 ``79`` / ``"79"`` / ``"CWE-79"`` 等格式。

        返回
        ----
        str:
            可直接拼接到提示词中的结构化文本，包含：
            - CWE 定义与核心特点
            - 关联关系、后果、缓解措施
        """
        entry = self.get(cwe_id)
        normalized_cwe = self._normalize_cwe_id(cwe_id)
        if entry is None:
            return (
                # f"[CWE CONTEXT]\n"
                f"- Requested CWE: CWE-{normalized_cwe}\n"
                f"- Result: The requested entry was not found in the CWE dataset.\n"
            )

        lines: List[str] = []
        # lines.append("[CWE CONTEXT]")
        lines.append(f"- CWE ID: CWE-{entry.id}")
        lines.append(f"- Name: {entry.name}")
        lines.append(f"- Type: {entry.entry_type}")
        # if entry.abstraction:
        #     lines.append(f"- Abstraction: {entry.abstraction}")
        # if entry.structure:
        #     lines.append(f"- Structure: {entry.structure}")
        # if entry.status:
        #     lines.append(f"- Status: {entry.status}")

        lines.append("- Definition:")
        if entry.description:
            lines.append(f"  {entry.description}")
        elif entry.summary:
            lines.append(f"  {entry.summary}")
        else:
            lines.append("  (No definition text provided in source XML)")

        if entry.extended_description:
            lines.append("- Detailed Description:")
            lines.append(f"  {entry.extended_description}")
        
        # if entry.related_weaknesses:
        #     lines.append("- Related CWE Relationships:")
        #     for rw in entry.related_weaknesses[:8]:
        #         rel = f"  - {rw.nature}: CWE-{rw.cwe_id}"
        #         extra: List[str] = []
        #         if rw.view_id:
        #             extra.append(f"View {rw.view_id}")
        #         if rw.ordinal:
        #             extra.append(rw.ordinal)
        #         if extra:
        #             rel += f" ({', '.join(extra)})"
        #         lines.append(rel)

        if entry.common_consequences:
            lines.append("- Typical Security Impacts:")
            for cons in entry.common_consequences[:6]:
                scopes = ", ".join([s for s in cons.scopes if s]) or "N/A"
                impacts = ", ".join([i for i in cons.impacts if i]) or "N/A"
                note = cons.note or "N/A"
                lines.append(f"  - Scope: {scopes}; Impact: {impacts}; Note: {note}")

        if entry.potential_mitigations:
            lines.append("- Mitigation Guidance:")
            for mit in entry.potential_mitigations[:6]:
                phases = ", ".join([p for p in mit.phases if p]) or "N/A"
                strategy = mit.strategy or "N/A"
                effectiveness = mit.effectiveness or "N/A"
                description = mit.description or "N/A"
                lines.append(
                    f"  - Phase: {phases}; Strategy: {strategy}; "
                    f"Effectiveness: {effectiveness}; Action: {description}"
                )

        # if entry.mapping_usage or entry.mapping_rationale:
        #     lines.append("- Mapping Notes:")
        #     if entry.mapping_usage:
        #         lines.append(f"  - Usage: {entry.mapping_usage}")
        #     if entry.mapping_rationale:
        #         lines.append(f"  - Rationale: {entry.mapping_rationale}")

        # lines.append("- Prompting Hint:")
        # lines.append(
        #     "  Use the definition + characteristics to infer root cause, then leverage CVE example "
        #     "details to ground exploit pattern, affected component, and practical mitigation."
        # )

        return "\n".join(lines)

    def list_ids(self) -> List[str]:
        """返回已加载的所有 CWE ID 列表（纯数字字符串）。"""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"CWEUtils(path={self._path!r}, loaded={len(self._entries)} entries)"

if __name__ == "__main__":
    cwe = CWEUtils("data/raw/cwe/cwec_v4.19.1.xml")
    print('=' * 50)
    print(cwe.get_prompt("119"))
    print('=' * 50)
    print(cwe.get_prompt("264"))
    print('=' * 50)
    print(cwe.get_prompt("399"))
    print('=' * 50)
    print(cwe.get_prompt("20"))
    print('=' * 50)
    print(cwe.get_prompt("200"))
    print('=' * 50)
    print(cwe.get_prompt("125"))
    print('=' * 50)
    print(cwe.get_prompt("362"))
    print('=' * 50)
    print(cwe.get_prompt("787"))
    print('=' * 50)
    print(cwe.get_prompt("476"))
    print('=' * 50)
    print(cwe.get_prompt("189"))
    print('=' * 50)
    print(cwe.get_prompt("416"))
    print('=' * 50)
    print(cwe.get_prompt("190"))
    print('=' * 50)
    print(cwe.get_prompt("310"))
    print('=' * 50)
    print(cwe.get_prompt("703"))
    print('=' * 50)