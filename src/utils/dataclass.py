"""
漏洞注入任务相关的数据封装类型。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class VulEntity:
    """
    漏洞实例记录（来自 retrieval_results 的单条命中）。
    """
    doc_id: int
    chunk_id: int
    cluster_id: str
    source: str
    total_chunks: int
    vul_code: str
    cwe_id: str
    cve_id: str
    cwe_desc: str
    cve_desc: str
    score: float
    vul_pattern: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VulEntity":
        """
        从字典构建漏洞实例对象，并做轻量类型归一。

        cwe_id / cve_id / cwe_desc 为可选字段（部分检索结果文件可能不包含），
        缺失时默认为空字符串。
        """
        return cls(
            doc_id=int(data["doc_id"]),
            chunk_id=int(data["chunk_id"]),
            cluster_id=str(data["cluster_id"]),
            source=str(data["source"]),
            total_chunks=int(data["total_chunks"]),
            vul_code=str(data["vul_code"]),
            cwe_id=str(data.get("cwe_id", "")),
            cve_id=str(data.get("cve_id", "")),
            cwe_desc=str(data.get("cwe_desc", "")),
            cve_desc=str(data.get("cve_desc", "")),
            score=float(data["score"]),
            vul_pattern=data.get("vul_pattern"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为与 JSON 兼容的字典结构。
        """
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "cluster_id": self.cluster_id,
            "source": self.source,
            "total_chunks": self.total_chunks,
            "vul_code": self.vul_code,
            "cwe_id": self.cwe_id,
            "cve_id": self.cve_id,
            "cwe_desc": self.cwe_desc,
            "cve_desc": self.cve_desc,
            "score": self.score,
            "vul_pattern": self.vul_pattern,
        }


@dataclass
class BFPEntity:
    """
    BFP（Bug-Fix Pair）记录。
    """
    id: str # 文档id
    language: str # 编程语言
    source: str # 数据集来源
    buggy_code: str 
    fixed_code: str
    vinj_code: Optional[str] = None
    retrieval_results: List[VulEntity] = field(default_factory=list) # 相关的CWE实例
    

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BFPEntity":
        """
        从字典构建 BFP 对象，并将 retrieval_results 封装为强类型列表。
        """
        retrieval_results = [
            VulEntity.from_dict(item)
            for item in data.get("retrieval_results", [])
        ]
        return cls(
            id=str(data["id"]),
            language=str(data["language"]),
            source=str(data["source"]),
            buggy_code=str(data["buggy_code"]),
            fixed_code=str(data["fixed_code"]),
            vinj_code=str(data.get("vinj_code", None)),
            retrieval_results=retrieval_results,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为与 JSON 兼容的字典结构。
        """
        return {
            "id": self.id,
            "language": self.language,
            "source": self.source,
            "buggy_code": self.buggy_code,
            "fixed_code": self.fixed_code,
            "vinj_code": self.vinj_code,
            "retrieval_results": [item.to_dict() for item in self.retrieval_results],
        }

# ─────────────────────────────────────────────────────────────────
#  检索对抗攻击日志记录
# ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievalAdvIterLog:
    """单轮迭代的日志记录。"""
    iteration: int
    total_iterations: int
    # token_changes: 列表，每个元组为 (position, old_token_id, new_token_id, old_token_text, new_token_text)
    token_changes: List[tuple] = field(default_factory=list)
    loss_before: float = 0.0
    loss_after: float = 0.0
    avg_sim_before: float = 0.0
    avg_sim_after: float = 0.0
    per_sample_sims: List[float] = field(default_factory=list)
    depth: int = 0
    updated: bool = False
    adv_text_preview: str = ""

    def format(self) -> str:
        lines = [
            f"{'='*10} Iteration {self.iteration} / {self.total_iterations} {'='*10}",
        ]
        if self.updated and self.token_changes:
            num_changes = len(self.token_changes)
            if num_changes == 1:
                # 单条修改保持简洁格式
                pos, old_tid, new_tid, old_text, new_text = self.token_changes[0]
                lines.append(
                    f"[Token Replace] Position {pos}: "
                    f"\"{old_text}\" -> \"{new_text}\" "
                    f"(token_id: {old_tid} -> {new_tid})"
                )
            else:
                # 多条修改显示总数并列出前5个
                lines.append(f"[Token Replace] {num_changes} changes:")
                for i, (pos, old_tid, new_tid, old_text, new_text) in enumerate(self.token_changes[:5]):
                    lines.append(
                        f"  Position {pos}: \"{old_text}\" -> \"{new_text}\" "
                        f"(token_id: {old_tid} -> {new_tid})"
                    )
                if num_changes > 5:
                    lines.append(f"  ... (total {num_changes} changes)")
        else:
            lines.append("[Token Replace] No update this iteration")

        loss_delta = self.loss_after - self.loss_before
        lines.append(
            f"[Loss] {self.loss_before:.6f} -> {self.loss_after:.6f} "
            f"(delta: {loss_delta:+.6f})"
        )

        sim_delta = self.avg_sim_after - self.avg_sim_before
        lines.append(
            f"[Avg Sim to Q+] {self.avg_sim_before:.6f} -> {self.avg_sim_after:.6f} "
            f"(delta: {sim_delta:+.6f})"
        )

        if self.per_sample_sims:
            sim_strs = ", ".join(f"{s:.4f}" for s in self.per_sample_sims)
            lines.append(f"[Per-Sample Sim] [{sim_strs}]")

        status = "reset" if self.updated else "increased"
        lines.append(f"[Depth] {self.depth} ({status})")

        if self.adv_text_preview:
            preview = self.adv_text_preview[:200]
            lines.append(f"[Adv Text Preview] {preview}")

        lines.append("")
        return "\n".join(lines)
