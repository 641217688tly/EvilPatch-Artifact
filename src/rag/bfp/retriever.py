import logging
from typing import Any, Dict, Iterator, List, Optional

from tqdm import tqdm

from src.models.retriever.Base import BaseEmbedder
from src.rag.bfp.milvus_client import BFPMilvusClient

logger = logging.getLogger(__name__)


class BFPRetriever:
    """
    混合检索器：协调稠密向量检索（语义匹配）与 BM25 检索（关键词匹配）。

    将嵌入模型（BaseEmbedder）与 APR 知识库客户端（BFPMilvusClient）组合，
    提供面向程序修复（APR）RAG 的端到端检索：索引与查询域均为缺陷代码（buggy_code），
    检索命中可携带 fixed_code、is_poisoned、source 等元数据。

    Attributes:
        embedder: BaseEmbedder 实例，用于生成代码嵌入
        milvus_client: BFPMilvusClient 实例，用于存储和检索向量
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        milvus_client: BFPMilvusClient,
    ):
        """
        初始化 APR 混合检索器。

        Args:
            embedder: BaseEmbedder 实例，配置好的代码嵌入模型
            milvus_client: BFPMilvusClient 实例，配置好的 Milvus 客户端

        Note:
            - embedder 与 milvus_client 的维度配置必须匹配
            - 建议在初始化前验证 embedder.dim == milvus_client.dim
        """
        self.embedder = embedder
        self.milvus_client = milvus_client

    @staticmethod
    def _build_entity(query: Dict[str, Any], query_target: str) -> Dict[str, Any]:
        """
        根据 query_target 从查询记录中提取对应的 entity 字段。

        Args:
            query: 单条查询记录
            query_target: 查询目标字段名，可选值为 "buggy_code"、"fixed_code"、"vul_code"

        Returns:
            entity 字典，包含查询侧关键元数据：
            - query_target == "vul_code"：id, cluster_id, source, vul_code, cwe_id, cve_id, cwe_desc, cve_desc
            - 其他（BFP 场景）：id, source, language, buggy_code, fixed_code
        """
        if query_target == "vul_code":
            return {
                "id": query["id"],
                "cluster_id": query.get("cluster_id"),
                "source": query.get("source", ""),
                "vul_code": query.get("vul_code", ""),
                "cwe_id": query.get("cwe_id", ""),
                "cve_id": query.get("cve_id", ""),
                "cwe_desc": query.get("cwe_desc", ""),
                "cve_desc": query.get("cve_desc", ""),
            }
        return {
            "id": query["id"],
            "source": query.get("source", ""),
            "language": query.get("language", ""),
            "buggy_code": query.get("buggy_code", ""),
            "fixed_code": query.get("fixed_code", ""),
        }

    @staticmethod
    def _parse_single_query_hits(hits) -> List[Dict[str, Any]]:
        """
        将单个查询的 hits（来自批量 dense_search / sparse_search）解析为扁平列表。

        用于批量检索中按索引拆解每个查询的结果。

        Args:
            hits: Milvus search 返回结果中单个查询的 hits（可迭代的 hit 对象）

        Returns:
            扁平结果列表（未去重），每项包含:
                doc_id, chunk_id, total_chunks, buggy_code, fixed_code,
                is_poisoned, source, score
        """
        parsed = []
        for hit in hits:
            entity = hit.get("entity", {})
            item = {
                "doc_id": int(entity.get("doc_id", 0)),
                "chunk_id": int(entity.get("chunk_id", 0)),
                "total_chunks": int(entity.get("total_chunks", 1)),
                "buggy_code": entity.get("buggy_code", ""),
                "fixed_code": entity.get("fixed_code", ""),
                "is_poisoned": bool(entity.get("is_poisoned", False)),
                "source": entity.get("source", ""),
                "score": round(float(hit["distance"]), 6),
            }
            if "dense_vector" in entity:
                item["dense_vector"] = entity["dense_vector"]
            parsed.append(item)
        return parsed

    @staticmethod
    def _dedup_by_doc_id(
        parsed: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        按 doc_id 去重，保留每个文档最高分的 chunk。

        Args:
            parsed: _parse_milvus_results 返回的扁平结果列表
            top_k: 去重后保留的最大结果数

        Returns:
            去重并按分数降序排列的前 top_k 个结果
        """
        best: Dict[int, Dict[str, Any]] = {}
        for item in parsed:
            doc_id = item["doc_id"]
            if doc_id not in best or item["score"] > best[doc_id]["score"]:
                best[doc_id] = item

        sorted_results = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    def iter_dense_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        embed_batch_size: int = 256,
        query_target: str = "buggy_code",
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        分批执行稠密向量检索，并按查询逐项返回结果。

        每个分片完成编码、Milvus 搜索和文档级去重后立即产生结果，适合
        只需聚合命中文档而不希望同时保存所有查询结果的离线任务。

        Args:
            queries: 查询列表，每项建议包含:
                - id: 查询唯一标识
                - buggy_code / fixed_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每条查询返回的去重后文档数上限
            embed_batch_size: 嵌入与搜索的分片大小，默认 256。
                控制单次 embed_queries 调用和单次 Milvus search RPC 的查询数量，
                防止超过 Milvus RPC 传输限制（默认 64MB）或 GPU 显存溢出。
            query_target: 查询目标字段名，默认 "buggy_code"，可选 "fixed_code" 或 "vul_code"
                - "buggy_code" / "fixed_code"：BFP 场景，查询记录含 buggy_code / fixed_code 等字段
                - "vul_code"：漏洞查询场景，查询记录含 vul_code / cwe_id / cve_id 等字段
            filter_expr: 可选的 Milvus 标量过滤表达式。
            output_fields: 可选的 Milvus 返回字段列表。

        Yields:
            与输入 queries 顺序对齐的结果，每项包含:
                - entity: 查询侧原始关键字段（字段集合由 query_target 决定）
                - retrieval_results: 检索命中列表（含 doc_id, chunk_id, total_chunks,
                    请求的返回字段以及 score）
        """
        effective_top_k = top_k or self.milvus_client.top_k
        n = len(queries)
        query_texts = [q.get(query_target, "") for q in queries]

        for shard_start in tqdm(
            range(0, n, embed_batch_size), desc="APR dense batch retrieval"
        ):
            shard_end = min(shard_start + embed_batch_size, n)
            shard_texts = query_texts[shard_start:shard_end]

            # 阶段 1: 批量编码 —— 一次调用嵌入模型处理整个分片
            shard_embeddings = self.embedder.embed_queries(shard_texts)
            shard_vectors = [emb.tolist() for emb in shard_embeddings]

            # 阶段 2: 批量搜索 —— 一次 Milvus RPC 传入多个查询向量
            # Milvus client.search(data=[v1, v2, ...]) 原生并行 ANN 搜索
            raw_results = self.milvus_client.dense_search(
                query_dense_list=shard_vectors,
                top_k=effective_top_k,
                filter_expr=filter_expr,
                output_fields=output_fields,
            )

            for i, hits in enumerate(raw_results):
                global_idx = shard_start + i
                parsed = self._parse_single_query_hits(hits)
                yield {
                    "entity": self._build_entity(
                        queries[global_idx],
                        query_target,
                    ),
                    "retrieval_results": self._dedup_by_doc_id(
                        parsed,
                        effective_top_k,
                    ),
                }

        logger.info(
            f"Dense batch retrieval completed: {n} queries, "
            f"top_k={effective_top_k}, shards={-(-n // embed_batch_size)}"
        )

    def dense_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        embed_batch_size: int = 256,
        query_target: str = "buggy_code",
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """批量稠密向量检索，返回与输入查询对齐的完整结果列表。"""
        return list(
            self.iter_dense_retrieve(
                queries=queries,
                top_k=top_k,
                embed_batch_size=embed_batch_size,
                query_target=query_target,
                filter_expr=filter_expr,
                output_fields=output_fields,
            )
        )

    def sparse_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        embed_batch_size: int = 256,
        query_target: str = "buggy_code",
    ) -> List[Dict[str, Any]]:
        """
        批量稀疏向量（BM25）检索，通过 Milvus 原生多文本批查询实现真正的批处理。

        本方法:
        1. 直接收集所有查询文本（无需嵌入模型编码，BM25 由 Milvus 服务端完成）
        2. 一次性将所有查询文本发送给 Milvus sparse search（利用 data=[t1, t2, ...] 原生批查询）
        3. 按查询索引拆解结果，逐查询去重并截断 Top-K

        对于超大批次，按 embed_batch_size 分片处理以控制单次 RPC 大小。

        Args:
            queries: 查询列表，每项建议包含:
                - id: 查询唯一标识
                - buggy_code / fixed_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每条查询返回的去重后文档数上限
            embed_batch_size: 每次发送给 Milvus 的查询分片大小，默认 256。
                控制单次 Milvus search RPC 的查询数量，防止超过 gRPC 传输限制。
            query_target: 查询目标字段名，默认 "buggy_code"，可选 "fixed_code" 或 "vul_code"
                - "buggy_code" / "fixed_code"：BFP 场景，查询记录含 buggy_code / fixed_code 等字段
                - "vul_code"：漏洞查询场景，查询记录含 vul_code / cwe_id / cve_id 等字段

        Returns:
            与输入 queries 对齐的结果列表，每项包含:
                - entity: 查询侧原始关键字段（字段集合由 query_target 决定）
                - retrieval_results: 检索命中列表（含 doc_id, chunk_id, total_chunks,
                    buggy_code, fixed_code, is_poisoned, source, score）
        """
        effective_top_k = top_k or self.milvus_client.top_k
        n = len(queries)

        query_texts = [q.get(query_target, "") for q in queries]

        all_per_query_hits: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

        for shard_start in tqdm(range(0, n, embed_batch_size), desc="APR sparse batch retrieval"):
            shard_end = min(shard_start + embed_batch_size, n)
            shard_texts = query_texts[shard_start:shard_end]

            raw_results = self.milvus_client.sparse_search(
                query_text_list=shard_texts,
                top_k=effective_top_k,
            )

            for i, hits in enumerate(raw_results):
                global_idx = shard_start + i
                parsed = self._parse_single_query_hits(hits)
                all_per_query_hits[global_idx] = self._dedup_by_doc_id(
                    parsed, effective_top_k
                )

        results = []
        for idx, query in enumerate(queries):
            result_item = {
                "entity": self._build_entity(query, query_target),
                "retrieval_results": all_per_query_hits[idx],
            }
            results.append(result_item)

        logger.info(
            f"Sparse batch retrieval completed: {n} queries, "
            f"top_k={effective_top_k}, shards={-(-n // embed_batch_size)}"
        )
        return results

    def hybrid_retrieve_single(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        对单个查询执行混合检索，按 doc_id 去重后返回 Top-K。

        Args:
            query_text: 查询文本，通常为待修复的缺陷代码（buggy_code）
            top_k: 返回的最相似结果数量，默认使用 Milvus 客户端配置
            alpha: 稠密向量权重（0.0–1.0），默认使用 Milvus 客户端配置

        Returns:
            去重后的检索结果列表，每项含 doc_id、buggy_code、fixed_code、
            is_poisoned、source、chunk 信息及 score
        """
        effective_top_k = top_k or self.milvus_client.top_k
        query_embedding = self.embedder.embed_queries([query_text])[0]

        raw_results = self.milvus_client.hybrid_search(
            query_dense=query_embedding.tolist(),
            query_text=query_text,
            top_k=effective_top_k,
            alpha=alpha,
        )

        parsed = []
        for hits in raw_results:
            parsed.extend(self._parse_single_query_hits(hits))
        return self._dedup_by_doc_id(parsed, effective_top_k)

    def hybrid_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        query_target: str = "buggy_code",
    ) -> List[Dict[str, Any]]:
        """
        批量混合检索，返回与输入对齐的结构化结果。

        Args:
            queries: 查询列表，每项建议包含:
                - id: 查询唯一标识
                - buggy_code / fixed_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每条查询返回的去重后文档数上限
            alpha: 稠密向量权重（0.0–1.0）
            query_target: 查询目标字段名，默认 "buggy_code"，可选 "fixed_code" 或 "vul_code"
                - "buggy_code" / "fixed_code"：BFP 场景，查询记录含 buggy_code / fixed_code 等字段
                - "vul_code"：漏洞查询场景，查询记录含 vul_code / cwe_id / cve_id 等字段

        Returns:
            与输入 queries 对齐的结果列表，每项包含:
                - entity: 查询侧原始关键字段（字段集合由 query_target 决定）
                - retrieval_results: 检索命中列表（含 doc_id, chunk_id, total_chunks,
                    buggy_code, fixed_code, is_poisoned, source, score）
        """
        results = []

        for query in tqdm(queries, desc="APR hybrid retrieval"):
            query_text = query.get(query_target)

            hits = self.hybrid_retrieve_single(
                query_text=query_text,
                top_k=top_k,
                alpha=alpha,
            )

            result_item = {
                "entity": self._build_entity(query, query_target),
                "retrieval_results": hits,
            }
            results.append(result_item)

        return results
