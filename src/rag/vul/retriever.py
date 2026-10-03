import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from src.models.retriever.Base import BaseEmbedder
from src.rag.vul.milvus_client import VulMilvusClient

logger = logging.getLogger(__name__)

class VulRetriever:
    """
    混合检索器：协调稠密向量检索（语义匹配）和 BM25 检索（关键词匹配）。

    该类作为高层业务逻辑封装，将嵌入模型（BaseEmbedder）和向量数据库客户端（VinjMilvusClient）
    组合在一起，提供端到端的漏洞代码检索能力。

    Attributes:
        embedder: BaseEmbedder 实例，用于生成代码嵌入
        milvus_client: VinjMilvusClient 实例，用于存储和检索向量
    
    Constants:
        CLUSTER_IDS: 漏洞知识库的固定聚类簇 ID 列表，共 5 个簇（KMeans k=5）
    """

    CLUSTER_IDS = ['0', '1', '2', '3', '4']

    def __init__(
        self,
        embedder: BaseEmbedder,
        milvus_client: VulMilvusClient,
    ):
        """
        初始化混合检索器。

        Args:
            embedder: BaseEmbedder 实例，配置好的代码嵌入模型
            milvus_client: VinjMilvusClient 实例，配置好的 Milvus 客户端

        Note:
            - embedder 和 milvus_client 的维度配置必须匹配
            - 建议在初始化前验证 embedder.dim == milvus_client.dim
        """
        self.embedder = embedder
        self.milvus_client = milvus_client

    @staticmethod
    def _parse_single_query_hits(hits) -> List[Dict[str, Any]]:
        """
        将单个查询的 hits（来自批量 dense_search / sparse_search）解析为扁平列表。

        用于批量检索中按索引拆解每个查询的结果。

        Args:
            hits: Milvus search 返回结果中单个查询的 hits（可迭代的 hit 对象）

        Returns:
            扁平结果列表（未去重），每项包含:
                doc_id, chunk_id, total_chunks, cluster_id, source, language,
                vul_code, fixed_code, diff, cwe_id, cve_id,
                cwe_desc, cve_desc, score
        """
        parsed = []
        for hit in hits:
            entity = hit.get("entity", {})
            parsed.append(
                {
                    "doc_id": int(entity.get("doc_id", 0)),
                    "chunk_id": int(entity.get("chunk_id", 0)),
                    "total_chunks": int(entity.get("total_chunks", 1)),
                    "cluster_id": str(entity.get("cluster_id", "")),
                    "source": entity.get("source", ""),
                    "language": entity.get("language", ""),
                    "vul_code": entity.get("vul_code", ""),
                    "fixed_code": entity.get("fixed_code", ""),
                    "diff": entity.get("diff", ""),
                    "cwe_id": entity.get("cwe_id", ""),
                    "cve_id": entity.get("cve_id", ""),
                    "cwe_desc": entity.get("cwe_desc", ""),
                    "cve_desc": entity.get("cve_desc", ""),
                    "score": round(float(hit["distance"]), 6),
                }
            )
        return parsed

    @staticmethod
    def _round_robin_merge(
        cluster_results: List[List[Dict[str, Any]]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        对多个簇的检索结果执行轮询交叉合并（Round-Robin Merge），保证结果多样性。

        策略：每轮从每个非空簇的头部依次取一条记录，遇到重复 doc_id 则跳过
        （保留分数更高的已有项），直到累积 top_k 条或所有簇均耗尽为止。

        Args:
            cluster_results: 按簇分组的检索结果列表，每个元素为一个簇的结果列表
                （结果已按分数降序排列）
            top_k: 最终保留的最大结果数

        Returns:
            轮询合并并按 doc_id 去重后的结果列表，长度不超过 top_k
        """
        queues = [list(results) for results in cluster_results]
        seen_doc_ids: Dict[int, float] = {}
        merged: List[Dict[str, Any]] = []

        while len(merged) < top_k:
            advanced = False
            for q in queues:
                if not q:
                    continue
                item = q.pop(0)
                doc_id = item["doc_id"]
                score = item["score"]
                if doc_id not in seen_doc_ids:
                    seen_doc_ids[doc_id] = score
                    merged.append(item)
                    advanced = True
                    if len(merged) >= top_k:
                        break
                elif score > seen_doc_ids[doc_id]:
                    # 更新为更高分的结果（替换已有项）
                    seen_doc_ids[doc_id] = score
                    for i, m in enumerate(merged):
                        if m["doc_id"] == doc_id:
                            merged[i] = item
                            break
            if not advanced:
                # 所有簇均已耗尽
                break

        return merged

    @staticmethod
    def _dedup_by_doc_id(
        parsed: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        按 doc_id 去重，保留每个文档最高分的 chunk。

        Args:
            parsed: _parse_single_query_hits 返回的扁平结果列表
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

    def hybrid_retrieve_single(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        cluster_search: bool = False,
        cwe_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        对单个查询执行混合检索，按 doc_id 去重后返回 Top-K 结果。

        Args:
            query_text: 查询文本（可以是自然语言描述或代码片段）
            top_k: 返回的最相似结果数量，默认使用 MilvusClient 的配置
            alpha: 稠密向量权重（0.0-1.0），默认使用 MilvusClient 的配置
            cluster_search: 是否启用分簇召回策略。为 True 时，对每个聚类簇独立检索
                后通过轮询交叉合并（Round-Robin Merge）保证结果多样性；
                为 False 时使用原有全库混合检索。
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件，仅从匹配的文档中检索

        Returns:
            List[Dict]: 去重后的检索结果列表，每个元素包含:
                - doc_id: 原始文档 ID（字符串）
                - cluster_id: 聚类簇 ID
                - cwe_id: CWE 标识符
                - cve_id: CVE 编号
                - score: 融合后的相似度得分（最高分 chunk 的得分）
        """
        effective_top_k = top_k or self.milvus_client.top_k
        query_embedding = self.embedder.embed_queries([query_text])[0]
        query_dense = query_embedding.tolist()

        if cluster_search:
            cluster_results: List[List[Dict[str, Any]]] = []
            for cid in self.CLUSTER_IDS:
                raw = self.milvus_client.hybrid_search(
                    query_dense=query_dense,
                    query_text=query_text,
                    top_k=effective_top_k,
                    alpha=alpha,
                    cluster_filter=f"cluster_id == '{cid}'",
                    cwe_list=cwe_list,
                )
                parsed: List[Dict[str, Any]] = []
                for hits in raw:
                    parsed.extend(self._parse_single_query_hits(hits))
                parsed.sort(key=lambda x: x["score"], reverse=True)
                cluster_results.append(parsed)
            return self._round_robin_merge(cluster_results, effective_top_k)

        raw_results = self.milvus_client.hybrid_search(
            query_dense=query_dense,
            query_text=query_text,
            top_k=effective_top_k,
            alpha=alpha,
            cwe_list=cwe_list,
        )

        parsed = []
        for hits in raw_results:
            parsed.extend(self._parse_single_query_hits(hits))
        deduped = self._dedup_by_doc_id(parsed, effective_top_k)
        return deduped


    def hybrid_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        cluster_search: bool = False,
        query_target: str = "fixed_code",
        cwe_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        对批量查询执行混合检索，返回结构化的结果列表。

        支持直接读取 proxy_query_set.jsonl 格式，保留所有原始字段。

        Args:
            queries: 查询列表，每个元素是包含以下字段的字典:
                - id: 查询唯一标识（用于结果关联）
                - fixed_code / buggy_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每个查询返回的最相似结果数量
            alpha: 稠密向量权重（0.0-1.0）
            cluster_search: 是否启用分簇召回策略，透传至 hybrid_retrieve_single
            query_target: 查询目标字段名，默认 "fixed_code"，可选 "buggy_code"
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件，仅从匹配的文档中检索

        Returns:
            List[Dict]: 结构化结果列表，每个元素包含:
                - id: 对应查询的 ID
                - language: 代码语言（如果输入中有）
                - buggy_code: 漏洞代码（如果输入中有）
                - fixed_code: 修复后的代码（如果输入中有，或来自 clean_code）
                - source: 数据来源（如果输入中有）
                - retrieval_results: 该查询的去重后检索结果列表
        """
        results = []

        for query in tqdm(queries, desc="Hybrid retrieval"):
            query_text = query.get(query_target)

            hits = self.hybrid_retrieve_single(
                query_text=query_text,
                top_k=top_k,
                alpha=alpha,
                cluster_search=cluster_search,
                cwe_list=cwe_list,
            )

            # 构建结果字典，保留所有原始字段
            result_item = {
                "id": query["id"],
                "source": query["source"],
                "buggy_code": query["buggy_code"],
                "fixed_code": query["fixed_code"],
                "retrieval_results": hits,
            }
            if query.get("language", "") != "":
                result_item["language"] = query.get("language")
            results.append(result_item)
        return results

    def dense_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        cluster_search: bool = False,
        embed_batch_size: int = 256,
        cluster_shard_size: int = 256,
        query_target: str = "fixed_code",
        cwe_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量稠密向量检索，通过 Jina 批量编码和 Milvus 原生多向量搜索实现真正的批处理。

        本方法:
        1. 一次性批量编码所有查询文本（embedder.embed_queries 批处理）
        2. 一次性将所有查询向量发送给 Milvus search（利用 data=[v1, v2, ...] 原生批查询）
        3. 按查询索引拆解结果，逐查询去重并截断 Top-K

        对于超大批次，按 embed_batch_size 分片处理以控制单次 RPC 大小与内存占用。

        Args:
            queries: 查询列表，每个元素是包含以下字段的字典:
                - id: 查询唯一标识（用于结果关联）
                - fixed_code / buggy_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每条查询返回的去重后文档数上限
            embed_batch_size: 嵌入的分片大小，默认 256。
                控制单次 embed_queries 调用和非分簇模式下单次 Milvus search RPC 的查询数量。
            cluster_search: 是否启用分簇召回策略。为 True 时，对每个聚类簇独立检索
                后通过轮询交叉合并（Round-Robin Merge）保证结果多样性；
                为 False 时使用原有全库稠密检索。
            cluster_shard_size: 分簇模式下每次向 Milvus 发送的查询向量数，默认 256。
                注意：过小的分片会导致 RPC 调用次数爆炸（n/shard × 簇数），反而触发 Milvus Lite
                的 gRPC keepalive `too_many_pings` 限流并引发 GOAWAY 断链。建议保持 256 以上，
                与 embed_batch_size 对齐。
            query_target: 查询目标字段名，默认 "fixed_code"，可选 "buggy_code"
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件，仅从匹配的文档中检索

        Returns:
            与输入 queries 对齐的结果列表，每项包含:
                id, source, language, buggy_code, fixed_code, retrieval_results
        """
        effective_top_k = top_k or self.milvus_client.top_k
        n = len(queries)

        # 使用 query_target 指定的字段作为查询文本
        query_texts = [q.get(query_target, "") for q in queries]

        # 所有查询向量（分片批量编码后拼接）
        all_vectors: List[List[float]] = []
        for shard_start in tqdm(
            range(0, n, embed_batch_size), desc="Vul dense embedding"
        ):
            shard_end = min(shard_start + embed_batch_size, n)
            shard_embeddings = self.embedder.embed_queries(query_texts[shard_start:shard_end])
            all_vectors.extend([emb.tolist() for emb in shard_embeddings])

        if cluster_search:
            # per_query_cluster_results[i][c_idx] 存放第 i 个查询在第 c_idx 个簇中的结果
            per_query_cluster_results: List[List[List[Dict[str, Any]]]] = [
                [[] for _ in self.CLUSTER_IDS] for _ in range(n)
            ]

            # 外层按 shard 遍历，内层按簇检索：一次 shard 向量服用 len(CLUSTER_IDS) 次，
            # 同时降低每轮进度粒度，避免只能看到 5 步的外层进度假象
            num_shards = -(-n // cluster_shard_size)
            total_steps = num_shards * len(self.CLUSTER_IDS)
            with tqdm(total=total_steps, desc="Vul dense cluster retrieval") as pbar:
                for shard_start in range(0, n, cluster_shard_size):
                    shard_end = min(shard_start + cluster_shard_size, n)
                    shard_vecs = all_vectors[shard_start:shard_end]
                    for c_idx, cid in enumerate(self.CLUSTER_IDS):
                        cfilter = f"cluster_id == '{cid}'"
                        raw_results = self.milvus_client.dense_search(
                            query_dense_list=shard_vecs,
                            top_k=effective_top_k,
                            cluster_filter=cfilter,
                            cwe_list=cwe_list,
                        )
                        for i, hits in enumerate(raw_results):
                            global_idx = shard_start + i
                            parsed = self._parse_single_query_hits(hits)
                            parsed.sort(key=lambda x: x["score"], reverse=True)
                            per_query_cluster_results[global_idx][c_idx] = parsed
                        pbar.update(1)

            all_per_query_hits: List[List[Dict[str, Any]]] = [
                self._round_robin_merge(per_query_cluster_results[i], effective_top_k)
                for i in range(n)
            ]
        else:
            all_per_query_hits = [[] for _ in range(n)]
            for shard_start in tqdm(
                range(0, n, embed_batch_size), desc="Vul dense batch retrieval"
            ):
                shard_end = min(shard_start + embed_batch_size, n)
                shard_vecs = all_vectors[shard_start:shard_end]

                raw_results = self.milvus_client.dense_search(
                    query_dense_list=shard_vecs,
                    top_k=effective_top_k,
                    cwe_list=cwe_list,
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
                "id": query["id"],
                "source": query.get("source", ""),
                "buggy_code": query.get("buggy_code", ""),
                "fixed_code": query.get("fixed_code", ""),
                "retrieval_results": all_per_query_hits[idx],
            }
            if query.get("language", "") != "":
                result_item["language"] = query.get("language")
            results.append(result_item)

        logger.info(
            f"Dense batch retrieval completed: {n} queries, "
            f"top_k={effective_top_k}, cluster_search={cluster_search}, "
            f"shards={-(-n // embed_batch_size)}"
        )
        return results

    def sparse_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        cluster_search: bool = False,
        embed_batch_size: int = 256,
        cluster_shard_size: int = 256,
        query_target: str = "fixed_code",
        cwe_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量稀疏向量（BM25）检索，通过 Milvus 原生多文本批查询实现真正的批处理。

        本方法:
        1. 直接收集所有查询文本（无需嵌入模型，BM25 由 Milvus 服务端完成）
        2. 一次性将所有查询文本发送给 Milvus sparse search（data=[t1, t2, ...] 原生批查询）
        3. 按查询索引拆解结果，逐查询去重并截断 Top-K

        对于超大批次，按 embed_batch_size 分片处理以控制单次 RPC 大小。

        Args:
            queries: 查询列表，每个元素是包含以下字段的字典:
                - id: 查询唯一标识（用于结果关联）
                - fixed_code / buggy_code: 查询字段（由 query_target 指定）
                - 其他字段（language, source 等）原样回传
            top_k: 每条查询返回的去重后文档数上限
            embed_batch_size: 非分簇模式下每次发送给 Milvus 的查询分片大小，默认 256。
                控制单次 Milvus search RPC 的查询数量，防止超过 gRPC 传输限制。
            cluster_search: 是否启用分簇召回策略。为 True 时，对每个聚类簇独立检索
                后通过轮询交叉合并（Round-Robin Merge）保证结果多样性；
                为 False 时使用原有全库 BM25 检索。
            cluster_shard_size: 分簇模式下每次向 Milvus 发送的查询文本数，默认 256。
                注意：过小的分片会导致 RPC 调用次数爆炸（n/shard × 簇数），反而触发 Milvus Lite
                的 gRPC keepalive `too_many_pings` 限流并引发 GOAWAY 断链。建议保持 256 以上，
                与 embed_batch_size 对齐。
            query_target: 查询目标字段名，默认 "fixed_code"，可选 "buggy_code"
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件，仅从匹配的文档中检索

        Returns:
            与输入 queries 对齐的结果列表，每项包含:
                id, source, language, buggy_code, fixed_code, retrieval_results
        """
        effective_top_k = top_k or self.milvus_client.top_k
        n = len(queries)

        # 使用 query_target 指定的字段作为查询文本
        query_texts = [q.get(query_target, "") for q in queries]

        if cluster_search:
            per_query_cluster_results: List[List[List[Dict[str, Any]]]] = [
                [[] for _ in self.CLUSTER_IDS] for _ in range(n)
            ]

            # 外层按 shard 遍历，内层按簇检索：一次 shard 复用 len(CLUSTER_IDS) 次，
            # 并提供更细粒度的进度反馈
            num_shards = -(-n // cluster_shard_size)
            total_steps = num_shards * len(self.CLUSTER_IDS)
            with tqdm(total=total_steps, desc="Vul sparse cluster retrieval") as pbar:
                for shard_start in range(0, n, cluster_shard_size):
                    shard_end = min(shard_start + cluster_shard_size, n)
                    shard_texts = query_texts[shard_start:shard_end]
                    for c_idx, cid in enumerate(self.CLUSTER_IDS):
                        cfilter = f"cluster_id == '{cid}'"
                        raw_results = self.milvus_client.sparse_search(
                            query_text_list=shard_texts,
                            top_k=effective_top_k,
                            cluster_filter=cfilter,
                            cwe_list=cwe_list,
                        )
                        for i, hits in enumerate(raw_results):
                            global_idx = shard_start + i
                            parsed = self._parse_single_query_hits(hits)
                            parsed.sort(key=lambda x: x["score"], reverse=True)
                            per_query_cluster_results[global_idx][c_idx] = parsed
                        pbar.update(1)

            all_per_query_hits: List[List[Dict[str, Any]]] = [
                self._round_robin_merge(per_query_cluster_results[i], effective_top_k)
                for i in range(n)
            ]
        else:
            all_per_query_hits = [[] for _ in range(n)]
            for shard_start in tqdm(
                range(0, n, embed_batch_size), desc="Vul sparse batch retrieval"
            ):
                shard_end = min(shard_start + embed_batch_size, n)
                shard_texts = query_texts[shard_start:shard_end]

                raw_results = self.milvus_client.sparse_search(
                    query_text_list=shard_texts,
                    top_k=effective_top_k,
                    cwe_list=cwe_list,
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
                "id": query["id"],
                "source": query.get("source", ""),
                "buggy_code": query.get("buggy_code", ""),
                "fixed_code": query.get("fixed_code", ""),
                "retrieval_results": all_per_query_hits[idx],
            }
            if query.get("language", "") != "":
                result_item["language"] = query.get("language")
            results.append(result_item)

        logger.info(
            f"Sparse batch retrieval completed: {n} queries, "
            f"top_k={effective_top_k}, cluster_search={cluster_search}, "
            f"shards={-(-n // embed_batch_size)}"
        )
        return results
