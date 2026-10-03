import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional

from tqdm import tqdm

from src.models.retriever.Base import BaseEmbedder
from src.utils.model import truncate_text

from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
)

# 获取模块级别的日志记录器
logger = logging.getLogger(__name__)


class BFPMilvusClient:
    """
    Milvus 向量数据库客户端封装类，用于程序修复（APR）RAG 知识库的混合检索。

    该类实现了稠密向量（Dense Vector）+ 稀疏向量（Sparse Vector，BM25）的混合搜索架构：
    以 buggy_code 作为检索与嵌入文本（与用户查询中的缺陷代码对齐），fixed_code 作为补丁元数据
    随检索结果返回；并提供 build_knowledge_base 完成集合创建、分块嵌入与批量入库。
    支持 Milvus Lite（本地文件）和 Milvus Server（远程服务）两种部署模式。

    Attributes:
        uri: Milvus 连接地址（本地文件路径或 HTTP 服务地址）
        collection_name: 集合名称
        dim: 稠密向量维度（自动从 model.retriever_config 读取，回退默认 896）
        dense_metric_type: 稠密向量距离度量类型（默认 COSINE）
        dense_index_type: 稠密向量索引类型（默认 HNSW，亦可配置 AUTOINDEX）
        dense_index_params: 稠密索引构建参数（如 HNSW 的 M 与 efConstruction）
        top_k: 默认返回的最相似结果数量
        alpha: 混合搜索中稠密向量的权重（0.0-1.0），1-alpha 为 BM25 权重
        client: pymilvus.MilvusClient 实例
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Milvus 客户端。

        dim 的解析优先级：
          1. database.dim（显式指定，兼容旧配置）
          2. model.retriever_config 指向的模型配置文件中的 dim
          3. 回退默认值 896

        Args:
            config: 配置字典，包含以下键:
                - database.uri (str): 必需，连接地址
                - database.collection_name (str): 必需，集合名称
                - model.retriever_config (str): 可选，模型配置文件路径（用于读取 dim）
                - database.dim (int): 可选，显式指定维度（优先于模型配置）

        Note:
            - 当 uri 不以 "http" 开头时，视为本地文件路径，会自动创建父目录
            - 使用本地文件时，Milvus Lite 会自动启用，所有数据存储在单个文件中
            - 对于大规模数据，建议使用 Milvus Server（Docker/K8s 部署）
        """
        database_config = config.get("database", {})
        retrieval_config = config.get("retrieval", {})
        
        # 从配置中提取参数
        self.uri = database_config.get("uri", "")
        self.collection_name = database_config.get("collection_name", "")
        self.dim = self._resolve_dim(database_config)
        
        self.dense_metric_type = retrieval_config.get("dense_metric_type", "COSINE")
        self.dense_index_type = retrieval_config.get("dense_index_type", "HNSW")
        self.dense_index_params = retrieval_config.get(
            "dense_index_params", {"M": 16, "efConstruction": 256}
        )
        self.top_k = retrieval_config.get("top_k", 10)
        self.alpha = retrieval_config.get("alpha_weight", 0.8)

        # 如果是本地文件路径，确保父目录存在
        if not self.uri.startswith("http"):
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)

        # 初始化 Milvus 客户端
        logger.info(f"Connecting to Milvus at {self.uri}")
        self.client = MilvusClient(uri=self.uri)
        logger.info("Milvus client connected")
        

    @staticmethod
    def _resolve_dim(database_config: Dict[str, Any]) -> int:
        """从 database.dim 或 model.retriever_config 解析稠密向量维度。

        路径解析顺序（对相对路径依次尝试）：
          1. 原始路径（已是绝对路径或当前目录下存在时直接使用）
          2. sys.path 中含 configs/ 子目录的项目根 + 相对路径
          3. 当前工作目录 + 相对路径
        """
        import sys
        import yaml
        
        retriever_cfg_path = database_config.get("model", "")
        if retriever_cfg_path:
            cfg = Path(retriever_cfg_path)
            candidates = [cfg]
            if not cfg.is_absolute():
                for p in sys.path:
                    root = Path(p)
                    if (root / "configs").exists():
                        candidates.append(root / cfg)
                        break
                candidates.append(Path.cwd() / cfg)

            for candidate in candidates:
                if candidate.exists():
                    try:
                        model_cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                        dim = model_cfg.get("dim")
                        if dim is not None:
                            logger.info(f"从模型配置 {candidate} 读取 dim={dim}")
                            return int(dim)
                    except Exception as e:
                        logger.warning(f"读取模型配置失败 ({candidate}): {e}")

            logger.warning(f"模型配置文件未找到: {retriever_cfg_path}")

        logger.warning("未能从配置中解析 dim")
        return -1

    def create_collection(self, drop_existing: bool = True) -> None:
        """
        创建 Milvus 集合（数据表）。

        该方法构建包含稠密向量、稀疏向量和元数据字段的 Schema，
        并创建相应的向量索引，支持高效检索。

        Args:
            drop_existing: 如果为 True，会先删除同名集合（如果有）

        Note:
            - 集合创建后会自动加载，可以直接进行插入和搜索操作
            - Schema 中的 enable_analyzer=True 启用文本分析器，支持 BM25 分词
            - BM25 函数会在插入时自动将 buggy_code 文本转换为稀疏向量

        Example:
            >>> client.create_collection(drop_existing=True)  # 重新创建
            >>> client.create_collection(drop_existing=False)  # 仅当不存在时创建
        """
        if drop_existing and self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)

        schema = self._build_schema()
        index_params = self._build_index_params()

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.collection_name}' created (dim={self.dim})")

    def _build_schema(self) -> CollectionSchema:
        """
        构建 Milvus 集合的 Schema（数据结构定义）。

        定义了 APR RAG 知识库所需的全部字段，包括:
        - 主键字段（id，自动生成）
        - 分块字段（doc_id, chunk_id, total_chunks）
        - 业务字段（buggy_code, fixed_code, is_poisoned, source）
        - 向量字段（dense_vector, sparse_vector）

        同时定义了 BM25 函数，用于自动将 buggy_code 文本转换为稀疏向量。

        Returns:
            CollectionSchema: Milvus 集合 Schema 对象

        Schema 字段详解:
            - id: INT64 类型主键，auto_id 由 Milvus 自动生成
            - doc_id: INT64，原始数据中的文档 ID（同一文档的多个 chunk 共享此 ID）
            - chunk_id: INT32，块序号（未分块时为 0，分块后从 0 递增）
            - total_chunks: INT32，该文档的总块数（未分块时为 1）
            - buggy_code: VARCHAR(65535)，缺陷代码文本（分块时为 chunk 文本），启用分析器用于分词
            - fixed_code: VARCHAR(65535)，对应补丁/修复后代码（整段重复写入各 chunk 行）
            - is_poisoned: BOOL，是否为投毒文档，默认 False
            - source: VARCHAR(32)，数据来源（语义枚举，如 CoCoNut、Codeflaws、DeepFix）
            - dense_vector: FLOAT_VECTOR(dim)，由嵌入模型对 buggy_code（chunk）编码得到
            - sparse_vector: SPARSE_FLOAT_VECTOR，由 BM25 对 buggy_code 生成

        BM25 函数:
            输入: buggy_code（缺陷代码文本或 chunk 文本）
            输出: sparse_vector（自动计算的 BM25 权重向量）
        """
        fields = [
            # 主键字段：Milvus 自动生成，支持分块后同一 doc_id 对应多行
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            # 原始文档 ID：来自 rag_data 等的唯一标识，分块后多行共享同一 doc_id
            FieldSchema(name="doc_id", dtype=DataType.INT64),
            # 块序号：未分块为 0，分块后从 0 递增
            FieldSchema(name="chunk_id", dtype=DataType.INT32),
            # 该文档的总块数：未分块为 1
            FieldSchema(name="total_chunks", dtype=DataType.INT32),
            # 缺陷代码（分块时为 chunk 文本），启用分析器以支持 BM25 分词
            FieldSchema(
                name="buggy_code",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            ),
            # 修复后代码（补丁），与检索字段 buggy_code 对应；分块时每行存整段 fixed_code
            FieldSchema(
                name="fixed_code",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            # 投毒标记：用于对抗实验场景
            FieldSchema(name="is_poisoned", dtype=DataType.BOOL),
            # 数据集来源
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
            # 稠密向量：语义嵌入，由 Jina 等对 buggy_code 编码生成
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            # 稀疏向量：由 BM25 函数从 buggy_code 自动生成
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        # BM25 函数：将 buggy_code 自动转换为稀疏向量
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=["buggy_code"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )

        return CollectionSchema(fields=fields, functions=[bm25_function])

    def _build_index_params(self):
        """
        构建向量索引参数。

        为稠密向量和稀疏向量分别创建索引，以支持高效的近似最近邻（ANN）搜索。

        Returns:
            IndexParams: 索引参数对象，包含两种向量的索引配置

        索引策略详解:
            稠密向量索引:
                - 默认 HNSW 或配置为 AUTOINDEX（由 dense_index_type 决定）
                - 距离度量: COSINE（余弦相似度），适合语义搜索

            稀疏向量索引（SPARSE_INVERTED_INDEX）:
                - 类型: 倒排索引，专为稀疏向量设计
                - 距离度量: BM25

        性能调优建议:
            - 高精度: 可增大 M、efConstruction
            - 高速度: 可减小 M，查询时使用较小 ef
        """
        index_params = self.client.prepare_index_params()

        dense_params = {
            "field_name": "dense_vector",
            "index_type": self.dense_index_type,
            "metric_type": self.dense_metric_type,
        }
        if self.dense_index_params:
            dense_params["params"] = self.dense_index_params
        index_params.add_index(**dense_params)

        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        return index_params
    
    def build_knowledge_base(
        self,
        embedder: BaseEmbedder,
        kb_data: List[Dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        """
        构建 APR RAG 知识库：仅对 buggy_code 分块、生成嵌入并批量插入 Milvus。

        对于超出嵌入模型 token 上限的 buggy_code，先通过 embedder.chunk_text() 分块，
        再为每个 chunk 独立生成嵌入。同一文档的所有 chunk 共享 doc_id；
        每条记录完整 fixed_code、is_poisoned、source 会写入每一 chunk 行（fixed 不按块切分）。

        Args:
            embedder: 用于 chunk_text / embed_documents 的嵌入模型（须与 self.dim 一致）
            kb_data: 知识库数据列表，每个元素典型字段:
                - id: 唯一标识符（必需）
                - buggy_code: 缺陷代码（必需）
                - fixed_code: 修复后代码（必需）
                - source: 可选，数据集名
                - is_poisoned: 可选，默认 False
            batch_size: 批处理大小（按原始文档数计），默认 5000。

        处理流程:
            1. 创建/重置 Milvus 集合（drop_existing=True）
            2. 按 batch_size 分批处理原始数据
            3. 对每条数据的 buggy_code 调用 chunk_text 分块
            4. 收集所有 chunk 文本，批量生成嵌入
            5. 组装 Milvus 文档并 truncate 长文本字段
            6. 批量插入（BM25 稀疏向量由 Milvus 自动计算）
        """
        self.create_collection(drop_existing=True)

        total = len(kb_data)
        chunked_count = 0
        total_chunks_inserted = 0
        logger.info(f"Building APR knowledge base with {total} entries (batch_size={batch_size})")

        for start in tqdm(range(0, total, batch_size), desc="Indexing APR KB"):
            batch = kb_data[start : start + batch_size]

            # 阶段1：按 buggy_code 分块，收集 chunk 与元数据
            chunk_records: List[Dict[str, Any]] = []
            for item in batch:
                chunks = embedder.chunk_text(item["buggy_code"])
                if chunks[0]["total_chunks"] > 1:
                    chunked_count += 1
                for chunk in chunks:
                    chunk_records.append(
                        {
                            "doc_id": int(item["id"]),
                            "chunk_id": chunk["chunk_id"],
                            "total_chunks": chunk["total_chunks"],
                            "text": chunk["text"],
                            "fixed_code": item.get("fixed_code") or "",
                            "is_poisoned": bool(item.get("is_poisoned", False)),
                            "source": item.get("source", "") or "",
                        }
                    )

            # 阶段2：批量生成嵌入
            texts = [r["text"] for r in chunk_records]
            embeddings = embedder.embed_documents(texts)

            # 阶段3：构建 Milvus 文档并插入
            docs = []
            for j, record in enumerate(chunk_records):
                docs.append(
                    {
                        "doc_id": record["doc_id"],
                        "chunk_id": record["chunk_id"],
                        "total_chunks": record["total_chunks"],
                        "buggy_code": truncate_text(record["text"]),
                        "fixed_code": truncate_text(record["fixed_code"]),
                        "is_poisoned": record["is_poisoned"],
                        "source": record["source"],
                        "dense_vector": embeddings[j].tolist(),
                    }
                )
            self.insert_documents(docs)
            total_chunks_inserted += len(docs)

        logger.info(
            f"APR knowledge base built: {total} docs → {total_chunks_inserted} rows "
            f"({chunked_count} docs were chunked)"
        )

    def insert_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        插入 APR 文档到 Milvus 集合。

        稠密向量用于语义搜索；buggy_code 会由 BM25 函数自动转换为稀疏向量。

        Args:
            docs: 文档列表，每个文档是包含以下字段的字典:
                - doc_id (int): 原始数据中的文档 ID（必需）
                - chunk_id (int): 块序号，未分块时为 0（必需）
                - total_chunks (int): 该文档的总块数，未分块时为 1（必需）
                - buggy_code (str): 缺陷代码或 chunk 文本（必需）
                - fixed_code (str): 修复后代码（必需）
                - is_poisoned (bool): 是否投毒（必需）
                - source (str): 数据来源（必需，可为空字符串）
                - dense_vector (List[float]): 稠密向量（必需）
                - id: 由 Milvus auto_id 自动生成，无需提供
                - sparse_vector: 由 BM25 函数自动生成，无需提供

        Returns:
            int: 成功插入的文档数量
        """
        res = self.client.insert(
            collection_name=self.collection_name,
            data=docs,
        )
        count = res["insert_count"]
        logger.info(f"Inserted {count} documents into '{self.collection_name}'")
        return count
    
    
    def clear_poisoned_docs(self) -> int:
        """
        清除 Milvus 集合中所有 is_poisoned 为 True 的文档（投毒数据）。

        利用 Milvus delete 的 filter 表达式功能，按布尔字段 is_poisoned 过滤并批量删除。
        删除操作在服务端执行，无需将数据拉取到客户端。

        Returns:
            int: 被删除的文档（行）数量

        Note:
            - 删除的是 Milvus 中的行（row），若原始文档被分块，则同一 doc_id 的
            所有 chunk 行只要 is_poisoned == True 都会被删除
            - 删除后数据不会立即从磁盘清除，Milvus 会在后台 compaction 时物理回收空间
            - 删除操作要求集合已加载（loaded），否则 Milvus 会报错

        Example:
            >>> deleted = client.clear_poisoned_docs()
            >>> print(f"已清除 {deleted} 条投毒数据")
        """
        if not self.client.has_collection(self.collection_name):
            logger.warning(
                f"Collection '{self.collection_name}' does not exist, skip clearing"
            )
            return 0

        # 先通过 query 统计待删除数量，用于日志记录
        poisoned = self.client.query(
            collection_name=self.collection_name,
            filter="is_poisoned == true",
            output_fields=["id"],
        )
        count = len(poisoned)

        if count == 0:
            logger.info(
                f"No poisoned documents found in '{self.collection_name}'"
            )
            return 0

        # 按 filter 表达式批量删除
        self.client.delete(
            collection_name=self.collection_name,
            filter="is_poisoned == true",
        )

        logger.info(
            f"Cleared {count} poisoned document(s) from '{self.collection_name}'"
        )
        return count

    def iter_dense_vectors(
        self,
        filter_expr: str = "is_poisoned == false",
        batch_size: int = 1000,
    ) -> Iterator[List[Dict[str, Any]]]:
        """分批遍历集合中的稠密向量及其文档标识。

        该接口面向需要扫描完整向量集合的通用离线审计任务，使用
        ``query_iterator`` 避免普通 ``query`` 的结果条数限制。每次迭代
        返回一个批次，调用方无需同时持有全部 Milvus 记录。

        Args:
            filter_expr: Milvus 标量过滤表达式。默认仅返回干净文档。
            batch_size: 每次从 Milvus 读取的最大行数，必须为正整数。

        Yields:
            包含 ``doc_id``、分块信息、``is_poisoned``、``source`` 和
            ``dense_vector`` 的字典列表。

        Note:
            无论遍历正常完成还是调用方/服务端抛出异常，底层 iterator 都会
            在 ``finally`` 中关闭。
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {batch_size!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=batch_size,
            filter=filter_expr,
            output_fields=[
                "doc_id",
                "chunk_id",
                "total_chunks",
                "is_poisoned",
                "source",
                "dense_vector",
            ],
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break

                normalized: List[Dict[str, Any]] = []
                for row in batch:
                    if isinstance(row, dict):
                        normalized.append(dict(row))
                    elif hasattr(row, "to_dict"):
                        normalized.append(dict(row.to_dict()))
                    else:
                        normalized.append(dict(row))
                yield normalized
        finally:
            iterator.close()
    

    def hybrid_search(
        self,
        query_dense: List[float],
        query_text: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Any:
        """
        执行混合搜索（稠密向量 + BM25 稀疏向量）。

        查询向量与查询文本通常均来自用户缺陷代码（与 buggy_code 域一致）。
        由于同一文档可能被分为多个 chunk 存储，搜索时内部 limit 会放大为
        top_k * 2，以确保按 doc_id 去重后仍有足够结果。去重由业务层完成。

        Args:
            query_dense: 查询的稠密向量（由嵌入模型对查询代码编码得到）
            query_text: 查询的原始文本（用于 BM25）
            top_k: 返回的最相似结果数量，默认使用初始化时的配置
            alpha: 稠密向量搜索的权重（0.0-1.0），1-alpha 为 BM25 权重

        Returns:
            Any: Milvus 返回的原始混合搜索结果（嵌套 hits 结构）。
        """
        top_k = top_k or self.top_k
        alpha = alpha if alpha is not None else self.alpha

        internal_limit = top_k * 2

        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={
                "metric_type": self.dense_metric_type,
                "params": {"ef": max(internal_limit, 64)},
            },
            limit=internal_limit,
        )

        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=internal_limit,
        )

        ranker = WeightedRanker(alpha, 1.0 - alpha)

        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=internal_limit,
            output_fields=[
                "doc_id",
                "chunk_id",
                "total_chunks",
                "buggy_code",
                "fixed_code",
                "is_poisoned",
                "source",
            ],
        )
    
    def dense_search(
        self,
        query_dense_list: List[List[float]],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> Any:
        """
        批量执行稠密向量搜索（利用 Milvus search 原生多向量批查询能力）

        Milvus client.search() 的 data 参数天然支持传入多个查询向量，
        服务端会并行执行 ANN 搜索，一次 RPC 返回所有查询的结果。
        返回结构为嵌套列表：外层长度 = nq（查询数），内层为每个查询的 hits。

        Args:
            query_dense_list: 查询稠密向量列表，每个元素为一个 float 列表，
                长度必须等于 self.dim。向量可由 BFP 的 buggy_code 或漏洞语料的 vul_code 编码生成。
            top_k: 每个查询返回结果数量上限（经 internal_limit 放大后由业务去重截断）
            filter_expr: 可选的 Milvus 标量过滤表达式。
            output_fields: 可选的返回字段列表。未指定时保持原有返回字段。

        Returns:
            Any: Milvus search 返回的原始嵌套结果，外层长度等于 len(query_dense_list)，
                每个元素对应一个查询的 hits 列表。
        """
        top_k = top_k or self.top_k
        internal_limit = top_k * 2

        search_params = {
            "metric_type": self.dense_metric_type,
            "params": {"ef": max(internal_limit, 64)},
        }
        
        effective_output_fields = output_fields or [
            "doc_id",
            "chunk_id",
            "total_chunks",
            "buggy_code",
            "fixed_code",
            "is_poisoned",
            "source",
        ]
        search_kwargs = {
            "collection_name": self.collection_name,
            "data": query_dense_list,
            "anns_field": "dense_vector",
            "search_params": search_params,
            "limit": internal_limit,
            "output_fields": effective_output_fields,
        }
        if filter_expr is not None:
            search_kwargs["filter"] = filter_expr

        results = self.client.search(
            **search_kwargs,
        )

        return results

    def sparse_search(
        self,
        query_text_list: List[str],
        top_k: Optional[int] = None,
    ) -> Any:
        """
        批量执行稀疏向量（BM25）搜索（利用 Milvus search 原生多文本批查询能力）。

        Milvus client.search() 的 data 参数支持传入多条原始文本字符串，
        服务端内置 BM25 函数自动将每条文本转换为稀疏向量后执行检索，
        一次 RPC 返回所有查询的结果。
        返回结构为嵌套列表：外层长度 = nq（查询数），内层为每个查询的 hits。

        Args:
            query_text_list: 查询文本列表，每个元素为一个原始代码/文本字符串，
                可来自 BFP 的 buggy_code 或漏洞语料的 vul_code。
            top_k: 每个查询返回结果数量上限（经 internal_limit 放大后由业务去重截断）

        Returns:
            Any: Milvus search 返回的原始嵌套结果，外层长度等于 len(query_text_list)，
                每个元素对应一个查询的 hits 列表。
        """
        top_k = top_k or self.top_k
        internal_limit = top_k * 2

        results = self.client.search(
            collection_name=self.collection_name,
            data=query_text_list,  # 原始文本列表，BM25 向量化由 Milvus 服务端完成
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25"},
            limit=internal_limit,
            output_fields=[
                "doc_id",
                "chunk_id",
                "total_chunks",
                "buggy_code",
                "fixed_code",
                "is_poisoned",
                "source",
            ],
        )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Returns:
            Dict: 如 row_count、data_size、index_size 等（具体字段依 Milvus 版本而定）
        """
        return self.client.get_collection_stats(self.collection_name)

    def close(self) -> None:
        """
        关闭 Milvus 客户端连接。

        Note:
            - 建议在程序结束时调用；关闭后需重新构造客户端实例方可继续使用
        """
        self.client.close()
        logger.info("Milvus client closed")
