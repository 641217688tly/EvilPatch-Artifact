import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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


class VulMilvusClient:
    """
    Milvus 向量数据库客户端封装类，用于漏洞代码的混合检索。

    该类实现了稠密向量（Dense Vector）+ 稀疏向量（Sparse Vector，BM25）的混合搜索架构，
    用于高效检索漏洞代码片段；并提供 build_knowledge_base 完成集合创建、分块嵌入与批量入库。
    支持 Milvus Lite（本地文件）和 Milvus Server（远程服务）两种部署模式。
    
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
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Milvus 客户端。

        Args:
            config: 配置字典，包含以下键:
                - uri (str): 必需，连接地址（如 "./milvus.db" 或 "http://localhost:19530"）
                - collection_name (str): 必需，集合名称
                - dim (int): 可选，稠密向量维度，默认 896
                - dense_metric_type (str): 可选，默认 "COSINE"
                - dense_index_type (str): 可选，默认 "HNSW"
                - dense_index_params (dict): 可选，HNSW 参数，默认 {"M": 16, "efConstruction": 256}
                - top_k (int): 可选，默认返回数量，默认 10
                - alpha_weight (float): 可选，稠密向量权重，默认 0.7

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
            - BM25 函数会在插入时自动将 vul_code 文本转换为稀疏向量

        Example:
            >>> client.create_collection(drop_existing=True)  # 重新创建
            >>> client.create_collection(drop_existing=False)  # 仅当不存在时创建
        """
        # 如果drop_existing为True，需要先删除已存在的同名集合
        if drop_existing and self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)

        # 构建 Schema 和索引参数
        schema = self._build_schema()
        index_params = self._build_index_params()

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.collection_name}' created (dim={self.dim})")

    def _build_schema(self) -> CollectionSchema:
        """
        构建 Milvus 集合的 Schema（数据结构定义）。

        定义了漏洞代码检索所需的全部字段，包括:
        - 主键字段（id，自动生成）
        - 分块字段（doc_id, chunk_id, total_chunks）
        - 元数据字段（vul_code, cwe_id, cve_id, cluster_id, source, language,
                      cwe_desc, cve_desc, fixed_code, diff）
        - 向量字段（dense_vector, sparse_vector）

        同时定义了 BM25 函数，用于自动将文本转换为稀疏向量。

        Returns:
            CollectionSchema: Milvus 集合 Schema 对象

        Schema 字段详解:
            - id: INT64 类型主键，auto_id 由 Milvus 自动生成
            - doc_id: INT64，原始数据中的文档 ID（同一文档的多个 chunk 共享此 ID）
            - chunk_id: INT32，块序号（未分块时为 0，分块后从 0 递增）
            - total_chunks: INT32，该文档的总块数（未分块时为 1）
            - vul_code: VARCHAR(65535)，漏洞代码文本（分块时为 chunk 文本），启用分析器用于分词
            - cwe_id: VARCHAR(32)，CWE 漏洞分类标识（如 CWE-120）
            - cve_id: VARCHAR(64)，CVE 漏洞编号（如 CVE-2021-12345）
            - cluster_id: VARCHAR(16)，聚类簇标识，用于分层检索
            - source: VARCHAR(32)，数据来源标识（如 BigVul, PrimeVul, CVEfixes）
            - language: VARCHAR(16)，编程语言标识（如 C, C++）
            - cwe_desc: VARCHAR(65535)，CWE 漏洞类型描述
            - cve_desc: VARCHAR(65535)，CVE 漏洞详细描述
            - fixed_code: VARCHAR(65535)，漏洞修复后的代码（与 vul_code 配对，不参与检索）
            - diff: VARCHAR(65535)，vul_code 与 fixed_code 的 unified diff 文本（不参与检索）
            - dense_vector: FLOAT_VECTOR(dim)，稠密语义向量，由嵌入模型生成
            - sparse_vector: SPARSE_FLOAT_VECTOR，稀疏关键词向量，由 BM25 生成

        BM25 函数:
            输入: vul_code（原始代码文本或 chunk 文本）
            输出: sparse_vector（自动计算的 BM25 权重向量）
            算法: 基于词频-逆文档频率（TF-IDF）的经典信息检索算法
        """
        fields = [
            # 主键字段：Milvus 自动生成，支持分块后同一 doc_id 对应多行
            FieldSchema(
                name="id", dtype=DataType.INT64,
                is_primary=True, auto_id=True,
            ),

            # 原始文档 ID：来自知识库数据的唯一标识，分块后多行共享同一 doc_id
            FieldSchema(name="doc_id", dtype=DataType.INT64),

            # 块序号：未分块为 0，分块后从 0 递增
            FieldSchema(name="chunk_id", dtype=DataType.INT32),

            # 该文档的总块数：未分块为 1
            FieldSchema(name="total_chunks", dtype=DataType.INT32),

            # 漏洞代码文本（分块时为 chunk 文本），启用分析器以支持 BM25 分词
            FieldSchema(
                name="vul_code",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            ),

            # CWE 标识符：漏洞类型分类（如 CWE-79 XSS, CWE-120 缓冲区溢出）
            FieldSchema(name="cwe_id", dtype=DataType.VARCHAR, max_length=32),

            # CVE 编号：漏洞唯一标识（如 CVE-2021-44228 Log4Shell）
            FieldSchema(name="cve_id", dtype=DataType.VARCHAR, max_length=64),

            # 聚类簇 ID：用于代码分层聚类检索
            FieldSchema(name="cluster_id", dtype=DataType.VARCHAR, max_length=16),

            # 数据来源标识：如 BigVul, PrimeVul
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),

            # CWE 漏洞类型描述：结构化 CWE 上下文信息
            FieldSchema(name="cwe_desc", dtype=DataType.VARCHAR, max_length=65535),

            # CVE 漏洞详细描述：结构化 CVE 上下文信息
            FieldSchema(name="cve_desc", dtype=DataType.VARCHAR, max_length=65535),

            # 编程语言：如 C、C++
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),

            # 修复后代码：与 vul_code 配对，仅作元数据存储，不参与 BM25 检索
            FieldSchema(name="fixed_code", dtype=DataType.VARCHAR, max_length=65535),

            # unified diff：vul_code 与 fixed_code 的差异文本，仅作元数据存储
            FieldSchema(name="diff", dtype=DataType.VARCHAR, max_length=65535),

            # 稠密向量：语义嵌入向量，由 Jina/CodeBERT 等模型生成
            FieldSchema(
                name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim
            ),

            # 稀疏向量：由 BM25 函数从 vul_code 自动生成
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        # BM25 函数：将 vul_code（或 chunk 文本）自动转换为稀疏向量
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=["vul_code"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )

        schema = CollectionSchema(fields=fields, functions=[bm25_function])
        return schema

    def _build_index_params(self):
        """
        构建向量索引参数。

        为稠密向量和稀疏向量分别创建索引，以支持高效的近似最近邻（ANN）搜索。

        Returns:
            IndexParams: 索引参数对象，包含两种向量的索引配置

        索引策略详解:
            稠密向量索引（HNSW）:
                - 类型: HNSW（Hierarchical Navigable Small World）
                - 特点: 图结构索引，查询速度快，召回率高
                - 参数:
                    - M: 每个节点的最大连接数（默认 16），越大图越稠密，查询越慢但召回越高
                    - efConstruction: 构建时的搜索深度（默认 256），越大索引质量越高
                - 距离度量: COSINE（余弦相似度），适合语义搜索

            稀疏向量索引（SPARSE_INVERTED_INDEX）:
                - 类型: 倒排索引，专为稀疏向量设计
                - 距离度量: BM25，基于概率的信息检索算法
                - 特点: 类似 Elasticsearch/Solr 的关键词搜索

        性能调优建议:
            - 对于高精度要求: 增大 M 到 32，efConstruction 到 512
            - 对于高速度要求: 减小 M 到 8，查询时使用较小的 ef
            - 对于高召回要求: 增大 ef（查询参数），可超过 top_k 数倍
        """
        # 准备索引参数对象
        index_params = self.client.prepare_index_params()

        # 配置稠密向量索引（HNSW 图索引）
        dense_params = {
            "field_name": "dense_vector",
            "index_type": self.dense_index_type,  # 默认 HNSW
            "metric_type": self.dense_metric_type,  # 默认 COSINE
        }
        if self.dense_index_params:
            # 将ANN的参数添加到索引参数中
            dense_params["params"] = self.dense_index_params
        index_params.add_index(**dense_params)

        # 配置稀疏向量索引（倒排索引 + BM25）
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",  # 专为稀疏向量优化的索引
            metric_type="BM25",  # BM25 排序算法
        )

        """
        index_params = {
            "indexes": [
                # 第一次 add_index 追加
                {"field_name": "dense_vector", "index_type": "HNSW", ...},
                # 第二次 add_index 追加
                {"field_name": "sparse_vector", "index_type": "SPARSE_INVERTED_INDEX", ...},
            ]
        }
        """
        return index_params

    def insert_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        插入漏洞代码文档到 Milvus 集合。

        文档数据会被自动索引，稠密向量用于语义搜索，
        vul_code 文本会被 BM25 函数自动转换为稀疏向量。

        Args:
            docs: 文档列表，每个文档是包含以下字段的字典:
                - doc_id (int): 原始数据中的文档 ID（必需）
                - chunk_id (int): 块序号，未分块时为 0（必需）
                - total_chunks (int): 该文档的总块数，未分块时为 1（必需）
                - vul_code (str): 漏洞代码文本或 chunk 文本（必需）
                - cwe_id (str): CWE 标识符（可选）
                - cve_id (str): CVE 编号（可选）
                - cluster_id (str): 聚类簇 ID（可选）
                - source (str): 数据来源标识（可选）
                - cwe_desc (str): CWE 漏洞类型描述（可选）
                - cve_desc (str): CVE 漏洞详细描述（可选）
                - dense_vector (List[float]): 稠密向量（必需）
                - id: 由 Milvus auto_id 自动生成，无需提供
                - sparse_vector: 由 BM25 函数自动生成，无需提供

        Returns:
            int: 成功插入的文档数量
        """
        # 执行插入操作
        res = self.client.insert(
            collection_name=self.collection_name,
            data=docs,
        )
        # 获取插入数量并记录日志
        count = res["insert_count"]
        logger.info(f"Inserted {count} documents into '{self.collection_name}'")
        return count

    def build_knowledge_base(
        self,
        embedder: BaseEmbedder,
        kb_data: List[Dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        """
        构建漏洞代码知识库：分块、生成嵌入并批量插入 Milvus。

        对于超出嵌入模型 token 上限的 vul_code，先通过 embedder.chunk_text() 分块，
        再为每个 chunk 独立生成嵌入。同一文档的所有 chunk 共享 doc_id。

        Args:
            embedder: 用于 chunk_text / embed_documents 的嵌入模型（须与 self.dim 一致）
            kb_data: 知识库数据列表，每个元素是包含漏洞信息的字典，必须包含:
                - id: 唯一标识符
                - vul_code: 漏洞代码文本（原始代码片段）
                - cwe_id: 可选，CWE 漏洞类型标识
                - cve_id: 可选，CVE 漏洞编号
                - cluster_id: 可选，聚类簇 ID
            batch_size: 批处理大小（按原始文档数计），默认 5000。

        处理流程:
            1. 创建/重置 Milvus 集合（drop_existing=True）
            2. 按 batch_size 分批处理原始数据
            3. 对每条数据调用 chunk_text 进行分块（大部分不需要分块）
            4. 收集所有 chunk 文本，批量生成嵌入
            5. 将嵌入关联回对应 chunk，构建 Milvus 文档（含 doc_id, chunk_id, total_chunks）
            6. 批量插入 Milvus（BM25 稀疏向量自动计算）
        """
        self.create_collection(drop_existing=True)

        total = len(kb_data)
        chunked_count = 0
        total_chunks_inserted = 0
        logger.info(f"Building knowledge base with {total} entries (batch_size={batch_size})")

        for start in tqdm(range(0, total, batch_size), desc="Indexing KB"):
            batch = kb_data[start : start + batch_size]

            chunk_records: List[Dict[str, Any]] = []
            for item in batch:
                chunks = embedder.chunk_text(item["vul_code"])
                if chunks[0]["total_chunks"] > 1:
                    chunked_count += 1
                for chunk in chunks:
                    chunk_records.append(
                        {
                            "doc_id": int(item["id"]),
                            "chunk_id": chunk["chunk_id"],
                            "cluster_id": str(item.get("cluster_id", "")),
                            "source": item.get("source", ""),
                            "language": item.get("language", "") or "",
                            "total_chunks": chunk["total_chunks"],
                            "text": chunk["text"],
                            "cwe_id": item.get("cwe_id", ""),
                            "cve_id": item.get("cve_id") or "",
                            "cwe_desc": item.get("cwe_desc") or "",
                            "cve_desc": item.get("cve_desc") or "",
                            "fixed_code": item.get("fixed_code", "") or "",
                            "diff": item.get("diff", "") or "",
                        }
                    )

            texts = [r["text"] for r in chunk_records]
            embeddings = embedder.embed_documents(texts)

            docs = []
            for j, record in enumerate(chunk_records):
                docs.append(
                    {
                        "doc_id": record["doc_id"],
                        "chunk_id": record["chunk_id"],
                        "cluster_id": record["cluster_id"],
                        "source": record["source"],
                        "language": record["language"],
                        "total_chunks": record["total_chunks"],
                        "vul_code": truncate_text(record["text"]),
                        "cwe_id": record["cwe_id"],
                        "cve_id": record["cve_id"],
                        "cwe_desc": truncate_text(record["cwe_desc"]),
                        "cve_desc": truncate_text(record["cve_desc"]),
                        "fixed_code": truncate_text(record["fixed_code"]),
                        "diff": truncate_text(record["diff"]),
                        "dense_vector": embeddings[j].tolist(),
                    }
                )
            self.insert_documents(docs)
            total_chunks_inserted += len(docs)

        logger.info(
            f"Knowledge base built: {total} docs → {total_chunks_inserted} rows "
            f"({chunked_count} docs were chunked)"
        )

    @staticmethod
    def _build_filter_expr(
        cluster_filter: Optional[str] = None,
        cwe_list: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        将 cluster_filter 和 cwe_list 两个可选过滤条件组合为单条 Milvus 过滤表达式。

        Args:
            cluster_filter: 聚类簇过滤表达式（如 "cluster_id == '2'"）
            cwe_list: CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                每项经格式校验（^CWE-\\d+$）后生效：
                - 单项：生成 ``cwe_id == 'CWE-XX'``（等值查询，利用索引优势）
                - 多项：生成 ``cwe_id in ['CWE-XX', 'CWE-YY']``（Milvus 原生 in 运算符）

        Returns:
            组合后的过滤表达式字符串，无有效条件时返回 None
        """
        parts: List[str] = []
        if cluster_filter:
            parts.append(cluster_filter)
        if cwe_list:
            valid = [c for c in cwe_list if re.match(r"^CWE-\d+$", c)]
            if len(valid) == 1:
                parts.append(f"cwe_id == '{valid[0]}'")
            elif len(valid) > 1:
                quoted = ", ".join(f"'{c}'" for c in valid)
                parts.append(f"cwe_id in [{quoted}]")
        return " and ".join(parts) if parts else None

    def hybrid_search(
        self,
        query_dense: List[float],
        query_text: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        cluster_filter: Optional[str] = None,
        cwe_list: Optional[List[str]] = None,
    ) -> Any:
        """
        执行混合搜索（稠密向量 + BM25 稀疏向量）。

        由于同一文档可能被分为多个 chunk 存储，搜索时内部 limit 会放大为
        top_k * 2，以确保按 doc_id 去重后仍有足够结果。去重由业务层完成。

        Args:
            query_dense: 查询的稠密向量（由嵌入模型生成）
            query_text: 查询的原始文本（用于 BM25 关键词搜索）
            top_k: 返回的最相似结果数量，默认使用初始化时的配置
            alpha: 稠密向量搜索的权重（0.0-1.0），1-alpha 为 BM25 权重，默认 0.85
            cluster_filter: 可选的 Milvus 过滤表达式（如 "cluster_id == '2'"），
                用于分簇检索时将搜索范围限制在指定簇内
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件（单项等值、多项 in 表达式）

        Returns:
            Any: Milvus 返回的原始混合搜索结果（嵌套 hits 结构）。
            结果解析和 doc_id 去重由业务层负责。
        """
        top_k = top_k or self.top_k
        alpha = alpha if alpha is not None else self.alpha

        # 放大内部检索量，为 chunk 去重留余量
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

        search_kwargs: Dict[str, Any] = dict(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=internal_limit,
            output_fields=[
                "doc_id", "chunk_id", "total_chunks",
                "vul_code", "fixed_code", "diff",
                "cwe_id", "cve_id", "cluster_id",
                "source", "language",
                "cwe_desc", "cve_desc",
            ],
        )
        filter_expr = self._build_filter_expr(cluster_filter, cwe_list)
        if filter_expr is not None:
            search_kwargs["filter"] = filter_expr

        results = self.client.hybrid_search(**search_kwargs)

        return results

    def dense_search(
        self,
        query_dense_list: List[List[float]],
        top_k: Optional[int] = None,
        cluster_filter: Optional[str] = None,
        cwe_list: Optional[List[str]] = None,
    ) -> Any:
        """
        执行稠密向量搜索（利用 Milvus search 原生多向量批查询能力）

        由于同一文档可能被分为多个 chunk 存储，搜索时内部 limit 会放大为
        top_k * 2，以确保按 doc_id 去重后仍有足够结果。去重由业务层完成。

        Args:
            query_dense_list: 查询向量列表（支持多向量批查询）
            top_k: 返回的最相似结果数量，默认使用初始化时的配置
            cluster_filter: 可选的 Milvus 过滤表达式（如 "cluster_id == '2'"），
                用于分簇检索时将搜索范围限制在指定簇内
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件（单项等值、多项 in 表达式）

        Returns:
            Any: Milvus 返回的原始搜索结果（嵌套 hits 结构）。
            结果解析和 doc_id 去重由业务层负责。
        """
        top_k = top_k or self.top_k
        internal_limit = top_k * 2 # 放大内部检索量，为 chunk 去重留余量

        search_params = {
            "metric_type": self.dense_metric_type,
            "params": {"ef": max(internal_limit, 64)},
        }

        search_kwargs: Dict[str, Any] = dict(
            collection_name=self.collection_name,
            data=query_dense_list,
            anns_field="dense_vector",
            search_params=search_params,
            limit=internal_limit,
            output_fields=[
                "doc_id", "chunk_id", "total_chunks",
                "vul_code", "fixed_code", "diff",
                "cwe_id", "cve_id", "cluster_id",
                "source", "language",
                "cwe_desc", "cve_desc",
            ],
        )
        filter_expr = self._build_filter_expr(cluster_filter, cwe_list)
        if filter_expr is not None:
            search_kwargs["filter"] = filter_expr

        results = self.client.search(**search_kwargs)

        return results

    def sparse_search(
        self,
        query_text_list: List[str],
        top_k: Optional[int] = None,
        cluster_filter: Optional[str] = None,
        cwe_list: Optional[List[str]] = None,
    ) -> Any:
        """
        批量执行稀疏向量（BM25）搜索（利用 Milvus search 原生多文本批查询能力）。

        Milvus client.search() 的 data 参数支持传入多条原始文本字符串，
        服务端内置 BM25 函数自动将每条文本转换为稀疏向量后执行检索，
        一次 RPC 返回所有查询的结果。
        返回结构为嵌套列表：外层长度 = nq（查询数），内层为每个查询的 hits。

        Args:
            query_text_list: 查询文本列表，每个元素为一个原始代码/文本字符串
            top_k: 每个查询返回结果数量上限（经 internal_limit 放大后由业务去重截断）
            cluster_filter: 可选的 Milvus 过滤表达式（如 "cluster_id == '2'"），
                用于分簇检索时将搜索范围限制在指定簇内
            cwe_list: 可选的 CWE 漏洞类型标识列表（如 ["CWE-20"] 或 ["CWE-20", "CWE-119"]），
                格式合法的项自动追加过滤条件（单项等值、多项 in 表达式）

        Returns:
            Any: Milvus search 返回的原始嵌套结果，外层长度等于 len(query_text_list)，
                每个元素对应一个查询的 hits 列表。
        """
        top_k = top_k or self.top_k
        internal_limit = top_k * 2

        search_kwargs: Dict[str, Any] = dict(
            collection_name=self.collection_name,
            data=query_text_list,  # 原始文本列表，BM25 向量化由 Milvus 服务端完成
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25"},
            limit=internal_limit,
            output_fields=[
                "doc_id", "chunk_id", "total_chunks",
                "vul_code", "fixed_code", "diff",
                "cwe_id", "cve_id", "cluster_id",
                "source", "language",
                "cwe_desc", "cve_desc",
            ],
        )
        filter_expr = self._build_filter_expr(cluster_filter, cwe_list)
        if filter_expr is not None:
            search_kwargs["filter"] = filter_expr

        results = self.client.search(**search_kwargs)

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Returns:
            Dict: 包含集合统计信息的字典，如:
                - row_count: 文档总数
                - data_size: 数据大小（字节）
                - index_size: 索引大小（字节）

        Example:
            >>> stats = client.get_collection_stats()
            >>> print(f"Total documents: {stats['row_count']}")
        """
        return self.client.get_collection_stats(self.collection_name)

    # ------------------------------------------------------------------
    # Stage 1 漏洞模式分析缓存（独立集合 vul_pattern_cache）
    # ------------------------------------------------------------------

    VUL_PATTERN_CACHE_COLLECTION = "vul_pattern_cache"

    def create_pattern_cache(self) -> None:
        """
        创建 Stage 1 漏洞模式分析缓存集合（如已存在则跳过）。

        Schema:
            - doc_id (INT64): 主键，对应知识库中漏洞样例的唯一标识
            - vul_pattern (VARCHAR 65535): LLM 生成的漏洞模式分析文本
            - _dummy_vec (FLOAT_VECTOR dim=2): Milvus 强制要求的向量字段（占位用，最小合法维度为 2）
        """
        if self.client.has_collection(self.VUL_PATTERN_CACHE_COLLECTION):
            logger.info(
                f"Stage 1 cache collection '{self.VUL_PATTERN_CACHE_COLLECTION}' already exists"
            )
            return

        schema = CollectionSchema(fields=[
            FieldSchema(
                name="doc_id", dtype=DataType.INT64,
                is_primary=True, auto_id=False,
            ),
            FieldSchema(
                name="vul_pattern", dtype=DataType.VARCHAR, max_length=65535,
            ),
            FieldSchema(
                name="_dummy_vec", dtype=DataType.FLOAT_VECTOR, dim=2,
            ),
        ])

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="_dummy_vec",
            index_type="AUTOINDEX",
            metric_type="L2",
        )

        self.client.create_collection(
            collection_name=self.VUL_PATTERN_CACHE_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Stage 1 cache collection '{self.VUL_PATTERN_CACHE_COLLECTION}' created")

    def get_pattern_cache(self, doc_id: int) -> Optional[str]:
        """
        查询 Stage 1 漏洞模式分析的缓存。

        Args:
            doc_id: 漏洞样例的文档 ID

        Returns:
            缓存的 vul_pattern 文本，未命中时返回 None
        """
        records = self.client.query(
            collection_name=self.VUL_PATTERN_CACHE_COLLECTION,
            filter=f"doc_id == {doc_id}",
            output_fields=["vul_pattern"],
        )
        if records and records[0].get("vul_pattern"):
            return records[0]["vul_pattern"]
        return None

    def set_pattern_cache(self, doc_id: int, summary: str) -> None:
        """
        写入 Stage 1 漏洞模式分析缓存（已有则覆盖）。

        Args:
            doc_id: 漏洞样例的文档 ID
            summary: LLM 生成的漏洞模式分析文本
        """
        self.client.upsert(
            collection_name=self.VUL_PATTERN_CACHE_COLLECTION,
            data=[{
                "doc_id": doc_id,
                "vul_pattern": summary,
                "_dummy_vec": [0.0, 0.0],
            }],
        )

    def clear_pattern_cache(self) -> None:
        """
        清空 Stage 1 漏洞模式分析缓存集合的全部记录。

        实现策略：若缓存集合存在则直接 drop（一并删除其索引/数据），
        以保证完全清空。调用方需在清空后重新调用 ``create_pattern_cache``
        重新创建集合。
        """
        if not self.client.has_collection(self.VUL_PATTERN_CACHE_COLLECTION):
            logger.info(
                f"Stage 1 cache collection '{self.VUL_PATTERN_CACHE_COLLECTION}' "
                f"does not exist, nothing to clear"
            )
            return
        self.client.drop_collection(self.VUL_PATTERN_CACHE_COLLECTION)
        logger.info(
            f"Stage 1 cache collection '{self.VUL_PATTERN_CACHE_COLLECTION}' dropped"
        )

    def close(self) -> None:
        """
        关闭 Milvus 客户端连接。

        释放与 Milvus 服务器的连接资源，对于本地 Milvus Lite 数据库，
        这会触发数据 flush 到磁盘的操作。

        Note:
            - 建议在程序结束时调用，确保数据持久化
            - 使用上下文管理器模式（with 语句）会自动调用此方法
            - 关闭后不能再使用该客户端实例，需要重新创建
        """
        self.client.close()
        logger.info("Milvus client closed")
