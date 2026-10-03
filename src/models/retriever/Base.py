import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    词嵌入模型的抽象基类，定义统一的编码器接口。

    所有具体的嵌入模型封装类（如 JinaCodeEmbedder）均应继承本类，
    并实现全部抽象方法。

    Attributes:
        model_name: HuggingFace 模型名称或本地路径
        device: 运行设备（如 "cuda"、"cpu"）
        max_length: 最大序列长度（token 数）
        batch_size: 批处理大小
        normalize: 是否对嵌入向量进行 L2 归一化
        dim: 目标嵌入维度
        chunk_overlap_tokens: 分块时相邻块的重叠 token 数
    """

    model_name: str
    device: str
    max_length: int
    batch_size: int
    normalize: bool
    dim: int
    chunk_overlap_tokens: int

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化嵌入模型。

        Args:
            config: 配置字典，至少包含 model_name 键；其余键由子类定义。
        """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        为文档（候选代码/文本）生成嵌入向量。

        Args:
            texts: 待编码的文档文本列表

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)
        """

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        为查询（用户输入）生成嵌入向量。

        Args:
            texts: 待编码的查询文本列表

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)
        """

    @abstractmethod
    def embed_token_ids(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        对已 tokenize 的序列执行前向传播，返回 L2 归一化的句子级嵌入。

        保留计算图梯度，供对抗攻击（AGGD/ABGS）反向传播使用。

        Args:
            token_ids: token ID 张量，形状 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状 (batch_size, seq_len)

        Returns:
            L2 归一化后的句子嵌入，形状 (batch_size, embed_dim)
        """

    @abstractmethod
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        使用模型分词器将文本转换为 token ID 列表。

        Args:
            text: 待分词的文本字符串
            add_special_tokens: 是否添加特殊 token，默认 True
        Returns:
            Token ID 列表
        """

    @abstractmethod
    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 token ID 列表还原为文本字符串。

        Args:
            token_ids: 待解码的 Token ID 列表
            skip_special_tokens: 是否跳过特殊 token，默认 True

        Returns:
            解码后的文本字符串
        """

    def get_instruct_tids(self, side: str = "query") -> List[int]:
        """
        返回当前任务对应侧（query/document）的 instruction prefix 的 token ID 列表。

        默认实现返回空列表，适用于无 prompt 前缀的通用嵌入模型。
        支持 prompt 的子类（如 JinaCodeEmbedder）应覆写本方法。

        Args:
            side: "query" 或 "document"

        Returns:
            prompt 的 token ID 列表；无 prompt 时为空列表
        """
        return []

    def get_instruct_prompt(self, side: str = "query") -> str:
        """
        返回当前任务对应侧（query/document）的 instruction prefix 原始文本。

        默认实现返回空字符串，适用于无 prompt 前缀的通用嵌入模型（如 GTE）。
        支持 prompt 的子类（Jina / Harrier）应覆写本方法。

        Args:
            side: "query" 或 "document"

        Returns:
            instruction prefix 的原始文本字符串；无 prompt 时为空字符串
        """
        return ""
    
    def get_model_name(self, with_provider: bool = True) -> str:
        """
        返回模型名称。

        Args:
            with_provider: 若为 True（默认），返回含 Provider 前缀的完整名称，
                           格式为 "{provider}/{model_name}"（如 "jinaai/jina-embeddings-v3"）；
                           若为 False，仅返回斜杠后的模型名称部分
                           （如 "jina-embeddings-v3"）。
                           当 model_name 中不含斜杠时，两种模式均返回完整字符串。

        Returns:
            模型名称字符串。
        """
        if with_provider:
            return self.model_name
        return self.model_name.split("/", 1)[-1]

    def chunk_text(
        self, text: str, max_tokens: int = 0, overlap_tokens: int = 0
    ) -> List[Dict[str, Any]]:
        """
        基于模型 tokenizer 对超长文本进行分块。

        如果文本 token 数不超过 max_tokens，直接返回单个 chunk；
        否则按 max_tokens 切分为多个块，相邻块之间有 overlap_tokens 个 token 的重叠，
        以保证语义连续性。

        Args:
            text: 待分块的文本
            max_tokens: 每个 chunk 的最大 token 数，0 表示使用 self.max_length
            overlap_tokens: 相邻 chunk 的重叠 token 数，0 表示使用 self.chunk_overlap_tokens

        Returns:
            List[Dict]: 分块结果列表，每个元素包含:
                - chunk_id (int): 块序号（从 0 开始）
                - text (str): 该块解码后的文本
                - token_count (int): 该块的 token 数
                - total_chunks (int): 该文档的总块数
        """
        max_tokens = max_tokens or self.max_length
        overlap_tokens = overlap_tokens or self.chunk_overlap_tokens

        tokenizer = self.model.tokenizer
        all_tokens = tokenizer.encode(text, add_special_tokens=True)
        total_token_count = len(all_tokens)

        if total_token_count <= max_tokens:
            return [
                {
                    "chunk_id": 0,
                    "text": text,
                    "token_count": total_token_count,
                    "total_chunks": 1,
                }
            ]

        # 步长 = max_tokens - overlap，确保至少前进 1 个 token
        stride = max(max_tokens - overlap_tokens, 1)
        chunks: List[Dict[str, Any]] = []
        start = 0

        while start < total_token_count:
            end = min(start + max_tokens, total_token_count)
            chunk_tokens = all_tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True,)
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                }
            )
            if end >= total_token_count:
                break
            start += stride

        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        logger.info(
            f"Chunked text ({total_token_count} tokens) into {total_chunks} chunks "
            f"(max_tokens={max_tokens}, overlap={overlap_tokens})"
        )
        return chunks


# 前缀 → (模块路径, 类名) 注册表，新增模型只需追加一行
_MODEL_NAME_REGISTRY: List[tuple] = [
    ("jinaai/",           "src.models.retriever.Jina",    "JinaCodeEmbedder"),
    ("alibaba-nlp/gte-",  "src.models.retriever.GTE",     "GTEEmbedder"),
    ("microsoft/harrier-","src.models.retriever.Harrier", "HarrierEmbedder"),
    ("qwen/qwen3-embedding-", "src.models.retriever.Qwen", "QwenEmbedder"),
]


def create_embedder(config: Dict[str, Any]) -> "BaseEmbedder":
    """
    根据配置字典中的 model_name 字段，自动实例化对应的 BaseEmbedder 子类。

    匹配规则：将 model_name 转为小写后，按 _MODEL_NAME_REGISTRY 中的前缀顺序依次匹配，
    命中第一个匹配项后延迟导入对应模块并返回实例。

    Args:
        config: 配置字典，至少包含 model_name 键。

    Returns:
        对应子类的实例（类型为 BaseEmbedder）。

    Raises:
        ValueError: model_name 未命中任何注册前缀时抛出。

    Example:
        >>> cfg = load_yaml("configs/models/jina-code-embeddings-0.5b.yml")
        >>> embedder = create_embedder(cfg)   # 返回 JinaCodeEmbedder 实例
        >>> cfg2 = load_yaml("configs/models/gte-modernbert-base.yml")
        >>> embedder2 = create_embedder(cfg2) # 返回 GTEEmbedder 实例
    """
    model_name = config.get("model_name", "").lower()
    for prefix, module_path, class_name in _MODEL_NAME_REGISTRY:
        if model_name.startswith(prefix):
            cls = getattr(importlib.import_module(module_path), class_name)
            return cls(config)
    raise ValueError(
        f"No embedder registered for model_name='{config.get('model_name')}'. "
        f"Add an entry to _MODEL_NAME_REGISTRY in src/models/retriever/Base.py."
    )
