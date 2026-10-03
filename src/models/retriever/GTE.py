import logging
import os
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.device import select_device
from src.models.retriever.Base import BaseEmbedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GTEEmbedder(BaseEmbedder):
    """
    基于 Alibaba-NLP/gte-modernbert-base 的文本嵌入生成器封装类。

    支持的模型:
        - Alibaba-NLP/gte-modernbert-base: 149M 参数，768 维，最大 8192 token

    模型特性:
        - 基于 ModernBERT 编码器架构，使用 CLS token pooling
        - 不使用 instruction prefix（query/document 侧编码对称）
        - 在代码检索（COIR）、长文档检索（LoCo）、通用文本检索（MTEB/BEIR）上均有竞争力
        - 支持 Flash Attention 2（需安装 flash_attn）

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备（cuda/cpu）
        max_length: 最大序列长度（模型支持最大 8192 token）
        batch_size: 批处理大小
        normalize: 是否对嵌入向量进行 L2 归一化
        dim: 目标嵌入维度（固定 768）
        torch_dtype: 模型权重数据类型（默认 bfloat16 以节省显存）
        model: SentenceTransformer 模型实例
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 GTE ModernBERT 嵌入模型。

        Args:
            config: 配置字典，包含以下键:
                - model_name (str): 必需，HuggingFace 模型路径
                - device (str): 可选，默认 "cuda"
                - max_length (int): 可选，默认 8192
                - batch_size (int): 可选，默认 8
                - normalize_embeddings (bool): 可选，默认 True
                - dim (int): 可选，默认 768
                - torch_dtype (str): 可选，默认 "bfloat16"
                - chunk_overlap_tokens (int): 可选，分块时相邻块的重叠 token 数，默认 1000

        Note:
            - GTE 使用 CLS token pooling，无需 padding_side="left"
            - GTE 是标准 ModernBERT 架构，无需 trust_remote_code=True
            - 若已安装 flash_attn，sentence-transformers 会自动启用 Flash Attention 2
        """
        self.model_name = config["model_name"]
        self.device = select_device(config)
        self.max_length = config.get("max_length", 8192)
        self.batch_size = config.get("batch_size", 8)
        self.normalize = config.get("normalize_embeddings", True)
        self.dim = config.get("dim", 768)
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 1000)

        # Hugging Face 端点：必须在首次 import huggingface_hub 之前写入 os.environ。
        # huggingface_hub.constants.ENDPOINT 在模块导入时一次性绑定，晚于本处设置则镜像不生效。
        hf_endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint).rstrip("/")

        # 延迟导入：避免 import 本模块时提前加载 huggingface_hub，导致上面 HF_ENDPOINT 来不及生效
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"torch_dtype": self.torch_dtype},
        )
        self.model.max_seq_length = self.max_length
        logger.info(f"Model loaded. dim={self.dim}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        为文档（候选代码/文本）生成嵌入向量。

        GTE 模型不区分 query/document 侧的编码策略，直接编码输入文本。

        Args:
            texts: 待编码的文档文本列表

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)

        Example:
            >>> embedder = GTEEmbedder({"model_name": "Alibaba-NLP/gte-modernbert-base"})
            >>> docs = ["def hello(): print('world')", "import pandas as pd"]
            >>> embeddings = embedder.embed_documents(docs)
            >>> embeddings.shape
            (2, 768)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        为查询（用户输入）生成嵌入向量。

        GTE 模型不区分 query/document 侧的编码策略，直接编码输入文本。

        Args:
            texts: 待编码的查询文本列表

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)

        Example:
            >>> embedder = GTEEmbedder({"model_name": "Alibaba-NLP/gte-modernbert-base"})
            >>> queries = ["how to implement quick sort", "what is a buffer overflow"]
            >>> embeddings = embedder.embed_queries(queries)
            >>> embeddings.shape
            (2, 768)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

    def embed_token_ids(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        对已 tokenize 的序列执行前向传播，返回 L2 归一化的句子级嵌入。

        与 ``encode`` 不同，本方法：
          - 接收已 tokenize 的 tensor，允许精确控制 token 级别的修改（对抗攻击需求）
          - 保留计算图梯度（不调用 ``.detach()``），供 AGGD/ABGS 反向传播使用
          - 不自动添加 prompt prefix（GTE 无 prompt，无需额外处理）

        Args:
            token_ids: token ID 张量，形状 ``(batch_size, seq_len)``
            attention_mask: 注意力掩码，形状 ``(batch_size, seq_len)``

        Returns:
            L2 归一化后的句子嵌入，形状 ``(batch_size, embed_dim)``
        """
        features = {"input_ids": token_ids, "attention_mask": attention_mask}
        output = self.model(features)
        emb = output["sentence_embedding"] if isinstance(output, dict) else output
        return F.normalize(emb, dim=-1)

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        使用模型的分词器将输入文本转换为 token ID 列表。

        Args:
            text: 待分词的文本字符串
            add_special_tokens: 是否添加特殊 token，默认 True
        Returns:
            Token ID 列表

        Example:
            >>> embedder = GTEEmbedder(config)
            >>> tokens = embedder.tokenize("print('hello')")
            >>> print(tokens)
            [123, 456, ...]
        """
        return self.model.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
        )

    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 token ID 列表还原为文本字符串。

        Args:
            token_ids: 待解码的 Token ID 列表
            skip_special_tokens: 是否跳过特殊 token，默认 True

        Returns:
            解码后的文本字符串

        Example:
            >>> embedder = GTEEmbedder(config)
            >>> tokens = embedder.tokenize("print('hello')")
            >>> text = embedder.detokenize(tokens)
            >>> print(text)
            "print('hello')"
        """
        return self.model.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )
    