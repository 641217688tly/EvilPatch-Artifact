import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from src.models.retriever.Base import BaseEmbedder
from src.utils.device import select_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class QwenEmbedder(BaseEmbedder):
    """Qwen/Qwen3-Embedding-0.6B 检索器封装。

    模型特性:
        - Decoder-only 架构，使用 last-token pooling + L2 归一化
        - ``add_special_tokens=True`` 时 tokenizer 自动在末尾添加
          ``<|endoftext|>``
        - 支持 Matryoshka 维度截断（32-1024）
        - 官方推荐为 query 侧添加 instruction prefix（可提升 1%~5% 检索
          性能），格式为 ``"Instruct: {task_description}\\nQuery:{query}"``；
          document 侧无需 prompt
        - 官方仅示范了 Web 搜索任务的 instruction，未提供 code2code
          任务的官方指令，``TASK_PROMPT_MAP`` 中的指令为按官方格式
          自定义的英文指令
        - 可通过配置 ``enable_instruct: false`` 全局关闭 instruction，
          关闭后 query/document 采用对称编码
    """

    # 任务类型到 instruction 字符串的映射表（document 侧统一为空字符串）。
    # 自定义指令不在模型内置 model.prompts 中，因此直接存储完整文本，
    # 通过 prompt= 参数传入 encode。
    TASK_PROMPT_MAP = {
        "code2code": {  # 代码相似度搜索：查找功能相似的代码实现
            "query": "Instruct: Given a code snippet, retrieve code snippets that are semantically similar to the query\nQuery:",
            "document": "",
        },
        "nl2code": {  # 自然语言搜代码：自然语言描述 → 对应代码实现
            "query": "Instruct: Given a natural language description, retrieve code snippets that implement the described functionality\nQuery:",
            "document": "",
        },
        "sts": {  # 语义文本相似度
            "query": "Instruct: Given a code snippet, retrieve semantically similar code or text\nQuery:",
            "document": "",
        },
        "bitext": {  # 双语文本挖掘
            "query": "Instruct: Retrieve parallel text pairs that are translations of each other\nQuery:",
            "document": "",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """初始化 Qwen3-Embedding 模型。

        Args:
            config: 配置字典，包含以下键:
                - model_name (str): 必需，HuggingFace 模型路径
                - device (str): 可选，默认 "cuda"
                - max_length (int): 可选，默认 32768
                - batch_size (int): 可选，默认 8
                - normalize_embeddings (bool): 可选，默认 True
                - dim (int): 可选，默认 1024（支持 32-1024 Matryoshka 截断）
                - task (str): 可选，默认 "code2code"
                - enable_instruct (bool): 可选，默认 True；False 时所有
                  方法均不添加 instruction（query/document 对称编码）
                - torch_dtype (str): 可选，默认 "bfloat16"
                - chunk_overlap_tokens (int): 可选，默认 1000
                - hf_endpoint (str): 可选，HuggingFace 镜像端点
        """
        self.model_name = config["model_name"]
        self.device = select_device(config)
        self.max_length = config.get("max_length", 32768)
        self.batch_size = config.get("batch_size", 8)
        self.normalize = config.get("normalize_embeddings", True)
        self.dim = config.get("dim", 1024)
        self.task = config.get("task", "code2code")
        self.enable_instruct = config.get("enable_instruct", True)
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 1000)

        # Hugging Face endpoint 必须在首次 import huggingface_hub 前设置。
        hf_endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint).rstrip("/")

        # 延迟导入，避免过早初始化 huggingface_hub endpoint。
        from sentence_transformers import SentenceTransformer

        logger.info("Loading model: %s on %s", self.model_name, self.device)
        self.model: "SentenceTransformer" = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"torch_dtype": self.torch_dtype},
            tokenizer_kwargs={"padding_side": "left"},
            truncate_dim=self.dim,
        )
        self.model.max_seq_length = self.max_length
        logger.info(
            "Model loaded. dim=%s, task=%s, enable_instruct=%s",
            self.dim,
            self.task,
            self.enable_instruct,
        )

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """为文档生成不含 instruction prefix 的嵌入。"""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """为查询生成嵌入。

        enable_instruct=True 时通过 ``prompt=`` 为 query 侧添加当前任务的
        instruction prefix（官方推荐，document 侧无需 prompt）；
        enable_instruct=False 时不添加任何前缀，与 document 侧对称编码。
        """
        prompt = (
            self.TASK_PROMPT_MAP[self.task]["query"] if self.enable_instruct else None
        )
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            prompt=prompt,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

    def embed_token_ids(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """对已 tokenize 的序列前向传播，并保留攻击算法所需梯度。"""
        features = {"input_ids": token_ids, "attention_mask": attention_mask}
        output = self.model(features)
        embedding = output["sentence_embedding"] if isinstance(output, dict) else output
        embedding = embedding[:, : self.dim]
        return F.normalize(embedding, p=2, dim=-1)

    def get_instruct_tids(self, side: str = "query") -> List[int]:
        """返回当前任务对应侧 instruction prefix 的 token ID 列表。

        用于对抗攻击场景：将 prompt prefix 拼接到代码 token 序列前端，
        确保 ``embed_token_ids`` 的前向传播与 ``embed_queries`` /
        ``embed_documents`` 的向量空间保持对齐。

        enable_instruct=False、document 侧或 prompt 为空时返回空列表。
        """
        if not self.enable_instruct:
            return []
        prompt_text = self.TASK_PROMPT_MAP[self.task].get(side, "")
        if not prompt_text:
            return []
        return self.model.tokenizer.encode(prompt_text, add_special_tokens=False)

    def get_instruct_prompt(self, side: str = "query") -> str:
        """返回当前任务对应侧的 instruction prefix 原始文本。
        
        Args:
            side: ``"query"`` 或 ``"document"``

        enable_instruct=False 或 document 侧（无 prompt）时返回空字符串。
        """
        if not self.enable_instruct:
            return ""
        return self.TASK_PROMPT_MAP[self.task].get(side, "")

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """使用模型 tokenizer 将文本转换为 token ID。"""
        return self.model.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
        )

    def detokenize(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """将 token ID 还原为文本。"""
        return self.model.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )
