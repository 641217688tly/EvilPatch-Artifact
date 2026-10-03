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


class HarrierEmbedder(BaseEmbedder):
    """
    基于 microsoft/harrier-oss-v1 的文本嵌入生成器封装类。

    支持的模型:
        - microsoft/harrier-oss-v1-270m: 270M 参数，640 维
        - microsoft/harrier-oss-v1-0.6b: 0.6B 参数，1024 维
        - microsoft/harrier-oss-v1-27b:  27B 参数，5376 维

    架构特点:
        - Decoder-only 架构（基于 Qwen3），使用 last-token pooling + L2 归一化
        - Query 侧必须添加 instruction prefix，否则性能下降；document 侧无需 prompt
        - Instruction 格式：``"Instruct: {task_description}\\nQuery: "``
        - 不支持 Matryoshka 维度截断

    支持的检索任务:
        - nl2code:   自然语言查询 → 代码片段
        - sts:       语义文本相似度
        - bitext:    双语文本挖掘

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备（cuda/cpu）
        max_length: 最大序列长度（模型支持最大 32768 token 的输入）
        batch_size: 批处理大小
        normalize: 是否对嵌入向量进行 L2 归一化
        dim: 嵌入维度（0.6b 模型固定 1024 维）
        task: 当前任务类型，决定使用哪种 instruction prefix
        torch_dtype: 模型权重数据类型（默认 bfloat16 以节省显存）
        model: SentenceTransformer 模型实例
    """

    # 任务类型到 instruction 字符串的映射表
    # Harrier 使用原始 instruction 字符串作为 prompt（通过 prompt= 参数传入 encode），
    # 而非 Jina 那样通过 prompt_name= 查表。document 侧统一为空字符串（无需 prompt）。
    TASK_PROMPT_MAP = {
        "code2code": {"query": "query", "document": "document"}, # 代码相似度搜索: 查找功能相似的代码实现
        "web_search": {"query": "web_search_query", "document": ""}, # 代码补全场景
        "sts": {"query": "sts_query", "document": ""}, # 技术问答检索
        "bitext_query": {"query": "bitext_query", "document": ""},
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Harrier 嵌入模型。

        Args:
            config: 配置字典，包含以下键:
                - model_name (str): 必需，HuggingFace 模型路径
                - device (str): 可选，默认 "cuda"
                - max_length (int): 可选，默认 32768
                - batch_size (int): 可选，默认 8
                - normalize_embeddings (bool): 可选，默认 True
                - dim (int): 可选，默认 1024（0.6b 模型）
                - task (str): 可选，默认 "code2code"
                - torch_dtype (str): 可选，默认 "bfloat16"
                - chunk_overlap_tokens (int): 可选，分块时相邻块的重叠 token 数，默认 1000
                - hf_endpoint (str): 可选，HuggingFace 镜像端点

        Note:
            - padding_side="left" 是必需的，因为模型使用 last-token pooling
            - bfloat16 在保持精度的同时显著降低显存占用
        """
        self.model_name = config["model_name"]
        self.device = select_device(config)
        self.max_length = config.get("max_length", 32768)
        self.batch_size = config.get("batch_size", 8)
        self.normalize = config.get("normalize_embeddings", True)
        self.dim = config.get("dim", 1024)  # 0.6b 模型固定 1024 维
        self.task = config.get("task", "code2code")
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 1000)

        hf_endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint).rstrip("/")

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"dtype": self.torch_dtype},  # Harrier 使用 dtype= 而非 torch_dtype=
            tokenizer_kwargs={"padding_side": "left"},  # last-token pooling 需要左填充
        )
        self.model.max_seq_length = self.max_length
        logger.info(f"Model loaded. dim={self.dim}, task={self.task}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        为文档（代码库中的候选代码）生成嵌入向量。

        Harrier 文档侧无需 instruction prefix，直接编码即可。

        Args:
            texts: 待编码的文档文本列表（如代码片段、函数定义等）

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)
        """
        embeddings = self.model.encode( # 默认 add_special_tokens=True
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        为查询（用户输入的缺陷代码）生成嵌入向量。

        Harrier query 侧必须添加 instruction prefix，通过 ``prompt=`` 参数
        直接传入 instruction 字符串（而非 prompt_name）。

        Args:
            texts: 待编码的查询文本列表

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)
        """
        prompt_name = self.TASK_PROMPT_MAP[self.task]["query"]

        embeddings = self.model.encode( # 默认 add_special_tokens=True
            texts,
            batch_size=self.batch_size,
            prompt_name=prompt_name,  # 直接传入 instruction 字符串
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings
    
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
          - 不自动添加 prompt prefix（调用方需自行拼接，参见 ``get_prompt_token_ids``）

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

    def get_instruct_tids(self, side: str = "query") -> List[int]:
        """
        返回当前任务对应侧（query/document）的 instruction prefix 的 token ID 列表。

        用于对抗攻击场景：将 prompt prefix 拼接到代码 token 序列前端，
        确保 ``embed_token_ids`` 的前向传播与 ``embed_queries`` / ``embed_documents``
        的向量空间保持对齐，同时通过 offset 机制冻结 prefix 段不参与优化。

        Args:
            side: ``"query"`` 或 ``"document"``

        Returns:
            prompt 的 token ID 列表（不含 special tokens）；
            document 侧或 prompt 为空时返回空列表
        """
        prompt_name = self.TASK_PROMPT_MAP[self.task].get(side, "")
        prompt_text = self.model.prompts.get(prompt_name, "")
        if not prompt_text:
            return []
        return self.model.tokenizer.encode(prompt_name, add_special_tokens=False)

    def get_instruct_prompt(self, side: str = "query") -> str:
        """
        返回当前任务对应侧的 instruction prefix 原始文本

        document 侧为空字符串（无需 prompt）。

        Args:
            side: ``"query"`` 或 ``"document"``

        Returns:
            instruction prefix 的原始文本字符串；document 侧或无 prompt 时为空字符串
        """
        prompt_name = self.TASK_PROMPT_MAP[self.task].get(side, "")
        return self.model.prompts.get(prompt_name, "")

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        使用模型的分词器将输入文本转换为 token ID 列表。

        Args:
            text: 待分词的文本字符串
            add_special_tokens: 是否添加特殊 token，默认 True
        Returns:
            List[int]: Token ID 列表
        """
        return self.model.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
        )

    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        使用模型的分词器将 Token ID 列表还原为文本字符串。

        Args:
            token_ids: 待解码的 Token ID 列表
            skip_special_tokens: 是否跳过特殊 token，默认 True

        Returns:
            str: 解码后的文本字符串
        """
        return self.model.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )
