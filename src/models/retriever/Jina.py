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


class JinaCodeEmbedder(BaseEmbedder):
    """
    基于 jinaai/jina-code-embeddings 的代码嵌入生成器封装类。

    支持的模型:
        - jinaai/jina-code-embeddings-0.5b: 494M 参数，最大 896 维
        - jinaai/jina-code-embeddings-1.5b: 1.54B 参数，最大 1536 维

    支持的检索任务:
        - nl2code: 自然语言查询 → 代码片段（如"如何读取CSV"→pandas代码）
        - code2code: 代码 → 相似代码实现（跨语言代码搜索）
        - code2nl: 代码 → 相关文档/注释
        - code2completion: 代码补全场景
        - qa: 技术问答检索

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备（cuda/cpu）
        max_length: 最大序列长度（模型支持最大 32768 token的输入）
        batch_size: 批处理大小
        normalize: 是否对嵌入向量进行 L2 归一化
        dim: 目标嵌入维度（支持 Matryoshka 截断：64/128/256/512/896/1536）
        task: 当前任务类型，决定使用哪种 instruction prefix
        torch_dtype: 模型权重数据类型（默认 bfloat16 以节省显存）
        model: SentenceTransformer 模型实例
    """

    # 任务类型到 prompt 名称的映射表，每个任务都有 query（查询侧）和 document（文档侧）两种 prompt
    # 这些 prompt 名称会被 sentence-transformers 映射为对应的 instruction prefix
    TASK_PROMPT_MAP = {
        "nl2code": {"query": "nl2code_query", "document": "nl2code_document"}, # 自然语言搜代码: "How to read CSV" → pandas.read_csv()
        "code2code": {"query": "code2code_query", "document": "code2code_document"}, # 代码相似度搜索: 查找功能相似的代码实现
        "code2nl": {"query": "code2nl_query", "document": "code2nl_document"}, # 代码 → 自然语言描述
        "code2completion": {"query": "code2completion_query", "document": "code2completion_document"}, # 代码补全场景
        "qa": {"query": "qa_query", "document": "qa_document"}, # 技术问答检索
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Jina 代码嵌入模型。

        Args:
            config: 配置字典，包含以下键:
                - model_name (str): 必需，HuggingFace 模型路径
                - device (str): 可选，默认 "cuda"
                - max_length (int): 可选，默认 32768
                - batch_size (int): 可选，默认 8
                - normalize_embeddings (bool): 可选，默认 True
                - dim (int): 可选，默认 896（0.5b 模型）
                - task (str): 可选，默认 "code2code"
                - torch_dtype (str): 可选，默认 "bfloat16"
                - chunk_overlap_tokens (int): 可选，分块时相邻块的重叠 token 数，默认 1000

        Note:
            - padding_side="left" 是必需的，因为模型使用 last-token pooling
            - trust_remote_code=True 允许加载模型的自定义代码
            - bfloat16 在保持精度的同时显著降低显存占用
        """
        # 从配置中提取参数，使用 get 方法提供默认值
        self.model_name = config["model_name"]  # 必需参数
        self.device = select_device(config)
        self.max_length = config.get("max_length", 8192)
        self.batch_size = config.get("batch_size", 8)
        self.normalize = config.get("normalize_embeddings", True)
        self.dim = config.get("dim", 896)  # 0.5b 模型默认 896 维
        self.task = config.get("task", "code2code")
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 1000)

        # Hugging Face 端点：必须在首次 import huggingface_hub 之前写入 os.environ。
        # huggingface_hub.constants.ENDPOINT 在模块导入时一次性绑定，晚于本处设置则镜像不生效。
        hf_endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint).rstrip("/")

        # 延迟导入：避免 import 本模块时提前加载 huggingface_hub，导致上面 HF_ENDPOINT 来不及生效
        from sentence_transformers import SentenceTransformer

        # 加载 SentenceTransformer 模型
        logger.info(f"Loading model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,  # 允许执行模型仓库中的自定义代码
            device=self.device,
            model_kwargs={"torch_dtype": self.torch_dtype},  # 使用bfloat16低精度节省显存
            tokenizer_kwargs={"padding_side": "left"},  # last-token pooling 需要左填充
        )
        # 设置模型的最大序列长度
        self.model.max_seq_length = self.max_length # 最大为32768
        logger.info(f"Model loaded. dim={self.dim}, task={self.task}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        为文档（代码库中的候选代码/注释）生成嵌入向量。

        使用 document 侧的 instruction prefix 对输入文本进行编码，
        使向量空间中的文档表示与查询表示形成非对称匹配。

        Args:
            texts: 待编码的文档文本列表（如代码片段、函数定义等）

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)

        Example:
            >>> embedder = JinaCodeEmbedder({"model_name": "jinaai/jina-code-embeddings-0.5b"})
            >>> docs = ["def hello(): print('world')", "import pandas as pd"]
            >>> embeddings = embedder.embed_documents(docs)
            >>> embeddings.shape
            (2, 896)
        """
        # 根据当前任务获取对应的 document prompt 名称
        # 例如 code2code 任务会使用 "Candidate code snippet:\n" 作为前缀
        prompt_name = self.TASK_PROMPT_MAP[self.task]["document"]

        # 调用 sentence-transformers 的 encode 方法生成嵌入
        embeddings = self.model.encode( # 默认 add_special_tokens=True
            texts,
            batch_size=self.batch_size, # 批量大小
            prompt_name=prompt_name,  # 自动添加任务特定的 instruction prefix
            show_progress_bar=True,  # 显示编码进度条
            normalize_embeddings=self.normalize,  # L2 归一化，便于余弦相似度计算
        )

        # Matryoshka 维度截断：如果输出维度大于目标维度，进行截断
        # 这允许使用更低的维度（如 256 维）来节省存储和计算，同时保持较高性能
        if embeddings.shape[1] > self.dim:
            embeddings = embeddings[:, : self.dim]

        return embeddings

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        为查询（用户输入的搜索条件）生成嵌入向量。

        使用 query 侧的 instruction prefix 对输入文本进行编码，
        与文档侧的编码形成非对称检索（asymmetric retrieval）架构。

        Args:
            texts: 待编码的查询文本列表（如自然语言描述、代码片段等）

        Returns:
            嵌入向量数组，形状为 (len(texts), dim)

        Example:
            >>> embedder = JinaCodeEmbedder({"model_name": "jinaai/jina-code-embeddings-0.5b"})
            >>> queries = ["how to read csv file", "sort list in python"]
            >>> embeddings = embedder.embed_queries(queries)
            >>> embeddings.shape
            (2, 896)

        Note:
            对于 nl2code 任务，查询是自然语言，文档是代码；
            对于 code2code 任务，查询和文档都是代码。
        """
        # 根据当前任务获取对应的 query prompt 名称
        # 例如 nl2code 任务会使用 "Find the most relevant code snippet given the following query:\n" 作为前缀
        prompt_name = self.TASK_PROMPT_MAP[self.task]["query"]

        # 生成查询嵌入
        embeddings = self.model.encode( # 默认 add_special_tokens=True
            texts,
            batch_size=self.batch_size,
            prompt_name=prompt_name,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

        # Matryoshka 维度截断
        if embeddings.shape[1] > self.dim:
            embeddings = embeddings[:, : self.dim]

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
        确保 ``embed_token_ids`` 的前向传播与 ``embed_documents`` / ``embed_queries``
        的向量空间保持对齐，同时通过 offset 机制冻结 prefix 段不参与优化。

        若模型未配置 prompt（或 prompt 为空字符串），返回空列表 ``[]``，
        下游代码可零改动兼容无 prompt 的嵌入模型。

        Args:
            side: ``"document"`` 或 ``"query"``，对应 TASK_PROMPT_MAP 中的键

        Returns:
            prompt 的 token ID 列表（不含 special tokens）；无 prompt 时为空列表
        """
        prompt_name = self.TASK_PROMPT_MAP[self.task][side]
        prompt_text = self.model.prompts.get(prompt_name, "")
        if not prompt_text:
            return []
        return self.model.tokenizer.encode(prompt_text, add_special_tokens=False)

    def get_instruct_prompt(self, side: str = "query") -> str:
        """
        返回当前任务对应侧的 instruction prefix 原始文本。

        Jina 的 prompt 文本由 SentenceTransformer 内部的 prompts 字典维护，
        通过 TASK_PROMPT_MAP 中存储的 prompt_name 进行查找。

        Args:
            side: ``"document"`` 或 ``"query"``，对应 TASK_PROMPT_MAP 中的键

        Returns:
            instruction prefix 的原始文本字符串；无 prompt 时为空字符串
        """
        prompt_name = self.TASK_PROMPT_MAP[self.task][side]
        return self.model.prompts.get(prompt_name, "")

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        使用模型的分词器将输入文本转换为 token ID 列表。

        Args:
            text: 待分词的文本字符串
            add_special_tokens: 是否添加特殊 token，默认 True
        Returns:
            List[int]: Token ID 列表

        Example:
            >>> embedder = JinaCodeEmbedder(config)
            >>> tokens = embedder.tokenize("print('hello')")
            >>> print(tokens)
            [123, 456, ...]
        """
        # SentenceTransformer 的 tokenizer 属性直接暴露了底层 Transformer 的 tokenizer
        # 使用 encode 方法直接返回 token ID 列表
        return self.model.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
        )
        
    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        使用模型的分词器将 Token ID 列表还原为文本字符串。

        这是 tokenize() 方法的逆操作。由于 BPE/WordPiece 等分词算法
        存在不可逆的文本规范化（如空格处理、大小写折叠），还原结果
        与原始输入可能存在细微差异（如空格位置），但语义内容一致。

        Args:
            token_ids: 待解码的 Token ID 列表（即 tokenize() 的返回值）
            skip_special_tokens: 是否跳过特殊 token（如 [CLS]、[SEP]、[PAD]），
                                默认 True；若需要保留特殊标记（如调试分词结构）
                                可设为 False

        Returns:
            str: 解码后的文本字符串

        Example:
            >>> embedder = JinaCodeEmbedder(config)
            >>> tokens = embedder.tokenize("print('hello')")
            >>> text = embedder.detokenize(tokens)
            >>> print(text)
            "print('hello')"

        Note:
            - tokenize() 默认会添加特殊 token（add_special_tokens=True），
            因此 detokenize() 默认 skip_special_tokens=True 与之对应，
            以确保还原结果不含多余的特殊标记。
            - 若上游 tokenize() 使用了 truncation=True，则超长部分已丢失，
            detokenize() 只能还原截断后的内容，无法恢复原始完整文本。
        """
        return self.model.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,  # 清理 BPE 引入的多余空格（如 "hello ." → "hello."）
        )