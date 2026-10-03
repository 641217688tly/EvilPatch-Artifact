import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


class QwenGenerator(BaseGenerator):
    """
    基于硅基流动（SiliconFlow）OpenAI 兼容 API 的 Qwen 文本生成器。

    通过硅基流动平台远程调用 Qwen 系列模型，例如:
        - Qwen/Qwen3-Coder-30B-A3B-Instruct
        - Qwen/Qwen3-235B-A22B-Instruct
        - 其他在硅基流动控制台可用的 Qwen 模型

    Qwen3 系列模型支持思考模式（Thinking Mode），可通过配置 enable_thinking 启用。
    启用后，模型在生成最终回答前会先输出推理过程（<think>...</think> 块），
    有助于提升复杂代码分析任务的输出质量。

    Attributes:
        model_name:       硅基流动平台上的模型 ID（如 "Qwen/Qwen3-Coder-30B-A3B-Instruct"）
        base_url:         API 基础 URL，默认 https://api.siliconflow.cn/v1
        temperature:      生成温度
        max_tokens:       默认最大生成 token 数
        timeout:          单次 API 请求超时时间（秒）
        retry_num:        上层应用层重试次数
        enable_thinking:  是否启用 Qwen3 思考模式（默认 False）
        client:           OpenAI 客户端实例
    """

    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        初始化 Qwen 文本生成器（硅基流动后端）。

        Args:
            config: 配置字典，支持两种格式:
                - model_name (str):       必需，硅基流动模型 ID
                - api_pool (list):        多线程，每项含 api_key，base_url 可选
                - api_key (str):          单条凭证，base_url 可选
                - temperature (float):    可选，默认 0.7
                - max_tokens (int):       可选，默认 4096
                - timeout (float):        可选，单次 API 请求超时秒数，默认 600
                - retry_num (int):        可选，默认 3
                - enable_thinking (bool): 可选，是否启用 Qwen3 思考模式，默认 False
            pool_index: api_pool 中使用哪个条目

        Example:
            >>> from src.utils.io import load_yaml
            >>> cfg = load_yaml("configs/models/qwen3-coder-30b-a3b-instruct.yml")
            >>> generator = QwenGenerator(cfg)
            >>> response = generator.generate("分析这段代码的安全性...")
        """
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 32768)
        self.timeout = BaseGenerator._resolve_timeout(config)
        self.retry_num = config.get("retry_num", 3)
        self.enable_thinking = config.get("enable_thinking", False)

        api_key, self.base_url, self.model_name = BaseGenerator._resolve_api_credentials(
            config, pool_index
        )

        logger.info(
            f"Initializing QwenGenerator: model={self.model_name}, "
            f"base_url={self.base_url}, pool_index={pool_index}, "
            f"enable_thinking={self.enable_thinking}, "
            f"timeout={self.timeout}s"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout,
        )
        logger.info("QwenGenerator initialized.")

    def _build_extra_body(self) -> Optional[Dict[str, Any]]:
        """
        构建请求的 extra_body 参数。

        Qwen3 模型通过 extra_body 中的 enable_thinking 字段控制思考模式。
        当 enable_thinking 为 False 时显式关闭，避免部分平台默认开启带来的
        额外延迟与 token 消耗。

        Returns:
            包含 enable_thinking 字段的字典；若模型不支持该参数可忽略。
        """
        return {"enable_thinking": self.enable_thinking}

    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """
        调用 Chat Completions API，返回单轮生成结果。

        若启用了思考模式（enable_thinking=True），模型返回的内容中包含
        <think>...</think> 推理块，本方法直接返回完整内容，由调用方按需解析。

        Args:
            prompt:     用户提示词
            max_tokens: 最大生成 token 数，0 表示使用默认值

        Returns:
            模型生成的文本内容，失败时返回空字符串
        """
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra_body=self._build_extra_body(),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen generation failed (model={self.model_name}): {e}")
            return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 0,
    ) -> str:
        """
        多轮对话接口，原样传入 messages（调用方自行包含 system 消息）。

        Args:
            messages:   消息列表，role 可为 system / user / assistant
            max_tokens: 最大生成 token 数，0 表示使用默认值

        Returns:
            模型生成的文本内容，失败时返回空字符串
        """
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra_body=self._build_extra_body(),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen chat failed (model={self.model_name}): {e}")
            return ""
