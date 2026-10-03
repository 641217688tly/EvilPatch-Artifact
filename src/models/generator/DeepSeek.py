import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


class DeepSeekGenerator(BaseGenerator):
    """
    基于硅基流动（SiliconFlow）OpenAI 兼容 API 的 DeepSeek 文本生成器。

    通过硅基流动平台远程调用 DeepSeek 系列模型，例如:
        - deepseek-ai/DeepSeek-V4-Flash
        - deepseek-ai/DeepSeek-V3
        - 其他在硅基流动控制台可用的 DeepSeek 模型

    Attributes:
        model_name:     硅基流动平台上的模型 ID（如 "deepseek-ai/DeepSeek-V4-Flash"）
        base_url:       API 基础 URL，默认 https://api.siliconflow.cn/v1
        temperature:    生成温度
        max_tokens:     默认最大生成 token 数
        timeout:        单次 API 请求超时时间（秒）
        retry_num:      上层应用层重试次数
        system_prompt:  可选；非空时 generate() 会附带 system 消息
        client:         OpenAI 客户端实例
    """

    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        初始化 DeepSeek 文本生成器（硅基流动后端）。

        Args:
            config: 配置字典，支持两种格式:
                - model_name (str): 必需，硅基流动模型 ID
                - api_pool (list):  多线程，每项含 api_key，base_url 可选
                - api_key (str):    单条凭证，base_url 可选
                - temperature (float): 可选，默认 0.7
                - max_tokens (int):    可选，默认 4096
                - timeout (float):     可选，单次 API 请求超时秒数，默认 600
                - retry_num (int):     可选，默认 3
                - system_prompt (str): 可选，generate() 使用的 system 消息
            pool_index: api_pool 中使用哪个条目

        Example:
            >>> from src.utils.io import load_yaml
            >>> cfg = load_yaml("configs/models/deepseek-v4-flash.yml")
            >>> generator = DeepSeekGenerator(cfg)
            >>> response = generator.generate("分析这段代码的安全性...")
        """
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = BaseGenerator._resolve_timeout(config)
        self.retry_num = config.get("retry_num", 3)

        api_key, self.base_url, self.model_name = BaseGenerator._resolve_api_credentials(
            config, pool_index
        )

        logger.info(
            f"Initializing DeepSeekGenerator: model={self.model_name}, "
            f"base_url={self.base_url}, pool_index={pool_index}, "
            f"timeout={self.timeout}s"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout,
        )
        logger.info("DeepSeekGenerator initialized.")

    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """
        调用 Chat Completions API，返回单轮生成结果。

        若配置了 system_prompt，则发送 [system, user] 消息；否则仅发送 user 消息。

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
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek generation failed (model={self.model_name}): {e}")
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
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek chat failed (model={self.model_name}): {e}")
            return ""
