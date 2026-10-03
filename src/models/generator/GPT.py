import logging
from typing import Any, Dict, List

from openai import OpenAI

from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


class GPTGenerator(BaseGenerator):
    """
    基于 OpenAI 兼容 API 的文本生成器封装类。

    支持所有遵循 OpenAI Chat Completions API 规范的模型，包括:
        - gpt-5-mini, gpt-4o 等 OpenAI 官方模型
        - 通过代理接口（如 bianxie.ai）访问的第三方兼容模型
        - 本地部署的 vLLM / Ollama 服务（设置 base_url=http://localhost:PORT/v1）
        - 通过代理接口访问的 Claude / DeepSeek / LLaMA 等模型

    Attributes:
        model_name:  模型名称（如 "gpt-5-mini"）
        base_url:    API 基础 URL
        temperature: 生成温度（0.0 为确定性输出，1.0 为高创造性）
        max_tokens:  默认最大生成 token 数
        timeout:     单次 API 请求超时时间（秒）
        retry_num:   上层应用层重试次数（SDK 内置重试之外的额外重试）
        client:      OpenAI 客户端实例
    """

    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        初始化 GPT 文本生成器。

        Args:
            config: 配置字典，支持两种格式:
                新格式（多线程）:
                - model_name (str): 必需，模型名称
                - api_pool (list):  必需，包含 {api_key, base_url} 的字典列表
                - temperature (float): 可选，默认 0.7
                - max_tokens (int):    可选，默认 4096
                - timeout (float):     可选，单次 API 请求超时秒数，默认 600
                - retry_num (int):     可选，默认 3
                旧格式（向后兼容）:
                - model_name (str): 必需
                - api_key (str):    必需
                - base_url (str):   必需
            pool_index: api_pool 中使用哪个条目（多线程时由线程编号决定）

        Example:
            >>> import yaml
            >>> with open("configs/models/gpt-5-mini.yml") as f:
            ...     config = yaml.safe_load(f)
            >>> generator = GPTGenerator(config, pool_index=0)
            >>> response = generator.generate("分析这段代码的安全性...")
        """
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 4096)
        self.timeout     = BaseGenerator._resolve_timeout(config)
        self.retry_num   = config.get("retry_num", 3)

        api_key, self.base_url, self.model_name = BaseGenerator._resolve_api_credentials(
            config, pool_index
        )

        logger.info(
            f"Initializing GPTGenerator: model={self.model_name}, "
            f"base_url={self.base_url}, pool_index={pool_index}, "
            f"timeout={self.timeout}s"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout,
        )
        logger.info("GPTGenerator initialized.")

    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """
        调用 Chat Completions API，返回单轮生成结果。

        Args:
            prompt:     用户提示词（user 角色消息）
            max_tokens: 最大生成 token 数，0 表示使用初始化时的默认值

        Returns:
            模型生成的文本内容，失败时返回空字符串

        Example:
            >>> response = generator.generate("解释 CWE-200 漏洞的注入机制", max_tokens=512)
            >>> print(response)
        """
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT generation failed (model={self.model_name}): {e}")
            return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 0,
    ) -> str:
        """
        多轮对话接口，支持传入完整的消息历史。

        Args:
            messages:   消息列表，每个元素为 {"role": "...", "content": "..."}
                        role 可为 "system" / "user" / "assistant"
            max_tokens: 最大生成 token 数，0 表示使用初始化时的默认值

        Returns:
            模型生成的文本内容，失败时返回空字符串

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a security expert."},
            ...     {"role": "user",   "content": "Explain buffer overflow."},
            ... ]
            >>> response = generator.chat(messages, max_tokens=512)
        """
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT chat failed (model={self.model_name}): {e}")
            return ""
