import logging
import math
from typing import Any, Dict, List

from openai import OpenAI

from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


class MiMoGenerator(BaseGenerator):
    """
    通过 OpenAI 兼容 API 调用 Xiaomi MiMo 文本生成模型。

    MiMo-V2.6-Flash 的按量付费地址为 https://api.xiaomimimo.com/v1。
    沿用项目的 max_tokens 配置/方法参数，请求时映射为 max_completion_tokens。
    仅返回最终回答 content，不将 reasoning_content 混入 APR 补丁。

    官方文档：
    https://mimo.mi.com/docs/zh-CN/quick-start/summary/first-api-call
    https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking
    """

    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        支持单条 api_key/base_url 或 api_pool，复用基类的模型名覆盖规则。

        可选配置：
            max_tokens: 默认 65536，包含思考与最终回答的总输出预算。
            thinking: {type: enabled/disabled}，默认 enabled（与官方一致）。
            temperature / top_p: 默认 1.0 / 0.95；仅关闭思考时可自定义。
            system_prompt: 默认空，仅 generate() 自动附加。
            timeout: 默认 600 秒，使用基类校验。
            retry_num: 默认 3，供上层读取；SDK 内置重试另为 3 次。

        采用非流式纯文本接口，不自动开启工具、联网搜索或批量推理服务。
        """
        self.max_tokens = config.get("max_tokens", 65536)
        self._validate_max_tokens(self.max_tokens)
        self.timeout = BaseGenerator._resolve_timeout(config)
        self.retry_num = config.get("retry_num", 3)
        self.system_prompt = config.get("system_prompt", "")
        if not isinstance(self.system_prompt, str):
            raise TypeError("MiMo system_prompt must be a string.")

        thinking = config.get("thinking", {})
        if not isinstance(thinking, dict):
            raise TypeError("MiMo thinking must be a mapping.")
        thinking_type = thinking.get("type", "enabled")
        if thinking_type not in ("enabled", "disabled"):
            raise ValueError("MiMo thinking.type must be enabled or disabled.")
        self.thinking = {"type": thinking_type}

        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.95)
        for name, value in (("temperature", self.temperature), ("top_p", self.top_p)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"MiMo {name} must be a finite non-negative number.")
        if self.top_p > 1:
            raise ValueError("MiMo top_p must not exceed 1.")
        if thinking_type == "enabled":
            if self.temperature != 1.0 or self.top_p != 0.95:
                logger.warning(
                    "MiMo thinking mode fixes temperature=1.0 and top_p=0.95; "
                    "custom sampling values are ignored. Set thinking.type=disabled "
                    "to use custom sampling."
                )
            self.temperature, self.top_p = 1.0, 0.95

        api_key, self.base_url, self.model_name = BaseGenerator._resolve_api_credentials(
            config, pool_index
        )
        logger.info(
            "Initializing MiMoGenerator: model=%s, base_url=%s, pool_index=%s, "
            "thinking=%s, timeout=%ss",
            self.model_name, self.base_url, pool_index, thinking_type, self.timeout,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_max_tokens(value: int) -> None:
        """输出预算必须是正整数；服务端还会校验模型输出及剩余上下文上限。"""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("MiMo max_tokens must be a positive integer.")

    def _build_extra_body(self) -> Dict[str, Any]:
        """thinking 不是 OpenAI 标准参数，必须通过 extra_body 传递。"""
        return {"thinking": dict(self.thinking)}

    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """单轮生成；max_tokens=0 使用配置值，失败时返回空字符串。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 0,
    ) -> str:
        """
        原样传入完整历史，不自动添加 system 消息或保存对话状态。

        保留调用方传入的历史 reasoning_content，但返回值只包含最终回答。
        API 错误或无最终回答时返回空字符串，供现有上层逻辑决定是否重试。
        """
        try:
            if type(max_tokens) is int and max_tokens == 0:
                max_tokens = self.max_tokens
            self._validate_max_tokens(max_tokens)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
                extra_body=self._build_extra_body(),
            )
            if not response.choices:
                logger.warning("MiMo returned no choices (model=%s).", self.model_name)
                return ""
            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning(
                    "MiMo output reached max_completion_tokens=%s (model=%s); "
                    "the final answer may be incomplete.",
                    max_tokens, self.model_name,
                )
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                logger.warning(
                    "MiMo returned no final content (model=%s); "
                    "reasoning_content is not used as a patch.",
                    self.model_name,
                )
                return ""
            return content
        except Exception as e:
            logger.error("MiMo chat failed (model=%s): %s", self.model_name, e)
            return ""
