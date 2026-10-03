import logging
import math
from typing import Any, Dict, List

from openai import OpenAI

from src.models.generator.Base import BaseGenerator

logger = logging.getLogger(__name__)


class GLMGenerator(BaseGenerator):
    """
    基于智谱 OpenAI 兼容 Chat Completions API 的 GLM 文本生成器。

    面向 GLM-5.3-Flash/FlashX：思考模式必须启用，reasoning_effort 支持
    low / high / max。接口只返回最终回答 content，不拼接 reasoning_content，
    避免推理文本被下游 APR 流程误当作补丁。

    官方文档：
    https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash
    """

    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        初始化生成器，沿用 BaseGenerator 的单凭证/API 池及模型名覆盖规则。

        除 model_name、api_key/base_url 或 api_pool 外，支持：
            temperature: 默认 1.0，范围 [0, 1]。
            top_p: 默认 0.95，范围 [0.01, 1]。
            max_tokens: 默认 65536，最大 131072；包含思考与回答的输出预算。
            reasoning_effort: 默认 max，可选 low / high / max。
            thinking: 默认 {type: enabled, clear_thinking: false}。
            system_prompt: 可选，仅 generate() 自动附加。
            timeout: 默认 600 秒，使用基类校验。
            retry_num: 默认 3，供上层重试逻辑读取；SDK 内置重试为 3 次。

        本类采用非流式请求，与项目现有返回字符串的生成器接口保持一致。
        """
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 65536)
        self.timeout = BaseGenerator._resolve_timeout(config)
        self.retry_num = config.get("retry_num", 3)
        self.system_prompt = config.get("system_prompt", "")
        self.reasoning_effort = config.get("reasoning_effort", "max")

        for name, value, minimum in (
            ("temperature", self.temperature, 0.0),
            ("top_p", self.top_p, 0.01),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not minimum <= value <= 1.0
            ):
                raise ValueError(f"GLM {name} must be a number in [{minimum}, 1].")
        self._validate_max_tokens(self.max_tokens)
        if self.reasoning_effort not in ("low", "high", "max"):
            raise ValueError("GLM-5.3-Flash reasoning_effort must be low, high or max.")

        thinking = config.get("thinking", {})
        if not isinstance(thinking, dict):
            raise TypeError("GLM thinking must be a mapping.")
        if thinking.get("type", "enabled") != "enabled":
            raise ValueError("GLM-5.3-Flash requires thinking.type='enabled'.")
        clear_thinking = thinking.get("clear_thinking", False)
        if not isinstance(clear_thinking, bool):
            raise TypeError("GLM thinking.clear_thinking must be a boolean.")
        self.thinking = {"type": "enabled", "clear_thinking": clear_thinking}

        api_key, self.base_url, self.model_name = BaseGenerator._resolve_api_credentials(
            config, pool_index
        )
        logger.info(
            "Initializing GLMGenerator: model=%s, base_url=%s, pool_index=%s, "
            "reasoning_effort=%s, timeout=%ss",
            self.model_name, self.base_url, pool_index,
            self.reasoning_effort, self.timeout,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_max_tokens(value: int) -> None:
        """防止将其他后端的超大输出预算直接发送至 GLM。"""
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 131072:
            raise ValueError("GLM max_tokens must be an integer in [1, 131072].")

    def _build_extra_body(self) -> Dict[str, Any]:
        """通过 extra_body 传递智谱参数，兼容不同版本的 OpenAI SDK。"""
        return {
            "thinking": dict(self.thinking),
            "reasoning_effort": self.reasoning_effort,
        }

    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """单轮文本生成；可选地附加 system_prompt，失败时返回空字符串。"""
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
        原样发送消息历史，返回最终回答；max_tokens=0 使用配置默认值。

        不自动保存对话状态。调用方若保留了历史 reasoning_content，可在
        assistant 消息中自行传入；本接口不会将它混入返回的补丁文本。
        API 错误或无最终回答时返回空字符串，供上层执行现有重试逻辑。
        """
        try:
            # 仅整数 0 是默认值标记，避免 False/None 等无效值被静默接受。
            if type(max_tokens) is int and max_tokens == 0:
                max_tokens = self.max_tokens
            self._validate_max_tokens(max_tokens)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
                extra_body=self._build_extra_body(),
            )
            if not response.choices:
                logger.warning("GLM returned no choices (model=%s).", self.model_name)
                return ""
            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning(
                    "GLM output reached max_tokens=%s (model=%s); "
                    "the final answer may be incomplete.",
                    max_tokens, self.model_name,
                )
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                logger.warning(
                    "GLM returned no final content (model=%s); "
                    "reasoning_content is not used as a patch.",
                    self.model_name,
                )
                return ""
            return content
        except Exception as e:
            logger.error("GLM chat failed (model=%s): %s", self.model_name, e)
            return ""
