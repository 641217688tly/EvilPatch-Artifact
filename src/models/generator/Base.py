import importlib
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 600.0


class BaseGenerator(ABC):
    """
    文本生成模型的抽象基类，定义统一的生成器接口。

    所有具体的生成模型封装类（如 GPTGenerator）均应继承本类，
    并实现全部抽象方法。

    所有具体后端均通过 OpenAI 兼容 HTTP 接入：
    - 远程 API：OpenAI / Claude（通过代理） / DeepSeek 等，设置对应的 base_url
    - 本地推理：vLLM / Ollama，设置 base_url=http://localhost:PORT/v1

    Attributes:
        model_name:  模型名称（如 "gpt-5-mini"、"deepseek-coder"）
        temperature: 生成温度（0.0 为确定性输出，1.0 为高创造性）
        max_tokens:  默认最大生成 token 数
        base_url:    OpenAI 兼容端点（远程 API 或本地 vLLM/Ollama 地址）
        timeout:     单次 OpenAI 兼容 API 请求的超时时间（秒）
        retry_num:   上层应用层重试次数（SDK 内置重试之外的额外重试）
    """

    model_name: str
    temperature: float
    max_tokens: int
    base_url: str
    timeout: float
    retry_num: int

    @abstractmethod
    def __init__(self, config: Dict[str, Any], pool_index: int = 0) -> None:
        """
        初始化生成模型。

        Args:
            config:     配置字典，至少包含 model_name 键；支持 api_pool 列表
                        或单条 api_key/base_url 两种格式。
            pool_index: api_pool 中使用哪个条目（多线程时由线程编号决定）
        """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 0) -> str:
        """
        单轮生成接口，接受纯文本 prompt，返回模型生成结果。

        Args:
            prompt:     用户提示词（user 角色消息）
            max_tokens: 最大生成 token 数，0 表示使用初始化时的默认值

        Returns:
            模型生成的文本内容，失败时返回空字符串
        """

    @abstractmethod
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
        """

    def get_model_name(self, with_provider: bool = True) -> str:
        """
        返回模型名称。

        Args:
            with_provider: 若为 True（默认），返回含 Provider 前缀的完整名称，
                           格式为 "{provider}/{model_name}"（如 "ollama/llama3"）；
                           若为 False，仅返回斜杠后的模型名称部分（如 "llama3"）。
                           当 model_name 中不含斜杠时，两种模式均返回完整字符串。

        Returns:
            模型名称字符串。
        """
        if with_provider:
            return self.model_name
        return self.model_name.split("/", 1)[-1]

    @staticmethod
    def _resolve_timeout(config: Dict[str, Any]) -> float:
        """读取并校验正有限的请求超时时间（秒）。"""
        value = config.get("timeout", DEFAULT_TIMEOUT_SECONDS)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "generator timeout must be a positive number of seconds, "
                f"got {value!r}"
            )
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "generator timeout must be a finite positive number of seconds, "
                f"got {value!r}"
            )
        return timeout

    @staticmethod
    def _resolve_api_credentials(
        config: Dict[str, Any], pool_index: int = 0
    ) -> Tuple[str, str, str]:
        """
        从配置字典中解析 API 凭证及调用所用的模型名，兼容两种 YAML 格式。

        每个 api_pool 条目可单独指定 ``model_name`` 以覆盖顶层默认值，用于不同
        中转站对同一模型采用不同 ID 的场景（如硅基流动用
        ``"deepseek-ai/DeepSeek-V4-Flash"``、部分代理用 ``"deepseek-v4-flash"``）。

        格式一（多线程 api_pool，可逐条覆盖 model_name）：
            model_name: "deepseek-ai/DeepSeek-V4-Flash"   # 顶层默认
            api_pool:
              - api_key: "sk-xxx"
                base_url: "https://api.siliconflow.cn/v1"
              - api_key: "sk-yyy"
                base_url: "https://api.example.com/v1"
                model_name: "deepseek-v4-flash"           # 覆盖顶层

        格式二（单条凭证）：
            model_name: "deepseek-v4-flash"
            api_key:  "sk-xxx"
            base_url: "https://api.example.com/v1"

        Args:
            config:     配置字典
            pool_index: 在 api_pool 中使用哪个条目，超出范围时取模

        Returns:
            ``(api_key, base_url, model_name)`` 三元组。其中 ``model_name`` 已
            应用 per-entry 覆盖逻辑（条目无 ``model_name`` 时回退至顶层默认）。

        Raises:
            KeyError: 配置中缺少必要的 ``api_key`` / ``base_url`` 字段，或两种
                格式下均无 ``model_name`` 时抛出。
        """
        top_model_name = config.get("model_name")
        pool = config.get("api_pool")
        if pool:
            entry = pool[pool_index % len(pool)]
            model_name = entry.get("model_name") or top_model_name
            if not model_name:
                raise KeyError(
                    "model_name missing: provide either a top-level "
                    "'model_name' or one inside this api_pool entry."
                )
            return entry["api_key"], entry["base_url"], model_name
        if not top_model_name:
            raise KeyError("top-level 'model_name' is required when api_pool is absent.")
        return config["api_key"], config["base_url"], top_model_name


# 前缀 → (模块路径, 类名) 注册表，新增模型只需追加一行
# 当前后端均通过 OpenAI 兼容 HTTP 调用，由各子类处理模型专用参数。
# 若后续接入独立 SDK，只需将对应前缀重定向至新的子类。
_MODEL_NAME_REGISTRY: List[tuple] = [
    ("gpt-",      "src.models.generator.GPT",      "GPTGenerator"),
    ("deepseek-", "src.models.generator.DeepSeek", "DeepSeekGenerator"),
    ("qwen/",     "src.models.generator.Qwen",     "QwenGenerator"),
    ("glm-",      "src.models.generator.GLM",      "GLMGenerator"),
    ("zai-org/glm-", "src.models.generator.GLM", "GLMGenerator"),
    ("mimo-",     "src.models.generator.MiMo",     "MiMoGenerator"),
    ("mimo/mimo-",     "src.models.generator.MiMo",     "MiMoGenerator"),
]


def create_generator(
    config: Dict[str, Any], pool_index: int = 0
) -> "BaseGenerator":
    """
    根据配置字典中的 model_name 字段，自动实例化对应的 BaseGenerator 子类。

    匹配规则：将 model_name 转为小写后，按 _MODEL_NAME_REGISTRY 中的前缀顺序依次匹配，
    命中第一个匹配项后延迟导入对应模块并返回实例。

    Args:
        config:     配置字典，至少包含 model_name 键。
        pool_index: api_pool 中使用哪个条目（多线程时由线程编号决定）

    Returns:
        对应子类的实例（类型为 BaseGenerator）。

    Raises:
        ValueError: model_name 未命中任何注册前缀时抛出。

    Example:
        >>> cfg = load_yaml("configs/models/gpt-5-mini.yml")
        >>> generator = create_generator(cfg)              # 返回 GPTGenerator 实例
        >>> generator_t1 = create_generator(cfg, pool_index=1)  # 多线程使用不同 API 凭证
    """
    model_name = config.get("model_name", "").lower()
    for prefix, module_path, class_name in _MODEL_NAME_REGISTRY:
        if model_name.startswith(prefix):
            cls = getattr(importlib.import_module(module_path), class_name)
            return cls(config, pool_index)
    raise ValueError(
        f"No generator registered for model_name='{config.get('model_name')}'. "
        f"Add an entry to _MODEL_NAME_REGISTRY in src/models/generator/Base.py."
    )
