"""
损失函数与梯度计算模块。

实现 EvilPatch 中用于检索器对抗攻击的均匀期望相似度、动态权重、
Soft-Min 及均匀损失与 Soft-Min 混合损失：

1. **期望相似度最大范式** (expected_sim)
   仅使用正样本的无条件期望相似度损失：

       L(a) = - (1/|Q+|) * Σ_{q+ ∈ Q+} Sim(q+, a)

   直接最大化对抗样本与正样本的平均余弦相似度。该范式在向量空间中
   对投毒对象施加纯粹的"拉力（Pulling Force）"，驱使对抗样本 a 的嵌入
   无条件向正样本集合 Q+ 的质心靠拢。

2. **动态权重变体** (dynamic_weight=True)
   对上述范式添加基于裕度的指数缩放权重（Margin-based Exponential Scaling），
   实现难例挖掘（Hard Example Mining）机制：

       ω(q+, a) = exp(α * (τ_stop - Sim(q+, a)))
       ω̂(q+, a) = ω(q+) / (Σ_{q∈Q+} ω(q) + ε)
       L_weighted(a) = - Σ_{q+} stopgrad(ω̂(q+, a)) * Sim(q+, a)

   当某正样本的相似度远低于 τ_stop 时，其权重指数级放大；
   当相似度超过 τ_stop 时，权重迅速衰减，从而将梯度集中在最难优化的样本上。
   权重在当前次反向传播中视为固定系数，梯度不穿过权重计算路径。

3. **Soft-Min 损失** (softmin)
   使用平滑最小值近似重点优化相似度最低的代理正样本：

       L_softmin(a) = T * log((1/|Q+|) * Σ exp(-Sim(q+, a) / T))

   温度 T 越小，损失越接近 ``-min Sim(q+, a)``；温度越大，梯度分布越均匀。

4. **均匀损失 + Soft-Min** (hybrid)
   使用 λ 控制平均相似度目标与最难正样本目标的权衡：

       L_hybrid(a) = (1-λ) * L_expected(a) + λ * L_softmin(a)

梯度获取方式：
  通过 PyTorch 的 backward hook 机制，在词嵌入层注册 hook，
  一次 backward 即可获取所有 token 位置的梯度 ∇_{e_{t_i}} L(a)，
  避免逐位置计算的 m 次 backward 开销（优化点 #3）。
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_FUNCS = ("expected_sim", "softmin", "hybrid")
_LOSS_FUNC_ALIASES = {
    "soft_min": "softmin",
    "uniform_softmin": "hybrid",
    "expected_sim_softmin": "hybrid",
}


def normalize_loss_func_name(loss_func: str) -> str:
    """规范化并校验检索攻击损失函数名称。"""
    normalized = str(loss_func).strip().lower().replace("-", "_")
    normalized = _LOSS_FUNC_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_LOSS_FUNCS:
        raise ValueError(
            f"不支持的 loss_func={loss_func!r}，可选值为: "
            f"{', '.join(SUPPORTED_LOSS_FUNCS)}"
        )
    return normalized


def _validate_softmin_temperature(temperature: float) -> float:
    """校验 Soft-Min 温度并返回 float 值。"""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(
            f"softmin_temperature 必须大于 0，当前值为 {temperature}"
        )
    return temperature


def _validate_hybrid_weight(hybrid_softmin_weight: float) -> float:
    """校验混合损失中的 Soft-Min 权重 λ。"""
    hybrid_softmin_weight = float(hybrid_softmin_weight)
    if not 0.0 <= hybrid_softmin_weight <= 1.0:
        raise ValueError(
            "hybrid_softmin_weight 必须位于 [0, 1]，"
            f"当前值为 {hybrid_softmin_weight}"
        )
    return hybrid_softmin_weight


def _compute_cosine_similarities(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
) -> torch.Tensor:
    """返回单个对抗样本与所有代理正样本的余弦相似度。"""
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0)
    adv_norm = F.normalize(adv_embedding, dim=-1)
    pos_norm = F.normalize(pos_embeddings, dim=-1)
    return (pos_norm @ adv_norm.T).squeeze(-1)


class GradientStorage:
    """
    梯度存储器 —— 通过 backward hook 捕获词嵌入层的输出梯度。

    在梯度搜索算法中，需要计算损失函数对每个 token 嵌入的梯度 ∇_{e_{t_i}} L(a)。
    由于 PyTorch 默认不保留中间层梯度，需要通过 register_full_backward_hook
    在反向传播时显式捕获。

    用法::

        word_emb = model[0].auto_model.embeddings.word_embeddings
        grad_storage = GradientStorage(word_emb)

        loss.backward()

        # grad shape: (batch_size, seq_len, embed_dim)
        grad = grad_storage.get()
    """

    def __init__(self, module: torch.nn.Module):
        self._stored_gradient: Optional[torch.Tensor] = None
        module.register_full_backward_hook(self._hook)

    def _hook(self, module, grad_input, grad_output):
        """backward hook: 保存词嵌入层输出方向的梯度。"""
        self._stored_gradient = grad_output[0] # grad_output[0]代表取batch维度的第一个样本的梯度（实际上batch_size=1）

    def get(self) -> Optional[torch.Tensor]:
        """
        返回上一次 backward 捕获的梯度。

        Returns:
            梯度张量 ``(batch_size, seq_len, embed_dim)``，
            若尚未执行 backward 则返回 None。
        """
        return self._stored_gradient


def compute_expected_sim_loss(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    计算期望相似度最大范式的损失（纯 forward，不涉及梯度计算）。

    L(a) = - (1/|Q+|) Σ_{q+} Sim(q+, a)

    该损失仅使用正样本。通过取负值将
    "最大化平均余弦相似度"转化为"最小化损失"问题。

    Args:
        adv_embedding: 对抗样本嵌入，形状 ``(embed_dim,)`` 或 ``(1, embed_dim)``
        pos_embeddings: 正样本嵌入集合，形状 ``(|Q+|, embed_dim)``

    Returns:
        标量损失值（越小表示对抗样本与正样本越匹配）
    """
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0)  # (1, dim)

    adv_norm = F.normalize(adv_embedding, dim=-1)   # (1, dim)
    pos_norm = F.normalize(pos_embeddings, dim=-1)   # (|Q+|, dim)

    # cos_sim(q+, a) 对每个正样本: (|Q+|,)
    sims = (pos_norm @ adv_norm.T).squeeze(-1)

    loss = -sims.mean()
    return loss


def compute_softmin_loss(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    计算 Soft-Min 损失。

    令 ``s_i = Sim(q_i^+, a)``，则：

        L_softmin(a) = T * log((1/|Q+|) * Σ_i exp(-s_i / T))

    最小化该损失等价于最大化代理正样本相似度的平滑最小值。使用
    ``logsumexp`` 实现，以避免温度较小时指数上溢。

    Args:
        adv_embedding: 对抗样本嵌入 ``(embed_dim,)`` 或 ``(1, embed_dim)``
        pos_embeddings: 正样本嵌入 ``(|Q+|, embed_dim)``
        temperature: Soft-Min 温度 T，必须大于 0；越小越关注最低相似度样本

    Returns:
        标量损失值
    """
    temperature = _validate_softmin_temperature(temperature)
    sims = _compute_cosine_similarities(adv_embedding, pos_embeddings)
    if sims.numel() == 0:
        raise ValueError("Soft-Min 损失至少需要一个代理正样本")
    return temperature * (
        torch.logsumexp(-sims / temperature, dim=-1) - math.log(sims.numel())
    )


def compute_hybrid_expected_softmin_loss(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
    temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
) -> torch.Tensor:
    """
    计算“均匀期望相似度 + Soft-Min”混合损失。

        L_hybrid(a) = (1-λ) * L_expected(a) + λ * L_softmin(a)

    ``λ=0`` 时退化为均匀期望相似度损失，``λ=1`` 时退化为 Soft-Min。
    """
    hybrid_softmin_weight = _validate_hybrid_weight(hybrid_softmin_weight)
    expected_loss = compute_expected_sim_loss(adv_embedding, pos_embeddings)
    softmin_loss = compute_softmin_loss(
        adv_embedding,
        pos_embeddings,
        temperature=temperature,
    )
    return (
        (1.0 - hybrid_softmin_weight) * expected_loss
        + hybrid_softmin_weight * softmin_loss
    )


def compute_dynamic_weights(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
    tau_stop: float,
    alpha: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    计算归一化动态权重 ω̂(q+, a)。

    通过基于裕度的指数缩放函数（Margin-based Exponential Scaling），
    对"落后"于 τ_stop 阈值的正样本赋予更高权重，实现难例挖掘：

        ω(q+, a) = exp(α · (τ_stop - Sim(q+, a)))
        ω̂(q+, a) = ω(q+) / (Σ_{q∈Q+} ω(q) + ε)

    当 Sim(q+, a) < τ_stop 时，裕度为正，权重指数级放大；
    当 Sim(q+, a) > τ_stop 时，裕度为负，权重迅速衰减至 (0, 1)。

    注意：返回前对权重执行 ``detach()``，使其在当前次反向传播中
    仅作为由当前相似度确定的固定加权系数。

    Args:
        adv_embedding: 对抗样本嵌入，形状 ``(embed_dim,)`` 或 ``(1, embed_dim)``，
                       需已经过 L2 归一化
        pos_embeddings: 正样本嵌入集合，形状 ``(|Q+|, embed_dim)``
        tau_stop: 早停相似度阈值，用作权重计算的裕度参考线
        alpha: 缩放系数，控制权重对裕度的敏感程度
        epsilon: 归一化分母的数值稳定项

    Returns:
        已 detach 的归一化权重向量 ``(|Q+|,)``，满足 sum ≈ 1
    """
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0)  # (1, dim)

    adv_norm = F.normalize(adv_embedding, dim=-1)   # (1, dim)
    pos_norm = F.normalize(pos_embeddings, dim=-1)   # (|Q+|, dim)

    # Sim(q+, a): (|Q+|,)
    sims = (pos_norm @ adv_norm.T).squeeze(-1)

    # ω(q+, a) = exp(α · (τ_stop - Sim(q+, a)))
    margins = tau_stop - sims  # (|Q+|,)
    raw_weights = torch.exp(alpha * margins)  # (|Q+|,)

    # ω̂(q+, a) = ω(q+) / (Σ ω(q) + ε)
    normalized_weights = raw_weights / (raw_weights.sum() + epsilon)

    # 只保留当前计算出的权重数值，阻止梯度经由权重路径回传。
    return normalized_weights.detach()


def compute_weighted_expected_sim_loss(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
    tau_stop: float,
    alpha: float,
) -> torch.Tensor:
    """
    计算带动态权重的期望相似度损失（纯 forward）。

    L_weighted(a) = - Σ_{q+} stopgrad(ω̂(q+, a)) · Sim(q+, a)

    与 ``compute_expected_sim_loss`` 的唯一差异在于将均匀平均 ``mean()``
    替换为动态权重加权求和 ``sum(stopgrad(ω̂) · ...)``。
    权重每次前向传播都会重新计算，但在当前次反向传播中不参与求导。

    Args:
        adv_embedding: 对抗样本嵌入 ``(embed_dim,)`` 或 ``(1, embed_dim)``
        pos_embeddings: 正样本嵌入 ``(|Q+|, embed_dim)``
        tau_stop: 早停相似度阈值
        alpha: 缩放系数

    Returns:
        标量损失值
    """
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0) # (embed_dim,) -> (1, embed_dim)

    adv_norm = F.normalize(adv_embedding, dim=-1) # (1, embed_dim)
    pos_norm = F.normalize(pos_embeddings, dim=-1) # (|Q+|, embed_dim)

    sims = (pos_norm @ adv_norm.T).squeeze(-1)  # (1, embed_dim) · (|Q+|, embed_dim)^T = (|Q+|,)

    w_hat = compute_dynamic_weights(
        adv_embedding, pos_embeddings, tau_stop, alpha
    )

    loss = -(w_hat * sims).sum()
    return loss


def compute_retrieval_attack_loss(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
    loss_func: str = "expected_sim",
    dynamic_weight: bool = False,
    tau_stop: float = 0.70,
    alpha: float = 1.0,
    softmin_temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
) -> torch.Tensor:
    """按配置统一分发检索对抗攻击损失函数。"""
    loss_func = normalize_loss_func_name(loss_func)
    if loss_func == "expected_sim":
        if dynamic_weight:
            return compute_weighted_expected_sim_loss(
                adv_embedding,
                pos_embeddings,
                tau_stop=tau_stop,
                alpha=alpha,
            )
        return compute_expected_sim_loss(adv_embedding, pos_embeddings)
    if loss_func == "softmin":
        return compute_softmin_loss(
            adv_embedding,
            pos_embeddings,
            temperature=softmin_temperature,
        )
    return compute_hybrid_expected_softmin_loss(
        adv_embedding,
        pos_embeddings,
        temperature=softmin_temperature,
        hybrid_softmin_weight=hybrid_softmin_weight,
    )


def compute_retrieval_attack_loss_and_grad(
    embedder,
    token_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    grad_storage: GradientStorage,
    loss_func: str = "expected_sim",
    dynamic_weight: bool = False,
    tau_stop: float = 0.70,
    alpha: float = 1.0,
    softmin_temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """一次 forward + backward 计算所选损失、token 梯度和句子嵌入。"""
    embedder.model.zero_grad()
    adv_emb = embedder.embed_token_ids(token_ids, attention_mask)
    loss = compute_retrieval_attack_loss(
        adv_emb.squeeze(0),
        pos_embeddings,
        loss_func=loss_func,
        dynamic_weight=dynamic_weight,
        tau_stop=tau_stop,
        alpha=alpha,
        softmin_temperature=softmin_temperature,
        hybrid_softmin_weight=hybrid_softmin_weight,
    )
    loss.backward()

    raw_grad = grad_storage.get()
    if raw_grad is None:
        raise RuntimeError("未捕获到梯度，请检查 GradientStorage 是否正确注册到词嵌入层")

    seq_len = token_ids.shape[1]
    grad = raw_grad[0, :seq_len, :]
    return loss.item(), grad, adv_emb.detach()


def compute_expected_sim_loss_and_grad(
    embedder,
    token_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    grad_storage: GradientStorage,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    一次 forward + backward 计算期望相似度损失并获取词嵌入层梯度。

    Args:
        embedder: BaseEmbedder 实例
        token_ids: 当前对抗样本的 token ID，形状 ``(1, seq_len)``
        attention_mask: 注意力掩码，形状 ``(1, seq_len)``
        pos_embeddings: 正样本嵌入，形状 ``(|Q+|, dim)``
        grad_storage: 已注册到词嵌入层的梯度存储器

    Returns:
        (loss_value, grad, adv_emb) 三元组:
          - loss_value: 期望相似度损失标量值
          - grad: 词嵌入梯度，形状 ``(seq_len, embed_dim)``
          - adv_emb: 当前对抗样本的句子嵌入，形状 ``(1, embed_dim)``，已 detach
    """
    embedder.model.zero_grad()

    adv_emb = embedder.embed_token_ids(token_ids, attention_mask)
    loss = compute_expected_sim_loss(adv_emb.squeeze(0), pos_embeddings)
    loss.backward()

    raw_grad = grad_storage.get()
    if raw_grad is None:
        raise RuntimeError("未捕获到梯度，请检查 GradientStorage 是否正确注册到词嵌入层")

    seq_len = token_ids.shape[1]
    grad = raw_grad[0, :seq_len, :]  # (seq_len, embed_dim)

    return loss.item(), grad, adv_emb.detach()


def compute_weighted_expected_sim_loss_and_grad(
    embedder,
    token_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    grad_storage: GradientStorage,
    tau_stop: float,
    alpha: float,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    一次 forward + backward 计算带动态权重的期望相似度损失并获取词嵌入层梯度。

    Args:
        embedder: BaseEmbedder 实例
        token_ids: 当前对抗样本的 token ID ``(1, seq_len)``
        attention_mask: 注意力掩码 ``(1, seq_len)``
        pos_embeddings: 正样本嵌入 ``(|Q+|, dim)``
        grad_storage: 已注册到词嵌入层的梯度存储器
        tau_stop: 早停相似度阈值
        alpha: 缩放系数

    Returns:
        (loss_value, grad, adv_emb) 三元组:
          - loss_value: 带动态权重的期望相似度损失标量值
          - grad: 词嵌入梯度，形状 ``(seq_len, embed_dim)``
          - adv_emb: 当前对抗样本的句子嵌入，形状 ``(1, embed_dim)``，已 detach
    """
    embedder.model.zero_grad()

    adv_emb = embedder.embed_token_ids(token_ids, attention_mask)
    loss = compute_weighted_expected_sim_loss(
        adv_emb.squeeze(0), pos_embeddings, tau_stop, alpha,
    )
    loss.backward()

    raw_grad = grad_storage.get()
    if raw_grad is None:
        raise RuntimeError("未捕获到梯度，请检查 GradientStorage 是否正确注册到词嵌入层")

    seq_len = token_ids.shape[1] # token_ids的形状是(batch_size=1, seq_len), shape[1]是当前这条序列的token长度
    grad = raw_grad[0, :seq_len, :] # raw_grad的形状是(batch_size, seq_len, embed_dim), raw_grad[0, :seq_len, :]则是取batch中第0条数据的前seq_len个位置的token的嵌入向量

    return loss.item(), grad, adv_emb.detach()


def compute_avg_similarity(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
) -> float:
    """
    计算对抗样本与所有正样本的平均余弦相似度。

    Args:
        adv_embedding: ``(embed_dim,)`` 或 ``(1, embed_dim)``
        pos_embeddings: ``(|Q+|, embed_dim)``

    Returns:
        平均余弦相似度标量值
    """
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0)
    adv_norm = F.normalize(adv_embedding, dim=-1)
    pos_norm = F.normalize(pos_embeddings, dim=-1)
    sims = (pos_norm @ adv_norm.T).squeeze(-1)
    return sims.mean().item()


def compute_per_sample_similarity(
    adv_embedding: torch.Tensor,
    pos_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    计算对抗样本与每个正样本的余弦相似度。

    Args:
        adv_embedding: ``(embed_dim,)`` 或 ``(1, embed_dim)``
        pos_embeddings: ``(|Q+|, embed_dim)``

    Returns:
        每个正样本的相似度向量 ``(|Q+|,)``
    """
    if adv_embedding.dim() == 1:
        adv_embedding = adv_embedding.unsqueeze(0)
    adv_norm = F.normalize(adv_embedding, dim=-1)
    pos_norm = F.normalize(pos_embeddings, dim=-1)
    return (pos_norm @ adv_norm.T).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
#  批量前向传播与批量损失计算（用于候选评估加速）
# ─────────────────────────────────────────────────────────────────

def compute_expected_sim_loss_batch(
    adv_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    批量计算期望相似度损失（纯 forward，无梯度）。

    对 B 个候选嵌入同时计算 ``- (1/|Q+|) Σ Sim(q+, a)``。

    Args:
        adv_embeddings: 批量对抗样本嵌入，形状 ``(B, embed_dim)``（已 L2 归一化）
        pos_embeddings: 正样本嵌入集合，形状 ``(|Q+|, embed_dim)``

    Returns:
        每个候选的损失向量，形状 ``(B,)``
    """
    adv_norm = F.normalize(adv_embeddings, dim=-1)   # (B, dim)
    pos_norm = F.normalize(pos_embeddings, dim=-1)   # (|Q+|, dim)

    # sims[b, q+] = cos_sim(q+, a_b)  →  (B, |Q+|)
    sims = adv_norm @ pos_norm.T                     # (B, |Q+|)

    # loss[b] = -mean over q+  →  (B,)
    loss = -sims.mean(dim=-1)
    return loss


def compute_softmin_loss_batch(
    adv_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """批量计算 Soft-Min 损失，返回形状 ``(B,)`` 的损失向量。"""
    temperature = _validate_softmin_temperature(temperature)
    if pos_embeddings.shape[0] == 0:
        raise ValueError("Soft-Min 损失至少需要一个代理正样本")
    adv_norm = F.normalize(adv_embeddings, dim=-1)
    pos_norm = F.normalize(pos_embeddings, dim=-1)
    sims = adv_norm @ pos_norm.T
    return temperature * (
        torch.logsumexp(-sims / temperature, dim=-1)
        - math.log(pos_embeddings.shape[0])
    )


def compute_hybrid_expected_softmin_loss_batch(
    adv_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
) -> torch.Tensor:
    """批量计算“均匀期望相似度 + Soft-Min”混合损失。"""
    hybrid_softmin_weight = _validate_hybrid_weight(hybrid_softmin_weight)
    expected_loss = compute_expected_sim_loss_batch(adv_embeddings, pos_embeddings)
    softmin_loss = compute_softmin_loss_batch(
        adv_embeddings,
        pos_embeddings,
        temperature=temperature,
    )
    return (
        (1.0 - hybrid_softmin_weight) * expected_loss
        + hybrid_softmin_weight * softmin_loss
    )


def compute_weighted_expected_sim_loss_batch(
    adv_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    tau_stop: float,
    alpha: float,
) -> torch.Tensor:
    """
    批量计算带动态权重的期望相似度损失（纯 forward，无梯度）。

    对 B 个候选嵌入同时计算加权期望相似度损失。

    Args:
        adv_embeddings: 批量对抗样本嵌入，形状 ``(B, embed_dim)``（已 L2 归一化）
        pos_embeddings: 正样本嵌入集合，形状 ``(|Q+|, embed_dim)``
        tau_stop: 早停阈值
        alpha: 缩放系数

    Returns:
        每个候选的损失向量，形状 ``(B,)``
    """
    adv_norm = F.normalize(adv_embeddings, dim=-1)   # (B, dim)
    pos_norm = F.normalize(pos_embeddings, dim=-1)   # (|Q+|, dim)

    sims = adv_norm @ pos_norm.T                     # (B, |Q+|)

    margins = tau_stop - sims                        # (B, |Q+|)
    raw_weights = torch.exp(alpha * margins)         # (B, |Q+|)
    w_hat = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-8)  # (B, |Q+|)
    w_hat = w_hat.detach()

    # loss[b] = -sum over q+ of (w_hat[b, q+] * sims[b, q+])  →  (B,)
    loss = -(w_hat * sims).sum(dim=-1)
    return loss


def compute_retrieval_attack_loss_batch(
    adv_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    loss_func: str = "expected_sim",
    dynamic_weight: bool = False,
    tau_stop: float = 0.70,
    alpha: float = 1.0,
    softmin_temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
) -> torch.Tensor:
    """按配置批量分发检索对抗攻击损失函数。"""
    loss_func = normalize_loss_func_name(loss_func)
    if loss_func == "expected_sim":
        if dynamic_weight:
            return compute_weighted_expected_sim_loss_batch(
                adv_embeddings,
                pos_embeddings,
                tau_stop=tau_stop,
                alpha=alpha,
            )
        return compute_expected_sim_loss_batch(adv_embeddings, pos_embeddings)
    if loss_func == "softmin":
        return compute_softmin_loss_batch(
            adv_embeddings,
            pos_embeddings,
            temperature=softmin_temperature,
        )
    return compute_hybrid_expected_softmin_loss_batch(
        adv_embeddings,
        pos_embeddings,
        temperature=softmin_temperature,
        hybrid_softmin_weight=hybrid_softmin_weight,
    )
