#!/usr/bin/env python3
"""
EvilPatch 检索器对抗攻击主入口（方案四 / Ultra）。

与 run_attack.py 的核心区别：
  - 输入/输出为单一的 poison_targets_set.json（嵌套 dict 格式）
  - 正样本直接从 entry["retrieval_attack"]["positive_samples"] 读取
  - 攻击结果原地写回 entry["retrieval_attack"]（与 vics_vinj.py 一致的模式）
  - 对抗优化器日志路径: {log_dir}/{日期}/{CWE-XX}_{id}_{OPTIMIZER}.txt（Naive 不写日志文件）

支持五种检索投毒策略（通过 attack.optimizer 配置切换）：
  - naive: 不进行任何对抗优化，原样使用 buggy_code
  - aggs:  近似贪婪梯度搜索（Approximate Greedy Gradient Search，改进版：整篇文档就地替换 + 安全词表）
  - abgs:  近似束梯度搜索（Approximate Beam Gradient Search，改进版：整篇文档就地替换 + 安全词表）
  - pabs:  位置感知束搜索（Position-Aware Beam Search，基线：插入定长对抗序列 + 位置搜索）
  - aggd:  近似贪婪梯度下降（Approximate Greedy Gradient Descent，原版论文算法：首部插入定长对抗序列）

使用示例:
    # 更新clash
    clashsub update

    # tmux工具下载
    sudo apt update
    sudo apt install tmux

    # ssh进程挂起
    tmux new -s EvilPatch # 之后ctrl+B，单点D键退出窗口
    
    # ssh进程重连
    tmux attach -t EvilPatch
    
    # ssh进程终止
    tmux kill-session -t EvilPatch # 或者tmux attach -t EvilPatch重连后ctrl+D终止

    # 避免HF连接错误:
    export HF_ENDPOINT=https://hf-mirror.com # unset HF_ENDPOINT( 可重置环境设置
    
    # 使用默认配置文件运行
    python -m src.pipeline.retrieval_attack.run_attack_ultra
    python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/naive.yml
    python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/aggs.yml
    python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/abgs.yml
    python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/pabs.yml
    python -m src.pipeline.retrieval_attack.run_attack_ultra --config configs/attack/retrieval/aggd.yml
"""

import argparse
from copy import deepcopy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _build_runtime_retriever_config(
    attack_config: Dict[str, Any],
    retriever_config: Dict[str, Any],
    job_label: str,
) -> Dict[str, Any]:
    """将攻击任务中的设备覆盖合并到检索器运行时配置。"""
    runtime_config = deepcopy(retriever_config)
    model_config = attack_config.get("model", {})
    for key in ("device", "device_pool"):
        if key in model_config:
            runtime_config[key] = deepcopy(model_config[key])

    selection_config = deepcopy(runtime_config.get("device_selection", {}) or {})
    selection_config.update(
        deepcopy(model_config.get("device_selection", {}) or {})
    )
    if selection_config:
        selection_config["owner_label"] = job_label
        runtime_config["device_selection"] = selection_config
    return runtime_config


def _resolve_project_root() -> Path:
    """向上查找包含 configs/ 的项目根目录并加入 sys.path。"""
    root = Path(os.path.abspath("")).resolve()
    while not (root / "configs").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _format_sample_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将样本列表规范化为输出格式（id/source/buggy_code/fixed_code），剥离 embedding。"""
    result = []
    for item in items:
        raw_id = item.get("id")
        try:
            normalized_id = int(raw_id) if raw_id is not None else None
        except (ValueError, TypeError):
            normalized_id = raw_id
        result.append({
            "id": normalized_id,
            "source": item.get("source", ""),
            "buggy_code": item.get("buggy_code", ""),
            "fixed_code": item.get("fixed_code", ""),
        })
    return result


def _atomic_save_target_data(
    target_data: Dict[str, list], target_file_path: str
) -> None:
    """Atomically persist the complete target dataset beside the destination."""
    destination = Path(target_file_path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(target_data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run_naive_attack(config: Dict[str, Any]) -> int:
    """Run the model-free baseline and persist each completed item atomically."""
    from src.pipeline.retrieval_attack.optimizer.naive_optimizer import (
        naive_optimize,
    )

    data_cfg = config.get("data", {}) or {}
    sampling_cfg = config.get("sampling", {}) or {}
    attack_cfg = config.get("attack", {}) or {}
    target_file_path = data_cfg.get("target_file_path")
    if not target_file_path:
        logger.error("data.target_file_path is required for the naive optimizer")
        return 1

    process_cwe = sampling_cfg.get("process_cwe", [])
    if not isinstance(process_cwe, list) or not process_cwe:
        logger.error("sampling.process_cwe must be a non-empty list")
        return 1

    try:
        with open(target_file_path, "r", encoding="utf-8") as handle:
            target_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load target file %s: %s", target_file_path, exc)
        return 1
    if not isinstance(target_data, dict):
        logger.error("Target file root must be a JSON object keyed by CWE")
        return 1

    pending: List[tuple] = []
    completed_count = 0
    for cwe_id in process_cwe:
        entries = target_data.get(cwe_id, [])
        if not entries:
            logger.warning("[%s] No target entries found", cwe_id)
            continue
        if not isinstance(entries, list):
            logger.error("[%s] Target entries must be a list", cwe_id)
            return 1
        for index, entry in enumerate(entries):
            retrieval_attack = (
                entry.get("retrieval_attack") if isinstance(entry, dict) else None
            )
            if isinstance(retrieval_attack, dict) and retrieval_attack.get(
                "poisoned_buggy_code"
            ):
                completed_count += 1
                continue
            pending.append((cwe_id, index, entry))

    if not pending:
        logger.info(
            "Naive baseline has no pending entries (already completed: %d)",
            completed_count,
        )
        return 0

    logger.info(
        "Naive baseline: pending=%d, already completed=%d",
        len(pending),
        completed_count,
    )
    success_count = 0
    fail_count = 0
    attack_snapshot = {
        key: deepcopy(value)
        for key, value in attack_cfg.items()
        if key != "frozen_token_dict"
    }

    for number, (cwe_id, _index, entry) in enumerate(pending, 1):
        had_retrieval_attack = (
            isinstance(entry, dict) and "retrieval_attack" in entry
        )
        previous_retrieval_attack = (
            deepcopy(entry.get("retrieval_attack"))
            if had_retrieval_attack
            else None
        )
        target_id = "unknown"
        if isinstance(entry, dict) and isinstance(entry.get("entity"), dict):
            target_id = entry["entity"].get("id", "unknown")
        try:
            if not isinstance(entry, dict):
                raise TypeError("target entry must be an object")
            entity = entry.get("entity")
            if not isinstance(entity, dict):
                raise TypeError("target entry must contain an entity object")

            result = naive_optimize(entity.get("buggy_code"))
            retrieval_attack = entry.get("retrieval_attack")
            if retrieval_attack is None:
                retrieval_attack = {}
                entry["retrieval_attack"] = retrieval_attack
            if not isinstance(retrieval_attack, dict):
                raise TypeError("retrieval_attack must be an object")

            formatted_positive_samples = None
            if "positive_samples" in retrieval_attack:
                positive_samples = retrieval_attack["positive_samples"]
                if not isinstance(positive_samples, list):
                    raise TypeError("retrieval_attack.positive_samples must be a list")
                formatted_positive_samples = _format_sample_list(positive_samples)

            retrieval_attack["poisoned_buggy_code"] = result["adv_text"]
            retrieval_attack["configs"] = {
                "data": target_file_path,
                "sampling": deepcopy(sampling_cfg),
                "attack": deepcopy(attack_snapshot),
            }
            if formatted_positive_samples is not None:
                retrieval_attack["positive_samples"] = formatted_positive_samples

            _atomic_save_target_data(target_data, target_file_path)
            success_count += 1
            logger.info(
                "[%d/%d] Naive target saved: CWE=%s id=%s",
                number,
                len(pending),
                cwe_id,
                target_id,
            )
        except Exception as exc:
            if isinstance(entry, dict):
                if had_retrieval_attack:
                    entry["retrieval_attack"] = previous_retrieval_attack
                else:
                    entry.pop("retrieval_attack", None)
            fail_count += 1
            logger.error(
                "[%d/%d] Naive target failed: CWE=%s id=%s: %s",
                number,
                len(pending),
                cwe_id,
                target_id,
                exc,
            )

    logger.info(
        "Naive baseline finished: success=%d, failed=%d, total=%d",
        success_count,
        fail_count,
        len(pending),
    )
    logger.info("Target file: %s", target_file_path)
    return 0 if fail_count == 0 else 1


def write_attack_log(
    log_dir: str,
    date_str: str,
    cwe_id: str,
    target_id: Any,
    result: Dict[str, Any],
    config: Dict[str, Any],
    optimizer_type: str = "aggd",
) -> str:
    """
    将对抗攻击日志写入文件。

    日志文件路径: {log_dir}/{date_str}/{cwe_id}_{target_id}_{OPTIMIZER}.txt

    Returns:
        日志文件的完整路径
    """
    log_path = Path(log_dir) / date_str
    log_path.mkdir(parents=True, exist_ok=True)

    safe_id = str(target_id).replace("/", "_").replace("\\", "_")
    optimizer_suffix = optimizer_type.upper()
    filename = f"{cwe_id}_{safe_id}_{optimizer_suffix}.txt"
    filepath = log_path / filename

    optimizer_labels = {
        "aggs": "AGGS 近似贪婪梯度搜索（改进·整篇就地）",
        "abgs": "ABGS 近似束梯度搜索（精英保留·独立 depth）",
        "pabs": "PABS 位置感知束搜索（基线·插入）",
        "aggd": "AGGD 近似贪婪梯度下降（原版·插入）",
    }
    optimizer_label = optimizer_labels.get(
        optimizer_type, f"{optimizer_type.upper()} 优化"
    )

    resolved_attack_cfg = result.get(
        "resolved_attack_config", config.get("attack", {})
    )
    sampling_cfg = config.get("sampling", {})
    configured_max_mutable = sampling_cfg.get("max_condidate_num", 6000)
    effective_max_mutable = min(
        configured_max_mutable,
        resolved_attack_cfg.get("n", config.get("attack", {}).get("n", 6000)),
    )
    runtime_model_cfg = config.get("model", {})
    reservation_info = runtime_model_cfg.get("device_reservation", {})
    lines = [
        f"{'=' * 60}",
        f"EvilPatch {optimizer_label} Retrieval Attack Log",
        f"{'=' * 60}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"CWE: {cwe_id}",
        f"Target ID: {target_id}",
        f"Optimizer: {optimizer_type.upper()}",
        f"Total Iterations: {result['total_iterations']}",
        f"Early Stopped: {result['early_stopped']}",
        f"Stop Reason: {result.get('stop_reason', 'unknown')}",
        f"Best Iteration: {result.get('best_iteration', '?')}",
        f"Final Loss: {result['final_loss']:.6f}",
        f"Final Avg Sim to Q+: {result['final_avg_sim']:.6f}",
        "",
        "── Config ──",
        f"  selected_device: {runtime_model_cfg.get('selected_device', '?')}",
        f"  physical_gpu_index: {reservation_info.get('physical_gpu_index', '?')}",
        f"  gpu_uuid: {reservation_info.get('gpu_uuid', '?')}",
        f"  device_selection: {runtime_model_cfg.get('resolved_device_selection', '?')}",
        f"  reservation_pid: {reservation_info.get('reservation_pid', '?')}",
        f"  reservation_waited_seconds: {reservation_info.get('reservation_waited_seconds', '?')}",
        f"  min_mutable_tokens: {sampling_cfg.get('min_condidate_num', 50)}",
        f"  max_mutable_tokens: {configured_max_mutable}",
        f"  effective_max_mutable_tokens: {effective_max_mutable}",
        f"  initial_mutable_tokens: {resolved_attack_cfg.get('initial_mutable_tokens', '?')}",
        f"  N (max iter): {resolved_attack_cfg.get('N', '?')}",
        f"  n (candidate size): {resolved_attack_cfg.get('n', '?')}",
        f"  dynamic_beam_width: {result.get('dynamic_beam_width', resolved_attack_cfg.get('dynamic_beam_width', '?'))}",
        f"  per_beam_budget: {result.get('per_beam_budget', resolved_attack_cfg.get('per_beam_budget', '?'))}",
        f"  elite_num: {resolved_attack_cfg.get('elite_num', '?')}",
        f"  effective_elite_num: {result.get('effective_elite_num', '?')}",
        f"  global_basic_budget: {resolved_attack_cfg.get('global_basic_budget', '?')}",
        f"  patience: {resolved_attack_cfg.get('patience', '?')}",
        f"  patience_loss_delta: {resolved_attack_cfg.get('patience_loss_delta', '?')}",
        f"  patience_similarity_delta: {resolved_attack_cfg.get('patience_similarity_delta', '?')}",
        f"  tau_stop: {resolved_attack_cfg.get('tau_stop', '?')}",
        f"  loss_func: {resolved_attack_cfg.get('loss_func', '?')}",
        f"  dynamic_weight: {resolved_attack_cfg.get('dynamic_weight', '?')}",
        f"  scaling_factor: {resolved_attack_cfg.get('scaling_factor', '?')}",
        f"  softmin_temperature: {resolved_attack_cfg.get('softmin_temperature', '?')}",
        f"  hybrid_softmin_weight: {resolved_attack_cfg.get('hybrid_softmin_weight', '?')}",
        f"  replacement_score: {resolved_attack_cfg.get('replacement_score', '?')}",
        f"  cache_strategy: {resolved_attack_cfg.get('cache_strategy', '?')}",
        "",
        "── Search Summary ──",
        f"  patience_counter: {result.get('patience_counter', '?')}",
        f"  max_depth_reached: {result.get('max_depth_reached', '?')}",
        f"  final_beam_depths: {result.get('final_beam_depths', '?')}",
        f"  total_candidates_generated: {result.get('total_candidates_generated', '?')}",
        f"  total_candidates_evaluated: {result.get('total_candidates_evaluated', '?')}",
        f"  total_fresh_evaluated: {result.get('total_fresh_evaluated', '?')}",
        f"  total_cache_replayed: {result.get('total_cache_replayed', '?')}",
        f"  total_cache_hits: {result.get('total_cache_hits', '?')}",
        f"  total_global_scanned: {result.get('total_global_scanned', '?')}",
        f"  total_sequence_duplicates: {result.get('total_sequence_duplicates', '?')}",
        f"  final_global_offsets: {result.get('final_global_offsets', '?')}",
        f"  best_avg_similarity: {result.get('best_avg_similarity', '?')}",
        f"  best_avg_similarity_iteration: {result.get('best_avg_similarity_iteration', '?')}",
        "",
        "── Final Adversarial Text ──",
        result["adv_text"],
        "",
        f"{'=' * 60}",
        "Iteration Details",
        f"{'=' * 60}",
        "",
    ]

    lines.extend(result.get("iteration_logs", []))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"攻击日志已保存: {filepath}")
    return str(filepath)


def main() -> int:
    _resolve_project_root()

    from src.utils.io import load_yaml
    from src.utils.log import setup_logging

    parser = argparse.ArgumentParser(
        description="EvilPatch 检索器对抗攻击 Ultra（方案四，所有参数通过配置文件控制）",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/attack/retrieval/archive/abgs.yml",
        help="配置文件路径（默认: configs/attack/retrieval/archive/abgs.yml）",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ──
    config = load_yaml(args.config)

    log_cfg = config.get("logging", {})
    verbose = log_cfg.get("verbose", False)
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    program_start = datetime.now()
    date_str = f"{program_start.year}-{program_start.month}-{program_start.day}"

    logger.info("=" * 60)
    logger.info("EvilPatch 检索器对抗攻击（Ultra / 方案四）")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"启动时间: {program_start.isoformat()}")

    optimizer_type = config.get("attack", {}).get("optimizer", "aggs")
    if optimizer_type == "naive":
        return _run_naive_attack(config)

    import torch

    from src.models.retriever.Base import create_embedder
    from src.pipeline.retrieval_attack.optimizer.aggs_optimizer import aggs_optimize
    from src.pipeline.retrieval_attack.optimizer.aggd_optimizer import aggd_optimize
    from src.pipeline.retrieval_attack.optimizer.abgs_optimizer import abgs_optimize
    from src.pipeline.retrieval_attack.optimizer.pabs_optimizer import pabs_optimize
    from src.pipeline.retrieval_attack.loss_func import GradientStorage
    from src.pipeline.retrieval_attack.vocab_filter import (
        build_safe_embedding_matrix,
        build_safe_vocab,
        get_word_embeddings,
    )
    from src.pipeline.retrieval_attack.data_prep_ultra import (
        build_pending_entries,
        prepare_attack_data,
    )
    from src.utils.model import probe_max_batch_size
    from src.utils.device import get_device_reservation_info

    seed = config.get("attack", {}).get("seed", 42)
    torch.manual_seed(seed)

    # ── 2. 初始化检索器模型 ──
    logger.info("正在加载检索器模型...")
    retriever_config_path = config.get("model", {}).get(
        "retriever_config", "configs/models/harrier-oss-v1-0.6b.yml"
    )
    retriever_config = _build_runtime_retriever_config(
        config,
        load_yaml(retriever_config_path),
        job_label=args.config,
    )
    embedder = create_embedder(retriever_config)
    st_model = embedder.model
    st_model.eval()

    default_batch_size = retriever_config.get("batch_size", 8)
    if "model" not in config:
        config["model"] = {}
    config["model"]["batch_size"] = default_batch_size
    device_reservation = get_device_reservation_info(embedder.device)
    config["model"]["selected_device"] = embedder.device
    config["model"]["device_reservation"] = device_reservation
    config["model"]["resolved_device_selection"] = deepcopy(
        retriever_config.get("device_selection", {})
    )
    if device_reservation:
        logger.info(
            "GPU 调度结果: device=%s, physical=%s, uuid=%s, "
            "pid=%s, waited=%ss",
            embedder.device,
            device_reservation.get("physical_gpu_index"),
            device_reservation.get("gpu_uuid"),
            device_reservation.get("reservation_pid"),
            device_reservation.get("reservation_waited_seconds"),
        )
    else:
        logger.info("GPU 调度结果: device=%s（无独占租约）", embedder.device)
    logger.info(f"候选评估批处理大小: batch_size={default_batch_size}")

    tokenizer = st_model.tokenizer
    logger.info(f"模型加载完成: {embedder.model_name}, 词表大小={tokenizer.vocab_size}")

    prefix_ids = embedder.get_instruct_tids(side="document")
    if prefix_ids:
        logger.info(f"Prompt prefix: {len(prefix_ids)} tokens（前向传播时拼接，优化时冻结）")
    else:
        logger.info("无 prompt prefix（模型不使用 instruction prefix）")

    # ── 3. 构建安全候选词表 V_safe ──
    logger.info("正在构建安全候选词表 V_safe...")
    use_safe_vocab = config.get("attack", {}).get("use_safe_vocab", True)
    frozen_token_dict = (
        config.get("attack", {}).get("frozen_token_dict", {}) if use_safe_vocab else {}
    )
    if use_safe_vocab:
        logger.info("安全词表模式：启用关键字/标点冻结过滤")
    else:
        logger.info("全词表模式（use_safe_vocab=false）：跳过冻结过滤")
    safe_ids_cpu = build_safe_vocab(tokenizer, frozen_token_dict, use_safe_vocab=use_safe_vocab)

    word_embeddings = get_word_embeddings(st_model)
    safe_ids, safe_emb_matrix = build_safe_embedding_matrix(
        word_embeddings, safe_ids_cpu, device=embedder.device
    )
    logger.info(f"V_safe 构建完成: {safe_ids.shape[0]} 个安全 token")

    # ── 4. 加载投毒目标文件 ──
    data_cfg = config.get("data", {})
    sampling_cfg = config.get("sampling", {})
    target_file_path = data_cfg.get(
        "target_file_path", "data/query/white/target/poison_targets_set.json"
    )

    logger.info(f"加载投毒目标文件: {target_file_path}")
    try:
        with open(target_file_path, "r", encoding="utf-8") as f:
            target_data: Dict[str, list] = json.load(f)
    except FileNotFoundError:
        logger.error(f"目标文件不存在: {target_file_path}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        return 1

    total_entries = sum(len(v) for v in target_data.values())
    logger.info(f"目标文件加载完成: {len(target_data)} 个 CWE 类别, {total_entries} 条数据")

    # ── 5. 构建待处理列表 ──
    process_cwe = sampling_cfg.get("process_cwe", [])
    if not process_cwe:
        logger.error("配置文件中 sampling.process_cwe 为空，无 CWE 类别可处理")
        return 1

    pending_entries = build_pending_entries(
        target_data=target_data,
        process_cwe=process_cwe,
        tokenizer=tokenizer,
        config=config,
    )
    if not pending_entries:
        logger.info("无待处理条目，退出")
        return 0

    # ── 6. 注册梯度存储器（循环外，一次性） ──
    optimizer_type = config.get("attack", {}).get("optimizer", "aggs")
    optimizer_labels = {
        "aggs": "AGGS 贪婪搜索",
        "abgs": "ABGS 束搜索（精英保留·独立 depth）",
        "pabs": "PABS 位置感知束搜索",
        "aggd": "AGGD 贪婪搜索",
    }
    optimizer_label = optimizer_labels.get(
        optimizer_type, f"未知优化器({optimizer_type})"
    )

    for param in st_model.parameters():
        param.requires_grad_(True)
    st_model.eval()

    logger.info("正在注册梯度存储器...")
    grad_storage = GradientStorage(word_embeddings)

    optimizer_map = {
        "aggs": aggs_optimize,
        "abgs": abgs_optimize,
        "pabs": pabs_optimize,
        "aggd": aggd_optimize,
    }
    if optimizer_type not in optimizer_map:
        logger.error(
            "未知 attack.optimizer=%r，可选值为 %s",
            optimizer_type,
            ", ".join(sorted(optimizer_map)),
        )
        return 1
    optimize_fn = optimizer_map[optimizer_type]
    log_dir = log_cfg.get("log_dir", "logs/retrieval_attack")

    # ── 7. 逐个投毒目标循环 ──
    success_count = 0
    fail_count = 0
    total = len(pending_entries)
    model_dtype = next(st_model.parameters()).dtype

    # 各 CWE 待处理总数与本次运行已完成计数（用于观测固定时间下各 CWE 的推进比例）
    from collections import Counter

    cwe_pending_total = Counter(cwe_id for cwe_id, _, _ in pending_entries)
    cwe_done = Counter()

    def _format_cwe_progress() -> str:
        return ", ".join(
            f"{cwe}={cwe_done[cwe]}/{cwe_pending_total[cwe]}"
            for cwe in sorted(cwe_pending_total)
        )

    def _save_target_file():
        """原子写回投毒目标文件（先写临时文件再 rename）。"""
        tmp_path = Path(target_file_path).with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(target_data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(target_file_path)

    for i, (cwe_id, idx, entry) in enumerate(pending_entries, 1):
        target_id = entry.get("entity", {}).get("id", "unknown")
        logger.info("=" * 60)
        logger.info(f"[{i}/{total}] 开始处理投毒目标 CWE={cwe_id} id={target_id}")
        logger.info(f"各 CWE 完成进度: {_format_cwe_progress()}")
        logger.info("=" * 60)

        try:
            # 7-1. 构建正/负样本嵌入
            attack_data = prepare_attack_data(
                entry=entry,
                embedder=embedder,
            )
            torch.cuda.empty_cache()

            pos_embeddings = attack_data["pos_embeddings"].to(
                device=embedder.device, dtype=model_dtype
            )
            logger.info(f"数据准备完成: |Q+|={pos_embeddings.shape[0]}")

            # 7-2. 探测当前投毒目标的最优 batch_size
            buggy_code = entry["entity"]["buggy_code"]
            poison_target_ids = tokenizer.encode(
                buggy_code, add_special_tokens=True
            )
            prompt_prefix_ids = embedder.get_instruct_tids(side="document")
            full_ids = prompt_prefix_ids + poison_target_ids
            adv_ids_probe = torch.tensor(
                [full_ids], dtype=torch.long, device=embedder.device
            )
            probed_bs = probe_max_batch_size(
                embedder,
                adv_ids_probe,
                safety_margin=0.85,
                fallback_batch_size=default_batch_size,
                max_probe_upper=8192,
            )
            config["model"]["batch_size"] = probed_bs
            logger.info(
                f"动态 batch_size 探测完成: seq_len={adv_ids_probe.shape[1]}, "
                f"probed={probed_bs}（默认={default_batch_size}）"
            )
            del adv_ids_probe
            torch.cuda.empty_cache()

            # 7-3. 执行优化
            logger.info(
                f"开始 {optimizer_label} 优化（optimizer={optimizer_type}）..."
            )
            t_start = time.time()
            result = optimize_fn(
                embedder=embedder,
                tokenizer=tokenizer,
                poison_buggy_code=buggy_code,
                pos_embeddings=pos_embeddings,
                safe_ids=safe_ids,
                safe_emb_matrix=safe_emb_matrix,
                grad_storage=grad_storage,
                frozen_token_dict=frozen_token_dict,
                config=config,
            )
            elapsed = time.time() - t_start

            logger.info(f"{optimizer_label} 优化完成，耗时 {elapsed:.1f}s")
            logger.info(f"最终损失: {result['final_loss']:.6f}")
            logger.info(f"最终平均相似度: {result['final_avg_sim']:.6f}")
            logger.info(f"总迭代轮数: {result['total_iterations']}")
            logger.info(f"早停: {result['early_stopped']}")

            # 7-4. 保存日志
            write_attack_log(
                log_dir, date_str, cwe_id, target_id, result, config, optimizer_type
            )

            # 7-5. 原地写回攻击结果
            attack_cfg_snapshot = {
                k: v
                for k, v in config.get("attack", {}).items()
                if k != "frozen_token_dict"
            }
            attack_cfg_snapshot.update(
                result.get("resolved_attack_config", {})
            )
            result_log_snapshot = {
                "init_loss": result.get("init_loss"),
                "final_loss": result["final_loss"],
                "init_sim": result.get("init_sim"),
                "final_sim": result.get("final_sim"),
                "init_avg_sim": result.get("init_avg_sim"),
                "final_avg_sim": result["final_avg_sim"],
                "total_iterations": result["total_iterations"],
                "early_stopped": result["early_stopped"],
                "start_time": result.get("start_time"),
                "end_time": result.get("end_time"),
                "selected_device": config.get("model", {}).get("selected_device"),
                "device_reservation": config.get("model", {}).get(
                    "device_reservation", {}
                ),
            }
            for key in (
                "stop_reason",
                "best_iteration",
                "patience_counter",
                "max_depth_reached",
                "final_beam_depths",
                "total_candidates_generated",
                "total_candidates_evaluated",
                "dynamic_beam_width",
                "per_beam_budget",
                "effective_elite_num",
                "total_fresh_evaluated",
                "total_cache_replayed",
                "total_cache_hits",
                "total_global_scanned",
                "total_sequence_duplicates",
                "max_local_depth",
                "final_local_depths",
                "final_global_offsets",
                "best_avg_similarity",
                "best_avg_similarity_iteration",
                "patience_loss_delta",
                "patience_similarity_delta",
            ):
                if key in result:
                    result_log_snapshot[key] = result[key]

            ra = entry["retrieval_attack"]
            ra["poisoned_buggy_code"] = result["adv_text"]
            ra["configs"] = {
                "data": target_file_path,
                "model": retriever_config_path,
                "device": {
                    "selected_device": config.get("model", {}).get(
                        "selected_device"
                    ),
                    "device_selection": config.get("model", {}).get(
                        "resolved_device_selection", {}
                    ),
                    "reservation": config.get("model", {}).get(
                        "device_reservation", {}
                    ),
                },
                "sampling": {
                    "process_cwe": process_cwe,
                    "min_condidate_num": sampling_cfg.get("min_condidate_num", 50),
                    "max_condidate_num": sampling_cfg.get("max_condidate_num", 6000),
                    "effective_max_condidate_num": min(
                        sampling_cfg.get("max_condidate_num", 6000),
                        config.get("attack", {}).get("n", 6000),
                    ),
                },
                "attack": attack_cfg_snapshot,
                "logs": result_log_snapshot,
            }
            ra["positive_samples"] = _format_sample_list(
                ra.get("positive_samples", [])
            )
            _save_target_file()
            success_count += 1
            cwe_done[cwe_id] += 1
            logger.info(
                f"[{cwe_id}][{target_id}] 写回成功（累计成功: {success_count}）"
            )
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(
                f"[{i}/{total}] 投毒目标 CWE={cwe_id} id={target_id} "
                f"处理失败: {e}，跳过"
            )
            fail_count += 1
            torch.cuda.empty_cache()
            continue
        finally:
            config["model"]["batch_size"] = default_batch_size

    logger.info("=" * 60)
    logger.info("全部投毒任务完成!")
    logger.info(f"成功: {success_count}  失败: {fail_count}  总计: {total}")
    logger.info(f"各 CWE 完成进度: {_format_cwe_progress()}")
    logger.info(f"投毒结果文件: {target_file_path}")
    logger.info(f"日志目录: {log_dir}/{date_str}")
    logger.info("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
