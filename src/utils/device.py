"""GPU 健康检查、外部占用检测与 EvilPatch 跨进程独占调度。"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger(__name__)
_PROCESS_STARTED_AT = datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class GPUProcess:
    """``nvidia-smi`` 报告的单个 GPU 计算进程。"""

    pid: int
    used_memory_mb: Optional[float]


@dataclass
class GPUInfo:
    """一张物理 GPU 的快照。"""

    index: int
    uuid: str
    memory_used_mb: float
    memory_free_mb: float
    memory_total_mb: float
    temperature: Optional[float]
    power_draw: Optional[float]
    processes: List[GPUProcess] = field(default_factory=list)


@dataclass
class GPUSnapshot:
    """GPU 设备与计算进程的同一次 ``nvidia-smi`` 视图。"""

    by_uuid: Dict[str, GPUInfo]
    by_index: Dict[int, GPUInfo]
    compute_query_available: bool


@dataclass
class DeviceReservation:
    """持有某张物理 GPU 跨进程文件锁的进程内租约。"""

    device: str
    physical_index: Optional[int]
    gpu_uuid: Optional[str]
    lock_key: str
    lock_path: Path
    lock_file: IO[str]
    acquired_at: str
    waited_seconds: float
    owner_label: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_device": self.device,
            "physical_gpu_index": self.physical_index,
            "gpu_uuid": self.gpu_uuid,
            "reservation_pid": os.getpid(),
            "reservation_hostname": socket.gethostname(),
            "reservation_process_started_at": _PROCESS_STARTED_AT,
            "reservation_acquired_at": self.acquired_at,
            "reservation_waited_seconds": round(self.waited_seconds, 3),
            "reservation_lock_path": str(self.lock_path),
            "reservation_owner_label": self.owner_label,
        }


@dataclass
class _SelectionAttempt:
    reservation: Optional[DeviceReservation]
    busy_reasons: List[str]
    invalid_reasons: List[str]


_ACTIVE_RESERVATIONS: Dict[str, DeviceReservation] = {}


def _parse_smi_optional_float(raw: str) -> Optional[float]:
    """解析 ``nvidia-smi`` CSV 数值；驱动占位符返回 ``None``。"""
    value = raw.strip()
    if not value:
        return None
    lower = value.lower()
    unavailable = (
        "n/a",
        "not supported",
        "unknown error",
        "not found",
    )
    if any(marker in lower for marker in unavailable):
        return None
    value = re.sub(r"\s*(mib|gib|w|c)\s*$", "", value, flags=re.IGNORECASE)
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_uuid(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    normalized = str(value).strip()
    return normalized.upper() if normalized else None


def _run_nvidia_smi(arguments: Sequence[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi 未找到，将使用 CUDA 显存信息作为占用兜底")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi 执行超时，将使用 CUDA 显存信息作为占用兜底")
    except Exception as exc:  # pragma: no cover - 驱动/子进程异常兜底
        logger.warning("nvidia-smi 探测异常: %s", exc)
    return None


def get_gpu_snapshot() -> Optional[GPUSnapshot]:
    """读取物理 GPU 清单和计算进程；不可用时返回 ``None``。"""
    gpu_result = _run_nvidia_smi([
        "--query-gpu=index,uuid,memory.used,memory.free,memory.total,"
        "temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if gpu_result is None or gpu_result.returncode != 0:
        if gpu_result is not None:
            logger.warning(
                "nvidia-smi GPU 查询返回非零退出码 %s: %s",
                gpu_result.returncode,
                gpu_result.stderr.strip(),
            )
        return None

    by_uuid: Dict[str, GPUInfo] = {}
    by_index: Dict[int, GPUInfo] = {}
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        try:
            if len(parts) < 7:
                raise ValueError("字段数不足7")
            index = int(parts[0])
            gpu_uuid = _normalize_uuid(parts[1])
            used = _parse_smi_optional_float(parts[2])
            free = _parse_smi_optional_float(parts[3])
            total = _parse_smi_optional_float(parts[4])
            if gpu_uuid is None or used is None or free is None or total is None:
                raise ValueError("UUID 或显存字段无法解析")
            info = GPUInfo(
                index=index,
                uuid=gpu_uuid,
                memory_used_mb=used,
                memory_free_mb=free,
                memory_total_mb=total,
                temperature=_parse_smi_optional_float(parts[5]),
                power_draw=_parse_smi_optional_float(parts[6]),
            )
            by_uuid[gpu_uuid] = info
            by_index[index] = info
        except (ValueError, IndexError) as exc:
            logger.warning("GPU 行解析失败 (line=%r): %s", line, exc)

    process_result = _run_nvidia_smi([
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    compute_query_available = bool(
        process_result is not None and process_result.returncode == 0
    )
    if compute_query_available and process_result is not None:
        for line in process_result.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            try:
                if len(parts) < 2:
                    raise ValueError("字段数不足2")
                gpu_uuid = _normalize_uuid(parts[0])
                pid = int(parts[1])
                used_memory = (
                    _parse_smi_optional_float(parts[2])
                    if len(parts) > 2
                    else None
                )
                if gpu_uuid in by_uuid:
                    by_uuid[gpu_uuid].processes.append(
                        GPUProcess(pid=pid, used_memory_mb=used_memory)
                    )
            except (ValueError, IndexError) as exc:
                logger.debug("GPU 计算进程行解析失败 (line=%r): %s", line, exc)
    elif process_result is not None:
        logger.warning(
            "nvidia-smi 计算进程查询失败，将使用 memory.used 兜底: %s",
            process_result.stderr.strip(),
        )

    return GPUSnapshot(
        by_uuid=by_uuid,
        by_index=by_index,
        compute_query_available=compute_query_available,
    )


def get_healthy_gpu_indices() -> Optional[Set[int]]:
    """保留旧接口：返回至少有 2 GiB 空闲显存的可见物理 GPU。"""
    snapshot = get_gpu_snapshot()
    if snapshot is None:
        return None
    healthy = {
        index
        for index, info in snapshot.by_index.items()
        if info.memory_free_mb >= 2048
    }
    logger.info("nvidia-smi 探测完成: 健康 GPU = %s", sorted(healthy))
    return healthy


def _logical_cuda_index(device: str) -> int:
    normalized = str(device).strip().lower()
    if normalized == "cuda":
        return 0
    if not normalized.startswith("cuda:"):
        raise ValueError(f"非法 CUDA 设备: {device!r}")
    try:
        index = int(normalized.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"非法 CUDA 设备: {device!r}") from exc
    if index < 0:
        raise ValueError(f"CUDA 设备索引不能为负数: {device!r}")
    return index


def _visible_device_token(logical_index: int) -> Optional[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tokens[logical_index] if logical_index < len(tokens) else None


def _resolve_gpu_info(
    device: str,
    snapshot: Optional[GPUSnapshot],
    properties: Any,
) -> Tuple[Optional[GPUInfo], Optional[int], Optional[str]]:
    """将 PyTorch 逻辑设备映射到 ``nvidia-smi`` 物理 GPU。"""
    logical_index = _logical_cuda_index(device)
    property_uuid = _normalize_uuid(getattr(properties, "uuid", None))
    if snapshot is not None and property_uuid in snapshot.by_uuid:
        info = snapshot.by_uuid[property_uuid]
        return info, info.index, info.uuid

    visible_token = _visible_device_token(logical_index)
    if visible_token:
        visible_uuid = _normalize_uuid(visible_token)
        if snapshot is not None and visible_uuid in snapshot.by_uuid:
            info = snapshot.by_uuid[visible_uuid]
            return info, info.index, info.uuid
        try:
            physical_index = int(visible_token)
        except ValueError:
            physical_index = None
        if snapshot is not None and physical_index in snapshot.by_index:
            info = snapshot.by_index[physical_index]
            return info, info.index, info.uuid
        if snapshot is None:
            if property_uuid is not None:
                return None, physical_index, property_uuid
            if physical_index is not None:
                return None, physical_index, None
            # nvidia-smi 不可用时，CUDA_VISIBLE_DEVICES 中的 UUID
            # 仍比逻辑索引更适合作为跨进程锁键。
            return None, None, visible_uuid

    if snapshot is not None and logical_index in snapshot.by_index:
        info = snapshot.by_index[logical_index]
        return info, info.index, info.uuid

    if snapshot is None:
        return None, logical_index, property_uuid
    return None, None, property_uuid


def _safe_lock_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _lock_file_nonblocking(handle: IO[str]) -> None:
    if os.name == "nt":  # pragma: no cover - Linux 为生产运行环境
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: IO[str]) -> None:
    try:
        if os.name == "nt":  # pragma: no cover - Linux 为生产运行环境
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass


def _read_lock_owner(lock_path: Path) -> Dict[str, Any]:
    try:
        with lock_path.open("r", encoding="utf-8") as handle:
            # Windows 的 msvcrt 锁定文件第 1 个字节。元数据从第 2 个
            # 字节开始写入，因此其他进程仍可读取持锁者信息；Linux 的
            # flock 不限制普通读取，跳过同一个哨兵字节也保持一致。
            handle.seek(1)
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _try_acquire_reservation_lock(
    lock_dir: Path,
    lock_key: str,
    metadata: Dict[str, Any],
) -> Tuple[Optional[IO[str]], Dict[str, Any], Path]:
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        probe_path = lock_dir / ".write_probe"
        with probe_path.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"GPU 独占锁目录不可写: {lock_dir}: {exc}"
        ) from exc

    lock_path = lock_dir / f"gpu-{_safe_lock_component(lock_key)}.lock"
    try:
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"GPU 独占锁文件无法打开: {lock_path}: {exc}") from exc
    try:
        _lock_file_nonblocking(handle)
    except BlockingIOError:
        handle.close()
        owner: Dict[str, Any] = {}
        # 抢锁失败可能紧跟在对方获锁之后，给持锁者一个极短的
        # 时间窗口写完元数据，避免日志偶发丢失 PID/配置路径。
        for _ in range(3):
            owner = _read_lock_owner(lock_path)
            if owner:
                break
            time.sleep(0.01)
        return None, owner, lock_path

    try:
        handle.seek(0)
        handle.truncate()
        # 第 1 个字节仅作为跨平台锁定哨兵，JSON 元数据从第 2 个字节开始。
        handle.write("\n")
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    except Exception:
        _release_lock_handle(handle)
        raise
    return handle, {}, lock_path


def _release_lock_handle(handle: IO[str]) -> None:
    _unlock_file(handle)
    try:
        handle.close()
    except OSError:
        pass


def release_device_reservations() -> None:
    """释放当前 Python 进程持有的全部 GPU 独占租约。"""
    for lock_key, reservation in list(_ACTIVE_RESERVATIONS.items()):
        _release_lock_handle(reservation.lock_file)
        logger.info(
            "已释放 GPU 独占租约: %s (physical=%s, uuid=%s)",
            reservation.device,
            reservation.physical_index,
            reservation.gpu_uuid,
        )
        _ACTIVE_RESERVATIONS.pop(lock_key, None)


def get_device_reservation_info(device: Optional[str] = None) -> Dict[str, Any]:
    """返回当前进程持有的 GPU 租约元数据。"""
    reservations = list(_ACTIVE_RESERVATIONS.values())
    if device is not None:
        reservations = [item for item in reservations if item.device == device]
    if not reservations:
        return {}
    return reservations[-1].as_dict()


atexit.register(release_device_reservations)


def _default_lock_dir(selection_config: Dict[str, Any]) -> Path:
    configured = selection_config.get("lock_dir")
    if configured:
        return Path(str(configured)).expanduser()
    return Path(tempfile.gettempdir()) / "evilpatch-gpu-locks"


def _torch_memory_info_mb(device: str) -> Tuple[float, float, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mb = free_bytes / 1024**2
    total_mb = total_bytes / 1024**2
    return total_mb - free_mb, free_mb, total_mb


def _format_process_reason(info: GPUInfo) -> Optional[str]:
    external = [process for process in info.processes if process.pid != os.getpid()]
    if not external:
        return None
    details = []
    for process in external:
        memory = (
            f"{process.used_memory_mb:.0f} MiB"
            if process.used_memory_mb is not None
            else "显存未知"
        )
        details.append(f"PID {process.pid} ({memory})")
    return "检测到外部计算进程 " + ", ".join(details)


def _owner_description(owner: Dict[str, Any]) -> str:
    pid = owner.get("pid", "?")
    label = owner.get("owner_label") or owner.get("command") or "unknown"
    return f"EvilPatch 独占锁由 PID {pid} 持有 ({label})"


def _selection_pool(config: Dict[str, Any]) -> List[str]:
    configured_pool = config.get("device_pool") or []
    if configured_pool:
        return [str(device) for device in configured_pool]
    fallback = str(config.get("device", "cuda"))
    if not fallback.startswith("cuda"):
        return []
    if fallback == "cuda":
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return [fallback]


def _attempt_exclusive_selection(
    config: Dict[str, Any],
    selection_config: Dict[str, Any],
    waited_seconds: float,
) -> _SelectionAttempt:
    pool = _selection_pool(config)
    min_free_mb = float(selection_config.get("min_free_memory_mb", 2048))
    max_idle_used_mb = float(
        selection_config.get("max_idle_memory_used_mb", 512)
    )
    occupied_policy = str(
        selection_config.get("occupied_policy", "any_process")
    ).strip().lower()
    lock_dir = _default_lock_dir(selection_config)
    owner_label = str(selection_config.get("owner_label", ""))

    if occupied_policy not in {"any_process", "only_evilpatch"}:
        raise ValueError(
            "device_selection.occupied_policy 必须为 "
            "'any_process' 或 'only_evilpatch'"
        )

    initial_snapshot = get_gpu_snapshot()
    busy_reasons: List[str] = []
    invalid_reasons: List[str] = []

    for raw_device in pool:
        try:
            logical_index = _logical_cuda_index(raw_device)
            device = f"cuda:{logical_index}"
            if logical_index >= torch.cuda.device_count():
                invalid_reasons.append(
                    f"{device}: PyTorch 仅可见 {torch.cuda.device_count()} 张 GPU"
                )
                continue
            properties = torch.cuda.get_device_properties(logical_index)
        except Exception as exc:
            invalid_reasons.append(f"{raw_device}: CUDA 设备属性读取失败 ({exc})")
            continue

        info, physical_index, gpu_uuid = _resolve_gpu_info(
            device, initial_snapshot, properties
        )
        if initial_snapshot is not None and info is None:
            invalid_reasons.append(
                f"{device}: 无法映射到 nvidia-smi 中的健康物理 GPU"
            )
            continue

        lock_key = gpu_uuid or (
            f"INDEX-{physical_index}"
            if physical_index is not None
            else f"LOGICAL-{logical_index}"
        )
        if lock_key in _ACTIVE_RESERVATIONS:
            busy_reasons.append(f"{device}: 当前进程已持有该 GPU 租约")
            continue

        acquired_at = datetime.now().astimezone().isoformat()
        metadata = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "owner_label": owner_label,
            "config_path": owner_label,
            "command": " ".join(sys.argv),
            "process_started_at": _PROCESS_STARTED_AT,
            "logical_device": device,
            "physical_gpu_index": physical_index,
            "gpu_uuid": gpu_uuid,
            "acquired_at": acquired_at,
        }
        lock_handle, owner, lock_path = _try_acquire_reservation_lock(
            lock_dir, lock_key, metadata
        )
        if lock_handle is None:
            busy_reasons.append(f"{device}: {_owner_description(owner)}")
            continue

        try:
            refreshed_snapshot = get_gpu_snapshot()
            refreshed_info, refreshed_index, refreshed_uuid = _resolve_gpu_info(
                device, refreshed_snapshot, properties
            )
            if refreshed_snapshot is None:
                # 二次 nvidia-smi 查询暂时不可用时，不复用获锁前的
                # 旧快照，改用当前 CUDA 显存信息做保守的占用判断。
                info = None
            elif refreshed_info is None:
                invalid_reasons.append(
                    f"{device}: 获锁后无法映射到物理 GPU"
                )
                _release_lock_handle(lock_handle)
                continue
            else:
                info = refreshed_info
                physical_index = refreshed_index
                gpu_uuid = refreshed_uuid

            if info is not None:
                used_mb = info.memory_used_mb
                free_mb = info.memory_free_mb
            else:
                used_mb, free_mb, _ = _torch_memory_info_mb(device)

            if occupied_policy == "any_process" and info is not None:
                process_reason = _format_process_reason(info)
                if process_reason:
                    busy_reasons.append(f"{device}: {process_reason}")
                    _release_lock_handle(lock_handle)
                    continue

            # 已能看到当前 PID 时，从 memory.used 中扣除它的显式
            # 占用，避免 PyTorch 在读取设备属性时初始化的小型 CUDA
            # 上下文导致当前进程误判自己。无法查询 PID 时仍使用
            # 全部 memory.used 作为保守兜底。
            occupancy_used_mb = used_mb
            if info is not None and refreshed_snapshot is not None:
                own_visible_memory = sum(
                    process.used_memory_mb or 0.0
                    for process in info.processes
                    if process.pid == os.getpid()
                )
                occupancy_used_mb = max(0.0, used_mb - own_visible_memory)

            process_query_has_visible_pids = bool(
                info is not None
                and refreshed_snapshot is not None
                and refreshed_snapshot.compute_query_available
                and info.processes
            )
            if (
                occupied_policy == "any_process"
                and not process_query_has_visible_pids
                and occupancy_used_mb > max_idle_used_mb
            ):
                busy_reasons.append(
                    f"{device}: 非当前进程已用显存 "
                    f"{occupancy_used_mb:.0f} MiB 超过空闲阈值 "
                    f"{max_idle_used_mb:.0f} MiB"
                )
                _release_lock_handle(lock_handle)
                continue
            if free_mb < min_free_mb:
                busy_reasons.append(
                    f"{device}: 可用显存 {free_mb:.0f} MiB 小于最低要求 "
                    f"{min_free_mb:.0f} MiB"
                )
                _release_lock_handle(lock_handle)
                continue

            temporary = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            del temporary

            reservation = DeviceReservation(
                device=device,
                physical_index=physical_index,
                gpu_uuid=gpu_uuid,
                lock_key=lock_key,
                lock_path=lock_path,
                lock_file=lock_handle,
                acquired_at=acquired_at,
                waited_seconds=waited_seconds,
                owner_label=owner_label,
            )
            _ACTIVE_RESERVATIONS[lock_key] = reservation
            logger.info(
                "获得 GPU 独占租约: %s (physical=%s, uuid=%s, "
                "pid=%s, waited=%.1fs, lock=%s)",
                device,
                physical_index,
                gpu_uuid,
                os.getpid(),
                waited_seconds,
                lock_path,
            )
            return _SelectionAttempt(reservation, busy_reasons, invalid_reasons)
        except RuntimeError as exc:
            _release_lock_handle(lock_handle)
            message = str(exc)
            if "out of memory" in message.lower():
                busy_reasons.append(f"{device}: CUDA 显存不足 ({message})")
            else:
                invalid_reasons.append(f"{device}: CUDA 运行时探测失败 ({message})")
        except Exception as exc:
            _release_lock_handle(lock_handle)
            invalid_reasons.append(f"{device}: 设备探测失败 ({exc})")

    return _SelectionAttempt(None, busy_reasons, invalid_reasons)


def _select_device_legacy(config: Dict[str, Any]) -> str:
    """保留原有“首张健康且可分配”设备选择行为。"""
    pool = config.get("device_pool") or []
    fallback = str(config.get("device", "cuda"))
    if not pool:
        if fallback.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退至 CPU")
            return "cpu"
        return fallback

    healthy_indices = get_healthy_gpu_indices()
    filtered_pool: List[str] = []
    for device in pool:
        try:
            index = _logical_cuda_index(str(device))
        except ValueError as exc:
            logger.warning("设备池排除 %s: %s", device, exc)
            continue
        if healthy_indices is None or index in healthy_indices:
            filtered_pool.append(str(device))
        else:
            logger.warning("设备池预筛选排除 %s（nvidia-smi 判定为不健康）", device)

    for device in filtered_pool:
        try:
            index = _logical_cuda_index(device)
            properties = torch.cuda.get_device_properties(index)
            temporary = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            del temporary
            logger.info(
                "设备池探测成功: %s (%s, %d MiB)",
                device,
                properties.name,
                properties.total_memory // 1024**2,
            )
            return device
        except Exception as exc:
            logger.warning("设备池跳过 %s: %s", device, exc)
    logger.warning("设备池中所有设备均不可用，回退至 CPU")
    return "cpu"


def select_device(config: Dict[str, Any]) -> str:
    """
    按 ``device_selection`` 选择 GPU，默认保留 legacy 兼容模式。

    ``exclusive`` 模式使用物理 GPU 文件锁隔离 EvilPatch 进程，并可
    根据 ``nvidia-smi`` 计算进程和显存占用排除其他 CUDA 任务。
    """
    selection_config = dict(config.get("device_selection") or {})
    mode = str(selection_config.get("mode", "legacy")).strip().lower()
    if mode == "legacy":
        return _select_device_legacy(config)
    if mode != "exclusive":
        raise ValueError("device_selection.mode 必须为 'legacy' 或 'exclusive'")

    on_busy = str(selection_config.get("on_busy", "fail")).strip().lower()
    if on_busy not in {"wait", "fail", "cpu"}:
        raise ValueError("device_selection.on_busy 必须为 wait/fail/cpu")
    if not torch.cuda.is_available():
        if on_busy == "cpu":
            logger.warning("CUDA 不可用，按配置回退至 CPU")
            return "cpu"
        raise RuntimeError(
            "独占模式下 CUDA 不可用；如需回退 CPU，请显式设置 "
            "device_selection.on_busy=cpu"
        )
    pool = _selection_pool(config)
    if not pool:
        fallback = str(config.get("device", "cuda"))
        if not fallback.startswith("cuda"):
            return fallback
        raise RuntimeError("独占模式下没有可见 CUDA 设备")

    poll_interval = float(selection_config.get("poll_interval_seconds", 10))
    if poll_interval <= 0:
        raise ValueError("device_selection.poll_interval_seconds 必须大于0")
    raw_timeout = selection_config.get("wait_timeout_seconds")
    wait_timeout = None if raw_timeout is None else float(raw_timeout)
    if wait_timeout is not None and wait_timeout < 0:
        raise ValueError("device_selection.wait_timeout_seconds 不能小于0")

    start = time.monotonic()
    last_reason_signature: Optional[Tuple[str, ...]] = None
    last_summary_at = -60.0
    while True:
        waited = time.monotonic() - start
        attempt = _attempt_exclusive_selection(
            config,
            selection_config,
            waited_seconds=waited,
        )
        if attempt.reservation is not None:
            return attempt.reservation.device

        reason_signature = tuple(attempt.busy_reasons + attempt.invalid_reasons)
        if reason_signature != last_reason_signature:
            for reason in attempt.invalid_reasons:
                logger.warning("GPU 不可用: %s", reason)
            for reason in attempt.busy_reasons:
                logger.info("GPU 已占用: %s", reason)
            last_reason_signature = reason_signature

        if not attempt.busy_reasons:
            details = "; ".join(attempt.invalid_reasons) or "无候选设备"
            raise RuntimeError(f"设备池中没有健康且可恢复的 GPU: {details}")
        if on_busy == "fail":
            raise RuntimeError("所有健康 GPU 当前均被占用")
        if on_busy == "cpu":
            logger.warning("所有健康 GPU 均被占用，按配置回退至 CPU")
            return "cpu"
        if wait_timeout is not None and waited >= wait_timeout:
            raise TimeoutError(
                f"等待空闲 GPU 超时：{waited:.1f}s >= {wait_timeout:.1f}s"
            )
        if waited - last_summary_at >= 60 or last_summary_at < 0:
            logger.info(
                "全部健康 GPU 当前均被占用，已等待 %.0f 秒；%.0f 秒后重试",
                waited,
                poll_interval,
            )
            last_summary_at = waited
        time.sleep(poll_interval)
