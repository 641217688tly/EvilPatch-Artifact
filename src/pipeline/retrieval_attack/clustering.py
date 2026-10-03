#!/usr/bin/env python3
"""
RACG 正样本查询集聚类预处理脚本

基于 yellowbrick 自动选择最佳簇数，支持 K-Means 和层次聚类算法。
为待优化的文档选择负样本进行语义聚类处理。

功能:
    1. 单文件或批量文件夹处理
    2. 使用 Jina Code Embeddings 对 buggy_code 进行语义嵌入
    3. 基于肘点法自动选择最优簇数
    4. 支持 K-Means 和层次聚类算法
    5. 动态构建输出路径并保存带 cluster_id 的结果
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.models.retriever.Base import BaseEmbedder, create_embedder

import numpy as np
import yaml

# 设置日志
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """配置日志输出格式和级别。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="RACG 正样本查询集聚类预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s -i data/bfp/white/raw/CWE20_RELEVANCE_0.75_TOP_10.json
  %(prog)s -i data/bfp/white/raw/ -c hierarchical -k 3,10
  %(prog)s -i data/bfp/black/raw/ -m configs/models/jina-code-embeddings-1.5b.yml
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入文件路径或文件夹路径",
    )
    parser.add_argument(
        "--model_config",
        "-m",
        default="configs/models/jina-code-embeddings-0.5b.yml",
        help="模型配置文件路径 (默认: configs/models/jina-code-embeddings-0.5b.yml)",
    )
    parser.add_argument(
        "--clustering",
        "-c",
        choices=["kmeans", "hierarchical"],
        default="kmeans",
        help="聚类算法选择 (默认: kmeans)",
    )
    parser.add_argument(
        "--k_range",
        "-k",
        default="2,15",
        help="肘点法搜索的k值范围，格式为'min,max' (默认: 2,15)",
    )
    parser.add_argument(
        "--default_k",
        "-d",
        type=int,
        default=5,
        help="无法检测肘点时的默认簇数 (默认: 5)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="data/bfp",
        help="输出目录基路径",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="强制覆盖已存在的输出文件",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出",
    )

    return parser.parse_args()


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    从文件名解析 CWE 参数。

    文件名格式: CWE{NUM}_RELEVANCE_{threshold}_TOP_{topk}[_{tag}].json
    示例: CWE20_RELEVANCE_0.75_TOP_10.json
          -> {"cwe_id": "CWE-20", "relevance_threshold": 0.75, "top_k": 10, "name_suffix": ""}
    示例: CWE20_RELEVANCE_0.8_TOP_10_VINJ.json（漏洞注入产物）
          -> name_suffix 为 "VINJ"

    Args:
        filename: 文件名（不含路径）

    Returns:
        包含 cwe_id, relevance_threshold, top_k 的字典

    Raises:
        ValueError: 文件名不符合预期格式
    """
    pattern = r"CWE(\d+)_RELEVANCE_([\d.]+)_TOP_(\d+)(?:_([A-Za-z][A-Za-z0-9]*))?\.json"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"文件名格式不符合预期: {filename}\n"
            f"期望格式: CWE{{NUM}}_RELEVANCE_{{threshold}}_TOP_{{topk}}[{{_tag}}].json"
        )

    cwe_num, threshold, top_k, name_suffix = match.groups()
    return {
        "cwe_id": f"CWE-{cwe_num}",
        "cwe_num": cwe_num,
        "relevance_threshold": float(threshold),
        "top_k": int(top_k),
        "name_suffix": name_suffix or "",
    }


def detect_dataset_type(input_path: str) -> str:
    """
    从输入路径检测数据集类型 (white 或 black)。

    Args:
        input_path: 输入文件或文件夹路径

    Returns:
        "white" 或 "black"
    """
    path_lower = input_path.lower()
    if "black" in path_lower:
        return "black"
    return "white"


def build_output_path(
    input_path: str,
    output_base: str,
    filename_params: Dict[str, Any],
) -> Path:
    """
    构建输出文件路径。

    输出路径格式: {output_base}/{dataset_type}/clustered/CWE{num}_RELEVANCE_{threshold}_TOP_{topk}[_tag]_clustered.json

    Args:
        input_path: 原始输入路径
        output_base: 输出目录基路径
        filename_params: 文件名解析结果

    Returns:
        完整的输出文件路径
    """
    dataset_type = detect_dataset_type(input_path)

    # 构建输出目录
    output_dir = Path(output_base) / dataset_type / "clustered"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建输出文件名
    cwe_num = filename_params["cwe_num"]
    threshold = filename_params["relevance_threshold"]
    top_k = filename_params["top_k"]
    tag = filename_params.get("name_suffix") or ""
    tag_part = f"_{tag}" if tag else ""

    output_filename = f"CWE{cwe_num}_RELEVANCE_{threshold}_TOP_{top_k}{tag_part}_clustered.json"
    return output_dir / output_filename


def load_json_data(filepath: Path) -> List[Dict[str, Any]]:
    """
    加载 JSON 数据文件。

    Args:
        filepath: JSON 文件路径

    Returns:
        数据列表

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析错误
    """
    if not filepath.exists():
        raise FileNotFoundError(f"输入文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"数据格式错误: 期望列表类型，得到 {type(data)}")

    return data


def extract_buggy_codes(data: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    """
    从数据中提取 buggy_code 字段。

    Args:
        data: 数据列表

    Returns:
        (code_list, valid_indices) 元组
        - code_list: 提取的 buggy_code 列表
        - valid_indices: 有效数据的索引列表（用于跳过无效项）

    Raises:
        ValueError: 没有有效的 buggy_code 字段
    """
    codes = []
    valid_indices = []

    for idx, item in enumerate(data):
        if "buggy_code" in item and item["buggy_code"]:
            codes.append(str(item["buggy_code"]))
            valid_indices.append(idx)
        else:
            logger.warning(f"第 {idx} 条数据缺少 buggy_code 字段，已跳过")

    if not codes:
        raise ValueError("没有有效的 buggy_code 数据可供聚类")

    return codes, valid_indices


def load_and_embed_data(
    filepath: Path,
    embedder: BaseEmbedder,
) -> Tuple[np.ndarray, List[int]]:
    """
    加载数据并生成语义嵌入。

    Args:
        filepath: 数据文件路径
        embedder: BaseEmbedder 实例

    Returns:
        (embeddings, valid_indices) 元组
        - embeddings: 嵌入向量数组 (n_samples, dim)
        - valid_indices: 有效数据的索引列表
    """
    logger.info(f"加载数据: {filepath}")
    data = load_json_data(filepath)
    logger.info(f"共加载 {len(data)} 条数据")

    # 提取 buggy_code
    codes, valid_indices = extract_buggy_codes(data)
    logger.info(f"有效 buggy_code 数量: {len(codes)}")

    # 生成嵌入
    logger.info("开始生成语义嵌入...")
    embeddings = embedder.embed_documents(codes)
    embeddings = np.asarray(embeddings)
    logger.info(f"嵌入完成，向量维度: {embeddings.shape}")

    return embeddings, valid_indices


def parse_k_range(k_range_str: str) -> Tuple[int, int]:
    """
    解析 k 值范围字符串。

    Args:
        k_range_str: 格式为 "min,max" 的字符串

    Returns:
        (k_min, k_max) 元组
    """
    try:
        k_min, k_max = map(int, k_range_str.split(","))
        if k_min >= k_max:
            raise ValueError("k_min 必须小于 k_max")
        return k_min, k_max
    except ValueError as e:
        raise ValueError(f"k_range 格式错误: {k_range_str}，应为 'min,max' 格式") from e


def perform_kmeans_clustering(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    default_k: int,
) -> Tuple[np.ndarray, int]:
    """
    使用 K-Means 进行聚类，基于肘点法自动选择最佳簇数。

    Args:
        X: 嵌入向量数组 (n_samples, dim)
        k_min: 最小簇数
        k_max: 最大簇数
        default_k: 无法检测肘点时的默认簇数

    Returns:
        (cluster_labels, optimal_k) 元组
    """
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer

    logger.info(f"使用肘点法选择最佳簇数 (范围: {k_min}-{k_max})...")

    # 调整 k_max 不超过样本数
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1) if n_samples > 2 else 2
    k_min = min(k_min, k_max)

    # 肘点法选择最佳簇数
    model = KMeans(random_state=42, n_init="auto")
    visualizer = KElbowVisualizer(
        model,
        k=(k_min, k_max),
        timings=False,
        force_model=True,
    )
    visualizer.fit(X)

    optimal_k = visualizer.elbow_value_
    if optimal_k is None:
        optimal_k = min(default_k, n_samples)
        logger.warning(f"未检测到明显肘点，使用默认簇数: {optimal_k}")
    else:
        logger.info(f"肘点法推荐的最佳簇数: {optimal_k}")

    # 使用最佳簇数进行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)

    return cluster_labels, optimal_k


def perform_hierarchical_clustering(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    default_k: int,
) -> Tuple[np.ndarray, int]:
    """
    使用层次聚类进行聚类，基于肘点法自动选择最佳簇数。

    Args:
        X: 嵌入向量数组 (n_samples, dim)
        k_min: 最小簇数
        k_max: 最大簇数
        default_k: 无法检测肘点时的默认簇数

    Returns:
        (cluster_labels, optimal_k) 元组
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.cluster import KMeans

    logger.info(f"执行层次聚类 (使用肘点法确定簇数)...")

    # 先使用肘点法确定最佳簇数
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1) if n_samples > 2 else 2
    k_min = min(k_min, k_max)

    # 使用 K-Means 肘点法确定最佳簇数（层次聚类本身不提供肘点法）
    model = KMeans(random_state=42, n_init="auto")
    visualizer = KElbowVisualizer(
        model,
        k=(k_min, k_max),
        timings=False,
        force_model=True,
    )
    visualizer.fit(X)

    optimal_k = visualizer.elbow_value_
    if optimal_k is None:
        optimal_k = min(default_k, n_samples)
        logger.warning(f"未检测到明显肘点，使用默认簇数: {optimal_k}")
    else:
        logger.info(f"肘点法推荐的最佳簇数: {optimal_k}")

    # 执行层次聚类
    logger.info("计算距离矩阵并执行层次聚类...")
    Y = pdist(X, metric="cosine")
    Z = hierarchy.linkage(Y, method="ward")
    cluster_labels = hierarchy.fcluster(Z, t=optimal_k, criterion="maxclust")

    # 将标签转换为 0-based
    cluster_labels = cluster_labels - 1

    return cluster_labels, optimal_k


def perform_clustering(
    X: np.ndarray,
    algorithm: str,
    k_range: str,
    default_k: int,
) -> Tuple[np.ndarray, int]:
    """
    执行指定的聚类算法。

    Args:
        X: 嵌入向量数组
        algorithm: 聚类算法 ("kmeans" 或 "hierarchical")
        k_range: k值范围字符串 "min,max"
        default_k: 默认簇数

    Returns:
        (cluster_labels, optimal_k) 元组
    """
    k_min, k_max = parse_k_range(k_range)

    if algorithm == "kmeans":
        return perform_kmeans_clustering(X, k_min, k_max, default_k)
    else:
        return perform_hierarchical_clustering(X, k_min, k_max, default_k)


def save_clustered_data(
    data: List[Dict[str, Any]],
    cluster_labels: np.ndarray,
    valid_indices: List[int],
    output_path: Path,
) -> None:
    """
    保存带 cluster_id 的数据。

    Args:
        data: 原始数据列表
        cluster_labels: 聚类标签数组
        valid_indices: 有效数据的索引列表
        output_path: 输出文件路径
    """
    # 为有效数据添加 cluster_id
    for i, idx in enumerate(valid_indices):
        data[idx]["cluster_id"] = int(cluster_labels[i])

    # 为无效数据添加默认值 -1
    for idx in range(len(data)):
        if idx not in valid_indices:
            data[idx]["cluster_id"] = -1

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"聚类结果已保存: {output_path}")

    # 输出各簇统计
    unique_labels = np.unique(cluster_labels)
    logger.info("各簇样本数分布:")
    for label in unique_labels:
        count = np.sum(cluster_labels == label)
        logger.info(f"  Cluster {label}: {count} 条样本")


def process_single_file(
    input_file: Path,
    args: argparse.Namespace,
    embedder: BaseEmbedder,
) -> bool:
    """
    处理单个文件。

    Args:
        input_file: 输入文件路径
        args: 命令行参数
        embedder: BaseEmbedder 实例

    Returns:
        处理是否成功
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理文件: {input_file}")
        logger.info(f"{'='*60}")

        # 解析文件名参数
        filename_params = parse_filename(input_file.name)
        suffix_info = (
            f", 文件名标记={filename_params['name_suffix']}"
            if filename_params.get("name_suffix")
            else ""
        )
        logger.info(
            f"文件参数: CWE={filename_params['cwe_id']}, "
            f"阈值={filename_params['relevance_threshold']}, "
            f"TopK={filename_params['top_k']}{suffix_info}"
        )

        # 构建输出路径
        output_path = build_output_path(
            str(input_file),
            args.output_dir,
            filename_params,
        )

        # 检查是否已存在
        if output_path.exists() and not args.force:
            logger.warning(f"输出文件已存在: {output_path}，使用 --force 强制覆盖")
            return False

        # 加载数据并生成嵌入
        embeddings, valid_indices = load_and_embed_data(input_file, embedder)

        if len(valid_indices) < 2:
            logger.error("有效样本数少于2，无法进行聚类")
            return False

        # 执行聚类
        cluster_labels, optimal_k = perform_clustering(
            embeddings,
            args.clustering,
            args.k_range,
            args.default_k,
        )

        logger.info(f"聚类完成: 共 {len(valid_indices)} 条数据分为 {optimal_k} 个簇")

        # 加载原始数据
        data = load_json_data(input_file)

        # 保存结果
        save_clustered_data(data, cluster_labels, valid_indices, output_path)

        return True

    except Exception as e:
        logger.error(f"处理文件失败: {input_file}")
        logger.error(f"错误: {e}")
        return False


def process_directory(
    input_dir: Path,
    args: argparse.Namespace,
    embedder: BaseEmbedder,
) -> Tuple[int, int]:
    """
    批量处理文件夹中的所有匹配文件。

    Args:
        input_dir: 输入文件夹路径
        args: 命令行参数
        embedder: BaseEmbedder 实例

    Returns:
        (成功数, 失败数) 元组
    """
    # 查找所有匹配的文件
    pattern = "CWE*_RELEVANCE_*_TOP_*.json"
    files = list(input_dir.glob(pattern))

    # 排除已聚类的文件
    files = [f for f in files if "_clustered" not in f.name]

    if not files:
        logger.warning(f"未找到匹配的文件: {input_dir / pattern}")
        return 0, 0

    logger.info(f"\n找到 {len(files)} 个待处理文件")

    success_count = 0
    fail_count = 0

    for i, input_file in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] 处理: {input_file.name}")
        if process_single_file(input_file, args, embedder):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def init_embedder(model_config_path: str) -> BaseEmbedder:
    """
    根据 YAML 配置初始化 BaseEmbedder 子类实例（经 create_embedder）。

    Args:
        model_config_path: 模型配置文件路径

    Returns:
        BaseEmbedder 实例
    """
    # 添加项目根目录到路径
    
    PROJECT_ROOT = Path(os.path.abspath('')).resolve()
    while not (PROJECT_ROOT / 'configs').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
        PROJECT_ROOT = PROJECT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.io import load_yaml

    # 加载模型配置
    config = load_yaml(model_config_path)
    logger.info(f"加载模型配置: {model_config_path}")
    logger.info(f"模型: {config.get('model_name', 'unknown')}")

    # 初始化嵌入器
    embedder = create_embedder(config)
    return embedder


def main() -> int:
    """主入口函数。"""
    args = parse_args()

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger.info("=" * 60)
    logger.info("RACG 正样本查询集聚类预处理脚本")
    logger.info("=" * 60)
    logger.info(f"聚类算法: {args.clustering}")
    logger.info(f"k值范围: {args.k_range}")
    logger.info(f"默认簇数: {args.default_k}")
    logger.info(f"模型配置: {args.model_config}")

    # 初始化嵌入器
    try:
        embedder = init_embedder(args.model_config)
    except Exception as e:
        logger.error(f"初始化嵌入器失败: {e}")
        return 1

    # 处理输入路径
    input_path = Path(args.input)

    if input_path.is_file():
        # 单文件处理
        success = process_single_file(input_path, args, embedder)
        success_count = 1 if success else 0
        fail_count = 0 if success else 1
    elif input_path.is_dir():
        # 批量处理
        success_count, fail_count = process_directory(input_path, args, embedder)
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return 1

    # 输出统计
    logger.info("\n" + "=" * 60)
    logger.info("处理完成统计")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"总计: {success_count + fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    # python -m src.pipeline.retrieval_attack.clustering -i data/bfp/white/vul_injected/CWE20_RELEVANCE_0.8_TOP_10_VINJ.json
    sys.exit(main())