"""
EvilPatch 检索器对抗攻击 Pipeline。

本包实现基于期望相似度最大范式（Expected Similarity Maximization）的
检索器对抗攻击，用于对 APR (Automated Program Repair) 系统的 RAG 知识库
进行投毒。

模块组成:
  - clustering:         正样本查询集语义聚类预处理
  - data_prep_ultra:    正样本选择与待处理投毒目标构建（方案四 / Ultra）
  - vocab_filter:       安全候选词表 V_safe 构建
  - loss_func:          损失函数与梯度计算（期望相似度 + 动态权重变体）
  - optimizer/
    - aggs_optimizer:   AGGS 改进贪婪梯度搜索优化器（整篇文档就地替换 + 安全词表）
    - abgs_optimizer:   ABGS 改进束梯度搜索优化器（整篇文档就地替换 + 安全词表）
    - pabs_optimizer:   PABS 位置感知束搜索优化器（基线：插入定长对抗序列 + 位置搜索）
    - aggd_optimizer:   AGGD 原版近似贪婪梯度下降优化器（基线：首部插入定长对抗序列）
  - run_attack_ultra:   CLI 入口与主流程编排（方案四 / Ultra）
"""
