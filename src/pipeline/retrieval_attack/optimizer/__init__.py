"""
EvilPatch 检索器对抗攻击优化器子包。

包含五种检索投毒策略：
  - naive_optimizer: 不做任何对抗优化，直接使用原始 buggy_code
  - aggs_optimizer:  AGGS 改进贪婪梯度搜索（整篇文档就地替换 + 安全词表）
  - abgs_optimizer:  ABGS 改进束梯度搜索（整篇文档就地替换 + 安全词表）
  - pabs_optimizer:  PABS 位置感知束搜索（基线：插入定长对抗序列 + 位置搜索）
  - aggd_optimizer:  AGGD 原版近似贪婪梯度下降（基线：首部插入定长对抗序列）
"""
