# 纯净版 ABGS：论文算法流程与命名建议

> 本文档描述 EvilPatch 主实验采用的纯净搜索设置，而非
> [`abgs_optimizer.py`](../src/pipeline/retrieval_attack/optimizer/abgs_optimizer.py)
> 中所有可选扩展的并集。对应实验设置为
> `loss_func=expected_sim`、`dynamic_weight=false`、
> `use_safe_vocab=false` 和 `elite_num=0`。
>
> **配置核对提示：**
> [`abgs.yml.example`](../configs/attack/retrieval/gte/abgs/abgs.yml.example)
> 当前仍以 `elite_num: 1` 为生效值。若使用该示例复现实验，应将其显式改为
> `elite_num: 0`，否则实际运行的是带父节点精英保留的扩展版本。

## 1. 精简范围

纯净版本保留以下核心机制：

1. 使用静态、均匀的期望相似度损失优化有毒 BFP 的 `buggy_code`；
2. 根据初始可扰动词元数和每轮候选评估预算动态确定束宽；
3. 使用一阶泰勒替换收益对离散词元替换进行排序；
4. 结合逐位置覆盖候选与全局高分候选；
5. 使用真实前向损失重新评估候选，并执行子节点式 Top-\(B\) 束更新；
6. 使用成功阈值、patience 和最大迭代轮数控制终止。

以下扩展在该实验设置下不参与有效搜索，因此不写入论文主算法：

| 移除的组件 | 在纯净设置下不生效的原因 |
|---|---|
| 动态损失权重 | `dynamic_weight=false`，所有代理正查询具有相同权重 |
| Soft-Min 与混合损失 | `loss_func=expected_sim` |
| 关键字、标点和 ASCII 安全词表 | `use_safe_vocab=false`；仅保留 prefix、special token 等必要的结构性硬约束 |
| 父节点精英保留 | `elite_num=0`，候选池中不直接加入父节点 |
| 每个束的局部深度递增 | 新一轮束全部由子节点重新构造，其局部深度始终从 0 开始 |
| 跨轮全局扫描偏移 | 新子节点的全局扫描位置始终从 0 开始 |
| 父节点候选缓存与缓存重放 | 父节点不会跨轮保留，其缓存没有下一轮消费者 |
| 跨轮梯度复用 | 每个入束子节点都是新状态，下一轮重新计算梯度 |
| Loss/AvgSim 双指标 patience | 静态均匀损失满足 \(\mathcal L(a)=-\operatorname{AvgSim}(a)\)；当两项改善阈值相同时，二者的 patience 判定等价 |

这里的“不使用安全词表”不意味着可以修改模型输入中的任意位置。当前实现仍然：

- 冻结文档编码器的 instruction prefix；
- 冻结 tokenizer 的 special token；
- 从替换候选集合中排除 special token，以及实现所排除的 added token。

除此之外，代码区域内的非 special token 均可被替换，候选来自完整的基础非 special
词表。

## 2. 问题定义与符号

攻击者优化有毒 BFP 的缺陷代码侧，使其在未知受害查询到来时更容易被检索。用户查询
不可见，因此使用历史数据构造的代理正查询集合 \(Q^+=\{q_r^+\}_{r=1}^{R}\)
近似目标查询分布。

| 符号 | 含义 |
|---|---|
| \(C\) | 待优化的初始 `buggy_code` |
| \(a=[t_1,\ldots,t_l]\) | 当前词元序列，包含被冻结的文档 prefix 和代码词元 |
| \(f_d(\cdot)\) | 白盒检索器的文档编码函数 |
| \(f_q(\cdot)\) | 检索器的查询编码函数 |
| \(Q^+\) | 代理正查询集合 |
| \(\mathcal V\) | 去除 special/added token 后的基础候选词表 |
| \(\mathcal M(a)\) | 当前序列中的可扰动位置集合 |
| \(m_0\) | 初始可扰动位置数 \(|\mathcal M(a_0)|\) |
| \(N\) | 最大迭代轮数 |
| \(n\) | 每轮最多执行真实前向评估的全新候选数 |
| \(g\) | 为全局候选预留的基础预算 `global_basic_budget` |
| \(B\) | 动态束宽 |
| \(K\) | 每个父节点的候选预算 |
| \(\tau_{\mathrm{stop}}\) | 单个代理查询相似度的成功阈值 |
| \(P\) | patience 轮数 |
| \(\delta\) | patience 所需的最小 Loss 改善量 |

## 3. 静态检索优化目标

令

$$
\mathbf z(a)=f_d(a),\qquad
\mathbf u_r=f_q(q_r^+).
$$

纯净版本最小化负平均余弦相似度：

$$
\mathcal L(a)
=-\frac{1}{R}\sum_{r=1}^{R}
\operatorname{cos}\bigl(\mathbf u_r,\mathbf z(a)\bigr).
$$

因此，

$$
\operatorname{AvgSim}(a)
=\frac{1}{R}\sum_{r=1}^{R}
\operatorname{cos}\bigl(\mathbf u_r,\mathbf z(a)\bigr)
=-\mathcal L(a).
$$

该目标对所有代理正查询均匀加权，不包含动态权重、难例挖掘、Soft-Min 或混合损失。

## 4. 精简后的算法流程

- 近似束梯度搜索法（Approximate Beam Gradient Search，ABGS）
  1. $$a_0 \leftarrow \operatorname{Prefix}\oplus\operatorname{Tokenizer}(C)=[t_1,t_2,\ldots,t_l]$$：使用干净缺陷代码 \(C\) 初始化词元序列 \(a_0\)，并冻结检索器所需的文档 prefix。
  2. $$\mathcal M(a_0)\leftarrow\{i\mid i\ge l_{\mathrm{prefix}},\ t_i\notin\mathcal V_{\mathrm{special}}\},\qquad m_0\leftarrow|\mathcal M(a_0)|$$：识别代码区域内的可扰动位置；替换候选词表 \(\mathcal V\) 为排除 special/added token 后的完整基础词表，并要求 \(0<m_0\le n\)。
  3. $$B\leftarrow\max\left(1,\left\lfloor\frac{n}{m_0+g}\right\rfloor\right),\qquad K\leftarrow\left\lfloor\frac{n}{B}\right\rfloor,\qquad BK\le n$$：根据初始可扰动位置数 \(m_0\)、每轮全新候选评估预算 \(n\) 和全局基础预算 \(g\) 动态确定束宽 \(B\) 与每个父节点的候选预算 \(K\)。
  4. $$\mathcal B\leftarrow\{a_0\},\qquad a^\star\leftarrow a_0,\qquad L^\star\leftarrow\mathcal L(a_0),\qquad c\leftarrow0$$：初始化大小为1的束集合、历史最优序列、历史最优损失和patience计数器。
  5. $$\textbf{for }j=1\textbf{ to }N\textbf{ do}$$：开始至多 \(N\) 轮搜索。
     1. $$\mathcal C_{\mathrm{pool}}\leftarrow\emptyset$$：为当前轮次初始化全局候选池。
     2. $$\textbf{for each }s\in\mathcal B\textbf{ do}$$：遍历当前束中的每个父状态。
        1. $$\mathcal M\leftarrow\mathcal M(s)$$：获取当前状态的可扰动位置集合。
        2. $$\mathcal L(s)=-\frac{1}{|Q^+|}\sum_{q^+\in Q^+}\operatorname{Sim}(q^+,s)$$：以所有代理正查询的静态、均匀平均相似度计算当前状态的检索损失。
        3. $$\mathbf g_i\leftarrow\nabla_{\mathbf e_{t_i}}\mathcal L(s),\qquad G_{v,i}\leftarrow(\mathbf e_v-\mathbf e_{t_i})^\top(-\mathbf g_i),\quad i\in\mathcal M,\ v\in\mathcal V\setminus\{t_i\}$$：计算各可扰动位置的梯度，并使用一阶泰勒近似得到词元替换的预测损失下降量。
        4. $$\mathcal C_{\mathrm{pos}}(s)\leftarrow\left\{s\left[t_i\leftarrow\underset{v\in\mathcal V,\ v\ne t_i}{\arg\max}\ G_{v,i}\right]\ \middle|\ i\in\mathcal M\right\}$$：为每个可扰动位置保留预测收益最高的一个替换，形成逐位置覆盖候选。
        5. $$K_{\mathrm{global}}\leftarrow\max(0,K-|\mathcal M|)$$：将单父节点预算中除逐位置覆盖候选外的剩余部分分配给全局候选。
        6. $$\mathcal C_{\mathrm{global}}(s)\leftarrow\operatorname{Top-}K_{\mathrm{global}}\!\left(\left\{s[t_i\leftarrow v]\mid i\in\mathcal M,\ v\in\mathcal V,\ v\ne t_i,\ s[t_i\leftarrow v]\notin\mathcal C_{\mathrm{pos}}(s)\right\};G_{v,i}\right)$$：在所有尚未入选的位置—词元组合中，按照 \(G_{v,i}\) 选择预测收益最高的全局替换候选。
        7. $$\mathcal C_{\mathrm{pool}}\leftarrow\mathcal C_{\mathrm{pool}}\cup\operatorname{Unique}\!\left(\mathcal C_{\mathrm{pos}}(s)\cup\mathcal C_{\mathrm{global}}(s)\right)$$：合并当前父状态的两类候选，并加入当前轮次的全局候选池。
     3. $$\mathcal C_{\mathrm{pool}}\leftarrow\operatorname{Unique}(\mathcal C_{\mathrm{pool}})$$：按照完整词元序列对不同父状态产生的候选统一去重。
     4. $$\textbf{if }\mathcal C_{\mathrm{pool}}=\emptyset\textbf{ then break}$$：若不存在合法子节点，则以候选空间耗尽结束搜索。
     5. $$\textbf{for each }x\in\mathcal C_{\mathrm{pool}},\qquad \mathcal L(x)=-\frac{1}{|Q^+|}\sum_{q^+\in Q^+}\operatorname{Sim}(q^+,x)$$：对每个去重后的候选执行真实前向计算；梯度分数不参与最终排序。
     6. $$\mathcal B\leftarrow\underset{x\in\mathcal C_{\mathrm{pool}}}{\operatorname{Top-}B}\ [-\mathcal L(x)]$$：选择真实损失最小的至多 \(B\) 个子节点更新束集合，不保留任何父节点精英。
     7. $$c\leftarrow\begin{cases}0,&\underset{x\in\mathcal B}{\min}\ \mathcal L(x)<L^\star-\delta,\\c+1,&\text{otherwise}\end{cases}$$：若本轮最优子节点未使历史最优损失改善至少 \(\delta\)，则累加patience计数器。
     8. $$a^\star\leftarrow\underset{x\in\{a^\star\}\cup\mathcal B}{\arg\min}\ \mathcal L(x),\qquad L^\star\leftarrow\mathcal L(a^\star)$$：独立更新历史最优状态，避免子节点式束搜索暂时退化时丢失已发现的最优结果。
     9. $$\textbf{for each }a\in\mathcal B,\quad \textbf{if }\forall q^+\in Q^+,\ \operatorname{Sim}(q^+,a)>\max\!\left(\operatorname{Sim}(q^+,a_0),\tau_{\mathrm{stop}}\right)\textbf{ then}$$：检查束中状态是否对全部代理正查询同时超过干净基线和停止阈值。
        1. $$\textbf{return }A=\operatorname{Tokenizer.decode}\!\left(a[l_{\mathrm{prefix}}:]\right)$$：满足成功条件时，去除文档 prefix 并返回当前对抗代码。
     10. $$\textbf{if }P>0\textbf{ and }c\ge P\textbf{ then break}$$：连续 \(P\) 轮无显著改善时提前终止；当 \(P=0\) 时禁用该条件。
  6. $$\textbf{return }A=\operatorname{Tokenizer.decode}\!\left(a^\star[l_{\mathrm{prefix}}:]\right)$$：达到最大迭代轮数、触发patience或候选空间耗尽后，返回去除文档 prefix 的历史最优对抗代码，而非最后一轮束中的机械最优项。

### 4.1 初始化与动态预算

首先对干净缺陷代码进行分词，并在其前部拼接检索器所需的文档 instruction prefix：

$$
a_0\leftarrow
\operatorname{Prefix}\oplus\operatorname{Tokenizer}(C).
$$

可扰动位置仅包含代码区域中的非 special token：

$$
\mathcal M(a_0)
=\{i\mid i\ge l_{\mathrm{prefix}},\ t_i\notin\mathcal V_{\mathrm{special}}\},
\qquad
m_0=|\mathcal M(a_0)|.
$$

实现要求 \(0<m_0\le n\)。根据初始可扰动位置数计算束宽和单束预算：

$$
B=\max\left(1,\left\lfloor
\frac{n}{m_0+g}
\right\rfloor\right),
\qquad
K=\left\lfloor\frac{n}{B}\right\rfloor.
$$

该设计使完整束形成后每轮的候选上界满足
\(B K\le n\)。搜索从单个初始状态开始，因此第一轮最多产生 \(K\) 个候选；
完整序列去重还可能进一步减少实际前向评估数。

初始化束和历史最优状态：

$$
\mathcal B_0=\{a_0\},\qquad
a^\star=a_0,\qquad
L^\star=\mathcal L(a_0),\qquad
c=0.
$$

其中 \(c\) 为 patience 计数器。

### 4.2 梯度近似与替换收益

在第 \(j\) 轮，对每个父状态 \(s\in\mathcal B_{j-1}\) 执行一次前向和反向传播，
得到所有可扰动位置的词嵌入梯度：

$$
\mathbf g_i
=\nabla_{\mathbf e_{t_i}}\mathcal L(s),
\qquad i\in\mathcal M(s).
$$

将位置 \(i\) 的当前词元 \(t_i\) 替换为候选词元 \(v\) 时，一阶泰勒展开为

$$
\mathcal L\bigl(s[t_i\leftarrow v]\bigr)
\approx
\mathcal L(s)
+\mathbf g_i^\top(\mathbf e_v-\mathbf e_{t_i}).
$$

由此定义预测损失下降量：

$$
G_{v,i}
=(\mathbf e_v-\mathbf e_{t_i})^\top(-\mathbf g_i).
$$

\(G_{v,i}\) 越大，替换越可能降低真实损失。算法屏蔽恒等替换 \(v=t_i\)。
该分数仅用于缩小离散搜索空间，最终束更新依据的仍是真实前向损失。

### 4.3 逐位置覆盖与全局高分补充

令 \(m_s=|\mathcal M(s)|\)。对于当前实现的纯净全词表设置，单词元替换不会改变
可扰动位置数，因而通常有 \(m_s=m_0\le K\)。

每个父状态首先为每个可扰动位置保留分数最高的一个替换，从而形成逐位置覆盖集合：

$$
\mathcal C_{\mathrm{pos}}(s)
=
\left\{
s\left[t_i\leftarrow
\underset{v\in\mathcal V,\ v\ne t_i}{\arg\max}\ G_{v,i}
\right]
\;\middle|\;
i\in\mathcal M(s)
\right\}.
$$

这一步不是带有跨轮 depth 的局部深搜，而是一次性的**位置覆盖候选生成**。

剩余全局预算为

$$
K_{\mathrm{global}}(s)=\max(0,K-m_s).
$$

随后将分数矩阵 \(G\) 中所有合法的 \((i,v)\) 对统一排序，从中选择尚未出现在
\(\mathcal C_{\mathrm{pos}}(s)\) 中的前 \(K_{\mathrm{global}}(s)\) 个替换：

$$
\mathcal C_{\mathrm{global}}(s)
=
\operatorname{Top}\text{-}K_{\mathrm{global}}(s)
\left\{
s[t_i\leftarrow v]
\;\middle|\;
i\in\mathcal M(s),\ v\in\mathcal V,\ v\ne t_i
\right\}.
$$

逐位置集合保障所有位置获得一次探索机会；全局集合则把剩余预算集中到预测收益最高
的替换上。当前父状态的候选集合为

$$
\mathcal C(s)
=\operatorname{Unique}\left(
\mathcal C_{\mathrm{pos}}(s)
\cup
\mathcal C_{\mathrm{global}}(s)
\right).
$$

### 4.4 真实评估与子节点式束更新

合并所有父状态的候选，并按照完整词元序列去重：

$$
\mathcal C_j
=\operatorname{Unique}
\left(
\bigcup_{s\in\mathcal B_{j-1}}\mathcal C(s)
\right).
$$

对 \(\mathcal C_j\) 中的每个候选执行真实检索器前向传播，计算
\(\mathcal L(x)\) 及其逐查询相似度。泰勒分数不参与最终排序：

$$
\mathcal B_j
=
\underset{x\in\mathcal C_j}{\operatorname{Top}\text{-}B}
\bigl[-\mathcal L(x)\bigr].
$$

等价地，保留真实损失最小的至多 \(B\) 个候选。由于 `elite_num=0`，
\(\mathcal B_{j-1}\) 中的父状态不会直接进入 \(\mathcal B_j\)；每个入束状态必须是
某个父状态的一步词元替换。该更新不要求子节点优于其父节点，因此搜索轨迹可以暂时
变差。为避免丢失已发现的最优结果，算法独立维护历史最优状态：

$$
a^\star
\leftarrow
\underset{x\in\{a^\star\}\cup\mathcal B_j}
{\arg\min}\ \mathcal L(x).
$$

### 4.5 终止条件

每轮束更新后依次检查以下条件。

**成功阈值。** 对任意 \(a\in\mathcal B_j\)，若其对每个代理正查询均满足

$$
\operatorname{cos}\bigl(f_q(q_r^+),f_d(a)\bigr)
>
\max\left(
\operatorname{cos}\bigl(f_q(q_r^+),f_d(a_0)\bigr),
\tau_{\mathrm{stop}}
\right),
\quad \forall r,
$$

则攻击达到优化器定义的成功条件并返回该状态。

**Patience。** 令

$$
L_j=\min_{a\in\mathcal B_j}\mathcal L(a).
$$

在静态均匀损失下，只需使用 Loss 判定 patience：

$$
c\leftarrow
\begin{cases}
0, & L_j<L^\star_{\mathrm{old}}-\delta,\\
c+1, & \text{otherwise}.
\end{cases}
$$

当 \(P>0\) 且 \(c\ge P\) 时终止。若 \(P=0\)，则禁用 patience 早停。

**其他终止条件。** 若不存在合法子节点，则判定候选空间耗尽；若迭代达到 \(N\)，
则按最大轮数终止。除成功阈值直接命中的情况外，算法最终解码并返回历史最优
\(a^\star\)，而不是机械地返回最后一轮的最优束状态。

## 5. 可直接转写为论文 Algorithm 的伪代码

```text
Input:
  clean buggy code C; proxy positive queries Q+
  white-box retriever (fq, fd); tokenizer vocabulary V
  maximum iterations N; fresh-evaluation budget n
  global base budget g; threshold τstop
  patience P; improvement threshold δ
Output:
  optimized buggy code A

1:  a0 ← Prefix ⊕ Tokenize(C)
2:  M ← code-side non-special-token positions in a0
3:  m0 ← |M|
4:  B ← max(1, floor(n / (m0 + g)))
5:  K ← floor(n / B)
6:  Beam ← {a0}; a* ← a0; L* ← L(a0); c ← 0
7:  for j = 1, ..., N do
8:      Cj ← ∅
9:      for each s ∈ Beam do
10:         compute L(s) and token-embedding gradients {gi}i∈M
11:         Gv,i ← (ev − eti)ᵀ(−gi) for all i ∈ M and v ∈ V \ {ti}
12:         Cpos ← the highest-scoring replacement at every position i
13:         Cglobal ← the highest-scoring max(0, K − |M|) remaining (i, v) pairs
14:         Cj ← Cj ∪ Cpos ∪ Cglobal
15:     end for
16:     Cj ← deduplicate Cj by complete token sequence
17:     evaluate every x ∈ Cj with the true forward loss L(x)
18:     if Cj = ∅ then break
19:     Beam ← the at most B children in Cj with the smallest true losses
20:     Lprev ← L*
21:     update a* and L* with the best historical loss
22:     if any beam satisfies every per-query success threshold then return Decode(beam)
23:     if min{L(x) | x ∈ Beam} < Lprev − δ then c ← 0 else c ← c + 1
24:     if P > 0 and c ≥ P then break
25: end for
26: return Decode(a*)
```

## 6. 预算与计算特性

- 每个父状态执行一次梯度计算。
- 每个父状态最多提出 \(K\) 个替换候选；完整束形成后，每轮真实评估数不超过
  \(BK\le n\)。
- 第一轮只有一个父状态，因此真实评估数上界是 \(K\)，而非 \(n\)。
- 完整序列去重使实际评估数可能小于上述上界。
- 替换分数矩阵的形状为 \(|\mathcal V|\times m_s\)，主要显存开销为
  \(O(|\mathcal V|m_s)\)。
- 候选真实损失可分批前向计算；`batch_size` 只影响执行吞吐，不改变搜索语义。

## 7. 算法命名

“近似束梯度搜索法（Approximate Beam Gradient Search, ABGS）”并非错误，但不够
精确，主要存在三个问题：

1. **“Approximate” 的修饰对象不明确。** 实际上近似的是候选替换的损失变化，而
   不是最终损失评估或 Top-\(B\) 选择。
2. **“Gradient Search” 容易被理解为连续空间梯度下降。** 当前方法使用梯度对离散
   词元替换排序，真正决定入束的是完整候选的前向损失。
3. **名称没有体现核心候选策略。** 相比通用的梯度束搜索，本方法更有辨识度的结构是
   “逐位置覆盖 + 全局高分补充 + 动态预算”。

此外，不建议直接改名为 **Gradient-Guided Beam Search (GGBS)**。该表述已被
[AgentPoison](https://arxiv.org/abs/2407.12784) 用于描述其离散触发器优化算法，
直接复用不利于在相关工作和消融实验中区分两种方法。

最终推荐使用**全局位置束搜索（GPBS）**作为算法命名
