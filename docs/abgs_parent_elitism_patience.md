- （ours）局部—全局混合近似束梯度搜索法（Approximate Beam Gradient Search，ABGS）
   1. $$a\leftarrow\operatorname{Tokenizer}(C)=[t_1,t_2,\dots,t_l]$$：使用干净代码 $$C$$ 初始化对抗样本；special token 和文档 instruction prefix 被冻结，不计入可扰动位置
   2. $$\mathcal M_a\leftarrow\{i\mid t_i\notin\mathcal K,\ i\in1\dots|a|\},\qquad m_0\leftarrow|\mathcal M_a|$$：识别初始可扰动位置，并保证 $$m_0\le n$$
      1. 例如，$$n=6000$$ 时，可扰动词元数为6000的样本可以参与优化，6001则在数据准备阶段被过滤
   3. $$B\leftarrow\max\left(1,\left\lfloor\frac{n}{m_0+g}\right\rfloor\right),\qquad K\leftarrow\left\lfloor\frac{n}{B}\right\rfloor$$：根据初始可扰动位置数动态计算 Beam 数 $$B$$ 和每个 Beam 的新候选评估预算 $$K$$，其中 $$g$$ 为 `global_basic_budget`
      1. 例如，$$n=6000,m_0=50,g=50$$ 时，$$B=60,K=100$$；$$m_0=6000$$ 时，$$B=1,K=6000$$
   4. $$\mathcal L(a)\leftarrow\operatorname{Loss}(a,Q^+),\qquad\operatorname{AvgSim}(a)\leftarrow\frac{1}{|Q^+|}\sum_{q^+\in Q^+}\operatorname{Sim}(q^+,a)$$：计算初始真实损失和平均相似度；`Loss` 可由配置选择 `expected_sim`、`softmin` 或 `hybrid`
   5. $$\mathcal B\leftarrow\{a\},\qquad L_{best}\leftarrow\mathcal L(a),\qquad S_{best}^{avg}\leftarrow\operatorname{AvgSim}(a),\qquad c\leftarrow0$$：初始化 Beam、历史最优指标和 patience 计数器；每个 Beam 同时保存自己的局部 depth、全局 offset、梯度和候选缓存
   6. $$\textbf{for }j=1\textbf{ To }N\textbf{ do}$$：开始第 $$j$$ 轮迭代
      1. $$\mathcal C_{pool}\leftarrow\operatorname{Top\text{-}E}_{s\in\mathcal B}\mathcal L(s)$$：仅将当前 Beam 中损失最小的 $$E=\min(\text{elite\_num},|\mathcal B|,B)$$ 个父节点加入候选池，其他父节点本身不保留，但仍可生成子节点
         1. 例如，$$|\mathcal B|=3,E=1$$ 时，只有最优父节点 $$b_1$$ 直接进池；$$b_2$$ 和 $$b_3$$ 必须依靠新子节点继续留在搜索中
      2. $$\textbf{for each }s\in\mathcal B\textbf{ do}$$：遍历所有父节点，包括未被精英保留的父节点
         1. $$\mathcal M_s\leftarrow\{i\mid t_i\notin\mathcal K,\ i\in1\dots|s|\},\qquad m_s\leftarrow|\mathcal M_s|$$：重新识别当前序列的可扰动位置
         2. $$n_s^{local}\leftarrow m_s,\qquad n_s^{global}\leftarrow\max(0,K-m_s)$$：为每个可扰动位置保留一个局部候选，剩余预算用于全局搜索，从而保证 $$n_s^{local}+n_s^{global}\le K$$ 且全部 Beam 的全新评估数不超过 $$n$$
         3. $$\nabla_{e_{t_i}}\mathcal L(s)\leftarrow\operatorname{GradientStorage.get}()$$：新子节点第一次被展开时计算梯度；未变化的精英父节点直接复用上一轮缓存的 loss、embedding 和 gradient
         4. $$\mathbf S\in\mathbb R^{|\mathcal V_{safe}|\times|\mathcal M_s|},\qquad S_{v,i}=(e_v-e_{t_i})^\top\bigl(-\nabla_{e_{t_i}}\mathcal L(s)\bigr)$$：计算修正后的一阶泰勒替换收益，并屏蔽 $$v=t_i$$ 的无效替换
            1. 例如，位置 $$i$$ 的候选投影为0.30、原词元投影为0.20，替换收益为0.10；位置 $$k$$ 分别为0.35和0.34，收益只有0.01，因此修正后应优先选择位置 $$i$$
         5. $$\mathcal X_i\leftarrow\mathcal V_{ranked}(i)[d_s^{local}:d_s^{local}+1]$$：局部分支在每个可扰动位置选择该 Beam 当前 depth 处的一个词元，共尝试 $$m_s$$ 个局部候选
            1. 例如，位置 $$i_1$$ 的排名为 $$[A,B,C]$$，$$d_s^{local}=0$$ 时选 $$A$$，若父节点被精英保留，下一轮 $$d_s^{local}=1$$ 时选 $$B$$
         6. $$\mathbf s_{ranked}\leftarrow\operatorname{argsort}(\operatorname{Flatten}(\mathbf S))^\downarrow$$：将所有 $$(i,v)$$ 组合按替换收益全局降序排列
         7. $$\textbf{while }fresh_s^{global}<n_s^{global}\textbf{ and }o_s^{global}<|\mathbf s_{ranked}|\textbf{ do}$$：全局分支从该 Beam 自己的扫描指针 $$o_s^{global}$$ 开始向后检查候选
            1. $$o_s^{global}\leftarrow o_s^{global}+1$$：每检查一个全局排名候选就推进指针，使下一轮不会从原位置重新扫描
            2. $$\textbf{if }(i,v)\text{ 已在当前轮候选池中 then continue}$$：局部与全局分支在同一轮命中相同替换时只保留一份，全局分支继续向后补位
            3. $$\textbf{if }(i,v)\in\mathcal H_s\text{ and admitted\_once=false then replay and continue}$$：若候选在历史缓存中但从未进入 Beam，就复用已有 loss、AvgSim 和 embedding 重新入池，同时继续向后寻找全新候选
               1. 例如，全局预算为50，扫描中命中8个未入 Beam 的缓存候选，则这8个候选免前向传播重新入池，全局分支仍继续向后找满50个全新候选
            4. $$\textbf{if }(i,v)\in\mathcal H_s\text{ and admitted\_once=true then continue}$$：已经进入过 Beam 的旧路径不再回流，避免搜索循环
            5. $$\mathcal C_{fresh}\leftarrow\mathcal C_{fresh}\cup\{s[t_i\leftarrow v]\},\qquad fresh_s^{global}\leftarrow fresh_s^{global}+1$$：从未评估过的候选占用一个真实新评估预算
         8. $$d_s^{local}\leftarrow d_s^{local}+1$$：若父节点 $$s$$ 属于本轮的 $$E$$ 个精英父节点，就在入池时预先推进其局部 depth；全局 offset 已在扫描中独立推进
      3. $$\textbf{for each }x\in\operatorname{Unique}(\mathcal C_{fresh})\textbf{ do}$$：按完整 token 序列去重后，仅对全新候选执行真实前向评估
         1. $$\mathcal L(x),\operatorname{AvgSim}(x),\operatorname{Sim}(x,Q^+)\leftarrow\operatorname{Forward}(x)$$：用真实损失而非泰勒近似分数决定候选是否入束，并把结果写入对应父节点的缓存
      4. $$\mathcal C_{pool}\leftarrow\operatorname{Unique}(\mathcal C_{pool})$$：对父节点、缓存重放候选和全新候选按完整 token 序列去重；同序列同 loss 时优先保留父节点状态
      5. $$\mathcal B\leftarrow\operatorname{Top\text{-}B}_{x\in\mathcal C_{pool}}\mathcal L(x)$$：从部分精英父节点和全部子节点中选择真实损失最小的 $$B$$ 个状态；新子节点的 $$d^{local}=0,o^{global}=0$$ 且候选缓存为空
      6. $$L_{iter}\leftarrow\min_{x\in\mathcal B}\mathcal L(x),\qquad S_{iter}^{avg}\leftarrow\max_{x\in\mathcal B}\operatorname{AvgSim}(x)$$：分别计算当前 Beam 中的最优损失和最大平均相似度
      7. $$c\leftarrow\begin{cases}0,&L_{iter}<L_{best}-\delta_L\ \lor\ S_{iter}^{avg}>S_{best}^{avg}+\delta_S\\c+1,&\text{otherwise}\end{cases}$$：Loss 或 AvgSim 任一指标显著改善就重置 patience，两者都未改善才累加
         1. 例如，$$\delta_L=\delta_S=10^{-4}$$，本轮 Loss 没有改善但 AvgSim 提升了0.002，则 $$c$$ 仍重置为0
      8. $$\textbf{if }\exists a\in\mathcal B,\ \forall q^+\in Q^+,\ \operatorname{Sim}(q^+,a)>\max\bigl(\operatorname{Sim}(q^+,C),\tau_{stop}\bigr)\textbf{ then}$$：若某个 Beam 对所有代理正样本都超过干净代码基线和成功阈值，立即返回该对抗代码
      9. $$\textbf{if }c\ge P\textbf{ then return }a_{best}$$：Loss 与 AvgSim 连续 $$P$$ 轮都未显著改善时触发 patience 早停；$$P=0$$ 表示禁用该早停
      10. $$\textbf{if }\mathcal C_{pool}\text{ 无子节点且所有入束状态的局部与全局排名均耗尽 then return }a_{best}$$：候选空间真正耗尽时直接结束，不等待 patience 产生空轮次
   7. $$\textbf{return }A=\operatorname{Tokenizer.decode}(a_{best})$$：达到 $$N$$ 轮时返回历史最优 Loss 样本，而不是被最终一轮的临时状态覆盖
