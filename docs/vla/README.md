# VLA branch

`vla` 分支的研究方向：改变 VLA 的训练方式。训练时让 VLA 预测"接触点–接近方向–路径点"计划，约束动作与计划一致，并加入 episode 内的轻量 affordance 状态，目标是在杂乱的桌面操作仿真中提高成功率、减少抓错物体、方向错误和碰撞。必须训练 VLA，这是和 `mem` 分支（环境记忆系统）的区别。`main` 上的文档是两者共享的 brainstorming 基础。

## 入口

1. [Pipeline v2（2026-11-10）](2026-11-10_VLA_branch_Pipeline_v2.md)：现行设计，包括方法、backbone 和 benchmark 选型、实验配置、阶段计划和后续检索清单。
2. [检索结果：backbone、benchmark 与近邻（2026-11-10）](2026-11-10_VLA_branch_检索结果_backbone_benchmark_近邻.md)：三个检索 agent 的原始报告。
3. [检索清单与 Pipeline v1（2026-11-10）](2026-11-10_VLA_branch_检索清单与Pipeline_v1.md)：已被取代，保留作历史记录。

后续检索结果、选型和实验记录都放在本目录，文件名以日期开头。
