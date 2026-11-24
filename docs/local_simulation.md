# 本机仿真：在 MuJoCo 里看一次测试是怎么跑的

Mac 上只跑仿真（CPU），不跑模型。三条命令都在 MuJoCo 自己的窗口里实时显示：可以旋转视角、暂停，也可以用鼠标拖动物体。

## 准备（只做一次）

```bash
bash envs/local/setup_sim_env.sh     # conda 环境 affordance-vla-sim：MuJoCo 3.2.3、robosuite 1.4.0，版本和服务器一致
bash scripts/local/fetch_sim_data.sh # LIBERO-PRO（GitHub 固定提交 eafdb80）+ 扰动任务文件 + 服务器评测日志（去掉图像）
```

Mac 上只放仿真需要的东西，都在 git 忽略的目录里：

- `third_party/LIBERO-PRO`：约 650 MB，和服务器上的副本逐文件一致，可以用 `VERIFY=1` 核对；
- `runs/libero_pro/<运行>/`：每条日志约 36 kB；
- `runs/plan_following/<数据集>/offsets.json`：抓取点和放置点的偏移。

数据集、模型和完整日志都只在服务器的 NAS 上。

## 三条命令

```bash
bash scripts/local/sim.sh scene  --suite libero_object_swap --task 2 --episode 0
bash scripts/local/sim.sh grasp  --suite libero_object_swap --task 2 --object ketchup_1
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
bash scripts/local/sim.sh list   runs/libero_pro/predicted_g1_hold_s10     # 列出可以回放的测试
```

- **scene**：打开指定任务集、任务、起始状态的场景，物理一直在跑。橙点是每个物体的抓取点，也就是示范里夹爪闭合时的位置，ORACLE 计划指向的就是这些点。
- **grasp**：用脚本控制器去抓指定物体。流程是：移到抓取点正上方 → 下降 → 闭合 → 抬起 → 移到放置点上方 → 放下 → 张开。橙点是抓取点，蓝点是放置点。最后报告两件事：抬起时有没有抓住，物体有没有进目标区域。这个控制器只读仿真里的真值，不看图像，用来观察仿真和计划点，不是策略。
- **replay**：把服务器上某次测试原样再跑一遍。用的是同一个任务文件、同一个起始状态、同一个随机种子和同一条指令；每一步执行的都是服务器上策略当时给出的动作。终端按步骤讲解，窗口里的橙点是策略当时拿到的计划接触点。最后和服务器的记录对比：成功与否、步数、末端轨迹偏差。

默认设置在 `configs/sim/local.yaml`。可以用 `--set` 改单项，例如：

- `--set viewer.realtime=0.5`：半速播放；
- `--set grasp.place=false`：只抓不放。

## 一次 LIBERO-PRO 测试的流程（replay 终端输出的就是这几段）

1. **任务**：任务集加任务编号，对应一个 BDDL 场景文件和它的目标判定，例如 `(in salad_dressing_1 basket_1_contain_region)`。

   - 位置扰动（`*_swap`）：把目标物体和另一个物体换了位置；
   - 任务扰动（`*_task`）：把指令和目标换成场景里的另一个物体。
2. **指令**：交给策略的那句话。任务扰动下，用的是扰动后任务文件里改过的说法。
3. **起始状态**：任务集预先记录的第 k 个初始状态，加上随机种子；先执行 10 步空动作，让物体落稳。
4. **闭环**：每次查询交给策略两样东西：

   - 两张相机图：第三人称和手腕视角，都旋转 180°，和训练时一样；
   - 指令；我们的方法还附带计划，也就是下一个夹爪事件、接触点和接近方向。

   策略一次返回 8 步动作（一个动作块），执行完再查询。用预测计划时，计划会一直保持到夹爪完成这次抓取或放下。
5. **结束**：满足以下任一条件就结束：

   - LIBERO 的目标判定成功；
   - 指令要的物体满足了目标；
   - 到了步数上限：Object 280 步，Spatial 220 步。

代码：

- 这套循环在 `affordance_vla/sim/episode.py`，服务器评测（`scripts/eval/rollout_libero_logged.py`）和本机回放用的是同一份；
- 讲解文字在 `affordance_vla/sim/narration.py`；
- 窗口在 `affordance_vla/sim/live_view.py`。

## 复现精度

已经核对过三条日志，涵盖成功和失败、有计划和没有计划，结果和步数都和服务器一致，末端轨迹偏差都在 0.5 mm 以内。不过 Mac 上的 MuJoCo 和服务器不保证逐位相同。如果某条回放和服务器结果不一致，终端会提示，以服务器记录为准。

## macOS 上的两点说明

- MuJoCo 的交互窗口要用 `mjpython` 启动，`scripts/local/sim.sh` 已经处理。
- robosuite 1.4 在 macOS 上做离屏渲染会创建隐藏窗口，这和 mjpython 冲突。本机工具改用 MuJoCo 自带的 CGL 上下文（`offscreen_without_windows`），不影响服务器。
