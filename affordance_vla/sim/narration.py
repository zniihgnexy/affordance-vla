"""The stages of a LIBERO-PRO test, printed in plain Chinese while the episode runs in the viewer.

`explain_scene` and `explain_loop` say what the test is before it starts (task file, goal check, instruction,
start state, how the policy is queried); `Narrator` is the `on_step` callback of
affordance_vla/sim/episode.run_episode: it prints each query, gripper command, grasp and goal event, and
draws the step in the viewer; `compare` sets the replay against the server's record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from affordance_vla.plan.plan_input import EVENT_COLUMN, GRASP_CODE, plan_contact
from affordance_vla.sim.live_view import LiveView, Marker
from affordance_vla.sim.logged_episode import Query

PERTURBATIONS = {"swap": "位置扰动：目标物体和另一个物体交换了位置",
                 "task": "任务扰动：指令和目标换成了场景里的另一个物体",
                 "object": "物体外观扰动", "lan": "指令措辞扰动"}
PLAN_KINDS = {"none": "不给计划", "oracle": "每次查询附上仿真真值算出的计划（ORACLE）", "zero": "附上空计划",
              "predicted": "服务器上的计划头根据画面预测计划"}


class WindowClosed(Exception):
    """The user closed the viewer during an episode."""


def goal_text(env) -> str:
    return " 且 ".join(f"({' '.join(term)})" for term in env.env.parsed_problem["goal_state"])


def explain_scene(suite: str, task, env, names: list[str], instruction: str, episode: int, seed: int) -> None:
    from libero.libero import get_libero_path

    kind = suite.rsplit("_", 1)[-1]
    print(f"[任务集] {suite}（{PERTURBATIONS.get(kind, '没有扰动的原版任务')}）")
    print(f"  场景文件：{Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file}")
    print(f"  任务的目标判定：{goal_text(env)}")
    print(f"  场景里可移动的物体：{', '.join(names)}")
    print(f"[指令] \"{instruction}\"")
    if instruction != task.language:
        print(f"  （原任务是 \"{task.language}\"，扰动后的任务文件把它改成了这句）")
    print(f"[起始状态] 任务集记录的第 {episode} 个初始状态；随机种子 {seed}")


def explain_loop(meta: dict, queries: list[Query], max_steps: int) -> None:
    horizon = len(queries[0].chunk) if queries else 0
    hold = "，预测的计划保持到夹爪完成这次动作" if meta.get("hold_plan") else ""
    print("[闭环] 先执行 10 步空动作让物体落稳；之后每次查询把两张相机图（第三人称 + 手腕，旋转 180°，和训练时一样）")
    print(f"  和指令交给策略，策略一次返回 {horizon} 步动作（一个动作块），执行完再查询。")
    print(f"  策略：{meta['policy']}；{PLAN_KINDS.get(meta.get('plan', 'none'), meta.get('plan'))}{hold}。")
    print(f"  本机不运行模型：每一步执行服务器上策略当时给出的动作。最多 {max_steps} 步。")


def plan_source(query: Query, run_plan: str) -> str:
    if run_plan == "oracle":
        return "仿真真值算出的计划（ORACLE）"
    if run_plan == "predicted":
        return "计划头在这次查询新预测的" if query.predicted else "保持之前预测的计划"
    return "随查询发送的计划"


def describe_plan(query: Query, ee_pos: np.ndarray, run_plan: str) -> tuple[str, np.ndarray | None]:
    """A line about the plan of a query, and the contact point it points at (None without a plan)."""
    if query.plan is None:
        return "没有计划输入（这个运行不给策略计划）", None
    contact = plan_contact(query.plan, ee_pos)
    if contact is None:
        return "计划：后面没有夹爪事件了", None
    event = "抓取" if query.plan[EVENT_COLUMN] == GRASP_CODE else "放下"
    reach_cm = 100 * float(np.linalg.norm(contact - ee_pos))
    text = (f"计划：下一个夹爪事件是{event}，接触点 {np.round(contact, 3)}，离夹爪 {reach_cm:.1f} cm"
            f"（{plan_source(query, run_plan)}）")
    return text, contact


class Narrator:
    """`on_step` callback: prints the events of the loop and shows each step in the viewer."""

    def __init__(self, queries: list[Query], names: list[str], run_plan: str, env, view: LiveView | None) -> None:
        self.queries = {query.step: (k, query) for k, query in enumerate(queries)}
        self.names, self.run_plan, self.env, self.view = names, run_plan, env, view
        self.contact: np.ndarray | None = None
        self.gripper = -1.0
        self.grasped = np.zeros(len(names), dtype=bool)
        self.in_goal = np.zeros(len(names), dtype=bool)

    @property
    def markers(self) -> tuple[Marker, ...]:
        return () if self.contact is None else (Marker(self.contact),)

    def __call__(self, step: int, obs: dict, log) -> None:
        if step in self.queries:
            k, query = self.queries[step]
            text, contact = describe_plan(query, log.rows["ee_pos"][-1], self.run_plan)
            self.contact = contact if contact is not None else self.contact
            print(f"  步 {step:3d}：第 {k + 1} 次查询 → 策略返回 {len(query.chunk)} 步的动作块；{text}", flush=True)
        if log.actions[-1][6] != self.gripper:
            self.gripper = log.actions[-1][6]
            print(f"  步 {step:3d}：夹爪{'闭合' if self.gripper > 0 else '张开'}", flush=True)
        grasped, in_goal = np.asarray(log.rows["grasped"][-1]), np.asarray(log.rows["in_goal"][-1])
        for index in np.flatnonzero(grasped & ~self.grasped):
            print(f"  步 {step:3d}：两根手指都接触到 {self.names[index]}（抓住）", flush=True)
        for index in np.flatnonzero(in_goal & ~self.in_goal):
            print(f"  步 {step:3d}：{self.names[index]} 满足目标条件", flush=True)
        self.grasped, self.in_goal = grasped, in_goal
        if self.view is not None and not self.view.show(self.env, self.markers):
            raise WindowClosed


def compare(meta: dict, logged_ee: np.ndarray, arrays: dict[str, np.ndarray], success: bool) -> None:
    """The replay against the server's record: outcome, steps and the end effector's path."""
    replayed = arrays["ee_pos"]
    steps = min(len(logged_ee), len(replayed))
    drift_mm = 1000 * np.linalg.norm(replayed[:steps] - logged_ee[:steps], axis=1)
    print(f"[对比] 服务器：{'成功' if meta['success'] else '失败'}、{meta['steps']} 步；"
          f"本机：{'成功' if success else '失败'}、{len(arrays['actions'])} 步；"
          f"末端轨迹偏差中位数 {np.median(drift_mm):.1f} mm，最大 {drift_mm.max():.1f} mm")
    if success != meta["success"]:
        print("  结果不一致：本机 MuJoCo 的数值和服务器不完全相同，回放累积了偏差；以服务器的记录为准。")
