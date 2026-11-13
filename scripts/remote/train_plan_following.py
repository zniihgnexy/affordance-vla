"""Teach the released starVLA QwenPI LIBERO action head to follow a plan (GPU: run through the queue).

A diagnostic, not the paper's configuration. The VLM stays frozen; trained are a plan encoder whose
output is added to every action token, low-rank updates of the action DiT, and the action decoder.
The plan input is the true plan (where the next gripper event happens, relative to the gripper).

Every sampled frame gives a demonstration sample (flow-matching loss on the demonstrated chunk). With
--counterfactual shift, frames on the way to a grasp are also paired with a contact moved by a few
centimetres; with shift+other, also with the grasp of another object of the scene and an instruction
that names it; with shift+other+release, frames on the way to the release are in addition paired with
a release contact moved by a few centimetres. Those samples have no demonstrated translation: it is
supervised by the plan–action consistency loss, the distance between the path of the one-step action
estimate and the moved path (or, with --supervision actions, by the rewritten actions themselves).

Usage (starvla env, from the starVLA repo root; PYTHONPATH = starVLA and this repo):
    python train_plan_following.py --ckpt .../steps_100000_pytorch_model.pt --base-vlm .../Qwen3-VL-4B-Instruct \
        --data $W/runs/plan_following/dataset --out $W/runs/plan_following/shift --counterfactual shift
Output: <out>/adapter.pt (trained parameters and settings), <out>/train.jsonl, <out>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_vla_demo import affine_unnormalise, encode, velocity  # noqa: E402

from affordance3d.path_loss import chunk_path, path_loss  # noqa: E402
from affordance3d.plan_consistency import plan_ahead  # noqa: E402
from affordance3d.plan_dataset import frame_offsets, load_tracks  # noqa: E402
from affordance3d.plan_input import PLAN_DIM, offsets_from_json  # noqa: E402
from affordance3d.plan_policy import PlanConditionedHead, dit_lora_match  # noqa: E402
from affordance3d.plan_samples import PlanSample, demo_sample, shifted_sample, without_plan  # noqa: E402

DEMO, SHIFT, OTHER = 0, 1, 2  # kinds of samples
KIND_NAMES = {DEMO: "demo", SHIFT: "shift", OTHER: "other"}
WARMUP_STEPS = 100
FINAL_LR_FRACTION = 0.1
APPROACH_SHARE = 0.5  # share of frames drawn from the approach to a grasp, where counterfactuals exist
MAX_START_DIFFERENCE = 0.05  # velocity units; bf16 rounding gives about 0.02 on the released checkpoint


@dataclass(frozen=True)
class Item:
    sample: PlanSample
    demo: int
    step: int
    instruction: str
    kind: int


def object_words(name: str) -> str:
    """'salad_dressing_1' -> 'salad dressing'."""
    return name.rsplit("_", 1)[0].replace("_", " ")


class PlanData:
    """Tracks, images and frame lists of a dataset written by build_plan_dataset.py."""

    def __init__(self, folder: Path, horizon: int, holdout: int) -> None:
        self.tracks = load_tracks(folder / "tracks.npz")
        self.rows = frame_offsets(self.tracks)
        self.agentview = np.load(folder / "agentview.npy")  # in memory: random reads from the NAS are slow
        self.wrist = np.load(folder / "wrist.npy")
        self.contact = offsets_from_json((folder / "offsets.json").read_text())
        self.scale = float(json.loads((folder / "stats.json").read_text())["motion_scale_m_per_unit"])
        self.horizon = horizon
        held_out = self._held_out(holdout)
        frames = [(d, t) for d, track in enumerate(self.tracks) for t in range(len(track.actions) - horizon + 1)]
        self.train = [frame for frame in frames if frame[0] not in held_out]
        self.test = [frame for frame in frames if frame[0] in held_out]
        self.train_approach = [frame for frame in self.train if self.on_approach(*frame)]
        self.test_approach = [frame for frame in self.test if self.on_approach(*frame)]

    def _held_out(self, count: int) -> set[int]:
        """The last `count` demonstrations of every instruction."""
        by_instruction: dict[str, list[int]] = {}
        for index, track in enumerate(self.tracks):
            by_instruction.setdefault(track.instruction, []).append(index)
        short = [text for text, group in by_instruction.items() if len(group) <= count]
        if count < 1 or short:
            raise SystemExit(f"--holdout must be at least 1 and leave training demonstrations for every task; "
                             f"it is {count}, and these tasks have no more than that: {short}")
        return {index for group in by_instruction.values() for index in group[len(group) - count:]}

    def on_approach(self, demo: int, step: int) -> bool:
        track = self.tracks[demo]
        plan = plan_ahead(track.labels, track.ee_pos[step], step)
        return plan is not None and plan.closing and shifted_sample(
            track, step, self.horizon, plan.polyline[-1], self.scale) is not None

    def example(self, demo: int, step: int, instruction: str) -> dict:
        row = int(self.rows[demo]) + step
        return {"image": [np.asarray(self.agentview[row]), np.asarray(self.wrist[row])], "lang": instruction}


def variants(data: PlanData, demo: int, step: int, mode: str, shift_radius: float, contact_noise: float,
             rng: np.random.Generator, plan_dropout: float = 0.0) -> list[Item]:
    """The demonstration sample of a frame and its counterfactual samples.

    On the way to a grasp the contact is moved within a disc ("shift") and, with "other" in the mode, to
    another object named by a changed instruction. With "release" in the mode the release contact is
    moved within the same disc while the object is carried. With probability `plan_dropout` (one draw per
    frame) the demonstration sample loses its plan input, labels unchanged. The counterfactual samples
    always keep their plan.
    """
    track = data.tracks[demo]
    noise = lambda: rng.normal(0.0, contact_noise, 3)  # noqa: E731
    demonstrated = demo_sample(track, step, data.horizon, noise())
    if plan_dropout > 0.0 and rng.uniform() < plan_dropout:
        demonstrated = without_plan(demonstrated)
    items = [Item(demonstrated, demo, step, track.instruction, DEMO)]
    plan = plan_ahead(track.labels, track.ee_pos[step], step)
    if mode == "none" or plan is None or not (plan.closing or "release" in mode):
        return items
    radius, angle = shift_radius * math.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * math.pi)
    moved = plan.polyline[-1] + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
    sample = shifted_sample(track, step, data.horizon, moved, data.scale, contact_noise=noise(),
                            allow_release="release" in mode)
    if sample is not None:
        items.append(Item(sample, demo, step, track.instruction, SHIFT))
    others = [name for name in track.names if name != track.target and name in data.contact.grasp]
    if plan.closing and "other" in mode and others:
        name = others[rng.integers(len(others))]
        contact = track.object_pos[step, track.names.index(name)] + data.contact.grasp[name]
        sample = shifted_sample(track, step, data.horizon, contact, data.scale,
                                approach=data.contact.grasp_approach[name], contact_noise=noise())
        instruction = track.instruction.replace(object_words(track.target), object_words(name))
        if sample is not None and instruction != track.instruction:
            items.append(Item(sample, demo, step, instruction, OTHER))
    return items


def collate(data: PlanData, items: list[Item], norm: tuple[np.ndarray, np.ndarray], supervision: str,
            device: str) -> dict:
    """VLM examples (one per distinct frame and instruction) and the tensors of the head's samples."""
    keys = list(dict.fromkeys((item.demo, item.step, item.instruction) for item in items))
    stack = lambda values, dtype: torch.as_tensor(np.stack(values), dtype=dtype, device=device)  # noqa: E731
    action_valid = np.stack([item.sample.action_valid for item in items])
    path_valid = np.stack([item.sample.path_valid for item in items])
    if supervision == "actions":  # ablation: relabelled actions instead of the path loss
        action_valid[:, :, :3] |= path_valid[:, :, None]
        path_valid = np.zeros_like(path_valid)
    return {
        "examples": [data.example(*key) for key in keys],
        "rows": torch.as_tensor([keys.index((i.demo, i.step, i.instruction)) for i in items], device=device),
        "kind": torch.as_tensor([item.kind for item in items], device=device),
        "actions": stack([(item.sample.actions - norm[0]) / norm[1] for item in items], torch.float32),
        "features": stack([item.sample.features for item in items], torch.float32),
        "start": stack([item.sample.start for item in items], torch.float32),
        "path_target": stack([item.sample.path_target for item in items], torch.float32),
        "path_valid": torch.as_tensor(path_valid, device=device),
        "action_valid": torch.as_tensor(action_valid, device=device),
    }


def draw_batch(data: PlanData, args: argparse.Namespace, rng: np.random.Generator) -> list[Item]:
    items: list[Item] = []
    for _ in range(args.frames):
        pool = data.train_approach if (rng.uniform() < APPROACH_SHARE and data.train_approach) else data.train
        demo, step = pool[rng.integers(len(pool))]
        items += variants(data, demo, step, args.counterfactual, args.shift_radius, args.contact_noise, rng,
                          args.plan_dropout)
    return items


def encode_rows(framework, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Frozen VLM states of the batch's examples, gathered per head sample."""
    with torch.no_grad():
        states, mask = encode(framework, batch["examples"])
    rows = batch["rows"]
    return [state[rows] for state in states], None if mask is None else mask[rows]


def path_of(batch: dict, normalised: torch.Tensor, norm: tuple[torch.Tensor, torch.Tensor],
            scale: float) -> torch.Tensor:
    """(S, H, 3) gripper positions asked for by normalised chunks that start at the batch's positions."""
    return chunk_path(batch["start"], norm[0][:3] + norm[1][:3] * normalised[..., :3], scale)


def losses(framework, plan_head: PlanConditionedHead, batch: dict, norm: tuple[torch.Tensor, torch.Tensor],
           scale: float, repeats: int) -> dict[str, torch.Tensor]:
    """Flow-matching loss on the supervised action entries and the path loss on the one-step estimate."""
    head = plan_head.head
    started = time.perf_counter()
    states, mask = encode_rows(framework, batch)
    torch.cuda.synchronize()
    vlm_seconds = time.perf_counter() - started
    tile = lambda value: value.repeat(repeats, *[1] * (value.dim() - 1))  # noqa: E731
    states, mask = [tile(state) for state in states], None if mask is None else tile(mask)
    actions, action_valid = tile(batch["actions"]), tile(batch["action_valid"]).float()
    noise = torch.randn_like(actions)
    t = head.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
    noisy = (1 - t) * noise + t * actions
    buckets = (t[:, 0, 0] * head.num_timestep_buckets).long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = plan_head.velocity(noisy, buckets, states, mask, tile(batch["features"]))
    predicted = predicted.float()
    flow = (((predicted - (actions - noise)) ** 2) * action_valid).sum() / action_valid.sum().clamp(min=1.0)
    estimate = noisy + (1 - t) * predicted
    tiled = {key: tile(batch[key]) for key in ("start", "path_target", "path_valid")}
    path = path_loss(path_of(tiled, estimate, norm, scale), tiled["path_target"], tiled["path_valid"])
    return {"flow": flow, "path": path, "vlm_seconds": vlm_seconds}


@torch.no_grad()
def open_loop(framework, plan_head: PlanConditionedHead, data: PlanData, items: list[Item], norm_np, norm,
              batch_size: int) -> dict[str, float]:
    """Sampled chunks against their targets on fixed held-out samples.

    demo: mean |error| of the translation actions (action units). shift / other: mean distance (cm) of
    the sampled chunk's path to the moved path, and the same distance for the demonstrated chunk
    (`*_if_ignored`: what a head that ignores the plan would score).
    """
    sums: dict[str, list[float]] = {}
    for first in range(0, len(items), batch_size):
        part = items[first:first + batch_size]
        batch = collate(data, part, norm_np, "path", "cuda")
        states, mask = encode_rows(framework, batch)
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = plan_head.sample(states, mask, batch["features"], torch.randn_like(batch["actions"]))
        chunk = chunk.float()
        reached = path_of(batch, chunk, norm, data.scale)
        distance = 100.0 * (reached - batch["path_target"]).norm(dim=-1)
        demo_chunks = torch.as_tensor(np.stack([(demo_sample(data.tracks[i.demo], i.step, data.horizon).actions
                                                 - norm_np[0]) / norm_np[1] for i in part]),
                                      dtype=torch.float32, device="cuda")
        ignored = 100.0 * (path_of(batch, demo_chunks, norm, data.scale) - batch["path_target"]).norm(dim=-1)
        action_error = (norm[1][:3] * (chunk - batch["actions"])[..., :3]).abs().mean(dim=(1, 2))
        for k, item in enumerate(part):
            valid = batch["path_valid"][k]
            if item.kind == DEMO:
                sums.setdefault("demo_action_error", []).append(float(action_error[k]))
            elif valid.any():
                name = KIND_NAMES[item.kind]
                sums.setdefault(f"{name}_path_cm", []).append(float(distance[k][valid].mean()))
                sums.setdefault(f"{name}_path_cm_if_ignored", []).append(float(ignored[k][valid].mean()))
    return {key: round(float(np.mean(values)), 4) for key, values in sorted(sums.items())}


def fixed_test_items(data: PlanData, args: argparse.Namespace) -> list[Item]:
    """Held-out samples that stay the same over the run: both kinds of frames, with every counterfactual."""
    rng = np.random.default_rng(1234)
    mode = "shift" if args.counterfactual == "none" else args.counterfactual
    frames = [data.test[i] for i in rng.choice(len(data.test), min(args.test_frames, len(data.test)), replace=False)]
    approach = data.test_approach
    frames += [approach[i] for i in rng.choice(len(approach), min(args.test_frames, len(approach)), replace=False)]
    return [item for demo, step in frames
            for item in variants(data, demo, step, mode, args.shift_radius, 0.0, rng, plan_dropout=0.0)]


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < WARMUP_STEPS:
        return args.lr * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(args.steps - WARMUP_STEPS, 1)
    return args.lr * (FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress)))


def unchanged_at_start(framework, plan_head: PlanConditionedHead, data: PlanData, reference_head_call) -> float:
    """Largest difference between the untrained plan head and the released head on one real frame."""
    demo, step = data.train[0]
    with torch.no_grad():
        states, mask = encode(framework, [data.example(demo, step, data.tracks[demo].instruction)])
        torch.manual_seed(0)
        noisy = torch.randn((1, plan_head.head.action_horizon, plan_head.head.action_dim), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ours = plan_head.velocity(noisy, torch.full((1,), 500, device="cuda"), states, mask,
                                      torch.zeros((1, PLAN_DIM), device="cuda"))
        return float((ours.float() - reference_head_call(noisy, states, mask).float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--counterfactual", choices=("none", "shift", "shift+other", "shift+other+release"),
                        default="shift")
    parser.add_argument("--supervision", choices=("path", "actions"), default="path",
                        help="translation of counterfactual samples: path loss, or relabelled actions (ablation)")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--frames", type=int, default=8, help="frames per step; each adds up to three samples")
    parser.add_argument("--repeats", type=int, default=2, help="noise draws per sample")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--path-weight", type=float, default=0.2)
    parser.add_argument("--shift-radius", type=float, default=0.18, help="metres; contacts move within this disc")
    parser.add_argument("--contact-noise", type=float, default=0.01, help="std (m) of the noise on the plan's contact")
    parser.add_argument("--plan-dropout", type=float, default=0.0,
                        help="probability that a demonstration sample is given no plan (labels unchanged)")
    parser.add_argument("--deep-plan", action="store_true",
                        help="also add the plan to the input of every later block of the action DiT "
                             "(the injection of Fast Plans, Faithful Actions), not only to the action tokens")
    parser.add_argument("--holdout", type=int, default=5, help="demonstrations per instruction kept for testing")
    parser.add_argument("--test-frames", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0.0 <= args.plan_dropout <= 1.0:
        parser.error(f"--plan-dropout is a probability, got {args.plan_dropout}")
    return args


def build(args: argparse.Namespace) -> tuple[torch.nn.Module, PlanConditionedHead, tuple[np.ndarray, np.ndarray],
                                             Callable[..., torch.Tensor]]:
    """The frozen framework, the plan head on top of its action head, and the action normalisation."""
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt, device="cuda", use_bf16=True,
        config_overrides=[f"framework.qwenvl.base_vlm={args.base_vlm}",
                          "framework.action_model.diffusion_model_cfg.use_canonical_forward=false"])
    framework, head = wrapper._framework, wrapper._framework.action_model
    norm_np = affine_unnormalise(wrapper.get_norm_processor())

    def reference(noisy: torch.Tensor, states: list[torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        # the released head's own pass, with its 32 learned tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return velocity(head, noisy, 500, states, mask)

    plan_head = PlanConditionedHead(head, args.rank, float(args.rank), dit_lora_match, deep=args.deep_plan)
    return framework, plan_head, norm_np, reference


def train(args: argparse.Namespace, framework, plan_head: PlanConditionedHead, data: PlanData, norm_np, norm,
          log) -> None:
    parameters = [parameter for parameter in plan_head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    spent = {"data": 0.0, "vlm": 0.0}  # seconds since the last log line
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)
        data_started = time.perf_counter()
        batch = collate(data, draw_batch(data, args, rng), norm_np, args.supervision, "cuda")
        spent["data"] += time.perf_counter() - data_started
        parts = losses(framework, plan_head, batch, norm, data.scale, args.repeats)
        spent["vlm"] += parts["vlm_seconds"]
        loss = parts["flow"] + args.path_weight * parts["path"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log({"step": step, "flow": round(float(parts["flow"]), 5), "path_cm": round(float(parts["path"]), 4),
                 "samples": len(batch["kind"]), "lr": round(learning_rate(step, args), 7),
                 "seconds": round(time.perf_counter() - started, 1),
                 "data_seconds": round(spent["data"], 1), "vlm_seconds": round(spent["vlm"], 1)})
            spent = {"data": 0.0, "vlm": 0.0}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    data = PlanData(args.data, horizon=8, holdout=args.holdout)
    framework, plan_head, norm_np, reference = build(args)
    if plan_head.head.action_horizon != data.horizon:
        raise SystemExit(f"the head predicts {plan_head.head.action_horizon} steps, the data assumes {data.horizon}")
    norm = tuple(torch.as_tensor(value, dtype=torch.float32, device="cuda") for value in norm_np)
    trained = sum(parameter.numel() for parameter in plan_head.parameters() if parameter.requires_grad)
    facts = {"args": {key: str(value) for key, value in vars(args).items()}, "motion_scale": data.scale,
             "train_frames": len(data.train), "train_approach_frames": len(data.train_approach),
             "test_frames": len(data.test), "demonstrations": len(data.tracks),
             "lora_layers": len(plan_head.lora_layers), "trained_parameters_millions": round(trained / 1e6, 2),
             "max_abs_difference_from_released_head": unchanged_at_start(framework, plan_head, data, reference)}
    print(json.dumps(facts, indent=2), flush=True)
    if facts["max_abs_difference_from_released_head"] > MAX_START_DIFFERENCE:
        raise SystemExit("the untrained plan head does not reproduce the released head: leaving out the learned "
                         "tokens or wrapping the layers changed its output")
    test_items = fixed_test_items(data, args)
    facts["open_loop_before"] = open_loop(framework, plan_head, data, test_items, norm_np, norm, 16)
    print("open loop before:", facts["open_loop_before"], flush=True)
    with (args.out / "train.jsonl").open("w") as handle:
        def log(record: dict) -> None:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(record, flush=True)

        train(args, framework, plan_head, data, norm_np, norm, log)
    facts["open_loop_after"] = open_loop(framework, plan_head, data, test_items, norm_np, norm, 16)
    facts["gpu_memory_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print("open loop after:", facts["open_loop_after"], flush=True)
    torch.save({"state": plan_head.adapter_state(), "rank": args.rank, "alpha": float(args.rank),
                "deep": args.deep_plan, "motion_scale": data.scale, "facts": facts}, args.out / "adapter.pt")
    (args.out / "metrics.json").write_text(json.dumps(facts, indent=2))
    print("TRAIN-OK", flush=True)


if __name__ == "__main__":
    main()
