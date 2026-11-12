"""Train a plan predictor on the frozen VLM of the released checkpoint (GPU: submit through the queue).

A plan predictor reads the VLM's hidden states and outputs the next gripper event, the contact point and
the approach direction. Two heads can be trained (--head):
  regress  affordance3d/plan_predictor.py: the contact's coordinates are regressed from the whole sequence
  grid     affordance3d/plan_grid.py: every image cell of the agent view is scored, with an offset inside
           the cell and a depth; the 3D contact follows from the camera calibration (--camera-log)
Targets are the oracle plans of the demonstration frames (affordance3d/plan_targets.py). Frames before
the grasp are also paired with an instruction that names another object of the scene, whose grasp point
is then the target; this needs no extra demonstrations. The grid head is also trained on shifted copies
of the agent view (the target cell moves with the image), so that it cannot rely on where things stand.

After training the predictor is tested without the simulator on held-out demonstrations (their own
instruction, and every other object of the scene named instead) and, with --pro, on the first frames of
LIBERO-PRO rollouts, where objects have traded places or another object is asked for.

Usage (starvla env, from the starVLA repo root; PYTHONPATH = starVLA and this repo):
    python train_plan_predictor.py --ckpt .../steps_100000_pytorch_model.pt --base-vlm .../Qwen3-VL-4B-Instruct \
        --data .../runs/plan_following/dataset10 --out .../runs/plan_predictor/g1 --head grid \
        --camera-log .../runs/plan_following/demos/<any>_demo0.npz --pro .../runs/libero_pro/released
Outputs in --out: predictor.pt (weights, sizes, head), train.jsonl, metrics.json, and predictions.npz
(images, instructions, predicted and oracle contacts of the test items, cell scores of the grid head).
With --evaluate-only an existing predictor.pt is tested again and nothing is trained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plan_dataset import resized  # noqa: E402
from probe_vla_demo import encode  # noqa: E402
from train_plan_following import PlanData, learning_rate, object_words  # noqa: E402

from affordance3d.plan_consistency import plan_ahead  # noqa: E402
from affordance3d.plan_grid import (DEPTH_REFERENCE_M, DEPTH_SCALE_M, NO_CELL, Camera,  # noqa: E402
                                    PlanGridPredictor, camera_of_log, contact_pixel, grid_contacts, grid_loss,
                                    grid_target, image_token_cells, shifted_image)
from affordance3d.plan_input import ContactOffsets  # noqa: E402
from affordance3d.plan_predictor import CONTACT_SCALE_M, PlanPredictor, predictor_loss  # noqa: E402
from affordance3d.plan_targets import GRASP, NONE, RELEASE, PlanTarget, oracle_target  # noqa: E402
from affordance3d.rollout_log import load_rollout_log  # noqa: E402

APPROACH_SHARE = 0.5  # share of training frames drawn from the approach to a grasp
EVENT_NAMES = {NONE: "none", GRASP: "grasp", RELEASE: "release"}
EVAL_BATCH = 16
IMAGE_TOKEN = "<|image_pad|>"  # placeholder of one image token in the VLM's input
MAX_SHIFT_PX = 56  # the agent view is moved by up to two cells in training (grid head)
SHIFT_SHARE = 0.5

Infer = Callable[[list["Item"]], tuple[np.ndarray, np.ndarray, "np.ndarray | None"]]  # events, contacts, cell scores


@dataclass(frozen=True)
class Item:
    example: dict  # what the VLM is given: the two images and the instruction
    target: PlanTarget
    kind: str  # "own": the demonstration's instruction; "other": another object named; "pro ...": LIBERO-PRO
    rivals: tuple[np.ndarray, ...]  # grasp points of the other objects of the scene
    shift: tuple[int, int] = (0, 0)  # pixels by which the agent view was moved (training of the grid head)


def grasp_points(names: tuple[str, ...], positions: np.ndarray, offsets: ContactOffsets) -> dict[str, np.ndarray]:
    """Oracle grasp point of every object of a scene that the demonstrations know how to grasp."""
    return {name: positions[k] + offsets.grasp[name] for k, name in enumerate(names) if name in offsets.grasp}


def frame_items(data: PlanData, demo: int, step: int, container: str, rng: np.random.Generator,
                others: float | str) -> list[Item]:
    """The frame under its own instruction and, before the grasp, under instructions naming other objects.

    `others` is the probability of adding one other object, or "all" to add every one of them.
    """
    track = data.tracks[demo]
    points = grasp_points(track.names, track.object_pos[step], data.contact)

    def item(name: str, kind: str) -> Item:
        instruction = (track.instruction if name == track.target
                       else track.instruction.replace(object_words(track.target), object_words(name)))
        rivals = tuple(point for other, point in points.items() if other != name)
        return Item(data.example(demo, step, instruction), oracle_target(track, step, data.contact, container, name),
                    kind, rivals)

    items = [item(track.target, "own")]
    ahead = plan_ahead(track.labels, track.ee_pos[step], step)
    names = [name for name in points if name != track.target]
    if ahead is None or not ahead.closing or not names:
        return items
    if others == "all":
        return items + [item(name, "other") for name in names]
    if rng.uniform() < others:
        items.append(item(names[rng.integers(len(names))], "other"))
    return items


def pro_items(folders: list[Path], offsets: ContactOffsets) -> list[Item]:
    """First frames of LIBERO-PRO rollouts that kept their images: the named object's grasp point is the target."""
    items = []
    for folder in folders:
        for path in sorted(folder.glob("task*_ep[01].npz")):
            with np.load(path) as stored:
                if "agentview" not in stored.files or "wrist" not in stored.files:
                    continue  # a log written without its images
            log = load_rollout_log(path)
            points = grasp_points(log.names, log.object_pos[0], offsets)
            if log.target not in points:
                continue
            target = PlanTarget(GRASP, points[log.target], np.asarray(offsets.grasp_approach[log.target], dtype=float))
            images = [resized(log.agentview[:1])[0], resized(log.wrist[:1])[0]]
            rivals = tuple(point for name, point in points.items() if name != log.target)
            items.append(Item({"image": images, "lang": log.meta["instruction"]}, target, f"pro {folder.name}", rivals))
    return items


def moved(item: Item, camera: Camera, rng: np.random.Generator) -> Item:
    """The item with its agent view shifted by a random amount, if its contact stays inside the image."""
    if rng.uniform() >= SHIFT_SHARE:
        return item
    dx, dy = (int(value) for value in rng.integers(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1, size=2))
    if item.target.event != NONE:
        u, v, _ = contact_pixel(item.target.contact, camera)
        if grid_target(u + dx, v + dy) is None:
            return item
    agent, wrist = item.example["image"]
    return replace(item, example={**item.example, "image": [shifted_image(np.asarray(agent), dx, dy), wrist]},
                   shift=(dx, dy))


def encode_with_cells(framework, examples: list[dict]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Hidden states and mask as probe_vla_demo.encode gives them, plus where the agent view's tokens sit."""
    from deployment.model_server.tools.image_tools import to_pil_preserve

    interface = framework.qwen_vl_interface
    inputs = interface.build_qwenvl_inputs(images=[to_pil_preserve(example["image"]) for example in examples],
                                           instructions=[example["lang"] for example in examples])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = interface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
    layers = len(framework.action_model.model.transformer_blocks)
    image_id = interface.processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    cells = image_token_cells(inputs["input_ids"], image_id)
    return list(outputs.hidden_states[-layers:]), inputs["attention_mask"].to(dtype=torch.bool), cells


def run_head(framework, predictor: torch.nn.Module, items: list[Item]) -> dict[str, torch.Tensor]:
    """The predictor's output for a batch; the VLM runs without gradients."""
    examples = [item.example for item in items]
    if isinstance(predictor, PlanGridPredictor):
        with torch.no_grad():
            states, mask, cells = encode_with_cells(framework, examples)
        return predictor([layer.detach() for layer in states], mask, cells)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        states, mask = encode(framework, examples)
    return predictor([layer.detach() for layer in states], mask)


def step_loss(output: dict[str, torch.Tensor], items: list[Item], camera: Camera | None) -> dict[str, torch.Tensor]:
    """The training loss of a batch for either head (the grid head is the one with a camera)."""
    def tensor(values: list, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device="cuda")

    events = [item.target.event for item in items]
    approach = tensor([item.target.approach for item in items])
    if camera is None:
        return predictor_loss(output, tensor(events, torch.long), tensor([item.target.contact for item in items]),
                              approach)
    cell, offset, depth = [], [], []
    for item in items:
        placed = None
        if item.target.event != NONE:
            u, v, metres = contact_pixel(item.target.contact, camera)
            placed = grid_target(u + item.shift[0], v + item.shift[1])
        if placed is None:  # no event, or the contact lies outside the image: no cell is trained for this item
            cell.append(NO_CELL), offset.append([0.0, 0.0]), depth.append(0.0)
        else:
            cell.append(placed[0]), offset.append(placed[1]), depth.append((metres - DEPTH_REFERENCE_M) / DEPTH_SCALE_M)
    return grid_loss(output, tensor(events, torch.long), tensor(cell, torch.long), tensor(offset), tensor(depth),
                     approach)


def make_infer(framework, predictor: torch.nn.Module, camera: Camera | None) -> Infer:
    @torch.no_grad()
    def infer(items: list[Item]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        output = run_head(framework, predictor, items)
        events = output["event"].argmax(dim=-1).cpu().numpy()
        if camera is None:
            return events, output["contact"].float().cpu().numpy() * CONTACT_SCALE_M, None
        return events, grid_contacts(output, camera), torch.softmax(output["cell"], dim=-1).float().cpu().numpy()

    return infer


def evaluate(infer: Infer, predictor: torch.nn.Module, items: list[Item]) -> dict[str, dict]:
    """Per kind of item and event: event accuracy, contact error (cm) and how often another object is nearer."""
    groups: dict[str, list[tuple[bool, float, bool]]] = {}
    predictor.eval()
    try:
        for start in range(0, len(items), EVAL_BATCH):
            chunk = items[start:start + EVAL_BATCH]
            events, contacts, _ = infer(chunk)
            for item, event, contact in zip(chunk, events, contacts):
                error = np.linalg.norm(contact - item.target.contact) if item.target.event != NONE else np.nan
                wrong = (item.target.event == GRASP and bool(item.rivals)
                         and min(np.linalg.norm(contact - rival) for rival in item.rivals) < error)
                key = f"{item.kind}: {EVENT_NAMES[item.target.event]}"
                groups.setdefault(key, []).append((bool(event == item.target.event), 100.0 * float(error),
                                                   bool(wrong)))
    finally:
        predictor.train()
    summary = {}
    for key, rows in sorted(groups.items()):
        errors = np.array([row[1] for row in rows])
        summary[key] = {"items": len(rows), "event_accuracy": round(float(np.mean([row[0] for row in rows])), 3)}
        if not np.isnan(errors).all():
            summary[key] |= {"contact_cm_mean": round(float(errors.mean()), 2),
                             "contact_cm_median": round(float(np.median(errors)), 2),
                             "contact_cm_p90": round(float(np.percentile(errors, 90)), 2),
                             "within_2cm": round(float((errors <= 2.0).mean()), 3),
                             "another_object_nearer": round(float(np.mean([row[2] for row in rows])), 3)}
    return summary


def save_predictions(infer: Infer, predictor: torch.nn.Module, items: list[Item], path: Path) -> None:
    """Images, instructions, predictions and targets of the given items, for figures."""
    predictor.eval()
    try:
        parts = [infer(items[start:start + EVAL_BATCH]) for start in range(0, len(items), EVAL_BATCH)]
    finally:
        predictor.train()
    arrays = {"kind": np.array([item.kind for item in items]),
              "instruction": np.array([item.example["lang"] for item in items]),
              "agentview": np.stack([np.asarray(item.example["image"][0]) for item in items]),
              "event": np.concatenate([part[0] for part in parts]),
              "contact": np.concatenate([part[1] for part in parts]),
              "target_event": np.array([item.target.event for item in items]),
              "target_contact": np.stack([item.target.contact for item in items])}
    if parts[0][2] is not None:
        arrays["cell_scores"] = np.concatenate([part[2] for part in parts])
    np.savez_compressed(path, **arrays)


def test_items(data: PlanData, container: str, frames: int) -> list[Item]:
    """A fixed set from the held-out demonstrations: any frame, and frames on the way to a grasp."""
    rng = np.random.default_rng(1234)
    chosen = [data.test[i] for i in rng.choice(len(data.test), min(frames, len(data.test)), replace=False)]
    approach = data.test_approach
    chosen += [approach[i] for i in rng.choice(len(approach), min(frames, len(approach)), replace=False)]
    return [item for demo, step in chosen for item in frame_items(data, demo, step, container, rng, "all")]


def train(args: argparse.Namespace, framework, predictor: torch.nn.Module, data: PlanData, container: str,
          camera: Camera | None, log) -> None:
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)
        items: list[Item] = []
        for _ in range(args.frames):
            pool = data.train_approach if rng.uniform() < APPROACH_SHARE else data.train
            demo, frame = pool[rng.integers(len(pool))]
            items += frame_items(data, demo, frame, container, rng, args.other_share)
        if camera is not None:
            items = [moved(item, camera, rng) for item in items]
        loss = step_loss(run_head(framework, predictor, items), items, camera)
        optimizer.zero_grad(set_to_none=True)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log({"step": step, "samples": len(items), "seconds": round(time.perf_counter() - started, 1),
                 **{key: round(float(value), 4) for key, value in loss.items()}})


def load_framework(args: argparse.Namespace):
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt, device="cuda", use_bf16=True,
        config_overrides=[f"framework.qwenvl.base_vlm={args.base_vlm}",
                          "framework.action_model.diffusion_model_cfg.use_canonical_forward=false"])
    return wrapper._framework


def build_predictor(args: argparse.Namespace, framework, example: dict) -> tuple[torch.nn.Module, dict]:
    """The untrained head for --head, sized from one real example. For the grid head the two ways of running
    the VLM are compared first: its cell positions are only right if both give the same states."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        states, mask = encode(framework, [example])
    config = {"head": args.head, "width": int(states[0].shape[-1]), "layers": len(states), "hidden": args.hidden}
    if args.head == "regress":
        return PlanPredictor(config["width"], config["layers"], config["hidden"]).to("cuda"), config
    with torch.no_grad():
        again, again_mask, _ = encode_with_cells(framework, [example])
    same = (len(again) == len(states) and again[-1].shape == states[-1].shape
            and (mask is None or torch.equal(again_mask, mask))
            and torch.allclose(again[-1].float(), states[-1].float(), atol=1e-2))
    if not same:
        raise SystemExit("encode_with_cells does not reproduce probe_vla_demo.encode: cell positions cannot be trusted")
    return PlanGridPredictor(config["width"], config["layers"], config["hidden"]).to("cuda"), config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--head", choices=("regress", "grid"), default="regress")
    parser.add_argument("--camera-log", type=Path,
                        help="any rollout log of the scenes: the grid head uses its camera calibration")
    parser.add_argument("--pro", type=Path, nargs="*", default=[],
                        help="run folders of run_libero_pro_gpu.sh; the first frames of their suites are an extra test")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=8, help="frames per step; each adds up to two samples")
    parser.add_argument("--other-share", type=float, default=1.0,
                        help="probability that a frame before the grasp is also given with another object named")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--holdout", type=int, default=5, help="demonstrations per instruction kept for testing")
    parser.add_argument("--test-frames", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="load predictor.pt from --out and only run the tests (nothing is trained)")
    parser.add_argument("--figure-items", type=int, default=120,
                        help="held-out items stored in predictions.npz, besides all benchmark items")
    args = parser.parse_args()
    if not 0.0 <= args.other_share <= 1.0:
        parser.error(f"--other-share is a probability, got {args.other_share}")
    if args.head == "grid" and args.camera_log is None:
        parser.error("--head grid needs --camera-log")
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    data = PlanData(args.data, horizon=8, holdout=args.holdout)
    container = json.loads((args.data / "stats.json").read_text())["container"]
    camera = camera_of_log(load_rollout_log(args.camera_log)) if args.head == "grid" else None
    framework = load_framework(args)
    held_out = test_items(data, container, args.test_frames)
    benchmark = pro_items([suite for run in args.pro for suite in sorted(run.iterdir()) if suite.is_dir()],
                          data.contact)
    predictor, config = build_predictor(args, framework, held_out[0].example)
    infer = make_infer(framework, predictor, camera)
    facts = {"args": {key: str(value) for key, value in vars(args).items()}, "config": config,
             "parameters_millions": round(sum(p.numel() for p in predictor.parameters()) / 1e6, 2),
             "train_frames": len(data.train), "held_out_items": len(held_out), "benchmark_items": len(benchmark)}
    print(json.dumps(facts, indent=2), flush=True)
    if args.evaluate_only:
        saved = torch.load(args.out / "predictor.pt", map_location="cuda", weights_only=True)
        predictor.load_state_dict(saved["state"])
    else:
        with (args.out / "train.jsonl").open("w") as handle:
            def log(record: dict) -> None:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(record, flush=True)

            train(args, framework, predictor, data, container, camera, log)
        torch.save({"config": config, "state": predictor.state_dict()}, args.out / "predictor.pt")
    save_predictions(infer, predictor, benchmark + held_out[:args.figure_items], args.out / "predictions.npz")
    facts["held_out"] = evaluate(infer, predictor, held_out)
    facts["benchmark"] = evaluate(infer, predictor, benchmark) if benchmark else {}
    facts["layer_mix"] = [round(float(w), 3) for w in torch.softmax(predictor.mix.detach(), dim=0).cpu()]
    facts["gpu_memory_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    (args.out / "metrics.json").write_text(json.dumps(facts, indent=2))
    print("held out:", json.dumps(facts["held_out"], indent=2))
    print("benchmark:", json.dumps(facts["benchmark"], indent=2))


if __name__ == "__main__":
    main()
