"""E1 of Pipeline v3: does the frozen VLM already point at the object the instruction names?

Nothing is trained. For the first frame of every LIBERO-PRO rollout that kept its images, the frozen VLM is run
once, as the plan head sees it. Two token sets are compared with the agent view's 64 cell tokens: the whole
instruction, and the words of the named object. The cosine similarity is merged over the tokens by mean or
by maximum, on every layer the plan head reads. Each 8 x 8 map is scored by the object under its best cell
(affordance_vla/analysis/semantic_probe.py). The trained grid plan head (--predictor) is scored the same way,
which splits its errors into "another object" and "the named object, but off". It also gets the 3D error of
its decoded contact.

Outputs in --out:
- probe.npz: per item, the maps (item, token set, merge, layer, 64), object coverages and the head's cell
  scores and contact;
- summary.json: the share of maps on the named object per token set, merge and layer, and the head's split.

Usage: through the GPU queue with scripts/vla/run_semantic_probe_gpu.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_plan_predictor import encode_with_cells, load_framework  # noqa: E402

from affordance_vla.analysis.semantic_probe import (  # noqa: E402
    MERGES,
    cell_coverage,
    merge_similarity,
    object_at_peak,
    object_phrase,
    text_token_indices,
)
from affordance_vla.heads.plan_grid import PlanGridPredictor, camera_of_log, grid_contacts  # noqa: E402
from affordance_vla.plan.plan_input import offsets_from_json  # noqa: E402
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log  # noqa: E402
from affordance_vla.training.plan_dataset import resized  # noqa: E402

TOKEN_SETS = ("instruction", "object")
NEAR_M = 0.02  # a head contact this close to the named object's grasp point is on target


def first_frames(runs: list[Path]) -> list[RolloutLog]:
    """Logs of episodes 0 and 1 of every suite of the runs that kept their images (same scenes in every run)."""
    logs, seen = [], set()
    for run in runs:
        for path in sorted(run.glob("*/task*_ep[01].npz")):
            key = (path.parent.name, path.name)
            with np.load(path) as stored:
                if key in seen or not {"agentview", "wrist", "seg"} <= set(stored.files):
                    continue
            seen.add(key)
            logs.append(load_rollout_log(path))
    return logs


def token_positions(tokenizer, input_ids: np.ndarray, instruction: str, phrase: str) -> dict[str, np.ndarray]:
    """Positions in the VLM's sequence of the instruction's tokens and of the named object's words."""
    pieces = [tokenizer.decode([int(token)]) for token in input_ids]
    words = text_token_indices(pieces, instruction)
    if len(words) == 0:
        raise ValueError(f"the instruction is not in the prompt: {instruction!r}")
    inside = text_token_indices(pieces[words[0]:words[-1] + 1], phrase)
    return {"instruction": words, "object": words[0] + inside}


def load_head(path: Path) -> PlanGridPredictor:
    saved = torch.load(path / "predictor.pt", map_location="cuda", weights_only=True)
    config = saved["config"]
    if config["head"] != "grid":
        raise SystemExit(f"{path} holds a {config['head']} head; the probe scores grid heads")
    head = PlanGridPredictor(config["width"], config["layers"], config["hidden"]).to("cuda")
    head.load_state_dict(saved["state"])
    return head.eval()


def probe(framework, head: PlanGridPredictor, camera, offsets, log: RolloutLog) -> dict:
    from deployment.model_server.tools.image_tools import to_pil_preserve

    example = {"image": [resized(log.agentview[:1])[0], resized(log.wrist[:1])[0]], "lang": log.meta["instruction"]}
    interface = framework.qwen_vl_interface
    input_ids = interface.build_qwenvl_inputs(images=[to_pil_preserve(example["image"])],
                                              instructions=[example["lang"]])["input_ids"][0].cpu().numpy()
    with torch.no_grad():
        states, mask, cells = encode_with_cells(framework, [example])
        output = head([layer.detach() for layer in states], mask, cells)
    stack = torch.stack([layer[0].float() for layer in states]).cpu().numpy()  # (L, T, D)
    cell_states = stack[:, cells[0].cpu().numpy()]
    positions = token_positions(interface.processor.tokenizer, input_ids, example["lang"], object_phrase(log.target))
    maps = np.full((len(TOKEN_SETS), len(MERGES), len(stack), cell_states.shape[1]), np.nan, np.float32)
    for s, name in enumerate(TOKEN_SETS):
        if len(positions[name]) == 0:
            continue  # the named object's words are not in this instruction
        for m, how in enumerate(MERGES):
            for layer in range(len(stack)):
                maps[s, m, layer] = merge_similarity(stack[layer, positions[name]], cell_states[layer], how)
    contact = grid_contacts(output, camera)[0]
    offset = offsets.grasp.get(log.target)  # the plan dataset knows the grasp points of its own suite only
    error = np.nan if offset is None else float(np.linalg.norm(contact - log.object_pos[0, log.target_index] - offset))
    return {"maps": maps, "coverage": cell_coverage(log.seg[0], len(log.names)),
            "head_scores": output["cell"][0].float().cpu().numpy(),
            "head_error_m": error, "object_words": int(len(positions["object"]))}


def summarise(records: list[dict], logs: list[RolloutLog]) -> dict:
    layers = records[0]["maps"].shape[2]
    on_target = np.full((len(records), len(TOKEN_SETS), len(MERGES), layers), np.nan)
    head = {"named object, within 2 cm": 0, "named object, further": 0, "named object, no grasp point known": 0,
            "another object": 0, "background": 0}
    for i, (record, log) in enumerate(zip(records, logs)):
        for s in range(len(TOKEN_SETS)):
            for m in range(len(MERGES)):
                for layer in range(layers):
                    scores = record["maps"][s, m, layer]
                    if not np.isnan(scores).any():
                        on_target[i, s, m, layer] = object_at_peak(scores, record["coverage"]) == log.target_index
        found = object_at_peak(record["head_scores"], record["coverage"])
        if found == log.target_index:
            error = record["head_error_m"]
            head["named object, no grasp point known" if np.isnan(error) else
                 "named object, within 2 cm" if error <= NEAR_M else "named object, further"] += 1
        else:
            head["background" if found is None else "another object"] += 1
    share = np.nanmean(on_target, axis=0)  # (token set, merge, layer)
    summary = {"items": len(records), "head": head, "maps": {}}
    for s, token_set in enumerate(TOKEN_SETS):
        for m, how in enumerate(MERGES):
            row = share[s, m]
            best = int(np.nanargmax(row))
            summary["maps"][f"{token_set}/{how}"] = {"best_layer": best, "best_share": round(float(row[best]), 3),
                                                      "share_by_layer": [round(float(v), 3) for v in row]}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--runs", required=True, type=Path, nargs="+", help="run folders of run_libero_pro_gpu.sh")
    parser.add_argument("--predictor", required=True, type=Path, help="run folder of a grid plan head (predictor.pt)")
    parser.add_argument("--camera-log", required=True, type=Path, help="a rollout log with the camera calibration")
    parser.add_argument("--data", required=True, type=Path, help="plan dataset whose offsets.json gives grasp points")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logs = first_frames(args.runs)
    if not logs:
        raise SystemExit("no first frames with images in the given runs")
    framework = load_framework(args)
    head = load_head(args.predictor)
    camera = camera_of_log(load_rollout_log(args.camera_log))
    offsets = offsets_from_json((args.data / "offsets.json").read_text())
    records = [probe(framework, head, camera, offsets, log) for log in logs]
    widest = max(len(log.names) for log in logs)
    coverage = np.zeros((len(records), widest, records[0]["coverage"].shape[1]), np.float32)
    for i, record in enumerate(records):
        coverage[i, :len(record["coverage"])] = record["coverage"]
    arrays = {
        "maps": np.stack([r["maps"] for r in records]).astype(np.float16), "coverage": coverage,
        "head_scores": np.stack([r["head_scores"] for r in records]),
        "head_error_m": np.array([r["head_error_m"] for r in records]),
        "object_words": np.array([r["object_words"] for r in records]),
        "target": np.array([log.target_index for log in logs]),
        "suite": np.array([log.path.parent.name for log in logs]),
        "file": np.array([log.path.name for log in logs]),
        "instruction": np.array([log.meta["instruction"] for log in logs]),
        "agentview": np.stack([log.agentview[0] for log in logs]), "token_sets": np.array(TOKEN_SETS),
        "merges": np.array(MERGES)}
    np.savez_compressed(args.out / "probe.npz", **arrays)
    summary = summarise(records, logs)
    summary["args"] = {key: str(value) for key, value in vars(args).items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "args"}, indent=2))


if __name__ == "__main__":
    main()
