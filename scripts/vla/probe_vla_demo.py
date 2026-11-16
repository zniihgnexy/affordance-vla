"""Probe the released starVLA QwenPI LIBERO checkpoint on one training episode (GPU: run through the queue).

Measurements for report/2026-11-10_vla_pipeline_demo.ipynb. Nothing is trained.
  1. model facts: parameters per module, the prompt, token counts, hidden-state shapes, time, memory;
  2. flow sampling: the action chunk after each Euler step, for several noise seeds, at one frame;
  3. open loop: predicted chunks along the recorded episode, with the true instruction and with
     instructions that name other objects of the same scene;
  4. training objective: the flow-matching loss and the one-step action estimate at fixed noise levels.

The episode is either a demonstration replayed in the simulator (`--log`, written by
rollout_libero_logged.py --replay-demo; object positions are then known for the same frames) or an
episode of the LeRobot training set (`--dataset`).

Usage (starvla env, from the starVLA repo root with it on PYTHONPATH):
    python probe_vla_demo.py --ckpt .../steps_100000_pytorch_model.pt --base-vlm .../Qwen3-VL-4B-Instruct \
        --log .../rollouts/demo_replay_ep0.npz --out .../runs/vla_pipeline_demo/probe
Output: <out>/probe.npz (arrays) and <out>/probe.json (facts and settings).

The sampling loop below repeats `LayerwiseFlowmatchingActionHead.predict_action` of starVLA @13d8ad4 so
that the intermediate steps can be kept; the script checks that both give the same chunk.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

IMAGE_SIZE = (224, 224)  # what starVLA's LIBERO client feeds the policy
NOISE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)  # flow time t: 0 = pure noise, 1 = the action chunk
IMAGE_PAD = "<|image_pad|>"


def read_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def load_episode(dataset: Path, task: str, episode: int | None) -> dict:
    """Frames, states and actions of one LeRobot episode of `task` (the first one unless `episode` is given)."""
    import pandas as pd

    episodes = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()]
    if episode is None:
        episode = next(e["episode_index"] for e in episodes if e["tasks"][0].lower() == task.lower())
    record = next(e for e in episodes if e["episode_index"] == episode)
    table = pd.read_parquet(dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    videos = dataset / "videos" / "chunk-000"
    return {"episode": episode, "instruction": record["tasks"][0],
            "actions": np.stack(table["action"]).astype(np.float64),
            "image": read_video(videos / "observation.images.image" / f"episode_{episode:06d}.mp4"),
            "wrist": read_video(videos / "observation.images.wrist_image" / f"episode_{episode:06d}.mp4")}


def load_log(path: Path) -> dict:
    """A replayed demonstration from rollout_libero_logged.py, with actions in the policy's convention."""
    with np.load(path) as log:
        meta = json.loads(str(log["meta"]))
        actions = log["actions"].astype(np.float64)
        # gripper: the simulator's command is -1 open / +1 close, the policy's output is 1 open / 0 close
        actions[:, 6] = (1.0 - actions[:, 6]) / 2.0
        steps = len(actions)
        return {"episode": Path(path).name, "instruction": meta["instruction"], "actions": actions,
                "image": log["agentview"][:steps], "wrist": log["wrist"][:steps]}


def policy_example(image: np.ndarray, wrist: np.ndarray, instruction: str) -> dict:
    from PIL import Image

    resized = [np.asarray(Image.fromarray(view).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR))
               for view in (image, wrist)]
    return {"image": resized, "lang": instruction}


def parameter_table(framework) -> dict[str, float]:
    """Millions of parameters per module (top level, and inside the action head)."""
    count = lambda module: sum(p.numel() for p in module.parameters()) / 1e6  # noqa: E731
    table = {name: count(module) for name, module in framework.named_children()}
    table.update({f"action_model.{name}": count(module) for name, module in framework.action_model.named_children()})
    table["total"] = count(framework)
    return {name: round(value, 2) for name, value in table.items() if value > 0}


def encode(framework, examples: list[dict]):
    """Layer-wise VLM hidden states and the attention mask for a batch of policy examples."""
    from deployment.model_server.tools.image_tools import to_pil_preserve

    images = [to_pil_preserve(example["image"]) for example in examples]
    states, mask = framework._encode_vl_hidden_states(images, [example["lang"] for example in examples])
    return states, None if mask is None else mask.to(dtype=torch.bool)


def velocity(head, actions: torch.Tensor, bucket: int, states, mask) -> torch.Tensor:
    """One pass of the action DiT: predicted velocity for the noisy chunk `actions` at time bucket `bucket`."""
    batch, device = actions.shape[0], actions.device
    timesteps = torch.full((batch,), bucket, device=device, dtype=torch.long)
    features = head.action_encoder(actions, timesteps)
    if head.config.add_pos_embed:
        features = features + head.position_embedding(torch.arange(features.shape[1], device=device)).unsqueeze(0)
    tokens = torch.cat((head.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1), features), dim=1)
    output = head.model(hidden_states=tokens, encoder_hidden_states=states, timestep=timesteps,
                        encoder_attention_mask=mask, return_pre_output=True)
    return head.action_decoder(output)[:, -head.action_horizon:]


def sample(head, states, mask, seed: int, keep_steps: bool = False):
    """Euler integration from noise to an action chunk, as in the head's predict_action."""
    torch.manual_seed(seed)
    batch = states[0].shape[0]
    actions = torch.randn((batch, head.action_horizon, head.action_dim), dtype=states[0].dtype, device=states[0].device)
    steps = head.num_inference_timesteps
    trace = [actions]
    for k in range(steps):
        bucket = int(k / float(steps) * head.num_timestep_buckets)
        actions = actions + velocity(head, actions, bucket, states, mask) / steps
        trace.append(actions)
    return torch.stack(trace, dim=1) if keep_steps else actions


def affine_unnormalise(processor) -> tuple[np.ndarray, np.ndarray]:
    """(offset, scale) with action = offset + scale · normalised, read off the training-time transform."""
    dims = sum(processor.action_key_dims.values())
    offset = processor.unapply_actions(np.zeros((1, dims)))[0]
    scale = processor.unapply_actions(np.ones((1, dims)))[0] - offset
    probe = np.linspace(-1.0, 1.0, 5)[:, None] * np.ones((1, dims))
    if not np.allclose(processor.unapply_actions(probe), offset + scale * probe, atol=1e-5):
        raise RuntimeError("the action normalisation is not affine; this probe needs to be adapted")
    return offset, scale


def timed(function, repeats: int = 3) -> float:
    """Median wall time in milliseconds, with the GPU synchronised."""
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-vlm", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="replayed demonstration (.npz from rollout_libero_logged.py)")
    source.add_argument("--dataset", type=Path, help="LeRobot LIBERO dataset directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", default="pick up the salad dressing and place it in the basket")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--other-objects", nargs="*",
                        default=["ketchup", "alphabet soup", "cream cheese", "milk", "tomato sauce"])
    parser.add_argument("--stride", type=int, default=4, help="open-loop query every this many steps")
    parser.add_argument("--trace-step", type=int, default=20, help="episode step used for the sampling trace")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    args.out.mkdir(parents=True, exist_ok=True)
    episode = load_log(args.log) if args.log else load_episode(args.dataset, args.task, args.episode)
    print(f"episode {episode['episode']}: {len(episode['actions'])} steps · {episode['instruction']}", flush=True)
    started = time.perf_counter()
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt, device="cuda", use_bf16=True,
        config_overrides=[f"framework.qwenvl.base_vlm={args.base_vlm}",
                          # the released checkpoint predates the canonical DiT forward (starVLA LIBERO README)
                          "framework.action_model.diffusion_model_cfg.use_canonical_forward=false"])
    measure(args, wrapper, episode, load_seconds=time.perf_counter() - started)


def model_facts(args: argparse.Namespace, wrapper, episode: dict, load_seconds: float, offset: np.ndarray,
                scale: np.ndarray) -> dict:
    """What does not depend on the observation: sizes, settings, memory after loading."""
    framework, head = wrapper._framework, wrapper._framework.action_model
    return {
        "checkpoint": args.ckpt, "episode": episode["episode"], "instruction": episode["instruction"],
        "load_seconds": round(load_seconds, 1), "parameters_millions": parameter_table(framework),
        "weight_dtype": str(next(framework.parameters()).dtype), "action_horizon": head.action_horizon,
        "action_dim": head.action_dim, "inference_steps": head.num_inference_timesteps,
        "dit_layers": len(head.model.transformer_blocks), "dit_width": head.input_embedding_dim,
        "future_tokens": head.future_tokens.weight.shape[0], "noise_levels": list(NOISE_LEVELS),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "unnormalise_offset": offset.tolist(), "unnormalise_scale": scale.tolist(),
    }


def observation_facts(framework, example: dict) -> dict:
    """The prompt the VLM receives for one observation, and its token counts."""
    from deployment.model_server.tools.image_tools import to_pil_preserve

    inputs = framework.qwen_vl_interface.build_qwenvl_inputs(images=[to_pil_preserve(example["image"])],
                                                           instructions=[example["lang"]])
    tokenizer = framework.qwen_vl_interface.processor.tokenizer
    ids = inputs["input_ids"][0].tolist()
    image_tokens = ids.count(tokenizer.convert_tokens_to_ids(IMAGE_PAD))
    prompt = tokenizer.decode(ids)
    while IMAGE_PAD * 2 in prompt:
        prompt = prompt.replace(IMAGE_PAD * 2, IMAGE_PAD)
    return {"prompt": prompt.replace(IMAGE_PAD, f"[{image_tokens // len(example['image'])} image tokens]"),
            "tokens_total": len(ids), "tokens_image": image_tokens, "tokens_text": len(ids) - image_tokens}


def timing_facts(wrapper, example: dict, states, mask) -> dict:
    """Time and peak memory of one policy call. The card is shared, so the times are upper bounds."""
    framework, head = wrapper._framework, wrapper._framework.action_model
    blank = torch.zeros((1, head.action_horizon, head.action_dim), dtype=states[0].dtype, device=states[0].device)
    torch.cuda.reset_peak_memory_stats()
    wrapper.predict_action(examples=[example])  # warm-up
    return {"ms_vlm": round(timed(lambda: encode(framework, [example])), 1),
            "ms_dit_step": round(timed(lambda: velocity(head, blank, 0, states, mask)), 1),
            "ms_policy_call": round(timed(lambda: wrapper.predict_action(examples=[example])), 1),
            "gpu_memory_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def sampling_trace(head, states, mask, seeds: int) -> tuple[np.ndarray, float]:
    """Normalised chunks after each Euler step, (seeds, steps + 1, horizon, dim), and the largest difference
    between this script's sampler and the head's own one for seed 0."""
    trace = torch.cat([sample(head, states, mask, seed, keep_steps=True) for seed in range(seeds)])
    torch.manual_seed(0)
    reference = head.predict_action(states, None, encoder_attention_mask=mask)
    return trace.float().cpu().numpy(), float((trace[0, -1] - reference[0]).abs().max())


def predict_chunks(framework, episode: dict, instruction: str, query_steps: list[int], batch: int) -> np.ndarray:
    """Normalised chunks (Q, horizon, dim) predicted at the query steps under one instruction."""
    chunks = []
    for steps in batches(query_steps, batch):
        states, mask = encode(framework, [policy_example(episode["image"][t], episode["wrist"][t], instruction)
                                          for t in steps])
        chunks.append(sample(framework.action_model, states, mask, seed=0).float().cpu().numpy())
    return np.concatenate(chunks)


def flow_objective(framework, episode: dict, query_steps: list[int], targets: np.ndarray,
                   batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Flow-matching loss (L, Q) and one-step chunk estimates (L, Q, horizon, dim) at the fixed noise levels.

    `targets` are the demonstrated chunks, normalised. Every batch draws its noise after re-seeding, so a
    chunk's noise depends on its position in the batch.
    """
    head = framework.action_model
    losses, estimates = [], []
    for rows in batches(list(range(len(query_steps))), batch):
        states, mask = encode(framework, [policy_example(episode["image"][query_steps[row]],
                                                         episode["wrist"][query_steps[row]], episode["instruction"])
                                          for row in rows])
        target = torch.as_tensor(targets[rows], dtype=states[0].dtype, device=states[0].device)
        torch.manual_seed(1)
        noise = torch.randn_like(target)
        batch_losses, batch_estimates = [], []
        for t in NOISE_LEVELS:
            noisy = (1 - t) * noise + t * target
            predicted_velocity = velocity(head, noisy, int(t * head.num_timestep_buckets), states, mask)
            batch_losses.append(((predicted_velocity - (target - noise)).float() ** 2).mean(dim=(1, 2)).cpu().numpy())
            batch_estimates.append((noisy + (1 - t) * predicted_velocity).float().cpu().numpy())
        losses.append(np.stack(batch_losses))
        estimates.append(np.stack(batch_estimates))
    return np.concatenate(losses, axis=1), np.concatenate(estimates, axis=1)


@torch.inference_mode()
def measure(args: argparse.Namespace, wrapper, episode: dict, load_seconds: float) -> None:
    framework, head = wrapper._framework, wrapper._framework.action_model
    offset, scale = affine_unnormalise(wrapper.get_norm_processor())
    horizon, steps = head.action_horizon, len(episode["actions"])
    if steps < horizon or not 0 <= args.trace_step < steps:
        raise SystemExit(f"the episode has {steps} steps: it needs at least {horizon}, and --trace-step "
                         f"({args.trace_step}) must lie inside it")
    instructions = [episode["instruction"]] + [args.task.replace("salad dressing", name) for name in args.other_objects]
    if len(set(instructions)) != len(instructions):
        raise SystemExit("--other-objects only works for the salad dressing task; adapt the instruction template")

    # 1–2. one observation: what the model reads, what a call costs, and the sampling trace
    example = policy_example(episode["image"][args.trace_step], episode["wrist"][args.trace_step],
                             episode["instruction"])
    states, mask = encode(framework, [example])
    trace, sampler_difference = sampling_trace(head, states, mask, args.seeds)
    facts = {**model_facts(args, wrapper, episode, load_seconds, offset, scale),
             **observation_facts(framework, example), "hidden_layers_used": len(states),
             "hidden_state_shape": list(states[0].shape), **timing_facts(wrapper, example, states, mask),
             "sampler_max_abs_diff": sampler_difference}

    # 3–4. along the episode: open-loop chunks per instruction, then the training objective on the same frames
    query_steps = list(range(0, steps - horizon + 1, args.stride))
    true_chunks = np.stack([episode["actions"][t:t + horizon] for t in query_steps])
    predicted = []
    for instruction in instructions:
        predicted.append(predict_chunks(framework, episode, instruction, query_steps, args.batch))
        print(f"open loop: {instruction}", flush=True)
    predicted = np.stack(predicted)
    losses, one_step = flow_objective(framework, episode, query_steps, (true_chunks - offset) / scale, args.batch)

    np.savez_compressed(
        args.out / "probe.npz", query_steps=np.asarray(query_steps), trace_step=args.trace_step,
        trace_normalised=trace, trace=offset + scale * trace, true_chunks=true_chunks,
        predicted=offset + scale * predicted, predicted_normalised=predicted, flow_loss=losses,
        one_step=offset + scale * one_step, instructions=np.asarray(instructions))
    (args.out / "probe.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps({k: v for k, v in facts.items() if k != "prompt"}, indent=2))
    print("PROBE-OK", flush=True)

if __name__ == "__main__":
    main()
