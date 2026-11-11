"""Policy server for a plan-conditioned head trained by train_plan_following.py (GPU: start it from a queue job).

starVLA's websocket server with one change: every example of a request must carry a `plan` vector
(affordance3d/plan_input.py), which is encoded and added to the action tokens before sampling. The
released checkpoint is loaded as usual and the trained parameters (`adapter.pt`) are put on top.

Usage (starvla env, from the starVLA repo root; PYTHONPATH = starVLA and this repo):
    python serve_plan_policy.py --ckpt_path .../steps_100000_pytorch_model.pt --adapter .../adapter.pt \
        --port 6694 --seed 7 --config_override framework.qwenvl.base_vlm=.../Qwen3-VL-4B-Instruct \
        --config_override framework.action_model.diffusion_model_cfg.use_canonical_forward=false
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_vla_demo import encode  # noqa: E402

from affordance3d.plan_input import PLAN_DIM  # noqa: E402
from affordance3d.plan_policy import PlanConditionedHead, dit_lora_match  # noqa: E402


def plans_of(examples: list[dict]) -> np.ndarray:
    """(B, PLAN_DIM) plan inputs of a request; every example must bring one."""
    missing = [index for index, example in enumerate(examples) if "plan" not in example]
    if missing:
        raise ValueError(f"examples {missing} carry no 'plan'; this server needs one per example "
                         "(rollout_libero_logged.py --plan oracle or --plan zero)")
    plans = np.stack([np.asarray(example["plan"], dtype=np.float32).reshape(-1) for example in examples])
    if plans.shape[1] != PLAN_DIM:
        raise ValueError(f"a plan has {plans.shape[1]} numbers, expected {PLAN_DIM}")
    return plans


def make_wrapper(args: argparse.Namespace) -> "PolicyServerWrapper":  # noqa: F821 (imported inside)
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    class PlanPolicyWrapper(PolicyServerWrapper):
        """PolicyServerWrapper whose sampling reads the plan of each example."""

        def __init__(self) -> None:
            super().__init__(ckpt_path=args.ckpt_path, device="cuda", use_bf16=True,
                             config_overrides=args.config_override)
            adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
            self._plan_head = PlanConditionedHead(self._framework.action_model, adapter["rank"], adapter["alpha"],
                                                  dit_lora_match)
            self._plan_head.load_adapter(adapter["state"])
            self._plan_head.eval()

        @torch.inference_mode()
        def predict_action(self, examples: list[dict], unnorm_key: str | None = None, **kwargs) -> dict:
            processor = self.get_norm_processor(unnorm_key)
            head = self._plan_head.head
            plans = torch.as_tensor(plans_of(examples), device="cuda")
            states, mask = encode(self._framework, examples)
            noise = torch.randn((len(examples), head.action_horizon, head.action_dim), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalised = self._plan_head.sample(states, mask, plans, noise)
            normalised = normalised.float().cpu().numpy()
            return {"actions": np.stack([processor.unapply_actions(chunk) for chunk in normalised])}

    return PlanPolicyWrapper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--config_override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    from deployment.model_server.seed_utils import set_seed_everywhere
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO, force=True)
    set_seed_everywhere(args.seed)
    wrapper = make_wrapper(args)
    metadata = {**wrapper.metadata, "seed": args.seed, "adapter": str(args.adapter), "plan_dim": PLAN_DIM}
    logging.info("plan policy server ready: %s", metadata)
    # the simulators run on the same machine; other users of the server have no business with this port
    WebsocketPolicyServer(policy=wrapper, host="127.0.0.1", port=args.port, idle_timeout=args.idle_timeout,
                          metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
