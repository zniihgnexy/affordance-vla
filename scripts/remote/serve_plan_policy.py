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
from affordance3d.plan_grid import (Camera, PlanGridPredictor, camera_of_log, grid_plan_inputs,  # noqa: E402
                                    image_token_cells)
from affordance3d.plan_predictor import PlanPredictor, plan_inputs  # noqa: E402
from affordance3d.rollout_log import load_rollout_log  # noqa: E402

IMAGE_TOKEN = "<|image_pad|>"  # placeholder of one image token in the VLM's input


def plans_of(examples: list[dict]) -> np.ndarray:
    """(B, PLAN_DIM) plan inputs of a request; every example must bring one."""
    missing = [index for index, example in enumerate(examples) if "plan" not in example]
    if missing:
        raise ValueError(f"examples {missing} carry no 'plan'; without a predictor this server needs one per "
                         "example (rollout_libero_logged.py --plan oracle or --plan zero), and with a predictor "
                         "a request must bring a plan for every example or for none")
    plans = np.stack([np.asarray(example["plan"], dtype=np.float32).reshape(-1) for example in examples])
    if plans.shape[1] != PLAN_DIM:
        raise ValueError(f"a plan has {plans.shape[1]} numbers, expected {PLAN_DIM}")
    return plans


def positions_of(examples: list[dict]) -> np.ndarray:
    """(B, 3) gripper positions of a request; needed to turn a predicted contact into the plan input."""
    missing = [index for index, example in enumerate(examples) if "ee_pos" not in example]
    if missing:
        raise ValueError(f"examples {missing} carry no 'ee_pos'; with --predictor every example needs one "
                         "(rollout_libero_logged.py --plan predicted)")
    return np.stack([np.asarray(example["ee_pos"], dtype=np.float32).reshape(3) for example in examples])


def agent_cells(framework, examples: list[dict], tokens: int) -> torch.Tensor:
    """Positions of the agent view's image tokens in the VLM input of each example (grid predictor).

    `tokens` is the length of the hidden states the positions will index: the input built here must have
    that length, or it is not the input the states were computed from.
    """
    from deployment.model_server.tools.image_tools import to_pil_preserve

    interface = framework.qwen_vl_interface
    inputs = interface.build_qwenvl_inputs(images=[to_pil_preserve(example["image"]) for example in examples],
                                           instructions=[example["lang"] for example in examples])
    if inputs["input_ids"].shape[1] != tokens:
        raise ValueError(f"the VLM input has {inputs['input_ids'].shape[1]} tokens but its hidden states {tokens}: "
                         "cell positions cannot be trusted")
    return image_token_cells(inputs["input_ids"], interface.processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN))


def load_predictor(args: argparse.Namespace) -> tuple[torch.nn.Module | None, Camera | None]:
    """The plan predictor named by --predictor (either head) and, for the grid head, its camera."""
    if not args.predictor:
        return None, None
    saved = torch.load(args.predictor, map_location="cpu", weights_only=True)
    config, camera = saved["config"], None
    if config.get("head", "regress") == "grid":
        if not args.camera_log:
            raise SystemExit("a grid predictor needs --camera-log")
        camera = camera_of_log(load_rollout_log(args.camera_log))
        predictor = PlanGridPredictor(config["width"], config["layers"], config["hidden"])
    else:
        predictor = PlanPredictor(config["width"], config["layers"], config["hidden"])
    predictor.load_state_dict(saved["state"])
    return predictor.to("cuda").eval(), camera


def make_wrapper(args: argparse.Namespace) -> "PolicyServerWrapper":  # noqa: F821 (imported inside)
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    class PlanPolicyWrapper(PolicyServerWrapper):
        """PolicyServerWrapper whose sampling reads the plan of each example."""

        def __init__(self) -> None:
            super().__init__(ckpt_path=args.ckpt_path, device="cuda", use_bf16=True,
                             config_overrides=args.config_override)
            adapter = torch.load(args.adapter, map_location="cpu", weights_only=True)
            self._plan_head = PlanConditionedHead(self._framework.action_model, adapter["rank"], adapter["alpha"],
                                                  dit_lora_match, deep=adapter.get("deep", False))
            self._plan_head.load_adapter(adapter["state"])
            self._plan_head.eval()
            self._predictor, self._camera = load_predictor(args)

        @torch.inference_mode()
        def predict_action(self, examples: list[dict], unnorm_key: str | None = None, **kwargs) -> dict:
            processor = self.get_norm_processor(unnorm_key)
            head = self._plan_head.head
            states, mask = encode(self._framework, examples)
            if self._predictor is None or any("plan" in example for example in examples):
                given = plans_of(examples)  # sent along: oracle plans, or predicted plans the client has kept
            elif self._camera is None:
                given = plan_inputs(self._predictor(states, mask), positions_of(examples))
            else:
                cells = agent_cells(self._framework, examples, states[0].shape[1])
                scored = self._predictor(states, mask, cells)
                given = grid_plan_inputs(scored, positions_of(examples), self._camera)
            plans = torch.as_tensor(given, device="cuda")
            noise = torch.randn((len(examples), head.action_horizon, head.action_dim), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalised = self._plan_head.sample(states, mask, plans, noise)
            normalised = normalised.float().cpu().numpy()
            # the plan inputs go back too, so that the simulator side can log what the head was given
            return {"actions": np.stack([processor.unapply_actions(chunk) for chunk in normalised]), "plans": given}

    return PlanPolicyWrapper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--predictor", type=Path,
                        help="predictor.pt of train_plan_predictor.py: the plan is predicted from the VLM's hidden "
                             "states for requests that carry `ee_pos` and no `plan`")
    parser.add_argument("--camera-log", type=Path,
                        help="a rollout log of the scenes; its camera calibration is needed by a grid predictor")
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
    metadata = {**wrapper.metadata, "seed": args.seed, "adapter": str(args.adapter), "plan_dim": PLAN_DIM,
                "predictor": str(args.predictor) if args.predictor else None}
    logging.info("plan policy server ready: %s", metadata)
    # the simulators run on the same machine; other users of the server have no business with this port
    WebsocketPolicyServer(policy=wrapper, host="127.0.0.1", port=args.port, idle_timeout=args.idle_timeout,
                          metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
