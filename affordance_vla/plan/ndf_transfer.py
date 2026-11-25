"""Move a demonstrated grasp onto another mesh with Neural Descriptor Fields (NDF, Simeonov et al. 2021,
arXiv 2112.05124), using the authors' public `multi_category_weights` (ShapeNet mugs, bowls and bottles); no
training. The code follows ndf_robot/opt/optimizer.py (OccNetOptimizer) for one demonstration: the descriptors of
query points around the gripper at the demonstrated grasp are the target; on the new mesh a rigid transform of the
query points is optimised, from several random starts, until their descriptors match (L1, Adam); the best start
gives the transferred grasp pose.

What differs from the reference, on purpose:
- one demonstration per transfer (a map is summed from many transferred grasps afterwards), no mean over demos;
  many transfers to one mesh run as one batch (every start of every grasp is independent: its own loss, its own
  Adam state);
- the shape is a point cloud sampled from the simulator's visual mesh, in the full mug rather than the partial view
  NDF was trained on; random starts are uniform rotations (scipy) instead of a healpix grid;
- the query points are a box around the Franka hand (its collision mesh's extents, hand frame moved to the grasp
  point), unscaled, while meshes are scaled to the diagonal NDF's mugs have (0.3 m), so hand and mug keep NDF's
  relative size and the descriptors stay in the range the network knows.
torch is needed; the NDF repository's `src` folder must be on the path (`load_model`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from affordance_vla.plan.interaction_labels import Grasp

NDF_DIAGONAL_M = 0.3  # ndf_robot/config/default_eval_cfg.py MESH_SCALE_DEFAULT: the mugs' bounding-box diagonal
# Franka hand collision mesh (ndf_robot/descriptions/franka_panda/meshes/collision/hand.obj) extents, in the hand
# frame (x thin, y along the closing line, z along the approach), shifted to the grasp point 0.1034 m ahead of the hand
HAND_BOX_MIN = np.array([-0.0316, -0.1040, -0.0259 - 0.1034])
HAND_BOX_MAX = np.array([0.0316, 0.1004, 0.0660 - 0.1034])
QUERY_POINTS = 500
SHAPE_POINTS = 1500
LATENT_DIM = 256


def ndf_scale(points: np.ndarray) -> float:
    """The factor that brings a mesh's bounding-box diagonal to NDF's mug size."""
    return NDF_DIAGONAL_M / float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def hand_rotation(grasp: Grasp) -> np.ndarray:
    """Columns: the hand's x, y (closing) and z (approach) axes in the object frame."""
    return np.stack([np.cross(grasp.closing, grasp.approach), grasp.closing, grasp.approach], axis=1)


def hand_points(grasp: Grasp, scale: float, rng: np.random.Generator, count: int = QUERY_POINTS) -> np.ndarray:
    """Query points: uniform in the hand box at a grasp. The object frame is scaled by `scale`; the box is not."""
    local = rng.uniform(HAND_BOX_MIN, HAND_BOX_MAX, size=(count, 3))
    return scale * np.asarray(grasp.centre) + local @ hand_rotation(grasp).T


class _CpuTorch:
    """`torch` for ndf_robot's layers, which build index tensors on `torch.device('cuda')` unconditionally."""

    def __getattr__(self, name):
        import torch

        return getattr(torch, name)

    @staticmethod
    def device(name):
        import torch

        return torch.device("cpu")


def load_model(weights: Path, ndf_src: Path, device: str = "cpu"):
    """The NDF occupancy network (pointnet encoder, 256-d latent) with the public weights, in eval mode."""
    import torch

    sys.path.insert(0, str(ndf_src))
    from ndf_robot.model import layers_equi
    from ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn import VNNOccNet

    if device == "cpu":
        layers_equi.torch = _CpuTorch()  # the layers' only use of the device; nothing else of the repository is changed
    model = VNNOccNet(latent_dim=LATENT_DIM, model_type="pointnet", return_features=True, sigmoid=True)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def _rotation_matrices(vectors):
    """(M, 3) rotation vectors to (M, 3, 3) rotation matrices (matrix exponential of the skew matrices)."""
    import torch

    zero = torch.zeros_like(vectors[:, 0])
    skew = torch.stack([zero, -vectors[:, 2], vectors[:, 1], vectors[:, 2], zero, -vectors[:, 0],
                        -vectors[:, 1], vectors[:, 0], zero], dim=1).reshape(-1, 3, 3)
    return torch.linalg.matrix_exp(skew)


@dataclass(frozen=True)
class Transfer:
    rotation: np.ndarray  # (3, 3) of the transform taking the demonstrated hand onto the new mesh (scaled frames)
    translation: np.ndarray  # (3,)
    loss: float  # descriptor L1 distance of the best start
    start_losses: np.ndarray  # (M,) of every start


def optimise_transfers(model, demos: Sequence[tuple[np.ndarray, np.ndarray]], target_points: np.ndarray,
                       rng: np.random.Generator, starts: int = 10, iterations: int = 250,
                       lr: float = 1e-2) -> list[Transfer]:
    """For every demonstration (mesh points, hand query points; scaled object frame), the rigid transform
    (x -> R x + t) that takes its hand onto the target mesh so that the query points' descriptors match the
    demonstration's. Meshes and query points are (N, 3) arrays in scaled object frames."""
    import torch
    from scipy.spatial.transform import Rotation

    device = next(model.parameters()).device
    count = len(demos)

    def cloud(points):
        picked = points[rng.permutation(len(points))[:SHAPE_POINTS]]
        return torch.from_numpy(picked - points.mean(axis=0)).float().to(device)

    with torch.no_grad():
        goals, queries, anchors = [], [], []
        for points, query in demos:
            query_t = torch.from_numpy(query - points.mean(axis=0)).float().to(device)
            latent = model.extract_latent({"point_cloud": cloud(points)[None]})
            goals.append(model.forward_latent(latent, query_t[None]))
            queries.append(query_t - query_t.mean(dim=0))
            anchors.append(points.mean(axis=0) + query_t.mean(dim=0).cpu().numpy())
        goal = torch.cat(goals)  # (G, N, F)
        turns = Rotation.random(count * starts, random_state=int(rng.integers(2**31))).as_matrix()
        start_rotation = torch.from_numpy(turns).float().to(device).reshape(count, starts, 3, 3)
        centred = torch.einsum("gmij,gnj->gmni", start_rotation, torch.stack(queries)).reshape(count * starts, -1, 3)
        clouds = torch.stack([cloud(target_points) for _ in range(count * starts)])
        target_latent = torch.cat([model.extract_latent({"point_cloud": chunk}) for chunk in clouds.split(40)])
    turn = torch.zeros(count * starts, 3, device=device, requires_grad=True)
    shift = (torch.rand(count * starts, 3, device=device) * 0.1).requires_grad_()
    optimiser = torch.optim.Adam([turn, shift], lr=lr)
    for _ in range(iterations):
        moved = centred @ _rotation_matrices(turn).transpose(1, 2) + shift[:, None]
        features = model.forward_latent(target_latent, moved).reshape(count, starts, *goal.shape[1:])
        losses = (features - goal[:, None]).abs().mean(dim=(2, 3))  # (G, M)
        optimiser.zero_grad()
        losses.mean(dim=1).sum().backward()  # every grasp's loss is the mean over its starts, as in the reference
        optimiser.step()
    with torch.no_grad():
        rotations = (_rotation_matrices(turn).reshape(count, starts, 3, 3) @ start_rotation).cpu().numpy().astype(float)
    losses, shifts = losses.detach().cpu().numpy(), shift.detach().cpu().numpy().reshape(count, starts, 3)
    target_centre, found = target_points.mean(axis=0), []
    for g in range(count):
        best = int(losses[g].argmin())
        rotation = rotations[g, best]
        # x -> R (x - anchor) + shift + centre of the target mesh, with the anchor the demonstrated query set's mean
        found.append(Transfer(rotation, shifts[g, best] + target_centre - rotation @ anchors[g],
                              float(losses[g, best]), losses[g]))
    return found


def transfer_grasps(model, grasps: Sequence[Grasp], source_points: Sequence[np.ndarray], target_points: np.ndarray,
                    rng: np.random.Generator, batch: int = 8, **options) -> list[tuple[Grasp, float]]:
    """The grasps placed on the target mesh by NDF, in its own (unscaled) frame, with their descriptor losses.
    `source_points[i]` is the mesh `grasps[i]` was made on; meshes are surface points in their body frames, metres."""
    s_target = ndf_scale(target_points)
    placed = []
    for first in range(0, len(grasps), batch):
        part = range(first, min(first + batch, len(grasps)))
        scales = [ndf_scale(source_points[i]) for i in part]
        demos = [(s * source_points[i], hand_points(grasps[i], s, rng)) for s, i in zip(scales, part)]
        moves = optimise_transfers(model, demos, s_target * target_points, rng, **options)
        for i, s, moved in zip(part, scales, moves):
            centre = (moved.rotation @ (s * np.asarray(grasps[i].centre)) + moved.translation) / s_target
            placed.append((Grasp(centre, moved.rotation @ grasps[i].closing, moved.rotation @ grasps[i].approach),
                           moved.loss))
    return placed
