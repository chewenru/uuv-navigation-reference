"""Swept body-box clearance against occupied cubes in one shared frame."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def path_clearances(paths, orientations, environment, vehicle, device):
    """GPU clearance lower bound for registered body boxes and occupied cubes.

    Pose-dependent world AABBs use Bullet's btTransformAabb rule |R| * half_size
    (Stonefish/3rdparty/LinearMath/btAabbUtil2.h), adapted to batched CUDA tensors.
    They still enclose the oriented boxes: not exact mesh/OBB collision distance.
    Across each linear-position/shortest-SLERP segment, subtract half the base
    translation plus the furthest corner's half-rotation travel. Thus checking
    only clear endpoints cannot admit an intervening translation/rotation hit.
    This bounds the sampled pose path, not model error or true vehicle motion.
    """
    import torch
    points = torch.as_tensor(environment['voxel_centres_m'], dtype=torch.float64, device=device)
    positions = torch.as_tensor(np.asarray(paths), dtype=torch.float64, device=device)
    rotations = torch.as_tensor(np.asarray(orientations), dtype=torch.float64, device=device)
    relative = rotations[:, :-1].transpose(-1, -2) @ rotations[:, 1:]
    angles = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1.,1.).acos()
    half_steps = .5*(positions[:, 1:]-positions[:, :-1]).norm(dim=-1)
    best = torch.full((len(positions),), float('inf'), dtype=torch.float64, device=device)
    for part in vehicle['primitives']:
        assert part['shape'] == 'box'
        half = np.asarray(part['size_xyz_m'])/2
        origin = np.asarray(part['origin_xyz_m'])
        local = Rotation.from_euler('xyz', part['origin_rpy_rad']).as_matrix()
        corners = np.asarray(list(itertools.product((-1.,1.), repeat=3)))*half
        sweep_radius = float(np.linalg.norm(corners @ local.T+origin, axis=1).max())
        centres = positions + rotations @ torch.as_tensor(origin, dtype=torch.float64, device=device)
        axes = rotations @ torch.as_tensor(local, dtype=torch.float64, device=device)
        extents = axes.abs() @ torch.as_tensor(half, dtype=torch.float64, device=device)
        gaps = (centres[:, :, None]-points[None, None]).abs()-extents[:, :, None]-environment['voxel_size_m']/2
        distances = gaps.clamp_min(0.).norm(dim=-1)+gaps.amax(dim=-1).clamp_max(0.)
        sweep = half_steps + 2*sweep_radius*torch.sin(angles/4)
        lower_bounds = torch.minimum(distances[:, :-1], distances[:, 1:])-sweep[:, :, None]
        best = torch.minimum(best, lower_bounds.amin(dim=(1,2)))
    return best.tolist()
