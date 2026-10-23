"""Reference feasibility, predicted execution clearance, and deterministic ranking."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import path_clearances


@dataclass(frozen=True)
class Selection:
    candidate_id: object
    details: dict


def integrate_motion(initial_state, predicted_states, times_s, *,
                     imu_to_body_xyzw, dvl_to_body_xyzw):
    """Trapezoidal gyro/DVL readout in the same fixed initial FLU frame as references.

    State channels: DVL[0:3], gyro[3:6], acceleration[6:9], pressure[9],
    altitude[10], previous actuator values[11:19]. Mount rotations map sensors to FLU.
    """
    predictions = np.asarray(predicted_states, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if predictions.shape[1:] != (len(times)-1, 19) or not np.isfinite(predictions).all():
        raise ValueError('state prediction shape or values invalid')
    states = np.concatenate((np.broadcast_to(initial_state, (len(predictions), 1, 19)), predictions), axis=1)
    gyro = Rotation.from_quat(imu_to_body_xyzw).apply(states[:, :, 3:6].reshape(-1, 3)).reshape(len(states), -1, 3)
    velocity = Rotation.from_quat(dvl_to_body_xyzw).apply(states[:, :, :3].reshape(-1, 3)).reshape(len(states), -1, 3)
    previous = Rotation.identity(len(states))
    positions = np.zeros((len(states), len(times), 3))
    rotations = np.empty((len(states), len(times), 3, 3))
    rotations[:, 0] = previous.as_matrix()
    for k, dt in enumerate(np.diff(times), start=1):
        current = previous * Rotation.from_rotvec(.5*(gyro[:, k-1]+gyro[:, k])*dt)
        positions[:, k] = positions[:, k-1] + .5*(previous.apply(velocity[:, k-1])+current.apply(velocity[:, k]))*dt
        rotations[:, k] = current.as_matrix()
        previous = current
    return positions, rotations


def select_candidate(candidates, references, *, environment, vehicle, margin_m,
                     prediction_ids, predicted_paths, predicted_rotations, device='cpu'):
    ids = [c.candidate_id for c in candidates]
    if not ids or len(set(ids)) != len(ids) or ids != list(prediction_ids):
        raise ValueError('candidate/prediction binding mismatch')
    if ids != [r.candidate_id for r in references]:
        raise ValueError('candidate/reference binding mismatch')
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError('nonnegative finite clearance margin required')
    if len(environment['voxel_centres_m']) == 0:
        raise ValueError('measured obstacle geometry required')
    times = references[0].times_s
    if any(not np.array_equal(r.times_s, times) for r in references):
        raise ValueError('candidate time grids differ')
    ref_paths = np.asarray([r.positions_m for r in references])
    ref_rotations = Rotation.from_quat(np.asarray([r.orientations_xyzw for r in references]).reshape(-1, 4)).as_matrix().reshape(len(ids), len(times), 3, 3)
    if np.shape(predicted_paths) != ref_paths.shape or np.shape(predicted_rotations) != ref_rotations.shape:
        raise ValueError('predicted pose grid differs from reference grid')
    if not np.isfinite(predicted_paths).all() or not np.isfinite(predicted_rotations).all():
        raise ValueError('nonfinite predicted poses')
    clearances = path_clearances(np.concatenate((ref_paths, predicted_paths)),
                                np.concatenate((ref_rotations, predicted_rotations)),
                                environment, vehicle, device)
    details = {}
    for i, cid in enumerate(ids):
        nominal, predicted = clearances[i], clearances[len(ids)+i]
        details[cid] = dict(reference_clearance_m=nominal, predicted_clearance_m=predicted,
                            risk_score=max(0., margin_m-predicted),
                            eligible=nominal > margin_m and predicted > margin_m)
    eligible = [c for c in candidates if details[c.candidate_id]['eligible']]
    chosen = min(eligible, key=lambda c: (details[c.candidate_id]['risk_score'],
                                        c.edit_cost, c.rank, c.candidate_id)) if eligible else None
    return Selection(None if chosen is None else chosen.candidate_id, details)
