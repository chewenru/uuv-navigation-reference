"""One observation -> shared candidate references -> prediction -> selection."""
import numpy as np
import torch
from .reference import build_reference
from .selection import integrate_motion, select_candidate


def decide(candidates, *, times_s, history, model, environment, vehicle, margin_m,
           imu_to_body_xyzw, dvl_to_body_xyzw, geometry_device='cpu'):
    if model.training:
        raise ValueError('put the supplied trained model in evaluation mode')
    references = [build_reference(c, times_s) for c in candidates]
    if environment['decision_ns'] != history.decision_ns:
        raise ValueError('geometry and prediction start times differ')
    if environment['frame_id'] != 'anchor_flu':
        raise ValueError('geometry must be registered to the same initial FLU anchor')
    current_valid = history.valid[0,-1,:6]
    current_age = history.decision_ns-history.source_ns[0,-1,:6]
    if not bool((current_valid & (current_age <= model.max_age_ns)).all()):
        raise ValueError('current gyro and DVL must be measured for path readout')
    with torch.inference_mode():
        prediction = model(history, references)
    if prediction['decision_ns'] != history.decision_ns:
        raise ValueError('prediction start does not match observation')
    paths, rotations = integrate_motion(history.values[0,-1,:19].detach().cpu().numpy(),
        prediction['public_state_sequence'].detach().cpu().numpy(), times_s,
        imu_to_body_xyzw=imu_to_body_xyzw, dvl_to_body_xyzw=dvl_to_body_xyzw)
    selection = select_candidate(candidates, references, environment=environment,
        vehicle=vehicle, margin_m=margin_m, prediction_ids=prediction['candidate_ids'],
        predicted_paths=paths, predicted_rotations=rotations, device=geometry_device)
    chosen = next((r for r in references if r.candidate_id == selection.candidate_id), None)
    return selection, chosen
