"""A single reference representation shared by prediction and control."""
import copy
import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def nominal_motion_duration_s(command):
    """Quintic duration: peak progress derivative 15/8 preserves the speed ceiling."""
    if 'native_reference' in command['action']:
        return command['action']['native_reference']['motion_duration_ns']/1e9
    if command['action']['kind'] != 'move':
        return 0.0
    delta = [y-x for x,y in zip(command['start_position_world_m'],
                              command['target_position_world_m'])]
    return (15.0/8.0)*math.sqrt(sum(x*x for x in delta))/command['action']['speed_mps']


def nominal_reference(command, *, elapsed_s):
    """Return position, orientation, velocity and acceleration at the given time."""
    elapsed = float(elapsed_s)
    if elapsed < 0:
        raise ValueError('negative elapsed time')
    if 'native_reference' in command['action']:
        from .reference import sample_native_reference as sample_reference
        return sample_reference(command, elapsed)
    c = command
    a = c['action']
    delta = [y-x for x,y in zip(c['start_position_world_m'],c['target_position_world_m'])]
    travel_s = nominal_motion_duration_s(c)
    if travel_s and elapsed < travel_s:
        u = elapsed/travel_s
        fraction = u*u*u*(10.0+u*(-15.0+6.0*u))
        rate = 30.0*u*u*(1.0-u)*(1.0-u)/travel_s
        second_rate = 60.0*u*(1.0-u)*(1.0-2.0*u)/(travel_s*travel_s)
    else:
        fraction = 1.0 if travel_s else 0.0
        rate = second_rate = 0.0
    position = (copy.deepcopy(c['target_position_world_m']) if travel_s and elapsed >= travel_s
                else [x+fraction*d for x,d in zip(c['start_position_world_m'],delta)])
    # World-z yaw rotation preserves the start roll/pitch. The same rule is
    # used by the shared prediction and execution reference.
    half = fraction*a['yaw_delta_rad']/2
    z,w = math.sin(half), math.cos(half)
    qx,qy,qz,qw = c['start_orientation_xyzw']
    quat = [w*qx-z*qy, w*qy+z*qx, w*qz+z*qw, w*qw-z*qz]
    velocity = [d*rate for d in delta]
    acceleration = [d*second_rate for d in delta]
    omega = a['yaw_delta_rad']*rate
    alpha = a['yaw_delta_rad']*second_rate
    return dict(position_m=position, orientation_xyzw=quat,
                velocity=velocity+[0.0,0.0,omega],
                acceleration=acceleration+[0.0,0.0,alpha])

def sample_native_reference(command, elapsed_s):
    """Playback, not interpolation: WM and PID read the exact same sample."""
    value = command['action']['native_reference']
    stamp = int(round(elapsed_s*1e9))
    sample = value['samples'][min(stamp//value['step_ns'], len(value['samples'])-1)]
    # Samples contain flat numeric arrays; copy only the four consumed fields.
    return dict(position_m=[a+b for a,b in zip(command['start_position_world_m'],
                                               sample['position_offset_world_m'])],
        orientation_xyzw=list(sample['orientation_xyzw']),
        velocity=list(sample['velocity']), acceleration=list(sample['acceleration']))


@dataclass(frozen=True)
class Reference:
    candidate_id: str
    times_s: np.ndarray
    positions_m: np.ndarray
    orientations_xyzw: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    def sample(self, elapsed_s):
        """Left-available playback: prediction and controller consume the same grid."""
        if elapsed_s < 0 or elapsed_s > self.times_s[-1]:
            raise ValueError('reference time outside its declared interval')
        i = min(np.searchsorted(self.times_s, elapsed_s, side='right')-1, len(self.times_s)-1)
        return dict(position_m=self.positions_m[i].tolist(),
                    orientation_xyzw=self.orientations_xyzw[i].tolist(),
                    velocity=self.velocities[i].tolist(), acceleration=self.accelerations[i].tolist())

    def conditions(self):
        """29 reference channels at interval starts in the fixed initial FLU frame."""
        n = len(self.times_s)-1
        values = np.zeros((n, 29), dtype=np.float32)
        mask = np.ones_like(values, dtype=bool)
        values[:, :3] = self.positions_m[:-1]
        values[:, 3:7] = self.orientations_xyzw[:-1]
        values[:, 7:13] = self.velocities[:-1]
        values[:, 13:19] = self.accelerations[:-1]
        # No applied future feedback overlay; unknown initial tracking error stays masked.
        mask[:, 25:28] = False
        values[:, 28] = self.times_s[:-1]
        return values, mask


def build_reference(candidate, times_s):
    """Existing quintic profile, expressed in a fixed initial body-FLU anchor."""
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0):
        raise ValueError('strictly increasing reference grid starting at zero required')
    if times[-1] > candidate.duration_s:
        raise ValueError('reference horizon exceeds candidate duration')
    command = dict(start_position_world_m=[0., 0., 0.],
                   target_position_world_m=list(candidate.offset_m),
                   start_orientation_xyzw=[0., 0., 0., 1.],
                   action=dict(kind='move', speed_mps=candidate.speed_mps,
                               yaw_delta_rad=candidate.yaw_delta_rad))
    if nominal_motion_duration_s(command) > candidate.duration_s:
        raise ValueError('quintic move cannot complete within candidate duration')
    rows = [nominal_reference(command, elapsed_s=float(t)) for t in times]
    return Reference(candidate.candidate_id, times.copy(),
                     np.asarray([r['position_m'] for r in rows]),
                     np.asarray([r['orientation_xyzw'] for r in rows]),
                     np.asarray([r['velocity'] for r in rows]),
                     np.asarray([r['acceleration'] for r in rows]))
