"""Action constraints and the explicit positive-speed upper-bound correction."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Limits:
    max_translation_m: float
    max_yaw_delta_rad: float
    max_speed_mps: float
    max_duration_s: float
    position_tolerance_m: float


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    offset_m: tuple
    speed_mps: float
    duration_s: float
    yaw_delta_rad: float = 0.0
    rank: int = 0
    edit_cost: float = 0.0


def constrain(candidate, limits, *, repair_speed=False):
    """Correct only the explicitly enabled speed ceiling; reject other violations."""
    c = candidate
    values = (*c.offset_m, c.speed_mps, c.duration_s, c.yaw_delta_rad)
    if len(c.offset_m) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('finite 3D candidate required')
    if repair_speed and c.speed_mps > limits.max_speed_mps:
        c = replace(c, speed_mps=limits.max_speed_mps,
                    edit_cost=c.edit_cost + (c.speed_mps-limits.max_speed_mps)/limits.max_speed_mps)
    distance = math.sqrt(sum(v*v for v in c.offset_m))
    if not limits.position_tolerance_m < distance <= limits.max_translation_m:
        raise ValueError('translation outside action limits')
    if not 0 < c.speed_mps <= limits.max_speed_mps:
        raise ValueError('speed outside action limits')
    if abs(c.yaw_delta_rad) > limits.max_yaw_delta_rad:
        raise ValueError('heading outside action limits')
    if not 0 < c.duration_s <= limits.max_duration_s:
        raise ValueError('duration outside action limits')
    return c
