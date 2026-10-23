"""Print a generated candidate and its reference; no model or robot is started."""
import math
import numpy as np
from uuv_navigation.constraints import Limits
from uuv_navigation.sampling import generate_candidates
from uuv_navigation.reference import build_reference


def main():
    limits = Limits(max_translation_m=.6, max_yaw_delta_rad=.5,
                    max_speed_mps=.5, max_duration_s=5., position_tolerance_m=.02)
    candidates = generate_candidates([.5,0.,0.], count=32, seed=7,
        speed_mps=.4, duration_s=3., max_angle_rad=math.pi/4,
        distance_fraction=.5, limits=limits)
    reference = build_reference(candidates[0], np.linspace(0.,3.,151))
    print(f'{len(candidates)} candidates')
    print(candidates[0])
    print(reference.sample(.5))


if __name__ == '__main__':
    main()
