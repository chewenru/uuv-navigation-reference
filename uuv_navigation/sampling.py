"""Bounded radial strata and opposing angular samples around a local guide."""
import math
import random


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sampling_domain(mean, policy, limits, *, lateral_pair=False, stratified=False):
    """A legal support derived BEFORE any random draw; no obstacle-dependent resampling.

    If n is the unit guide and theta < asin(|n_i|), coordinate i cannot
    cross zero. Half that angle leaves numerical margin for all guided axes.
    Radial support also reserves a tiny inward numerical margin; it is NOT
    a post-draw action correction. Empty/invalid support is rejected, not filled.
    """
    radius = math.sqrt(sum(x*x for x in mean))
    minimum = limits['position_tolerance_m']; maximum = policy['max_sample_distance_m']
    require(minimum < radius <= maximum, 'nominal move outside bounded support')
    unit = [x/radius for x in mean]
    sign_angle = .5*min(math.asin(min(1.,abs(x))) for x in unit if x != 0.)
    tangent = None
    if lateral_pair:
        horizontal = math.hypot(unit[0], unit[1])
        tangent = [-unit[1]/horizontal, unit[0]/horizontal, 0.] if horizontal else [1., 0., 0.]
        # Planar left/right offsets do not vary the vertical component. Small
        # nonzero z must not collapse horizontal alternatives to a tiny cone.
        sign_angle = min([math.pi/2]+[.5*math.atan(abs(x/t)) for x,t in zip(unit,tangent) if x and t])
    angle = min(policy['max_direction_angle_rad'], sign_angle)
    fraction = policy['distance_fraction']
    eight = stratified
    # When both spreads are zero, preserve the exact nominal duplicate. Do not
    # manufacture diversity. Otherwise construct an interior radial support.
    exact = fraction == 0. and angle == 0.
    safe_max = maximum if exact else maximum-(maximum-minimum)*1e-12
    upper = min(radius*(1.+fraction), safe_max)
    lower = min(max(radius*(1.-fraction), (radius+minimum)/2.), upper)
    if eight and fraction > 0:
        lower = min(max(radius*fraction, minimum+(radius-minimum)*1e-6), upper)
    require(minimum < lower <= upper <= maximum, 'empty bounded sampling support')
    return dict(kind=('four_radial_strata_opposite_planar_sides_v1' if eight else
        ('opposite_lateral_half_angle_pair_uniform_radius_v1' if lateral_pair else
         'uniform_solid_angle_cone_and_uniform_radius_v2')),
        nominal_distance_m=radius, axis_body_flu=unit,
        distance_min_m=lower, distance_max_m=upper, max_angle_rad=angle,
        exact_nominal_duplicate=exact,
        **({'lateral_tangent_body_flu': tangent} if lateral_pair else {}))


def sample_stratified_offsets(mean, domain, seed, budget=8, *, full_3d=False):
    """Radially stratified candidates with opposing angular directions.

    The default keeps the original planar left/right sampler unchanged. The
    explicit native 3D branch uses antipodal azimuth pairs around the guide,
    spanning the full declared cone without changing radial/theta draws.
    """
    rng = random.Random(seed); rows = []
    unit = domain['axis_body_flu']
    tangent = domain.get('lateral_tangent_body_flu')
    width = domain['distance_max_m'] - domain['distance_min_m']
    first_side = 1 if seed % 2 == 0 else -1
    strata=budget//2
    order=(3,0,2,1) if budget==8 else (strata-1,0,*range(strata-2,0,-1))
    if full_3d:
        index = min(range(3), key=lambda i: abs(unit[i]))
        reference = [float(i == index) for i in range(3)]
        first = [unit[1]*reference[2]-unit[2]*reference[1],
                 unit[2]*reference[0]-unit[0]*reference[2],
                 unit[0]*reference[1]-unit[1]*reference[0]]
        norm = math.sqrt(sum(x*x for x in first))
        first = [x/norm for x in first]
        second = [unit[1]*first[2]-unit[2]*first[1],
                  unit[2]*first[0]-unit[0]*first[2],
                  unit[0]*first[1]-unit[1]*first[0]]
    for pair_index, stratum in enumerate(order):
        for side in (first_side, -first_side):
            draws = [rng.random() for _ in range(3)]
            length = domain['distance_min_m'] + width*(stratum+draws[0])/strata
            theta = domain['max_angle_rad']*(.25+.375*(draws[1]+draws[2]))
            if full_3d:
                phi = pair_index*math.pi/strata + (0. if side == first_side else math.pi)
                transverse = [first[i]*math.cos(phi)+second[i]*math.sin(phi) for i in range(3)]
                vector = [unit[i]*math.cos(theta)+transverse[i]*math.sin(theta) for i in range(3)]
            else:
                vector = [unit[i]*math.cos(theta)+side*tangent[i]*math.sin(theta) for i in range(3)]
            norm = math.sqrt(sum(x*x for x in vector))
            offset = list(mean) if domain['exact_nominal_duplicate'] else [length*x/norm for x in vector]
            row = dict(uniform_01=draws, offset_m=offset, radial_stratum=stratum)
            row['angular_side' if full_3d else 'lateral_side'] = side
            if full_3d:
                row['azimuth_rad'] = phi
            rows.append(row)
    return rows


def generate_candidates(guide_m, *, count, seed, speed_mps, duration_s,
                        max_angle_rad, distance_fraction, limits):
    """The guide is supplied by the task/VLM interface in the current body frame."""
    from .constraints import Candidate, constrain
    require(type(count) is int and count >= 4 and count % 2 == 0,
            'an even candidate count of at least four is required')
    require(0 <= max_angle_rad <= math.pi/2 and 0 <= distance_fraction <= 1,
            'invalid angular or radial spread')
    policy = dict(max_sample_distance_m=limits.max_translation_m,
                  max_direction_angle_rad=max_angle_rad, distance_fraction=distance_fraction)
    domain = sampling_domain(guide_m, policy, vars(limits), stratified=True)
    rows = sample_stratified_offsets(guide_m, domain, seed, budget=count, full_3d=True)
    return [constrain(Candidate(f'candidate_{i:03d}', tuple(row['offset_m']),
                               speed_mps, duration_s, rank=i), limits)
            for i, row in enumerate(rows)]
