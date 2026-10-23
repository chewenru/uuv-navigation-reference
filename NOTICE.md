# Implementation notes

- Candidate sampling, the quintic reference profile, reference-grid playback,
  swept body-box geometry and ranking are extracted or adapted from existing
  project implementations.
- The visual/recurrent prediction core retains the CNN, history fusion,
  conditional GRU, state-increment readout and displacement-head computations;
  its public interface uses ordinary tensors and references.
- Pose-dependent AABB extents follow the `abs(R) * half_size` construction in
  Bullet's `btTransformAabb` (`LinearMath/btAabbUtil2.h`), also used by Stonefish.
- Numerical and tensor operations use NumPy, SciPy and PyTorch. ROS and simulator
  source code are not redistributed by this package.
