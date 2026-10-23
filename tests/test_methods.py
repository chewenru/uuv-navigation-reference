import math
import unittest
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from uuv_navigation.constraints import Candidate, Limits, constrain
from uuv_navigation.sampling import generate_candidates
from uuv_navigation.reference import build_reference, nominal_motion_duration_s
from uuv_navigation.selection import integrate_motion, select_candidate
from uuv_navigation.geometry import path_clearances
from uuv_navigation.controller import ReferencePlayer


class MethodsTest(unittest.TestCase):
    def setUp(self):
        self.limits = Limits(.6, .5, .5, 5., .02)
        self.candidates = generate_candidates([.5,0.,0.], count=32, seed=7,
            speed_mps=.4, duration_s=3., max_angle_rad=math.pi/4,
            distance_fraction=.5, limits=self.limits)
        self.times = np.linspace(0.,3.,151)
        self.vehicle = {'primitives':[dict(shape='box', size_xyz_m=[.2,.2,.2],
            origin_xyz_m=[0.,0.,0.], origin_rpy_rad=[0.,0.,0.])]}

    def test_sampler_bounds_and_repeatability(self):
        second = generate_candidates([.5,0.,0.], count=32, seed=7,
            speed_mps=.4, duration_s=3., max_angle_rad=math.pi/4,
            distance_fraction=.5, limits=self.limits)
        self.assertEqual(self.candidates, second)
        self.assertEqual(len(self.candidates),32)
        for c in self.candidates:
            length = np.linalg.norm(c.offset_m)
            self.assertGreater(length,.02)
            self.assertLessEqual(length,.6)
            self.assertLessEqual(math.acos(c.offset_m[0]/length), math.pi/4)

    def test_explicit_speed_repair_only(self):
        fast = replace(self.candidates[0],speed_mps=.8)
        with self.assertRaises(ValueError): constrain(fast,self.limits)
        repaired = constrain(fast,self.limits,repair_speed=True)
        self.assertEqual(repaired.speed_mps,.5)
        self.assertEqual(repaired.offset_m,fast.offset_m)
        self.assertAlmostEqual(repaired.edit_cost,.6)
        with self.assertRaises(ValueError):
            constrain(replace(fast,offset_m=(2.,0.,0.)),self.limits,repair_speed=True)

    def test_reference_endpoint_and_speed(self):
        ref = build_reference(self.candidates[0],self.times)
        np.testing.assert_allclose(ref.positions_m[0],0.)
        np.testing.assert_allclose(ref.positions_m[-1],self.candidates[0].offset_m)
        np.testing.assert_allclose(ref.velocities[-1],0.)
        self.assertLessEqual(np.linalg.norm(ref.velocities[:,:3],axis=1).max(),.4+1e-10)
        c,m = ref.conditions()
        self.assertEqual(c.shape,(150,29))
        self.assertFalse(m[:,25:28].any())
        self.assertTrue(m[:,19:25].all())
        np.testing.assert_array_equal(c[:,19:25],0.)

    def test_impossible_duration_rejected(self):
        with self.assertRaises(ValueError):
            build_reference(replace(self.candidates[0],duration_s=.1),[0.,.1])

    def test_sweep_rejects_obstacle_between_endpoints(self):
        paths = np.array([[[-1.,0.,0.],[1.,0.,0.]]])
        rotations = np.broadcast_to(np.eye(3),(1,2,3,3)).copy()
        environment = dict(voxel_centres_m=[[0.,0.,0.]],voxel_size_m=.2)
        self.assertLess(path_clearances(paths,rotations,environment,self.vehicle,'cpu')[0],0.)

    def test_risk_selection_and_candidate_binding(self):
        refs = [build_reference(c,self.times) for c in self.candidates[:2]]
        paths = np.asarray([r.positions_m for r in refs])
        rotations = np.broadcast_to(np.eye(3),(2,len(self.times),3,3))
        env = dict(voxel_centres_m=[[5.,0.,0.]],voxel_size_m=.2)
        kwargs = dict(environment=env,vehicle=self.vehicle,margin_m=.1,
                      prediction_ids=[c.candidate_id for c in self.candidates[:2]],
                      predicted_paths=paths,predicted_rotations=rotations)
        result = select_candidate(self.candidates[:2],refs,**kwargs)
        self.assertEqual(result.candidate_id,self.candidates[0].candidate_id)
        # Make only the first predicted execution enter an occupied cube.
        unsafe = paths.copy(); unsafe[0,:,0] = np.linspace(0.,5.,len(self.times))
        unsafe[0,:,1:] = 0.
        result = select_candidate(self.candidates[:2],refs,**dict(kwargs,predicted_paths=unsafe))
        self.assertEqual(result.candidate_id,self.candidates[1].candidate_id)
        with self.assertRaises(ValueError):
            select_candidate(self.candidates[:2],refs,**dict(kwargs,prediction_ids=list(reversed(kwargs['prediction_ids']))))
        bad = dict(kwargs,environment=dict(voxel_centres_m=[[0.,0.,0.]],voxel_size_m=.2))
        self.assertIsNone(select_candidate(self.candidates[:2],refs,**bad).candidate_id)

    def test_motion_readout(self):
        state = np.zeros(19); state[0]=.2
        predictions = np.broadcast_to(state,(2,3,19))
        paths,rotations = integrate_motion(state,predictions,[0.,.1,.2,.3],
            imu_to_body_xyzw=[0.,0.,0.,1.],dvl_to_body_xyzw=[0.,0.,0.,1.])
        np.testing.assert_allclose(paths[:,-1,0],.06)
        np.testing.assert_allclose(rotations,np.broadcast_to(np.eye(3),rotations.shape))

    def test_control_uses_same_grid_and_absolute_start(self):
        ref = build_reference(self.candidates[0],self.times)
        sent = []
        player = ReferencePlayer(ref,start_ns=1_000_000_000,
            anchor_position_m=[1.,2.,3.],anchor_orientation_xyzw=[0.,0.,0.,1.],
            frame_id='local',publish=sent.append)
        with self.assertRaises(ValueError): player.tick(1_010_000_000)
        self.assertEqual(len(sent),0)
        player.tick(1_000_000_000)
        message = player.tick(1_510_000_000)
        np.testing.assert_allclose(message['position_m'],np.array(ref.sample(.51)['position_m'])+[1.,2.,3.])
        self.assertEqual(message['candidate_id'],ref.candidate_id)
        self.assertEqual(len(sent),2)
        with self.assertRaises(ValueError): player.tick(999_999_999)
        with self.assertRaises(ValueError): player.tick(4_000_000_001)
        self.assertEqual(len(sent),2)


if __name__ == '__main__':
    unittest.main()
