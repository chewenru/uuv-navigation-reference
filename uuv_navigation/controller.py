"""Publish the selected reference through the caller's existing control transport."""
from scipy.spatial.transform import Rotation


class ReferencePlayer:
    def __init__(self, reference, *, start_ns, anchor_position_m,
                 anchor_orientation_xyzw, frame_id, publish):
        self.reference = reference
        self.start_ns = start_ns
        self.anchor_position = anchor_position_m
        self.anchor_rotation = Rotation.from_quat(anchor_orientation_xyzw)
        self.frame_id = frame_id
        self.publish = publish
        self.started = False

    def tick(self, now_ns):
        if not self.started and now_ns != self.start_ns:
            raise ValueError('first execution tick must match the prediction start')
        sample = self.reference.sample((now_ns-self.start_ns)/1e9)
        rotation = self.anchor_rotation
        sample['position_m'] = (rotation.apply(sample['position_m'])+self.anchor_position).tolist()
        sample['orientation_xyzw'] = (rotation*Rotation.from_quat(sample['orientation_xyzw'])).as_quat().tolist()
        for field in ('velocity', 'acceleration'):
            sample[field] = list(rotation.apply(sample[field][:3])) + list(rotation.apply(sample[field][3:]))
        message = dict(candidate_id=self.reference.candidate_id, start_ns=self.start_ns,
                       stamp_ns=now_ns, frame_id=self.frame_id, **sample)
        self.publish(message)
        self.started = True
        return message
