"""Call from an application supplying observations, a trained model and transport."""
from uuv_navigation.pipeline import decide
from uuv_navigation.controller import ReferencePlayer


def select_reference_player(candidates, *, times_s, history, model,
                            environment, vehicle, margin_m, imu_to_body_xyzw,
                            dvl_to_body_xyzw, anchor_position_m,
                            anchor_orientation_xyzw, frame_id, publish):
    selection, reference = decide(candidates, times_s=times_s, history=history,
        model=model, environment=environment, vehicle=vehicle, margin_m=margin_m,
        imu_to_body_xyzw=imu_to_body_xyzw, dvl_to_body_xyzw=dvl_to_body_xyzw)
    if reference is None:
        return selection, None
    player = ReferencePlayer(reference, start_ns=history.decision_ns,
        anchor_position_m=anchor_position_m, anchor_orientation_xyzw=anchor_orientation_xyzw,
        frame_id=frame_id, publish=publish)
    return selection, player
