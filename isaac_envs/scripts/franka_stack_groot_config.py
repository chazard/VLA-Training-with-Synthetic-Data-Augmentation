"""
Modality configuration for Franka stack cube visuomotor task (for Groot model training).

This config maps the dataset structure (defined in meta/modality.json) to GR00T's
data processing pipeline. It defines:
- Video: ego_view (table cam) and wrist_view (wrist cam)
- State: eef_pos (3D) + eef_quat (4D) = end-effector pose
- Action: eef_delta (6D: position delta + axis-angle rotation delta) + gripper (1D)
- Language: task description annotation
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


franka_stack_config = {
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=[
            "ego_view",   # Table/external camera (matches meta/modality.json)
            "wrist_view",  # Wrist-mounted camera (matches meta/modality.json)
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Current state
        modality_keys=[
            "eef_pos",   # End-effector position (3D) - matches meta/modality.json
            "eef_quat",  # End-effector orientation quaternion (4D) - matches meta/modality.json
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # Predict 16 steps into the future (action horizon)
        modality_keys=[
            "eef_delta",  # End-effector pose delta: 6D (position + axis-angle rotation) - matches meta/modality.json
            "gripper",    # Gripper control: 1D - matches meta/modality.json
        ],
        action_configs=[
            # eef_delta: 6D relative pose change (position delta + axis-angle rotation delta)
            # Actions are stored as deltas in the dataset (relative pose changes)
            # Use ABSOLUTE representation - the stored delta is what we want the model to predict
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # Treat stored delta as the target action
                type=ActionType.NON_EEF,  # Use NON_EEF since it's 6D (EEF expects 9D with rot6d)
                format=ActionFormat.XYZ_ROTVEC,  # Position (3D) + axis-angle rotation vector (3D) = 6D total
            ),
            # gripper: 1D absolute control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # Gripper is absolute position
                type=ActionType.NON_EEF,  # Not end-effector space
                format=ActionFormat.DEFAULT,  # Standard format
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],  # Current annotation
        modality_keys=["annotation.human.action.task_description"],  # Matches meta/modality.json
    ),
}

# Register the configuration for NEW_EMBODIMENT tag
register_modality_config(franka_stack_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
