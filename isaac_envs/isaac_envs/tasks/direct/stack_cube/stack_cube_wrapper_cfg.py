
"""Custom stack cube visuomotor env config with dict-driven distribution parameters and active event term control."""

from __future__ import annotations

import copy
from typing import Any

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import stack_cube_events

# -----------------------------------------------------------------------------
# Event term names: all changeable terms, and default active subset
# -----------------------------------------------------------------------------

# Event terms with continuous or range-based params (set_distribution_parameters).
ALL_CHANGEABLE_EVENT_TERMS: list[str] = [
    "randomize_cube_1_position",
    "randomize_cube_2_position",
    "randomize_cube_3_position",
    "randomize_light_intensity_color",
    "randomize_table_cam_orientation",
    "randomize_wrist_cam_orientation",
    "randomize_franka_joint_state",
    "randomize_cube_1_scale",
    "randomize_cube_2_scale",
    "randomize_cube_3_scale",
]

# Event term that is always active (cannot be disabled)
ALWAYS_ACTIVE_EVENT_TERM = "init_franka_arm_pose"

# Discrete event terms (e.g. choice from texture list); can still be toggled active and have params set.
DISCRETE_EVENT_TERMS: list[str] = [
    "randomize_background_texture",
    "randomize_table_visual_material",
    "randomize_robot_arm_visual_texture",
]

# Default: only these run on environment reset. Discrete texture terms are off by default.
# Note: init_franka_arm_pose is always active (ensures consistent robot initialization).
# randomize_franka_joint_state is available but not active by default (for deterministic resets).
DEFAULT_ACTIVE_EVENT_TERMS: list[str] = [
    "randomize_cube_1_position",
    "randomize_cube_2_position",
    "randomize_cube_3_position",
    "randomize_light_intensity_color",


    #"randomize_franka_joint_state",
    #"randomize_table_cam_orientation",
    #"randomize_wrist_cam_orientation",
]

# Default event parameter ranges - all default values for event randomization
DEFAULT_EVENT_RANGES = {
    "cube": {
        "pose_range": {
            "x": (0.3, 0.6),
            "y": (-0.25, 0.25),
            "z": (0.0203, 0.0203),
            "yaw": (-1.0, 1.0), 
        },
        "min_separation": 0.2,
    },
    "light": {
        "intensity_range": (1500.0, 10000.0),
        "color_variation": 0.2,
    },
    "franka_joint": {
        "pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        "randomize_mean": 0.0,
        "randomize_std": 0.02,
    },
    "camera": {
        "pitch_range": (-0.05, 0.05),  # Small default: ±0.05 rad ≈ ±3°
        "yaw_range": (-0.05, 0.05),
    },
    #TODO: cube scaling keeps reseting after the first frame -- need to fix bug before we can use it in experiments
    "cube_scale": {
        "scale_percent_range": (-10.0, 10.0),  # -10% to +10% (0.9x–1.1x)
        "base_z_height": 0.0203,  # Cube half-height at scale=1 (m); cube center z = table_surface_z + this * scale
        "table_surface_height": 0.0,  # Table top z above env origin (m). Use 0 when table top is at env origin.
    },
}

def _default_light_textures() -> list[str]:
    """Default HDR list for dome/background light (minimal so randomization has at least one choice)."""
    return [f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr"]

def _default_table_textures() -> list[str]:
    """Default texture list for table (minimal default)."""
    return [f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png"]

def _default_robot_textures() -> list[str]:
    """Default texture list for robot arm (minimal default)."""
    return [f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png"]

def _build_event_term(term_name: str) -> EventTerm:
    """Build a single event term by name. Raises ValueError if term_name is not in allowed lists."""
    allowed = ALL_CHANGEABLE_EVENT_TERMS + DISCRETE_EVENT_TERMS + [ALWAYS_ACTIVE_EVENT_TERM]
    if term_name not in allowed:
        raise ValueError(f"Unknown event term '{term_name}'. Allowed: {allowed}")

    if term_name == "randomize_cube_1_position":
        return EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": copy.deepcopy(DEFAULT_EVENT_RANGES["cube"]["pose_range"]),
                "min_separation": DEFAULT_EVENT_RANGES["cube"]["min_separation"],
                "asset_cfgs": [SceneEntityCfg("cube_1")],
            },
        )
    elif term_name == "randomize_cube_2_position":
        return EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": copy.deepcopy(DEFAULT_EVENT_RANGES["cube"]["pose_range"]),
                "min_separation": DEFAULT_EVENT_RANGES["cube"]["min_separation"],
                "asset_cfgs": [SceneEntityCfg("cube_2")],
            },
        )
    elif term_name == "randomize_cube_3_position":
        return EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": copy.deepcopy(DEFAULT_EVENT_RANGES["cube"]["pose_range"]),
                "min_separation": DEFAULT_EVENT_RANGES["cube"]["min_separation"],
                "asset_cfgs": [SceneEntityCfg("cube_3")],
            },
        )
    elif term_name == "randomize_light_intensity_color":
        return EventTerm(
            func=stack_cube_events.randomize_scene_lighting_domelight,
            mode="reset",
            params={
                "intensity_range": DEFAULT_EVENT_RANGES["light"]["intensity_range"],
                "color_variation": DEFAULT_EVENT_RANGES["light"]["color_variation"],
                "textures": _default_light_textures(),
                "default_intensity": 3000.0,
                "default_color": (0.75, 0.75, 0.75),
                "default_texture": "",
                "components": ["intensity", "color"],
            },
        )
    elif term_name == "randomize_table_cam_orientation":
        return EventTerm(
            func=stack_cube_events.randomize_camera_orientation,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "pitch_range": DEFAULT_EVENT_RANGES["camera"]["pitch_range"],
                "yaw_range": DEFAULT_EVENT_RANGES["camera"]["yaw_range"],
            },
        )
    elif term_name == "randomize_wrist_cam_orientation":
        return EventTerm(
            func=stack_cube_events.randomize_camera_orientation,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "pitch_range": DEFAULT_EVENT_RANGES["camera"]["pitch_range"],
                "yaw_range": DEFAULT_EVENT_RANGES["camera"]["yaw_range"],
                "perturb_in_local_frame": True,
            },
        )
    elif term_name == "randomize_background_texture":
        return EventTerm(
            func=stack_cube_events.randomize_scene_lighting_domelight,
            mode="reset",
            params={
                "intensity_range": DEFAULT_EVENT_RANGES["light"]["intensity_range"],
                "color_variation": DEFAULT_EVENT_RANGES["light"]["color_variation"],
                "textures": _default_light_textures(),
                "default_intensity": 3000.0,
                "default_color": (0.75, 0.75, 0.75),
                "default_texture": "",
                "components": ["texture"],
            },
        )
    elif term_name == "randomize_table_visual_material":
        return EventTerm(
            func=stack_cube_events.randomize_visual_texture_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table"),
                "textures": _default_table_textures(),
                "default_texture": (
                    f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
                ),
                "force": True,
            },
        )
    elif term_name == "randomize_robot_arm_visual_texture":
        return EventTerm(
            func=stack_cube_events.randomize_visual_texture_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "textures": _default_robot_textures(),
                "default_texture": "",
                "force": True,
            },
        )
    elif term_name == "init_franka_arm_pose":
        return EventTerm(
            func=stack_cube_events.init_franka_arm_pose,
            mode="reset",
            params={
                "default_pose": copy.deepcopy(DEFAULT_EVENT_RANGES["franka_joint"]["pose"]),
            },
        )
    elif term_name == "randomize_franka_joint_state":
        return EventTerm(
            func=franka_stack_events.randomize_joint_by_gaussian_offset,
            mode="reset",
            params={
                "mean": DEFAULT_EVENT_RANGES["franka_joint"]["randomize_mean"],
                "std": DEFAULT_EVENT_RANGES["franka_joint"]["randomize_std"],
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    elif term_name == "randomize_cube_1_scale":
        return EventTerm(
            func=stack_cube_events.randomize_cube_scale_and_height,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube_1"),
                "scale_percent_range": DEFAULT_EVENT_RANGES["cube_scale"]["scale_percent_range"],
                "base_z_height": DEFAULT_EVENT_RANGES["cube_scale"]["base_z_height"],
                "table_surface_height": DEFAULT_EVENT_RANGES["cube_scale"]["table_surface_height"],
            },
        )
    elif term_name == "randomize_cube_2_scale":
        return EventTerm(
            func=stack_cube_events.randomize_cube_scale_and_height,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube_2"),
                "scale_percent_range": DEFAULT_EVENT_RANGES["cube_scale"]["scale_percent_range"],
                "base_z_height": DEFAULT_EVENT_RANGES["cube_scale"]["base_z_height"],
                "table_surface_height": DEFAULT_EVENT_RANGES["cube_scale"]["table_surface_height"],
            },
        )
    elif term_name == "randomize_cube_3_scale":
        return EventTerm(
            func=stack_cube_events.randomize_cube_scale_and_height,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube_3"),
                "scale_percent_range": DEFAULT_EVENT_RANGES["cube_scale"]["scale_percent_range"],
                "base_z_height": DEFAULT_EVENT_RANGES["cube_scale"]["base_z_height"],
                "table_surface_height": DEFAULT_EVENT_RANGES["cube_scale"]["table_surface_height"],
            },
        )
    else:
        raise ValueError(f"Event term '{term_name}' not implemented in _build_event_term")

def _build_events_from_term_names(term_names: list[str]) -> object:
    """Build an events object containing only the specified event terms.
    
    Always includes init_franka_arm_pose even if not in term_names.
    """
    events = type("EventCfg", (), {})()
    # Always add init_franka_arm_pose first (always active)
    setattr(events, ALWAYS_ACTIVE_EVENT_TERM, _build_event_term(ALWAYS_ACTIVE_EVENT_TERM))
    # Add all other requested terms
    for name in term_names:
        if name != ALWAYS_ACTIVE_EVENT_TERM:  # Avoid duplicate
            setattr(events, name, _build_event_term(name))
    return events

# -----------------------------------------------------------------------------
# Custom env config
# -----------------------------------------------------------------------------

@configclass
class CustomCubeStackEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """
    Visuomotor stack cube config with:
    - set_distribution_parameters(event_term_name -> param ranges/values) to set event params.
    - active_event_terms: only these events run on reset; default is DEFAULT_ACTIVE_EVENT_TERMS.
    - For deterministic values use (v, v) for range params (e.g. intensity_range=(5000, 5000)).
    - init_franka_arm_pose is always active (cannot be disabled) to ensure consistent robot initialization.
    """

    active_event_terms: list[str] = list(DEFAULT_ACTIVE_EVENT_TERMS)

    def __post_init__(self):
        super().__post_init__()
        # Replace events with only the active terms (so resets only trigger these).
        self.events = _build_events_from_term_names(self.active_event_terms)

    def set_distribution_parameters(self, params_dict: dict[str, dict[str, Any]]) -> None:
        """
        Set parameter ranges or fixed values for event terms.

        Args:
            params_dict: Map from event term name to a dict of param name -> value.
                Use (min, max) for ranges; (v, v) for deterministic. Pass full dicts (e.g. pose_range) to replace.

        Only updates terms that are currently in active_event_terms.
        """
        allowed = ALL_CHANGEABLE_EVENT_TERMS + DISCRETE_EVENT_TERMS
        for term_name, param_updates in params_dict.items():
            if term_name not in allowed:
                continue
            if not hasattr(self.events, term_name):
                continue
            term_cfg = getattr(self.events, term_name)
            for key, val in param_updates.items():
                term_cfg.params[key] = val

    def set_active_event_terms(self, term_names: list[str]) -> None:
        """
        Change which event terms are active and rebuild the events object.

        Args:
            term_names: List of event term names to activate. Must all be in ALL_CHANGEABLE_EVENT_TERMS
                or DISCRETE_EVENT_TERMS. Note: init_franka_arm_pose is always active and will be included
                automatically.
        """
        allowed = ALL_CHANGEABLE_EVENT_TERMS + DISCRETE_EVENT_TERMS
        for name in term_names:
            if name not in allowed:
                raise ValueError(f"Unknown event term '{name}'. Allowed: {allowed}")
        self.active_event_terms = list(term_names)
        self.events = _build_events_from_term_names(self.active_event_terms)

    def get_default_event_ranges(self) -> dict[str, Any]:
        """Return a deep copy of the default event parameter ranges for this env.
        Used by domain samplers (e.g. adversarial CEM) so they can apply to any env
        that exposes this interface."""
        return copy.deepcopy(DEFAULT_EVENT_RANGES)

    def get_default_active_event_terms(self) -> list[str]:
        """Return the default list of active event term names for this env.
        Used by domain samplers so they can apply to any env that exposes this interface."""
        return list(DEFAULT_ACTIVE_EVENT_TERMS)
