"""
Event functions that extend or override Isaac Lab stack task events for our environment.

Use these in stack_cube_wrapper_cfg instead of modifying third_party/IsaacLab.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def init_franka_arm_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float] | torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set default Franka joint pose and write it to the sim for reset envs.

    Isaac Lab runs scene.reset() before events, so the built-in set_default_joint_pose
    only updates the data buffer and never writes to sim for this reset. This version
    updates the buffer and writes to sim so the Franka actually resets to the default pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pose = torch.tensor(default_pose, dtype=torch.float32, device=env.device)
    # Update default for all envs so future scene resets use it
    asset.data.default_joint_pos = pose.unsqueeze(0).repeat(env.num_envs, 1)
    # Write to sim for the envs we are resetting so this reset shows the default pose
    joint_pos = asset.data.default_joint_pos[env_ids]
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_vel.zero_()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    default_intensity: float = 3000.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    components: list[str] | None = None,
):
    """Randomize dome light intensity, color, and/or background texture.

    When :attr:`components` is provided (e.g. ["intensity", "color"] or ["texture"]),
    only those aspects are randomized and eval_mode is ignored. When :attr:`components`
    is None, delegates to Isaac Lab's version (eval_mode / eval_type behavior).
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(default_intensity)

    color_attr = light_prim.GetAttribute("inputs:color")
    color_attr.Set(default_color)

    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
    texture_file_attr.Set(default_texture)

    use_components = components is not None and len(components) > 0
    if not use_components:
        # Delegate to Isaac Lab original (respects eval_mode / eval_type)
        franka_stack_events.randomize_scene_lighting_domelight(
            env=env,
            env_ids=env_ids,
            intensity_range=intensity_range,
            color_variation=color_variation,
            textures=textures,
            default_intensity=default_intensity,
            default_color=default_color,
            default_texture=default_texture,
            asset_cfg=asset_cfg,
        )
        return

    do_intensity = "intensity" in components
    do_color = "color" in components
    do_texture = "texture" in components

    if do_intensity:
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        intensity_attr.Set(new_intensity)

    if do_color:
        new_color = franka_stack_events.sample_random_color(base=default_color, variation=color_variation)
        color_attr.Set(new_color)

    if do_texture:
        new_texture = random.sample(textures, 1)[0]
        texture_file_attr.Set(new_texture)

#TODO:: check this out
def randomize_visual_texture_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    textures: list[str],
    default_texture: str = "",
    texture_rotation: tuple[float, float] = (0.0, 0.0),
    force: bool = False,
):
    """Randomize the visual texture of bodies on an asset.

    When :attr:`force` is True, randomization runs regardless of eval_mode.
    When False, delegates to Isaac Lab's version (eval_mode / eval_type behavior).
    """
    if not force:
        franka_stack_events.randomize_visual_texture_material(
            env=env,
            env_ids=env_ids,
            asset_cfg=asset_cfg,
            textures=textures,
            default_texture=default_texture,
            texture_rotation=texture_rotation,
        )
        return

    # force=True: run regardless of eval_mode
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize visual texture material with scene replication enabled."
            " For stable USD-level randomization, please disable scene replication"
            " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
        )

    texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)
    asset = env.scene[asset_cfg.name]

    body_names = asset_cfg.body_names
    if isinstance(body_names, str):
        body_names_regex = body_names
    elif isinstance(body_names, list):
        body_names_regex = "|".join(body_names)
    else:
        body_names_regex = ".*"

    if not hasattr(asset, "cfg"):
        prims_group = rep.get.prims(path_pattern=f"{asset.prim_paths[0]}/visuals")
    else:
        prims_group = rep.get.prims(path_pattern=f"{asset.cfg.prim_path}/{body_names_regex}/visuals")

    with prims_group:
        rep.randomizer.texture(
            textures=textures, project_uvw=True, texture_rotate=rep.distribution.uniform(*texture_rotation)
        )


# Cache of nominal (default) world orientation per camera sensor, so we perturb from a fixed
# reference each reset and avoid hysteresis. Key: sensor name; value: quat (4,) wxyz on CPU.
_camera_nominal_quat_cache: dict[str, torch.Tensor] = {}


def randomize_camera_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    pitch_range: tuple[float, float] = (-0.05, 0.05),
    yaw_range: tuple[float, float] = (-0.05, 0.05),
    perturb_in_local_frame: bool = False,
):
    """Randomize camera orientation by perturbing pitch and yaw from a fixed nominal.

    Uses a cached nominal orientation (from the first run, when the camera is still at USD default)
    so orientations are i.i.d. across resets with no hysteresis. For a camera attached to the
    robot (e.g. wrist): set perturb_in_local_frame=True so the perturbation is in the camera's
    frame. For a fixed camera (e.g. table): use perturb_in_local_frame=False so pitch/yaw are
    perturbed in world frame.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    from isaaclab.sensors import Camera

    camera: Camera = env.scene.sensors[sensor_cfg.name]
    name = sensor_cfg.name

    current_positions = camera.data.pos_w[env_ids].clone()
    current_orientations = camera.data.quat_w_world[env_ids].clone()

    # Use a fixed nominal orientation each time to avoid correlation across resets.
    if name not in _camera_nominal_quat_cache:
        # First time: current pose is the USD default; cache it for future resets.
        _camera_nominal_quat_cache[name] = current_orientations[0].detach().clone().cpu()
    nominal = _camera_nominal_quat_cache[name].to(device=env.device, dtype=current_orientations.dtype)
    nominal = nominal.unsqueeze(0).expand(len(env_ids), 4)

    pitch_perturb = math_utils.sample_uniform(
        pitch_range[0], pitch_range[1], (len(env_ids),), device=env.device
    )
    yaw_perturb = math_utils.sample_uniform(
        yaw_range[0], yaw_range[1], (len(env_ids),), device=env.device
    )
    roll_perturb = torch.zeros_like(pitch_perturb)
    orientation_deltas = math_utils.quat_from_euler_xyz(roll_perturb, pitch_perturb, yaw_perturb)

    if perturb_in_local_frame:
        new_orientations = math_utils.quat_mul(nominal, orientation_deltas)
    else:
        new_orientations = math_utils.quat_mul(orientation_deltas, nominal)

    camera.set_world_poses(
        positions=current_positions, orientations=new_orientations, env_ids=env_ids, convention="world"
    )


def randomize_cube_scale_and_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_percent_range: tuple[float, float],
    base_z_height: float,
    table_surface_height: float = 0.0,
):
    """Randomize cube scale on USD and set z so the cube bottom sits on the table surface.

    - Samples scale from scale_percent_range (e.g. -10..10 -> 0.9..1.1) and applies it to the cube prim.
    - table_surface_height: table top z above env origin (m). Cube center z = env_origin.z + table_surface_height + base_z_height * scale.
    - Only updates scale and z; x,y are left as set by position randomization (run this event after position events).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]

    # Sample scale as percentage -> multiplier (e.g. -10% -> 0.9, +10% -> 1.1)
    scale_percent = math_utils.sample_uniform(
        scale_percent_range[0], scale_percent_range[1], (len(env_ids),), device=env.device
    )
    scale_factor = 1.0 + (scale_percent / 100.0)
    scale_factor_cpu = scale_factor.cpu()

    # Z: table surface = env_origin.z + table_surface_height; cube center = table_surface + half_height at this scale
    table_surface_z = env.scene.env_origins[env_ids, 2] + table_surface_height
    half_height = base_z_height * scale_factor
    new_z_world = table_surface_z + half_height

    # Write pose to sim FIRST so any physics→USD sync uses the correct pose.
    # Then apply USD scale LAST so nothing after us overwrites it (e.g. set_transforms can push transform and reset scale).
    pos = asset.data.root_pos_w[env_ids].clone()
    pos[:, 2] = new_z_world
    quat = asset.data.root_quat_w[env_ids].clone()
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    # Apply uniform scale to each cube prim on the stage (after pose write so we are last to touch the prim)
    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids_cpu.tolist()):
            prim_path = prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            s = float(scale_factor_cpu[i].item())
            scale_vec = Gf.Vec3f(s, s, s)
            if prim.IsA(UsdGeom.Xformable):
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                scale_op = None
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                if scale_op is None:
                    scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
                scale_op.Set(scale_vec)
