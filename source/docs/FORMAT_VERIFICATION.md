# Data Format Verification: Agent ↔ GR00T Config

## Summary
All data formats align correctly between the agent code and GR00T modality configuration.

---

## 1. Video Format ✓

**Config (`franka_stack_groot_config.py`):**
- Keys: `["ego_view", "wrist_view"]`
- `delta_indices=[0]` → T=1
- Expected: `(B, T=1, H, W, 3)` uint8

**Agent (`agent.py`):**
- Maps: `table_cam` → `ego_view`, `wrist_cam` → `wrist_view` ✓
- Input: `(B, H, W, 3)` from `obs["policy"]`
- Conversion: `x[:, None, ...]` → `(B, T=1, H, W, 3)` ✓
- Dtype: `np.uint8` ✓

**GR00T Expects:**
- `dict[str, np.ndarray[np.uint8, (B, T, H, W, 3)]]` ✓

---

## 2. State Format ✓

**Config (`franka_stack_groot_config.py`):**
- Keys: `["eef_pos", "eef_quat"]`
- `delta_indices=[0]` → T=1
- Expected: `eef_pos` (3D), `eef_quat` (4D) as separate keys
- Shape: `(B, T=1, D)` float32

**Agent (`agent.py`):**
- Extracts: `eef_pos`, `eef_quat` from `obs["policy"]` ✓
- Input: `(B, D)` or `(D,)` from environment
- Conversion: `(B, D)` → `(B, T=1, D)` via `x[:, None, :]` ✓
- Maps to separate keys: `{"eef_pos": ..., "eef_quat": ...}` ✓
- Order matches config: `state_keys_cfg[i]` maps to `state_blocks[i]` ✓
- Dtype: `np.float32` ✓

**GR00T Expects:**
- `dict[str, np.ndarray[np.float32, (B, T, D)]]` with separate keys ✓

---

## 3. Language Format ✓

**Config (`franka_stack_groot_config.py`):**
- Key: `["annotation.human.action.task_description"]`
- `delta_indices=[0]` → T=1
- Expected: `list[list[str]]` with shape `(B, T=1)`

**Agent (`agent.py`):**
- Reads: `task` key from `obs["policy"]` or uses default `"stack the cubes"` ✓
- Formats: `[[str], [str], ...]` for each batch item ✓
- Maps to config key: `lang_key = lang_cfg.modality_keys[0]` → `"annotation.human.action.task_description"` ✓
- Shape: `(B, T=1)` ✓

**GR00T Expects:**
- `dict[str, list[list[str]]]` with shape `(B, T)` ✓

---

## 4. Action Format ✓

**Config (`franka_stack_groot_config.py`):**
- Keys: `["eef_delta", "gripper"]`
- `delta_indices=list(range(0, 16))` → T=16 (action horizon)
- Expected: `eef_delta` (6D), `gripper` (1D) as separate keys
- Shape: `(B, T=16, D)` float32
- Total action dim: 6 + 1 = 7D

**Agent (`agent.py`):**
- Extracts: first timestep from each key: `arr[:, 0, :]` → `(B, D)` ✓
- Concatenates: `torch.cat([eef_delta_step0, gripper_step0], dim=-1)` → `(B, 7)` ✓
- Order matches config: uses `action_cfg.modality_keys` order ✓
- Verifies: `combined_action.shape[-1] == self.action_dim` (7) ✓
- Dtype: `torch.get_default_dtype()` (converted from float32) ✓

**GR00T Returns:**
- `dict[str, np.ndarray[np.float32, (B, T=16, D)]]` with separate keys ✓

**Environment Expects:**
- `(B, 7)` tensor: `[eef_delta (6D), gripper (1D)]` ✓

---

## Key Mappings Summary

| Modality | Env Key | Config Key | Agent Mapping |
|----------|---------|------------|---------------|
| Video | `table_cam` | `ego_view` | ✓ |
| Video | `wrist_cam` | `wrist_view` | ✓ |
| State | `eef_pos` | `eef_pos` | ✓ |
| State | `eef_quat` | `eef_quat` | ✓ |
| Language | `task` (or default) | `annotation.human.action.task_description` | ✓ |
| Action | N/A | `eef_delta` (6D) + `gripper` (1D) | ✓ (concatenated) |

---

## Verification Status: ✅ ALL FORMATS ALIGN

All data formats correctly match between:
1. Environment observations → Agent processing
2. Agent processing → GR00T policy input
3. GR00T policy output → Agent processing
4. Agent processing → Environment actions
