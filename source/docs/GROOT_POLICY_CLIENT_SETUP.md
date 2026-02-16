# GR00T policy server and client setup

Summary from `third_party/groot/README.md` and `getting_started/policy.md`.


## 1. Start the policy server (GPU machine)

From the **GROOT repo root**:

```bash
# With a trained model
uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /path/to/checkpoint \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555

# Replay mode (no model): replays actions from a LeRobot dataset for debugging
uv run python gr00t/eval/run_gr00t_server.py \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --port 5555 \
  --execution-horizon 8
```

Parameters:

- `--embodiment-tag`: e.g. `GR1`, `OXE_GOOGLE`, `LIBERO_PANDA`, `NEW_EMBODIMENT` (for custom Franka stack use `NEW_EMBODIMENT` and a modality config).
- `--model-path`: checkpoint dir (or HuggingFace id like `nvidia/GR00T-N1.6-3B`).
- `--host`: `127.0.0.1` (local only) or `0.0.0.0` (accept external connections).
- `--port`: default `5555`.

No custom server code is required.

## 2. Client: PolicyClient

On the **Isaac / rollout side** (this repo)

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555, timeout_ms=15000, strict=False)
if not policy.ping():
    raise RuntimeError("Cannot connect to policy server!")

# Same interface as a local policy
modality_configs = policy.get_modality_config()
action, info = policy.get_action(observation)
info = policy.reset(options=None)
```

Observation and action formats are defined in `getting_started/policy.md`.

## 3. Multiple parallel environments (batch B)

The policy API supports **batched inference**. You send one batch per step:

- **Observation**
  - `video`: dict of arrays shape `(B, T, H, W, 3)` with `B = num_envs`.
  - `state`: dict of arrays `(B, T, D)`.
  - `language`: `(B, 1)` list of lists of strings (e.g. `[["stack the cubes"]] * B`).
- **Action**
  - Returned as dict of arrays `(B, action_horizon, action_dim)`.
  - For `env.step()` you typically use the first step: `action["action_name"][:, 0, :]` → shape `(B, action_dim)`.

So the client does **not** call the server once per env; it sends one observation batch of size `B = num_envs` and receives one action batch.

---

## 4. Comparison rollout (model vs replay)

Two parallel sim envs in lockstep: **env 0** = model server, **env 1** = replay server. Both start at the same demo/step from the HDF5 dataset.

### Policy server commands (run from GROOT repo root)

**Model server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path /path/to/checkpoint --device cuda:0 --host 127.0.0.1 --port 5555
```

**Replay server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py --dataset-path /path/to/lerobot_dataset --embodiment-tag NEW_EMBODIMENT --host 127.0.0.1 --port 5556 --execution-horizon 1
```

Use different ports (5555 and 5556). `--execution-horizon 1` keeps replay in lockstep with the sim.

### Prerequisites

- HDF5 dataset (Franka stack format) for loading initial/step state.
- LeRobot dataset with episode index matching HDF5 demo index (e.g. demo_0 = episode 0).
- Both servers running before starting the rollout script.

### Run comparison rollout

**Terminal 1:** Start model server (command above, port 5555).

**Terminal 2:** Start replay server (command above, port 5556).

**Terminal 3:**
```bash
python source/scripts/run_agent_rollout.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --agent groot --demo_replay_comparison --hdf5-path /path/to/franka_stack_dataset.hdf5 --demo-index 0 --start-step 0 --model-port 5555 --replay-port 5556
```

| Argument | Description |
|----------|-------------|
| `--demo_replay_comparison` | Enable demo vs replay comparison (requires `--hdf5-path`, `--agent groot`). |
| `--hdf5-path` | Path to Franka stack HDF5 (required with `--demo_replay_comparison`). |
| `--demo-index` | Demo in HDF5 (e.g. 0 = demo_0); must match LeRobot episode. |
| `--start-step` | Step within demo (0 = start). Replay server is reset to this episode/step. |
| `--model-port`, `--replay-port` | Must match the two servers (defaults 5555, 5556). |

Demo index alignment: HDF5 demo_0, demo_1, ... map to LeRobot episodes 0, 1, ... by the converter.

