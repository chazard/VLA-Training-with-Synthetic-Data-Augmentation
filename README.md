# Active Learning for Robust Visuomotor Cube Stacking with MimicGen + GR00T (WIP)

## Overview

This project tests whether we can systematically improve robustness of a visuomotor stacking policy by combining:

- **Isaac Lab Mimic (MimicGen)** for synthetic demonstration generation
- **GR00T** fine-tuning on the resulting dataset
- **Failure-region search** in a continuous perturbation space using Cross-Entropy Method (CEM)
- An **active learning loop** that targets synthetic data generation toward discovered failure pockets

**Primary task target:** `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0` — Stack three cubes (bottom to top: blue, red, green) with Franka (IK-relative control). The blueprint env is used for the NVIDIA Isaac GR00T blueprint for synthetic manipulation motion generation.

## Problem Statement

MimicGen is a tool for generating multiple synthetic demonstrations from a given real demonstration for a task by stitching together motions under randomized domains. This is typically done with uniform sampling, which after some number of samples yields diminishing returns, particularly as we increase the dimensionality of our domain (curse of dimensionality). In a given randomized domain, we can vary features such as:

- Object pose/layout configurations (geometry/contact)
- Camera viewpoint shifts (yaw/pitch)
- Lighting intensity/color changes (appearance)

By randomizing these uniformly we can get pretty good coverage of our domain, however, we can expect to see pockets in this randomized domain where performance drops off. In that case we would want to robustify our policy by sampling in these low performing regions and generating synthetic samples in these pockets to improve our policy. The goal of this project is to implement an active learning loop that outperforms the basedline uniform sampled mimicgen on a VLA imitation learned policy.

Please note that this project is currently a work in progress ...

## Milestones

### Milestone 1 — Baseline MimicGen → GR00T

- Collect/obtain demonstration HDF5 for cube stacking
- Annotate demos for Mimic
- Generate synthetic dataset via MimicGen (uniform sampling)
- Convert HDF5 → LeRobot
- Fine-tune GR00T baseline
- Evaluate baseline on nominal conditions + held-out randomized eval

### Milestone 2 — Verify "there is a problem"

- Define continuous perturbation space θ (geometry + camera + lighting)
- Run failure search (CEM) over θ on the baseline policy
- Build a reproducible "failure suite" of θ values and seeds

### Milestone 3 — Active Learning Loop

- Condition MimicGen randomization on elite θ regions found by CEM (cross entropy method)
- Regenerate targeted synthetic data
- Fine-tune GR00T on expanded dataset
- Re-evaluate on the frozen failure suite and report improvement

### Milestone 4 — Extensions

- Extend domain perturbations (camera intrinsics, cube scale, etc.)
- Extend to additional tasks/scenes (multi-task training) to prove scalability of active learning methodology
- Extend to other VLA foundation models like pi-0

## Isaac Lab Mimic (MimicGen) Baseline

### What MimicGen is doing here

We treat Isaac Lab Mimic as a baseline data multiplier:

1. Start from a small set of demonstrations (teleop or provided dataset).
2. Annotate demonstrations (phase/subtask structure).
3. Generate many synthetic rollouts by perturbing scene parameters at reset (domain randomization event terms) and replaying/warping motion.

In this repo, we use MimicGen to produce an HDF5 dataset containing actions, states, and camera observations.

### Example Franka stack HDF5 structure

The dataset at `data/franka_stack_dataset.hdf5` can be inspected with:

```bash
python scripts/print_hdf5_structure.py
```

Typical structure (example `data/demo_0/`):

- `data/demo_0/actions`: (T, 7)
- `data/demo_0/obs/`:
  - `table_cam`: image tensors
  - `wrist_cam`: image tensors
  - Proprio + object state fields
- `data/demo_0/states/`:
  - Robot joint/root states
  - `cube_1` / `cube_2` / `cube_3` root pose/velocity

Image tensors are written to:

- `data/demo_*/obs/table_cam`
- `data/demo_*/obs/wrist_cam`

## Running the agent

The custom stack cube task `Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0` supports pluggable agents. See `docs/AGENT_INTERFACE.md` for the observation/action contract.

### Rollout (run_agent_rollout.py)

Run the sim with a built-in or GR00T agent:

```bash
python scripts/run_agent_rollout.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --agent random
python scripts/run_agent_rollout.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --agent groot
```

With `--agent groot` the GR00T policy server must be running (see `docs/GROOT_POLICY_CLIENT_SETUP.md`). For demo vs replay comparison (two envs in lockstep):

```bash
python scripts/run_agent_rollout.py --agent groot --demo_replay_comparison --hdf5-path /path/to/data.hdf5 --demo-index 0 --start-step 0
```

### Evaluation (eval_stack_cube.py)

Run N episodes and report success rate and mean reward:

```bash
python scripts/eval_stack_cube.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --num_episodes 20 --agent groot
python scripts/eval_stack_cube.py --agent groot --num_episodes 20 --headless
```

Use `--headless` for no GUI. For GR00T evaluation the policy server must be running.

## GR00T Baseline Training

We fine-tune GR00T on the Mimic-generated dataset.

### Convert HDF5 → LeRobot

```bash
python scripts/convert_franka_stack_to_lerobot.py \
  --input data/generated_dataset_small.hdf5 \
  --output data/generated_dataset_small_lerobot \
  --eval-proportion 0.1
```

### Train GR00T

```bash
cd ../third_party/groot
conda activate groot

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/generated_dataset_small_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/configs/franka_stack_groot_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir /home/chris/groot_training_results \
  --save-total-limit 5 \
  --save-steps 2000 \
  --max-steps 10000 \
  --use-wandb \
  --global-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --optim paged_adamw_8bit \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  --num-shards-per-epoch 10000
```

Note that the above training command is suitable for training with a single 24 GB gpu (eg an RTX 4090) since we do training with Adam in 8 bits. For more information, see [README_GR00T_TRAINING.md](README_GR00T_TRAINING.md).



# Active Learning Loop (Gradient-Guided MimicGen)

## Core Idea

Instead of searching for binary failure regions, we use an **acquisition
function based on the magnitude of the policy's training loss
gradient**.

We ask:

> Which mutated expert trajectories would most improve the current
> policy if we trained on them?

We approximate this using the squared L2 norm of the gradient of the
flow-matching loss with respect to the policy's action head.

This turns active learning into a **training-signal maximization
problem**, not a failure-rate maximization problem.

------------------------------------------------------------------------

## Notation

Let:

-   D = {d₁, ..., d\_\|D\|} be the seed demonstrations.
-   φ be a continuous mutation vector (geometry + perception
    parameters).
-   G(d, φ) be the MimicGen generator that produces a mutated expert
    trajectory τ\_{d,φ}.

For each demonstration d, we maintain a learned mutation distribution:

p(φ \| d; η_d)

where η_d = (μ_d, Σ_d) parameterizes a diagonal Gaussian over mutation
parameters.

------------------------------------------------------------------------

## Acquisition Score (Gradient Magnitude)

For a generated expert trajectory τ\_{d,φ}, we compute:

s(d, φ) = \|\| ∇*{θ_head} L_diff(θ; τ*{d,φ}) \|\|²
Efficient Gradient Approximation
where:

- L_diff is the diffusion head training loss.
- Sample $m$ anchor timesteps from the trajectory.
- Extract local action chunks of horizon $H$.
Intuition:

-   Large gradient magnitude ⇒ trajectory would significantly change the
    current policy.
-   Small gradient magnitude ⇒ trajectory is redundant / already well
    learned.

This becomes our active learning acquisition score.

------------------------------------------------------------------------

## Efficient Gradient Approximation

To reduce compute:

1.  Sample m anchor timesteps from the trajectory.

2.  Extract local action chunks of horizon H.

3.  Approximate:

    L_FM(θ; τ) ≈ (1/m) Σ_j L_FM(θ; c\_{t_j}, a\_{t_j:t_j+H-1})

4.  Perform a single backward pass.

5.  Compute squared L2 norm over action-head gradients.

Noise seeds in diffusion are fixed during scoring to reduce variance.

------------------------------------------------------------------------

## Per-Demonstration Mutation Search (CEM-Style)

For each demonstration d, we optimize its mutation distribution to
produce high-gradient trajectories.

### Initialization (per demo)

1.  Sample M_init mutations: φ_i \~ p(φ \| d)
2.  Generate trajectories τ\_{d,φ_i}
3.  Compute scores s_i = s(d, φ_i)
4.  Select top-K elites
5.  Define demo score: S(d) = mean_elite(s_i)
6.  Update distribution toward elite mutations.

------------------------------------------------------------------------

### Distribution Update

Using exponential moving average:

μ_d ← (1 - α) μ_d + α μ_elite

Σ_d ← (1 - α) Σ_d + α Σ_elite + εI

With:

-   Diagonal covariance
-   Variance floor σ_min²
-   Small isotropic noise injection εI

This keeps exploration alive.

------------------------------------------------------------------------

## Demonstration Selection (Explore--Exploit)

At each active learning step, we choose which demonstration to expand.

P(d) = (1 - ε) \* \[S(d)\^β / Σ\_{d'} S(d')\^β\] + ε / \|D\|

-   β controls exploitation sharpness.
-   ε ensures uniform exploration.

------------------------------------------------------------------------

## Data Aggregation Step

For selected demo d:

1.  Sample mutations from updated p(φ \| d)

2.  Generate expert trajectories via MimicGen

3.  Add:

    -   Elite trajectories
    -   Small fraction of non-elite trajectories (for diversity)

4.  Fine-tune GR00T on aggregated dataset

5.  Smooth demo score:

    S(d) ← ρ S(d) + (1 - ρ) mean_elite(s_i)

------------------------------------------------------------------------

This creates an implicit curriculum:
-   Early training focuses on broadly informative mutations.
-   Later training concentrates on hard, high-gradient regions.
-   Each demonstration develops its own mutation specialization.
-   MimicGen becomes adaptively guided rather than uniformly random.



## Dataset Generation Commands

### Annotate demos (Mimic)

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --auto \
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/annotated_franka_stack_dataset.hdf5
```

### Generate a small synthetic dataset

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --num_envs 5 \
  --generation_num_trials 20 \
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/annotated_franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/generated_dataset_small.hdf5
```

### Generate a larger dataset (headless)

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --headless \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --num_envs 20 \
  --generation_num_trials 1000 \
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/annotated_franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/isaac_envs/data/generated_dataset.hdf5
```

## Status

- **Mimic demo environment and annotation:** working
- **Mimic dataset generation:** working
- **HDF5 → LeRobot conversion:** working
- **GR00T fine-tuning:** working
- **CEM failure search:** in progress
- **Active learning orchestrator:** in progress
