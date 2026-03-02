# Active Learning for Robust Visuomotor Policies with MimicGen + GR00T (WIP)

## Overview

This project tests whether we can systematically improve robustness of a visuomotor stacking policy by combining:

- **Isaac Lab Mimic (MimicGen)** for synthetic demonstration generation
- **GR00T** fine-tuning on the resulting dataset
- **Failure-region search** in a continuous perturbation space using Cross-Entropy Method (CEM)
- An **active learning loop** (in progress) that targets synthetic data generation toward discovered failure pockets

This project is currently a work in progress ...

**Primary task:** `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0` — Stack three cubes (bottom to top: blue, red, green) with Franka (IK-relative control). The blueprint environment aligns with the NVIDIA GR00T synthetic manipulation workflow.

## Problem Statement

MimicGen generates synthetic demonstrations by replaying expert trajectories under randomized scene configurations. Standard practice applies uniform domain randomization, varying factors such as:
- Object pose/layout (geometry/contact)
- Lighting intensity and color
- Camera viewpoint shifts

While uniform sampling provides coverage, it suffers from:
- Curse of dimensionality as perturbation space grows
- Diminishing returns from blind augmentation
- Under-sampling of structured “failure pockets”

If performance degrades in localized regions of perturbation space, uniform augmentation is inefficient. Instead, we aim to identify difficult regions in perturbation space and generate targeted synthetic data to improve robustness.


## Milestones

### Milestone 1 — Baseline MimicGen → GR00T

- Collect/obtain demonstration HDF5 for cube stacking
- Annotate demos for Mimic
- Generate synthetic dataset via MimicGen (uniform sampling)
- Convert HDF5 → LeRobot
- Fine-tune GR00T baseline
- Evaluate baseline on nominal conditions + held-out randomized eval

### Milestone 2 — Search for failure pockets in our domain setup

- Define continuous perturbation space θ (geometry + camera + lighting)
- Run failure search (CEM) over θ on the baseline policy
- Build a reproducible "failure suite" of θ values and seeds

### Milestone 3 — Gradient-Guided Active Learning

- Condition MimicGen randomization on elite θ regions found by CEM (cross entropy method)
- Regenerate targeted synthetic data
- Fine-tune GR00T on expanded dataset
- Re-evaluate on the frozen failure suite and report improvement

### Milestone 4 — Extensions

- Extend domain perturbations (camera intrinsics, cube scale, etc.)
- Extend to additional tasks/scenes (multi-task training) to prove scalability of active learning methodology
- Extend to other VLA models like pi-0

## Isaac Lab Mimic (MimicGen) Baseline

### MimicGen Data Pipeline

We treat Isaac Lab Mimic as a baseline data multiplier:

1. Start from a small set of teleop demonstrations.
2. Annotate demonstrations (phase/subtask structure).
3. Generate many synthetic rollouts by perturbing scene parameters at reset and replaying/warping motion.

This produces an HDF5 dataset containing actions, states, and camera observations. After dataset generation, we finetine the Groot model on a NEW_EMBODIMENT head to get an initial baseline policy trained on the real + synthetic data combination.

## Intermediate Step: CEM Failure-Pocket Search (Pre–Active Learning)

Before introducing gradient-guided data selection, we first verify that meaningful robustness gaps exist in the continuous perturbation space with respect to the baseline policy trained on uniformly sampled MimicGen data.

---

### Perturbation Space

We define a continuous perturbation vector:

$$
\theta \in \mathbb{R}^d
$$

Encoding reset-time environment parameters such as:

- Cube initial poses (with geometric feasibility constraints)
- Relative cube spacing
- Lighting variation
- (Optionally) Camera extrinsics jitter

---

### Optimization Objective

For a given perturbation $\theta$, we estimate:

$$
J(\theta) = \mathbb{E}_{s}[\mathbf{1}_{\text{fail}}(\pi, \theta, s)]
$$

Where:

- $s$ denotes stochastic rollout seeds
- $\mathbf{1}_{\text{fail}}$ is a binary failure indicator
- Expectation is approximated with multiple rollouts per $\theta$

The goal is to maximize failure probability.

---

### CEM Search Procedure

We maintain a sampling distribution over perturbations:

$$
q_t(\theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\theta \mid \mu_k, \Sigma_k)
$$

At each iteration:

1. Sample $N$ perturbations $\theta_i \sim q_t$
2. Reject infeasible configurations (e.g., cube overlap)
3. Evaluate failure rate $J(\theta_i)$
4. Select top $\rho\%$ elite perturbations
5. Fit the mixture model to elites via EM
6. Smoothly update mixture parameters
7. Repeat

We use diagonal covariance and variance floors to prevent premature collapse.

---

### Output: Failure Suite

The result of this stage is a reproducible **failure suite**:

- High-failure perturbations
- Associated seeds
- Estimated failure probabilities

The failure suite is frozen and reused for evaluation in later stages.

---

# Active Learning Loop (Gradient-Guided MimicGen) -- In Development

## Motivation

Uniform domain randomization treats all perturbations equally.  
However, not all synthetic trajectories contribute equally to learning.

We instead aim to generate synthetic data that maximizes expected training signal for the current policy.

This reframes active learning as a **training-signal maximization problem**, rather than a pure failure-rate maximization problem.

---

## Core Idea

For a seed demonstration $d$ and mutation parameters $\phi$,  
MimicGen produces a trajectory:

$$
\tau_{d,\phi} = G(d, \phi)
$$

We define an acquisition score:

$$
s(d, \phi) =
\left\|
\nabla_{\theta_{\text{head}}}
L_{\text{diff}}(\tau_{d,\phi})
\right\|^2
$$

Where:

- $L_{\text{diff}}$ is the diffusion head loss
- Gradients are computed with respect to trainable action-head / adapter parameters
- Diffusion noise seeds are fixed during scoring to reduce variance

**Interpretation:**

- Large gradient norm ⇒ trajectory would significantly update the model  
- Small gradient norm ⇒ trajectory is redundant or already well learned  

---

## Efficient Gradient Approximation

To reduce computational cost:

1. Sample $m$ anchor timesteps from the trajectory  
2. Extract local action chunks of horizon $H$  
3. Approximate:

$$
L_{\text{diff}}(\tau)
\approx
\frac{1}{m}
\sum_j
L_{\text{diff}}(c_{t_j}, a_{t_j:t_j+H-1})
$$

4. Perform a single backward pass  
5. Compute squared L2 norm over action-head gradients  

This provides a low-cost proxy for expected parameter update magnitude.

---

## Per-Demonstration Mutation Search

For each demonstration $d$, we maintain a mutation distribution:

$$
p(\phi \mid d; \eta_d)
$$

parameterized as a diagonal Gaussian (or mixture).

### Iterative Update

1. Sample mutations $\phi_i \sim p(\phi \mid d)$  
2. Generate trajectories $\tau_{d,\phi_i}$  
3. Compute acquisition scores $s_i = s(d, \phi_i)$  
4. Select elite mutations  
5. Update distribution parameters toward elites using CEM-style smoothing  
μ_d ← (1 - α) μ_d + α μ_elite
Σ_d ← (1 - α) Σ_d + α Σ_elite + εI

This allows each demonstration to specialize toward high-impact regions.

---

## Demonstration Selection (Explore–Exploit)

At each active learning iteration, we select which demonstration to expand.

Define a demo score:

$$
S(d) = \text{mean elite acquisition score}
$$

We sample demonstrations according to:

$$
P(d)=(1 - \varepsilon)\frac{S(d)^\beta}{\sum_{d'} S(d')^\beta}+\frac{\varepsilon}{|D|}
$$

Where:

- $\beta$ controls exploitation sharpness  
- $\varepsilon$ ensures exploration  

---

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

---

## Expected Outcome

This creates an implicit curriculum:
-   Early training focuses on broadly informative mutations.
-   Later training concentrates on high-gradient (difficult) regions.
-   Each demonstration develops its own mutation specialization.
-   MimicGen becomes adaptively guided rather than uniformly random.

----------------------------------------------------------------------

## Results

Below is an example small set of MimicGen-generated trajectories in our dataset:

<video controls width="640">
  <source src="source/docs/generated_dataset_small_obs.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video
  <a href="source/docs/generated_dataset_small_obs.mp4">here</a>.
</video>

## Evaluation Summary

Evaluations are done on the custom cube stacking environment where we can vary environment setup variables like cube positioning, lighting, camera extrinsics jitter, etc.

| Experiment name | Varied parameters        | Training demonstrations | MimicGen success rate | Eval success rate | Additional notes            |
|-----------------|--------------------------|-------------------------|-----------------------|-------------------|-----------------------------|
| Baseline        | cube positions, lighting | 1000                    | 33.6%                 | 53%               | No adversarial augmentation |

----------------------------------------------------------------------
## Dataset Generation Commands

### Annotate demos (for MimicGen)

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --auto \
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/annotated_franka_stack_dataset.hdf5
```

### Generate a small synthetic dataset

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --num_envs 5 \
  --generation_num_trials 20 \
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/annotated_franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/generated_dataset_small.hdf5
```

### Play back demonstration
```bash
python -m robomimic.scripts.playback_dataset \
    --dataset /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/generated_dataset_small.hdf5 \
    --use-obs \
    --render_image_names table_cam wrist_cam \
    --video_path PATH_TO_SAVE_VIDEO \
    --n 5
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
  --input_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/annotated_franka_stack_dataset.hdf5 \
  --output_file /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/generated_dataset.hdf5
```
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

Run N rounds and report success rate and mean reward:

```bash
python scripts/eval_stack_cube.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --num_rounds 20 --agent groot
python scripts/eval_stack_cube.py --agent groot --num_rounds 20 --headless
```

Use `--headless` for no GUI. For GR00T evaluation the policy server must be running.

### Adversarial CEM domain search (eval_stack_cube.py)

Run evaluation with CEM-driven domain randomization to search for perturbation parameters that lower success (failure-pocket search):

```bash
python scripts/eval_stack_cube.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 10 --num_rounds 50 --agent groot --run_adversarial_domain_rand_search --num_cem_clusters 3
```

Use `--headless` for no GUI. Each round prints success rate and mean reward; the domain sampler updates the search distribution between rounds. The final failure pockets are the resulting GMM distribution in perturbation space.

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
  --dataset-path /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/data/generated_dataset_small_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/scripts/franka_stack_groot_config.py \
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

### Run GR00T Policy Server

Start the GR00T inference server with your trained model:

```bash
cd /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/third_party/groot
conda activate groot

python gr00t/eval/run_gr00t_server.py \
    --model-path /home/chris/groot_training_results/checkpoint-2000 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source/scripts/franka_stack_groot_config.py \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555
```

**Note:** Replace `checkpoint-2000` with the desired checkpoint (e.g., `checkpoint-4000`, `checkpoint-6000`, `checkpoint-8000`, `checkpoint-10000`) based on your training progress.

### Run GR00T Agent in Isaac Lab

In a separate terminal, run the Isaac Lab environment with the GR00T agent:

```bash
cd /home/chris/VLA-Training-with-Synthetic-Data-Augmentation/source
python scripts/run_agent_rollout.py \
    --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 \
    --num_envs 4 \
    --agent groot \
    --enable_cameras
```

The agent will connect to the GR00T server at `localhost:5555` (default port).

Note that the above training command is suitable for training with a single 24 GB gpu (eg an RTX 4090) since we do training with Adam in 8 bits. For more information, see [README_GR00T_TRAINING.md](README_GR00T_TRAINING.md).

## Status

- **Mimic demo environment and annotation:** complete
- **Mimic dataset generation:** complete
- **HDF5 → LeRobot conversion:** complete
- **Baseline policy training:** complete 
- **CEM search infrastructure:** complete  
- **CEM Failure suite generation:** in progress  
- **Active learning orchestrator:** in progress
- **Empirical validation vs uniform baseline:** pending

