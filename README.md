# MAGRPO: Accelerated MARL Training for Fluid Antenna-Assisted Wireless Network Optimization

This repository contains the code for the project and paper:

Wanzhe Wang, Tong Zhang, Hao Xu, Shuai Wang, Rui Wang, and Kai-Kit Wong, "MAGRPO: Accelerated MARL Training for Fluid Antenna-Assisted Wireless Network Optimization."

The project studies distributed optimization of fluid antenna positions, beamforming, and power allocation in a downlink fluid antenna system (FAS) wireless network. The problem is formulated as a decentralized partially observable Markov decision process (Dec-POMDP), where each base station (BS) is an agent trained under centralized training and decentralized execution (CTDE).

Compared with MAPPO, the proposed MAGRPO removes critic usage in the main training stage and uses group-relative advantage estimation instead. According to the paper, this keeps testing sum-rate comparable to MAPPO while reducing training time by about 30%-40%.

## What This Repository Contains

- A paper-oriented FAS environment implemented as a local Gymnasium task: `fas_env-v0`
- A MAGRPO learner implementation with MAPPO-style warm-up and GRPO-style policy updates
- Training and evaluation entrypoints built on top of EPyMARL
- Environment configs for different BS and user-count settings used in the study
- Sacred and TensorBoard logging support for experiment tracking

This codebase is built on top of EPyMARL. The generic MARL infrastructure is retained, but the primary workflow in this repository is the MAGRPO training pipeline for the FAS problem.

## Main Components

- `src/fas_env/`
  Paper-style wireless environment and Gymnasium registration for `fas_env-v0`
- `src/learners/magrpo_continuous_learner.py`
  Two-stage MAGRPO learner
- `src/config/algs/magrpo_continuous_m4_k2.yaml`
  Public MAGRPO configuration for the main `M=4`, `K=2` setting
- `src/config/envs/`
  FAS environment presets such as `fas_env.yaml` and `fas_env_3bs_k2.yaml`
- `src/main.py`
  Main experiment entrypoint
- `src/run.py`
  Training loop, logging, checkpoint loading, and evaluation flow

## Problem Setting

The released environment follows the paper's system model:

- `N` BSs, each serving `K` single-antenna users
- Each BS is equipped with `M` fluid antennas
- Each agent jointly controls antenna positions and beamforming
- Execution is decentralized, so BSs do not communicate at test time
- Training uses a network-wide sum-rate reward under CTDE
- The spacing constraint between fluid antennas is enforced inside the environment

The default public MAGRPO config uses:

- `N = 2`
- `K = 2`
- `M = 4`
- trajectory length `T = 5`
- group size `G = 16`
- warm-up steps `grpo_warmup_steps = 1000000`
- total training steps `t_max = 8000000`

## Installation

1. Create and activate a Python environment.
2. Install a PyTorch build matching your CUDA or CPU setup.
3. Install project dependencies:

```bash
pip install -r requirements.txt
```

4. Optional: install extra upstream EPyMARL environment dependencies:

```bash
pip install -r env_requirements.txt
```

For the local FAS environment released in this repository, `requirements.txt` is the main dependency list.

## Quick Start

Train the released MAGRPO setting:

```bash
python src/main.py --config=magrpo_continuous_m4_k2 --env-config=fas_env
```

Train a 3-BS variant:

```bash
python src/main.py --config=magrpo_continuous_m4_k2 --env-config=fas_env_3bs_k2
```

Override selected environment parameters from the command line:

```bash
python src/main.py --config=magrpo_continuous_m4_k2 --env-config=fas_env with env_args.n_bs=4 env_args.m_antennas=3
```

Note on config precedence:

- default config
- environment config
- algorithm config
- command-line overrides after `with ...`

Because the algorithm config can overwrite parts of the environment config, command-line overrides are the safest way to change the final experiment setting.

If you override `env_args.k_users`, also keep user-angle configuration consistent with the number of users.

## Evaluation

Evaluate a saved checkpoint:

```bash
python src/main.py --config=magrpo_continuous_m4_k2 --env-config=fas_env with checkpoint_path="results/models/<run_name>" load_step=8000000 evaluate=True
```

`checkpoint_path` should point to a run directory that contains timestep subdirectories such as `2000000`, `4000000`, or `8000000`.

## Logging and Outputs

By default, results are written to:

- Sacred logs: `results/sacred/<experiment_name>/<env_key>`
- TensorBoard logs: `results/tb_logs/<unique_token>`
- Model checkpoints: `results/models/<unique_token>/<timestep>` when `save_model=True`
- Exported best points: `results/exports` when `export_best_point=True`

## MAGRPO Training Flow in This Codebase

The released learner implements the following training pipeline:

1. A MAPPO-style warm-up stage trains actor and critic networks for `grpo_warmup_steps`.
2. The warm-up actor is frozen as a reference policy.
3. Training switches to group-relative policy optimization without critic updates.
4. Returns are normalized inside trajectory groups, with optional KL regularization to the reference policy.

This logic is implemented in `src/learners/magrpo_continuous_learner.py`.

## Repository Scope

This public release is centered on the MAGRPO/FAS workflow described in the paper. The repository still contains inherited EPyMARL modules such as controllers, runners, critics, and utility code because they are part of the training stack used here.

## License and Acknowledgement

This repository includes code derived from EPyMARL and PyMARL. See `LICENSE` and `NOTICE` for the applicable license and attribution information.
