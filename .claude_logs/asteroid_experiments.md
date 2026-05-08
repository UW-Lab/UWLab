# ASTEROID / DAgger experiments — summary

Timeline of ASTEROID in-context-exploration runs on the UR5e + Robotiq 2F-85 peg-in-hole task in `/mnt/storage/lti/UWLab/`. All runs use the same expert checkpoint:
`logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-02_01-43-39/exported/policy.pt`.

## Pipeline knobs (orchestrator)

`run_incontext_exploration.py` with `--schedule fixed`:

| Iter | Episode length (s) | Horizons (min, max) | Sampling ratio | LR |
|---|---|---|---|---|
| 0 | 8.0 | (0.2, 0.5) | (1.0,) | 1e-4 |
| 1 | 8.0 | (0.3, 0.7) | (0.25, 0.75) | 1e-5 |
| 2 | 8.0 | (0.4, 0.9) | (0.2, 0.3, 0.5) | 1e-5 |
| 3 | 8.0 | (0.5, 0.95) | (0.1, 0.2, 0.3, 0.4) | 1e-5 |

## Reset-state datasets (peg, `ObjectAnywhereEEAnywhere`)

Each dataset is sampled once via `scripts_v2/tools/record_reset_states.py` and lives at `<dataset>/Resets/Peg/resets_ObjectAnywhereEEAnywhere.pt`. The data-collection task points at one of these via the `dataset_dir` field in `data_collection_tactile_cfg.py` → `TactileEventCfg.reset_from_reset_states.params`.

| Dataset | Pose distribution | Notes |
|---|---|---|
| `reset_states_dataset_iter2` | wider | initial wide ranges |
| `reset_states_dataset_iter3` | narrower; **peg upright** (pitch=0); EE side-grasp (pitch=π/2, yaw=π) | `pose_range`: x=(0.4,0.5), y=(0.09,0.11), z=(0.01,0.03), yaw=(-π/8, π/8). 10k states. |
| `reset_states_dataset_small` | same XY/Z window as iter3; **peg lying flat** (pitch=π/2) | resampled to 10 states. EE same as iter3. |

### Lessons on flushing
`TorchDatasetFileHandler.flush()` originally re-pickled the entire growing dict on every successful episode (O(N²) wall-clock cost). Patched in `source/uwlab/uwlab/utils/datasets/torch_dataset_file_handler.py` to make `flush()` a no-op and only write the dataset once in `close()`. Resample throughput went from decaying (~22/s → ~1/s) to a steady ~18/s.

## Results so far

| Pipeline | Reset dataset | iter-0 | iter-1 | iter-2 | iter-3 | Best |
|---|---|---|---|---|---|---|
| 4-iter, eval=64, 10k demos | `reset_states_dataset_iter2` (wider) | 18.75% | 18.75% | 21.88% | 25.00% | iter-3: 25.00% |
| 4-iter, eval=64, 10k demos | `reset_states_dataset_iter3` (narrower, peg upright) | 28.12% | **39.06%** | 21.88% | 32.81% | iter-1: 39.06% |
| 4-iter, eval=64, 20k demos, 512 envs | `reset_states_dataset_small` (peg upright, 20-state) | **42.19%** | (died at 84% of iter-1 collect) | — | — | iter-0: 42.19% |

Key observations:
- **Reset distribution is the dominant lever**: narrower distribution lifted iter-0 from 18.75% → 28.12% → 42.19% across the three datasets.
- iter-2 (horizon (0.3, 0.7)) consistently regresses on this expert. Optimal student often peaks at iter-1 or iter-3.
- Doubling demos (10k → 20k) lifted iter-0 by ~14pp on the same distribution shape.

## Operational gotchas

- **`--no_video` is mandatory for the current orchestrator.** The local working tree of `run_incontext_exploration.py` (uncommitted) bakes `--save_video --enable_cameras` into `collect_demos`'s Hydra command unconditionally, and `visualize_demos` calls a non-existent `scripts_v2/tools/conversions/convert_zarr_video.py`. Without `--no_video`, the pipeline crashes after iter-0 demo collection.
- **Default checkpoint selection**: `_expected_train_checkpoint` default `step` was changed from `40_000` → `20_000` because the policy overfits past 30k. Affects both the explorer ckpt for next-iter collect and the eval ckpt.
- **`--save_video` flag silently dropped** by `collect_demos.py` (it accepts `--video`, not `--save_video`), but is then misparsed by Hydra. With `--no_video` this whole branch is skipped.
- **GPU 0 instability**: the previous fresh-pipeline run died via SIGKILL during iter-1 collect (no traceback in log; container `dmesg` is locked down so root cause unconfirmed).

## Next experiment

**Goal**: rerun the 4-iteration pipeline against the new flat-peg reset distribution (`reset_states_dataset_small`, 10 states, peg lying on its side at `pitch=π/2`). This breaks the tight pose prior the prior runs benefited from, so this is the hardest reset distribution yet.

**Pending**: ASTEROID launch is **not yet running** — the prior step (resample 10 flat-peg states) was in cleanup phase when the shell errored out. Once the shell is back:

1. Confirm `reset_states_dataset_small/Resets/Peg/resets_ObjectAnywhereEEAnywhere.pt` was rewritten today after 07:14 (the new flat-peg sample).
2. Launch from scratch on GPU 2 via `.claude_logs/_launch_small.sh` (already configured with `--no_video`, `--num_demos 20000 --num_data_envs 512 --num_eval_envs 16 --num_eval_episodes 64 --max_iterations 4 --schedule fixed`).
3. Output dir: `logs/incontext_exploration_tactile/incontext_tactile_peg_small_512envs_20kdemos_eval64/<timestamp>/`.
4. Estimated runtime: ~70 min/iteration × 4 = ~5h end-to-end.

## Real-robot integration (not yet exercised)

`/mnt/storage/lti/diffusion_policy/` (the real-robot working clone) was extended so that ASTEROID checkpoints load and run in `eval_real_robot.py`:

- `diffusion_policy/policy/transformer_image_policy.py` — verbatim copy of the GPT2-backed `TransformerImagePolicy` from the UWLab training fork (also includes the `DPTImagePolicy`/`AAWRImagePolicy` siblings).
- `diffusion_policy/model/vision/multi_image_obs_encoder.py` — appended `FlattenObsEncoder` (the lowdim-only encoder used by `tactile_lowdim` shape_meta).
- `eval_real_robot.py` — added `build_policy_input(...)` helper that detects a `TransformerImagePolicy` via `isinstance` and accumulates a per-episode rolling history, feeding `(1, T, *)` obs + `(1, T)` `attention_mask`. All `policy.reset()` call-sites now also `obs_history.clear()`. Other policy types fall back to the existing `unsqueeze(0)` path.

The existing real-robot eval script already handles the 7-D `RelCartesianOSC` action layout (`action[:6]` delta scaled by `CARTESIAN_SCALE`, `action[6]` gripper), so no action-mapping changes were needed. The `robodiff` conda env will need `pip install transformers` before running an ASTEROID checkpoint, since the new policy file imports `GPT2Config, GPT2Model`.
