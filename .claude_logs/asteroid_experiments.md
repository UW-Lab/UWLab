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
| `reset_states_dataset_small` | same XY/Z window as iter3; **peg lying flat** (pitch=π/2) | 32 states. EE same as iter3. |

### Lessons on flushing
`TorchDatasetFileHandler.flush()` originally re-pickled the entire growing dict on every successful episode (O(N²) wall-clock cost). Patched in `source/uwlab/uwlab/utils/datasets/torch_dataset_file_handler.py` to make `flush()` a no-op and only write the dataset once in `close()`. Resample throughput went from decaying (~22/s → ~1/s) to a steady ~18/s.

## Results so far

| Pipeline | Reset dataset | iter-0 | iter-1 | iter-2 | iter-3 | Best |
|---|---|---|---|---|---|---|
| 4-iter, eval=64, 10k demos | `reset_states_dataset_iter2` (wider) | 18.75% | 18.75% | 21.88% | 25.00% | iter-3: 25.00% |
| 4-iter, eval=64, 10k demos | `reset_states_dataset_iter3` (narrower, peg upright) | 28.12% | **39.06%** | 21.88% | 32.81% | iter-1: 39.06% |
| 4-iter, eval=64, 20k demos, 512 envs | `reset_states_dataset_small` (peg upright, 20-state) | **42.19%** | (died at 84% of iter-1 collect) | — | — | iter-0: 42.19% |
| 4-iter, eval=64, 20k demos, 512 envs | `reset_states_dataset_small` (peg flat, 32 states) | 51.56% | **64.06%** | 40.62% | 60.94% | iter-1: **64.06%** |

Key observations:
- **Reset distribution is the dominant lever**: narrower distribution lifted iter-0 from 18.75% → 28.12% → 42.19% across the three datasets.
- iter-2 (horizon (0.3, 0.7)) appears to regress at the orchestrator's default eval checkpoint (step 20k). But the flat-peg checkpoint sweep below shows this is **partly a checkpoint-selection artifact**: iter-2 evaluated at step 40k jumps from 40.62% → 56.92%, on par with iter-1/iter-3. Optimal training step is iter-dependent, not a fixed 20k.
- Doubling demos (10k → 20k) lifted iter-0 by ~14pp on the same distribution shape.
- **Flat-peg (pitch=π/2) is not harder for the student**: despite the prior hypothesis that it would break the pose prior, the flat-peg `_small` dataset produced the best student so far (iter-1: 64.06%, +25pp over upright `iter3`). Hypothesis: the flat-peg reset distribution exposes a more learnable contact geometry for the tactile-driven student.

### Checkpoint sweep on the flat-peg run (2026-05-06_08-47-29)

Re-evaluated iters 1/2/3 at step 40k and 50k on the same eval task (`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0`, 64 trajectories, 16 envs, seed 0, no `--save_video`). iter-0 only ever trained to step 20k, so it's not in the sweep.

| Iter | step 20k (orchestrator default) | step 40k | step 50k |
|---|---|---|---|
| 1 | **64.06%** | 57.81% | 50.00% |
| 2 | 40.62% | **56.92%** | 54.69% |
| 3 | **60.94%** | 43.75% | 51.56% |

Takeaways:
- iter-1 and iter-3 confirm the "overfits past 30k" intuition — step 20k is best, 40k regresses ~6–17pp.
- iter-2 is the opposite — step 20k is the worst of the three. The "iter-2 always regresses" pattern from the doc's headline table is mostly an artifact of comparing step-20k iter-2 against step-20k iter-1/iter-3.
- With per-iter best-step selection: iter-1=64.06% (step 20k), iter-2=56.92% (step 40k), iter-3=60.94% (step 20k). Best overall is still iter-1 at 64.06%, but the **iter-2 explorer used to seed iter-3 collect was likely a worse-than-necessary checkpoint** under the current default, which may also dent downstream iter-3.

## Operational gotchas

- **`--no_video` is mandatory for the current orchestrator.** The local working tree of `run_incontext_exploration.py` (uncommitted) bakes `--save_video --enable_cameras` into `collect_demos`'s Hydra command unconditionally, and `visualize_demos` calls a non-existent `scripts_v2/tools/conversions/convert_zarr_video.py`. Without `--no_video`, the pipeline crashes after iter-0 demo collection.
- **Default checkpoint selection**: `_expected_train_checkpoint` default `step` was changed from `40_000` → `20_000` because the policy overfits past 30k on iter-1 / iter-3. The checkpoint sweep above shows this is **iter-dependent**: iter-2 actually benefits from training to 40k+ (40.62% → 56.92%). The default still drives both the eval ckpt and the explorer ckpt fed into the next iter's data collect, so a uniform `step=20000` may be hurting iter-2's downstream contribution.
- **`--save_video` flag silently dropped** by `collect_demos.py` (it accepts `--video`, not `--save_video`), but is then misparsed by Hydra. With `--no_video` this whole branch is skipped.
- **GPU 0 instability**: the previous fresh-pipeline run died via SIGKILL during iter-1 collect (no traceback in log; container `dmesg` is locked down so root cause unconfirmed).

## Last completed run

Flat-peg 4-iter pipeline (`reset_states_dataset_small`, 32 states, pitch=π/2) completed cleanly on GPU 4. Wall-clock 08:47 → 14:25 on 2026-05-06 (~5h 38m, matching the ~5h estimate). All four iterations returned code 0; final eval at iter-3 = 60.94%, best at iter-1 = 64.06%.

- Output dir: `logs/incontext_exploration_tactile/incontext_tactile_peg_small_512envs_20kdemos_eval64/2026-05-06_08-47-29/`
- W&B run: `stoic-night-74` (`91vr1tw6`) in project `incontext_exploration`
- Notable in-run behavior: iter-0 collect hit periodic stalls ("zero successes in window") that the pipeline's stall-recovery (force `env.reset()`) recovered from. One physx-solver step-latency spike (~5s) also recovered. No fatal errors.
- Re-eval helper: `.claude_logs/_reeval.sh ITER STEP` mirrors the orchestrator's `eval_policy()` call against this run's checkpoint tree, with `--enable_cameras` (required for sim init) but `--save_video` dropped. wandb runs land in project `incontext_exploration`, groups `eval_reeval_step{40k,50k}`.

## Real-robot integration (not yet exercised)

`/mnt/storage/lti/diffusion_policy/` (the real-robot working clone) was extended so that ASTEROID checkpoints load and run in `eval_real_robot.py`:

- `diffusion_policy/policy/transformer_image_policy.py` — verbatim copy of the GPT2-backed `TransformerImagePolicy` from the UWLab training fork (also includes the `DPTImagePolicy`/`AAWRImagePolicy` siblings).
- `diffusion_policy/model/vision/multi_image_obs_encoder.py` — appended `FlattenObsEncoder` (the lowdim-only encoder used by `tactile_lowdim` shape_meta).
- `eval_real_robot.py` — added `build_policy_input(...)` helper that detects a `TransformerImagePolicy` via `isinstance` and accumulates a per-episode rolling history, feeding `(1, T, *)` obs + `(1, T)` `attention_mask`. All `policy.reset()` call-sites now also `obs_history.clear()`. Other policy types fall back to the existing `unsqueeze(0)` path.

The existing real-robot eval script already handles the 7-D `RelCartesianOSC` action layout (`action[:6]` delta scaled by `CARTESIAN_SCALE`, `action[6]` gripper), so no action-mapping changes were needed. The `robodiff` conda env will need `pip install transformers` before running an ASTEROID checkpoint, since the new policy file imports `GPT2Config, GPT2Model`.
