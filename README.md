# Pixel-Based Robotic Grasping in MuJoCo

An online reinforcement-learning experiment with a simulated Franka Panda, RGB observations, PyTorch, and Stable-Baselines3. This repository documents the **executed v2 STANDARD run** in [Pixel_Based_Robotic_Grasping_MuJoCo_v2.ipynb](Pixel_Based_Robotic_Grasping_MuJoCo_v2.ipynb), including its negative results.

**Outcome:** the simulator's grasp detector recorded grasps for Pixel SAC and State SAC, but **no agent completed a successful stable lift**. The pipeline ran end to end; reliable grasp-and-lift behavior was not demonstrated.

## Recorded results

One training seed (11) was used for each agent. All three selected policies completed the same 24-episode held-out test bank.

| Agent | Policy observations | Episodes with a detected grasp | Stable lifts | Mean return |
|---|---|---:|---:|---:|
| Pixel SAC | RGB + robot proprioception | 1/24 (4.17%) | 0/24 (0%) | 0.493537 |
| Pixel PPO | RGB + robot proprioception | 0/24 (0%) | 0/24 (0%) | -1.446346 |
| State SAC | Privileged simulator state | 11/24 (45.83%) | 0/24 (0%) | 2.696873 |

All test episodes timed out. Recorded collision and drop rates were zero for all three agents; these detector results do not establish safe or reliable manipulation. Median time to the first detected grasp was 0.8 seconds for Pixel SAC and State SAC, conditional on episodes that registered a grasp.

The notebook reports **49.44 minutes** total runtime, **72 main test episodes**, and **18 shifted-condition episodes**. Preflight checks and the separate SMOKE integration run passed. All three main agents performed optimizer updates, and no budget reductions were reported.

### What counts as a grasp?

A recorded grasp requires bilateral finger contact, the object within 6 cm of the grasp center, and relative speed below 0.25 m/s, continuously for at least 0.1 seconds. Contact detection also applies a minimum contact-force threshold. The episode-level grasp flag remains true once this condition has occurred.

**A detected grasp does not require lifting the object off the table.** It can therefore record a brief table-supported hold that does not look like a completed pickup. Stable-lift success is stricter: a secure grasp with the object at least 10 cm above its settled starting height, maintained for 0.5 seconds. None of the test episodes met that criterion.

These counts come from saved notebook outputs and the implemented detector. They are not an independent frame-by-frame confirmation of the videos.

### Why the videos do not show a successful pickup

The video generator illustrates **Pixel SAC only**, not PPO or State SAC. Its category named `successful_grasp` actually selects episodes with `stable_lift == True`. Consequently, this run printed **“No qualifying episode in the illustrated seed's bank.”**

The displayed categories were `failed_grasp`, `difficult_start`, `unseen_appearance`, and `camera_shift`. They are representative selections, not a replay of every test episode; the single Pixel SAC episode with a detected grasp is not specifically selected by that event. State SAC's 11 detected grasps are not represented by this video generator. Unseen-geometry evaluation was not run in STANDARD.

## Training and validation

| Agent | Online training steps | Recorded optimizer updates | Training time | Validation evaluations | Final curriculum stage |
|---|---:|---:|---:|---:|---:|
| Pixel SAC | 15,194 | 7,341 | 16.93 min | 4 | 1 |
| Pixel PPO | 3,072 | 120 | 2.47 min | 1 | 0 |
| State SAC | 5,278 | 2,383 | 1.97 min | 1 | 0 |

These are total training-run counts; test results use the validation-selected checkpoint, which can precede the final training step. Optimizer-update counts are algorithm-specific and are not equal units of compute across SAC and PPO.

Pixel SAC advanced from curriculum stage 0 to 1 at step 8,000. Its four validation grasp rates were 8.3%, 8.3%, 16.7%, and 0%; every validation stable-lift rate was zero. Its latest printed rolling training grasp rate reached 55%, but this reflects exploratory actions on the training curriculum, not deterministic held-out performance. PPO's validation grasp rate was 0%; State SAC's was 16.7%. No agent satisfied the stable-performance early-stopping criterion.

The earlier v1 run recorded zero test grasps for all three agents, and its PPO model received zero optimizer updates. V2 provided substantially more training and recorded some grasp events. Because initial conditions, rewards, actuation parameters, evaluation seeds, and evaluation-bank sizes changed, this is **not a controlled comparison isolating the cause of improvement**.

## Generalization and failures

Pixel SAC completed six episodes in each of three shifted conditions:

| Condition | Level | Stable lifts |
|---|---:|---:|
| Position outside the nominal training range | 1 | 0/6 |
| Object/table appearance change | 3 | 0/6 |
| Camera pose shift | 5 | 0/6 |

This run does not demonstrate successful stable-lift generalization. The saved textual summary reports stable-lift rates for these conditions; it does not provide their grasp-event counts.

The most frequent classified failure was failure to approach for Pixel SAC (11/24, 45.8%) and PPO (19/24, 79.2%). State SAC's most frequent categories were tied: object pushed away and poor alignment/failed bilateral contact (8/24 each, 33.3%). Failure categories are ordered heuristics, so an episode that registered a grasp can later be classified under another failure. They should not be treated as independent causal diagnoses.

## Method

- **Simulation:** MuJoCo Panda arm, physical parallel fingers, tabletop, and procedurally randomized objects. Physics runs at 500 Hz; the policy acts at 20 Hz for at most 200 steps per episode.
- **Actions:** bounded Cartesian XYZ and yaw increments plus continuous gripper aperture. Inverse kinematics uses robot state and the commanded target, not object pose.
- **Observations:** pixel policies receive an 84×84 RGB image and 16 robot-proprioceptive features. State SAC receives a separate 35-dimensional privileged observation.
- **Learning:** online SAC and PPO with randomly initialized networks. Pixel encoders use four convolutional layers and a 256-dimensional latent representation. No demonstrations, pretrained perception, or supervised image dataset are used.
- **Rewards and curriculum:** distance progress, proximity-weighted closure progress, physical milestones, and action/collision/drop penalties. Object placement and randomization widen through the curriculum. Simulator state is used for rewards and evaluation even though it is excluded from pixel-policy inputs.
- **Protocol:** fixed, disjoint training/validation/test seed domains; validation-based checkpoint selection; frozen selections before test evaluation.

The scripted preflight diagnostic physically lifted the object approximately 14.6 cm. That diagnostic checks mechanics; it does not train the policies, and its success is **not a learned-policy result**. Likewise, the notebook's static discussion of an earlier local State SAC probe is separate from the executed Colab results reported here.

## Reproducing the experiment

The executed notebook can be viewed directly without rerunning it. To reproduce the experiment, open it in Google Colab, select a GPU runtime, keep `MODE = "STANDARD"`, and run all cells. Re-execution produces a new measurement and is not guaranteed to reproduce the exact numbers above.

The saved run used CUDA, PyTorch `2.11.0+cu128`, and MuJoCo `3.3.5`. Setup pins Gymnasium `1.2.0` and Stable-Baselines3 `2.7.0`; it retains the runtime's installed PyTorch. The GPU model is not identified in the saved textual output. Dependency installation and pinned Panda asset download require internet access.

Artifacts were written to `/content/panda_pixels_v2/standard`, including configuration banks, calibration/budget records, training logs, checkpoints, selection manifests, episode-level evaluations, figures, videos, and generated summaries. `USE_DRIVE` and replay persistence were both disabled, so these runtime files need separate export to survive disposal of the Colab session. Notebook outputs alone do not contain all checkpoint or replay data.

`RESUME = True` reuses compatible checkpoints and frozen selections when their artifact directory still exists. A new independent run requires a fresh experiment name. `SMOKE` is an integration test; `FULL` is a larger, multi-session experiment and was not executed here.

## Limitations

1. **The full task remains unsolved in this run.** All agents had zero stable lifts and 100% test timeouts. Grasp-event rates should not be presented as successful pickup rates.
2. **Grasp detection is a simulator proxy.** Brief bilateral contact near the object can qualify without lifting it. No independent visual annotation or real-world grasp validation was performed.
3. **Training was limited and unequal.** The agents received about 16.93, 2.47, and 1.97 minutes of training respectively. All met the v2 minimum exposure, but that does not imply convergence or a fair equal-compute algorithm comparison.
4. **Evidence is small and single-seed.** Each test episode changes the environment configuration, not the training seed. With 24 test episodes, one grasp changes the reported rate by 4.17 percentage points. Shifted banks contain only six episodes each.
5. **Validation was unstable.** Pixel SAC's grasp rate fell from 16.7% to 0% at the last validation, and no validation achieved a stable lift. A selected checkpoint should not be mistaken for consistently reliable behavior.
6. **The state baseline has privileged information.** Its higher grasp rate does not isolate a perception bottleneck because training exposure and optimization differ. The notebook's zero stable-lift “perception gap” simply reflects that both agents scored zero on that metric; it does not demonstrate equivalent perception or control.
7. **Generalization and ablations are incomplete.** Only three shifts were tested. Multi-seed replication, the four FULL ablations, unseen geometry, and RGB-D experiments were not run.
8. **Simulation and task simplifications limit transfer.** Primitive objects, a fixed external camera, restricted Cartesian/yaw control, dense reward shaping, idealized actuation/contact, and a fixed near-workspace robot start differ from real robotic manipulation. No hardware deployment is claimed.
9. **Reproduction is not bitwise guaranteed.** GPU execution, runtime dependencies, and resumed optimization can differ. Omitting replay persistence changes SAC's optimization history after resumption.

The defensible result is an end-to-end implemented and executed vision-based RL pipeline with limited detected grasp behavior, alongside a clear failure to learn the complete stable-lift task within this run's budget.

## Implementation and attribution

The environment integration, controller, observation checks, experiment budgeting, evaluation protocol, and reporting are custom. SAC/PPO optimization is provided by Stable-Baselines3; Panda assets come from MuJoCo Menagerie.

- [MuJoCo Menagerie Panda assets and license](https://github.com/google-deepmind/mujoco_menagerie/tree/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/franka_emika_panda)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html)

Asset and dependency licenses remain applicable; this README does not assign a project-wide license.
