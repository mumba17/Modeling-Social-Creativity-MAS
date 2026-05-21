# Modeling Social Creativity

A training-free multi-agent simulation of social creativity. Agents generate visual artifacts using quaternion-based expression trees, score them for novelty using a shared pretrained ResNet-18 feature extractor with per-agent kNN repositories, and share interesting work with peers. Because the extractor is fixed and the kNN distance computation runs as a batched GPU operation, per-step cost stays sub-linear in population size and simulations of hundreds of agents fit on a single consumer GPU.

This codebase accompanies the ICCC 2026 paper *A Training-Free Novelty Detection Framework for Large-Scale Multi-Agent Social Creativity Simulations*. The underlying model is based on Saunders and Gero's *Digital Clockwork Muse* (2001) and extends the original BSc thesis (Motz, 2025). The published paper will be added to the repository if the Camera-Ready copy is released.
## Overview

Each simulation step runs a fixed pipeline.

1. **Generate.** Every agent breeds its current expression tree with one drawn from personal memory (subtree crossover plus 5% point mutation), producing a new 32x32 RGB artifact.
2. **Evaluate.** Artifacts are rendered, passed through ResNet-18 Layer 2 (with optional PCA reduction), and scored for novelty via batched kNN against each agent's personal repository.
3. **Self-evaluate.** Novelty is normalised to [0, 1] via a rolling percentile window, then mapped to interest through the agent's Wundt curve.
4. **Share.** If interest clears the dynamic self-approval threshold τC, the agent shares the artifact with K random peers.
5. **Interact.** Recipients evaluate shared artifacts against their own memory. Pieces clearing the domain threshold τD enter the shared domain. Pieces that exceed the recipient's current interest are adopted as that recipient's new reference expression.
6. **Boredom.** Agents whose cumulative interest falls below τB sample a piece from the shared domain to restart their trajectory.
7. **Update thresholds.** τC, τD, τB are recomputed from rolling percentile windows (80th, 80th, 10th).

The first 15 steps are warmup while per-agent repositories and the rolling window populate, and the figure generator excludes them by default.

## Installation

Clone the repository.

```bash
git clone https://github.com/mumba17/Modeling-Social-Creativity-MAS.git
cd Modeling-Social-Creativity-MAS
```

Check your CUDA version.

```bash
nvidia-smi
```

`requirements.txt` defaults to CUDA 12.8 (`cu128`). For a different version, edit the `--index-url` line before installing. Remove the line entirely for CPU-only.

| CUDA Version | URL suffix |
|---|---|
| 12.8 | `cu128` |
| 12.6 | `cu126` |
| 11.8 | `cu118` |
| CPU only | Remove the `--index-url` line |

```bash
pip install -r requirements.txt
```

Verify the GPU is visible.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Python 3.10+ and a CUDA-capable GPU are recommended. The paper experiments were run on an NVIDIA RTX 3090, AMD Ryzen 5 5600X, and 32 GB RAM.

## Usage

Run with paper defaults (250 agents, 2500 steps, K=5, raw 128d features).

```bash
python main.py
```

### Examples

Quick smoke test.

```bash
python main.py --num_agents 50 --num_steps 100
```

Paper Figure 1a configuration (small panel).

```bash
python main.py --num_agents 50 --num_steps 2500 --share_count 5 --seed 42
```

Paper Figure 1b configuration (large panel).

```bash
python main.py --num_agents 250 --num_steps 2500 --share_count 25 --seed 42
```

PCA-16 features. The paper reports this matches the full 128d AUROC for kin-vs-unrelated discrimination at one eighth the dimensionality, so it is a reasonable default for larger simulations or longer runs.

```bash
python main.py --pca_dims 16 --pca_calibration_samples 5000
```

Uniform novelty preferences (collapses the population onto a single Wundt centre at p=0.5).

```bash
python main.py --uniform_novelty_pref
```

Save rendered artifacts as PNGs for visual inspection.

```bash
python main.py --save_images --image_output_dir output/images
```

Per-function timing breakdown.

```bash
python main.py --num_agents 100 --num_steps 50 --time_it
```

Multi-seed sweep.

```bash
for seed in 1 2 3 4 5; do
    python main.py --seed $seed --log_dir runs/sweep_seed_$seed
done
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_agents` | int | 250 | Number of agents in the simulation |
| `--num_steps` | int | 2500 | Number of simulation steps |
| `--share_count` | int | 5 | K, peers each agent shares with per step |
| `--mutation_rate` | float | 0.05 | Per-node mutation probability |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--pca_dims` | int | 128 | Target PCA dimensionality. Set to 16 for the compressed variant validated in the paper. Setting it equal to or above the raw backbone width (128) skips PCA and uses raw features. |
| `--pca_calibration_samples` | int | 5000 | Random artifacts used to fit PCA |
| `--uniform_novelty_pref` | flag | off | All agents get preferred_novelty = 0.5 instead of a draw from N(0.5, 0.15) clipped to [0.1, 0.9] |
| `--use_personal_threshold` | flag | off | Track and apply per-agent sharing thresholds rather than the global τC |
| `--use_static_noise` | flag | off | Replace expression rendering with random RGB noise (debug control) |
| `--save_images` | flag | off | Save rendered artifact PNGs |
| `--image_output_dir` | str | logs/<run>/images | Output directory for saved images |
| `--log_dir` | str | logs/run_<timestamp> | Override the log output directory |
| `--time_it` | flag | off | Enable per-function timing |
| `--strict_integrity_mode` / `--no_strict_integrity_mode` | flag | on | Enforce deterministic synchronisation for trace-stable runs |

## Output

Logs are written to `logs/run_YYYYMMDD_HHMMSS/` or the path passed to `--log_dir`.

### events.csv

Per-artifact event log. Event types are `generation`, `share`, `boredom_adoption`, and `step_end`.

| Column | Description |
|---|---|
| `event_type` | One of the four event types above |
| `step` | Simulation step number |
| `agent_id` | Agent that triggered the event |
| `artifact_id` | Unique artifact identifier |
| `expression` | S-expression string of the artifact's expression tree |
| `novelty` | Normalised novelty score (0 to 1) |
| `interest` | Hedonic interest score |
| `creator_id` | Original creator (lineage root), preserved across copies and adoptions |
| `evaluator_id` | `recipient_id` for `share`, `agent_id` for `generation` and `boredom_adoption` |
| `domain_size` | Shared domain size when the row was logged |
| `sender_id` / `recipient_id` | For share events |
| `accepted` | Whether the recipient accepted the artifact into the shared domain |
| `parent1_id` / `parent2_id` | Parent artifact IDs (if bred) |

`step_end` rows aggregate dynamic thresholds, accept and reject counts, and system-wide counters (`total_self_evals`, `total_other_evals`, `total_shares`, `total_domain_adoptions`) for regression checks across runs.

### agent_init.csv

Initial agent configuration. One row per agent with `agent_id`, `preferred_novelty`, `reward_mean`, `punishment_mean`.

### agent_state.csv

Periodic agent state snapshots, written every 10 steps. Cumulative interest, kNN memory size, share counts, adoption counts, average novelty and interest generated.

### TensorBoard

Real-time visualisation of per-agent and system-wide metrics.

```bash
tensorboard --logdir logs
```

## Reproducing Paper Ablation

The paper aggregates over five seeds per (N, K) condition across N in {50, 100, 250} and K in {1, 5, 25}.

```bash
for N in 50 100 250; do
    for K in 1 5 25; do
        for seed in 1 2 3 4 5; do
            python main.py --num_agents $N --share_count $K --seed $seed \
                --log_dir runs/pop${N}_share${K}_seed${seed}
        done
    done
done
```

## Architecture

| Module | Role |
|---|---|
| `main.py` | Entry point, CLI parsing, logger and scheduler setup |
| `scheduler.py` | Simulation loop, all phase orchestration |
| `framework.py` | Abstract base classes for Agent, Artifact, Scheduler, Logger |
| `genart.py` | Quaternion math, expression trees, GPU stack-machine renderer |
| `features.py` | ResNet-18 Layer 2 extractor with optional PCA projection |
| `knn.py` | Batched GPU kNN against per-agent feature repositories |
| `wundtcurve.py` | Hedonic interest mapping from novelty |
| `logger.py` | Async CSV writers, TensorBoard writer, composite fan-out |
| `timing_utils.py` | Profiling decorator and aggregation |

## Performance

If VRAM becomes an issue (not unthinkable at large scales), try reducing the `chunk_size` in `knn.py`. Also attempt to use smaller PCA such as 16, to reduce the amount of required VRAM for the kNN. Also try to reduce the repository size per agent.


## License

MIT. See [LICENSE](LICENSE).

## Citation

The paper introducing the framework was accepted at ICCC 2026; the formal citation will follow with the published proceedings. Until then, please cite the BSc thesis the simulation is built on.

```bibtex
@thesis{motz2025socialcreativity,
  author     = {Luuk Motz},
  title      = {Modeling Social Creativity with Large-Scale Multi-Agent Systems and Emergent Properties of Collective Creative Processes},
  school     = {Leiden Institute of Advanced Computer Science (LIACS), Leiden University},
  year       = {2025},
  month      = {January},
  type       = {Bachelor's Thesis},
  supervisor = {Rob Saunders, Tessa Verhoef},
  url        = {https://theses.liacs.nl/pdf/2024-2025-MotzLLuuk.pdf}
}
```

## Author

[Luuk Motz](https://luukmotz.nl) | [GitHub](https://github.com/mumba17)
```
