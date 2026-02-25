# Modeling Social Creativity

A multi-agent simulation of social creativity based on the DIFI (Domain, Individual, Field, Interaction) model. Agents generate visual artifacts using quaternion-based expression trees, evaluate them through novelty search and hedonic (Wundt curve) evaluation, and share interesting work with their peers.

In ```paper/``` you will find the original bachelor thesis that details the model. This codebase is based on the original paper but has been mostly rewritten and extended. Key differences that deviate from the paper are highlighted in the code and where necessary the appropriate paragraph in the paper is referenced for context. 

## Overview

Each simulation step follows a fixed pipeline (Algorithm 1):

1. **Generate** — Every agent breeds its current expression tree with one from memory (subtree crossover + mutation), producing a new visual artifact.
2. **Evaluate** — Artifacts are rendered to 32x32 images, passed through a ResNet-18 feature extractor, and scored for novelty via GPU-accelerated kNN.
3. **Self-evaluate** — Normalized novelty is mapped to interest through the agent's personal Wundt curve.
4. **Share** — Agents whose interest exceeds a dynamic threshold share with N random peers.
5. **Interact** — Recipients evaluate received artifacts against their own memory; all seen shared artifacts are added to recipient memory (with duplicate filtering), accepted artifacts enter the shared domain, and recipient expression adoption is optional via CLI.
6. **Boredom** — Agents with declining cumulative interest either explore the domain or retreat to known high-interest work.
7. **Update thresholds** — System-wide sharing, acceptance, and boredom thresholds are recalculated from rolling percentile windows.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Modeling-Social-Creativity.git
cd Modeling-Social-Creativity
```

2. Check your CUDA version:
```bash
nvidia-smi
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` defaults to **CUDA 12.8** (`cu128`). If your system uses a different version, edit the `--index-url` line before installing:

| CUDA Version | URL suffix |
|---|---|
| 12.8 | `cu128` |
| 12.6 | `cu126` |
| 11.8 | `cu118` |
| CPU only | Remove the `--index-url` line |

To verify GPU is available after installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Requirements:** Python 3.10+, PyTorch 2.7 and a CUDA-capable GPU strongly recommended.

## Usage

### Basic Run

Run the simulation with default parameters (250 agents, 2000 steps):

```bash
python main.py
```

### Usage Examples

**Quick test run** with fewer agents and steps:
```bash
python main.py --num_agents 50 --num_steps 100
```

**Paper-equivalent run** with Euclidean distance metric and classic boredom mode:
```bash
python main.py --num_agents 250 --num_steps 2000 --distance_metric euclidean --boredom_mode classic
```

**Higher mutation rate** with uniform novelty preferences:
```bash
python main.py --mutation_rate 0.15 --uniform_novelty_pref
```

**PCA-reduced features** (64d, as used in paper experiments):
```bash
python main.py --feature_dims 64 --pca_calibration_samples 500
```

**Save artifact images** for visual inspection:
```bash
python main.py --save_images --image_output_dir output/images
```

**Enable recipient adoption** of accepted shared artifacts:
```bash
python main.py --adopt-shared-expression
```

**Custom logging directory** with a fixed seed:
```bash
python main.py --log_dir runs/experiment_01 --seed 123
```

**Performance profiling** with per-function timing:
```bash
python main.py --num_agents 100 --num_steps 50 --time_it
```

### All Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_agents` | int | 250 | Number of agents in the simulation |
| `--num_steps` | int | 2000 | Number of simulation steps |
| `--share_count` | int | 5 | Number of peers each agent shares with |
| `--mutation_rate` | float | 0.05 | Per-node mutation probability |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--feature_dims` | int | 0 | Feature dimensionality (0 = raw 128d, >0 = PCA) |
| `--pca_calibration_samples` | int | 500 | Calibration samples for PCA fitting |
| `--distance_metric` | str | cosine | kNN distance metric (`cosine` or `euclidean`) |
| `--boredom_mode` | str | extended | Boredom mechanism (`classic` or `extended`) |
| `--adopt-shared-expression` | flag | off | Recipients adopt accepted shared artifacts as current expression (default keeps own expression) |
| `--uniform_novelty_pref` | flag | off | Give all agents identical novelty preference (0.5) |
| `--use_static_noise` | flag | off | Replace rendering with random noise (debug) |
| `--save_images` | flag | off | Save rendered artifact PNGs |
| `--image_output_dir` | str | None | Directory for saved images |
| `--log_dir` | str | None | Override log output directory |
| `--time_it` | flag | off | Enable per-function timing instrumentation |

## Output

Logs are saved to `logs/run_YYYYMMDD_HHMMSS/` (or the path set by `--log_dir`).

### events.csv

Per-artifact event log with columns:

| Column | Description |
|---|---|
| `event_type` | `generation`, `share`, or `boredom_adoption` |
| `step` | Simulation step number |
| `agent_id` | Agent that triggered the event |
| `artifact_id` | Unique artifact identifier |
| `expression` | S-expression string of the artifact's expression tree |
| `novelty` | Normalized novelty score (0–1) |
| `interest` | Hedonic interest score (0–1) |
| `creator_id` | Original creator (lineage root) of the artifact, preserved across copies/adoptions |
| `evaluator_id` | Evaluating agent (`recipient_id` for `share`, `agent_id` for `generation`/`boredom_adoption`) |
| `domain_size` | Size of shared domain at the time the event row is logged |
| `sender_id` / `recipient_id` | For share events |
| `accepted` | Whether the recipient accepted the artifact into the domain |
| `parent1_id` / `parent2_id` | Parent artifact IDs (if bred) |

### agent_init.csv

Initial agent configuration: `agent_id`, `preferred_novelty`, `reward_mean`, `punishment_mean`.

### agent_state.csv

Periodic agent state snapshots (every 10 steps): cumulative interest, kNN memory size, share/adoption counts, average novelty and interest generated.

### TensorBoard

Real-time metrics visualization:

```bash
tensorboard --logdir logs
```

Tracks per-agent novelty/interest, domain size, dynamic thresholds, and interaction statistics.

## Architecture

| Module | Role |
|---|---|
| `main.py` | Entry point, CLI argument parsing, logging setup |
| `scheduler.py` | Simulation loop (Algorithm 1), all phase orchestration |
| `framework.py` | Abstract base classes: `Agent`, `Artifact`, `Scheduler`, `Logger` |
| `genart.py` | Quaternion math, expression trees, GPU stack-machine renderer |
| `features.py` | ResNet-18 Layer 2 feature extraction with optional PCA |
| `knn.py` | GPU-accelerated kNN memory for novelty estimation |
| `wundtcurve.py` | Wundt curve hedonic evaluation model |
| `logger.py` | Thread-safe CSV, TensorBoard, and composite loggers |
| `timing_utils.py` | Performance profiling decorator and statistics |


## License

MIT License. See [LICENSE](LICENSE) for details.

### Happy simulating! For questions or issues, please open an issue on GitHub.

## Author

[Luuk Motz]
