# DLAI_deep_DYN

DLAI 2025 project: DYN Audio Representation for Efficient Learning and Inference.

This repository implements DYN Audio Representation, a block-floating-point inspired audio representation (8-bit sample + 6-bit block gain) for neural audio models.

## Project Structure

The source code of the project is organized in the following way, in the `src` folder:

```
src
├── datasets
├── dyn_encoder
├── models
└── utils
```

- `datasets/` – dataset wrapper class for LibriSpeech
- `dyn_encoder/` – DYN block (and per sample) encoder and decoder
- `models/` – baseline and DYN model definitions
- `utils/` – utilities for training and evaluation

Training and testing scripts are located directly in the `src/` folder.

## Installation and Setup

### 1. Prerequisites

- **uv:** This project uses `uv` for fast dependency management. Install it following the instructions at [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).

### 2. Clone the repository

```bash
git clone https://github.com/Lorenzoantonelli/DLAI_deep_DYN.git
cd DLAI_deep_DYN
```

### 3. Install dependencies

Use `uv` to install the project's dependencies:

```bash
uv sync --frozen
```

### 4. Download the checkpoints

Checkpoints are available in this folder: [download](https://drive.google.com/drive/folders/18LOJ82B2k1FF8OExjHydlnBbL_xDedh4?usp=sharing). Download them into the Checkpoints/ folder.

### 5. Run the tests

Run all evaluation scripts and save results to a CSV (customize `file_name` in the script if needed):

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### 6. Train a model

Baseline models:

```
uv run src/train_baseline.py -h  # show all options
```

There are also two predefined training scripts for the baselines, that you can run with:

```
uv run src/train_baseline_16bit.py
uv run src/train_baseline_mu_law.py
```

To train a DYN model, run the following command with the model you want to train:

```
uv run src/train_dyn.py
```
