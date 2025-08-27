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

> [!WARNING]  
> You may see a warning when loading the dataset with torchaudio **2.8.0**:
>
> _"UserWarning: In 2.9, this function's implementation will be changed to use `torchaudio.load_with_torchcodec` under the hood. Some parameters like `normalize`, `format`, `buffer_size`, and `backend` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead."_
>
> This warning comes from **`torchaudio.datasets.LIBRISPEECH`** itself and cannot be disabled in the dataset wrapper.  
> The issue is tracked here: [pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902).
>
> The scripts in this repository work fine with **torchaudio 2.7.1** and **2.8.0** despite the warning, it will probably work with version **2.9.0** as well.

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

Linux/macOS:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Windows:

```bash
.\run_all_experiments.bat
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
