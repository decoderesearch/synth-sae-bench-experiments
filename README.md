# SynthSAEBench

This repo contains code for the paper "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data".

## Structure

Experiments for the paper are in the `experiments` directory. The extended SAE classes used in the paper (`XStandardTrainingSAE` and `XJumpReLUTrainingSAE` containing L0 autotuning) are in the `saes` directory.

## Setup

This project uses uv for package management. To install the dependencies, run:

```bash
uv sync
```

## Running Experiments

To run the experiments, use the `uv run` command. For example, to run the superposition experiment, run:

```bash
uv run experiments/sweeps/sweep_superposition.py
```

## Loading the benchmark model

The main benchmark model is on Huggingface at [decoderesearch/synth-sae-bench-16k-v1](https://huggingface.co/decoderesearch/synth-sae-bench-16k-v1). To load the model, run:

```bash
from sae_lens.synthetic.synthetic_model import SyntheticModel

model = SyntheticModel.from_pretrained("decoderesearch/synth-sae-bench-16k-v1")
```

## Development

### Linting and Formatting

This project uses ruff for linting and formatting. To run the linting and formatting, run:

```bash
uv run ruff check .
uv run ruff format .
```

### Testing

This project uses pytest for testing. To run the tests, run:

```bash
uv run pytest
```

### Type Checking

This project uses pyright for type checking. To run the type checking, run:

```bash
uv run pyright
```
