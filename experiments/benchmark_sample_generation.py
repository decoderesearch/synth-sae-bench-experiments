# %%
"""Benchmark sample generation speed for synthetic models with varying feature counts."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sae_lens.synthetic.firing_magnitudes import (
    FoldedNormalMagnitudeConfig,
    LinearMagnitudeConfig,
)
from sae_lens.synthetic.synthetic_model import (
    HierarchyConfig,
    LowRankCorrelationConfig,
    SyntheticModel,
    SyntheticModelConfig,
    ZipfianFiringProbabilityConfig,
)

# %%

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_BATCHES = 5
BENCHMARK_BATCHES = 100
BATCH_SIZE = 1024
HIDDEN_DIM = 768


@dataclass
class BenchmarkResult:
    num_features: int
    total_root_nodes: int
    model_creation_time: float
    samples_per_second: float
    time_per_batch_ms: float
    total_samples: int
    total_time: float


def create_model_config(
    num_features: int,
    hidden_dim: int = HIDDEN_DIM,
    branching_factor: int = 4,
    max_depth: int = 3,
) -> SyntheticModelConfig:
    """Create a synthetic model config with a 3-level hierarchy covering all features."""
    features_per_root = branching_factor**max_depth  # 4^3 = 64
    total_root_nodes = num_features // features_per_root

    if total_root_nodes * features_per_root != num_features:
        raise ValueError(
            f"num_features ({num_features}) must be divisible by "
            f"features_per_root ({features_per_root})"
        )

    return SyntheticModelConfig(
        num_features=num_features,
        hidden_dim=hidden_dim,
        firing_probability=ZipfianFiringProbabilityConfig(
            max_prob=0.4, min_prob=5e-4, exponent=0.5
        ),
        hierarchy=HierarchyConfig(
            total_root_nodes=total_root_nodes,
            branching_factor=branching_factor,
            mutually_exclusive_portion=1.0,
            mutually_exclusive_min_depth=0,
            compensate_probabilities=True,
            scale_children_by_parent=True,
            max_depth=max_depth,
        ),
        mean_firing_magnitudes=LinearMagnitudeConfig(start=5.0, end=4.0),
        std_firing_magnitudes=FoldedNormalMagnitudeConfig(
            mean=0.5,
            std=0.5,
        ),
        correlation=LowRankCorrelationConfig(correlation_scale=0.1, rank=25),
        orthogonalization=None,
        bias=0.5,
        seed=42,
    )


def benchmark_model(
    num_features: int,
    batch_size: int = BATCH_SIZE,
    warmup_batches: int = WARMUP_BATCHES,
    benchmark_batches: int = BENCHMARK_BATCHES,
    device: str = DEVICE,
) -> BenchmarkResult:
    """Benchmark sample generation speed for a model with the given number of features."""
    # Create config
    cfg = create_model_config(num_features)

    # Time model creation
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    model = SyntheticModel(cfg, device=device)
    torch.cuda.synchronize() if device == "cuda" else None
    model_creation_time = time.perf_counter() - start_time

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            _ = model.sample_with_features(batch_size)
        torch.cuda.synchronize() if device == "cuda" else None

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(benchmark_batches):
            _ = model.sample_with_features(batch_size)
        torch.cuda.synchronize() if device == "cuda" else None
    total_time = time.perf_counter() - start_time

    total_samples = batch_size * benchmark_batches
    samples_per_second = total_samples / total_time
    time_per_batch_ms = (total_time / benchmark_batches) * 1000

    return BenchmarkResult(
        num_features=num_features,
        total_root_nodes=cfg.hierarchy.total_root_nodes if cfg.hierarchy else 0,
        model_creation_time=model_creation_time,
        samples_per_second=samples_per_second,
        time_per_batch_ms=time_per_batch_ms,
        total_samples=total_samples,
        total_time=total_time,
    )


def format_num_features(n: int) -> str:
    """Format number of features for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# %%

# Feature counts: powers of 2 from 128 to ~1.3M
# Each must be divisible by 64 (4^3) for the 3-level hierarchy
# 128, 256, 512, 1024, 2048, 4096, 8192, 16383, 22768, 65536, 131072, 262144, 524288, 1048576
FEATURE_COUNTS = [128 * (2**i) for i in range(15)]  # 128 to 2,097,152

# Filter to those <= ~1.3M
FEATURE_COUNTS = [n for n in FEATURE_COUNTS if n <= 1_400_000]

print(f"Testing feature counts: {[format_num_features(n) for n in FEATURE_COUNTS]}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Warmup batches: {WARMUP_BATCHES}")
print(f"Benchmark batches: {BENCHMARK_BATCHES}")
print()

# %%

results: list[BenchmarkResult] = []

for num_features in FEATURE_COUNTS:
    print(f"Benchmarking {format_num_features(num_features)} features...", end=" ")
    try:
        result = benchmark_model(num_features)
        results.append(result)
        print(
            f"✓ {result.samples_per_second:,.0f} samples/sec "
            f"({result.time_per_batch_ms:.2f} ms/batch, "
            f"model creation: {result.model_creation_time:.2f}s)"
        )
    except Exception as e:
        print(f"✗ Error: {e}")

# %%

# Print summary table
print("\n" + "=" * 90)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 90)
print(
    f"{'Features':<12} {'Roots':<8} {'Samples/sec':<15} {'ms/batch':<12} "
    f"{'Model creation':<15}"
)
print("-" * 90)

for r in results:
    print(
        f"{format_num_features(r.num_features):<12} {r.total_root_nodes:<8} "
        f"{r.samples_per_second:>12,.0f}   {r.time_per_batch_ms:>8.2f} ms   "
        f"{r.model_creation_time:>10.2f} s"
    )

print("=" * 90)

# Save results to JSON
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

results_json = {
    "config": {
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "benchmark_batches": BENCHMARK_BATCHES,
        "hidden_dim": HIDDEN_DIM,
    },
    "results": [asdict(r) for r in results],
}

results_path = PLOTS_DIR / "benchmark_results.json"
with open(results_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {results_path}")
