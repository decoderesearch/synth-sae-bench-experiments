# %%

"""
Create the SynthSAEBench-16k benchmark model.
"""

from sae_lens.synthetic.firing_magnitudes import (
    FoldedNormalMagnitudeConfig,
    LinearMagnitudeConfig,
)
from sae_lens.synthetic.synthetic_model import (
    HierarchyConfig,
    LowRankCorrelationConfig,
    OrthogonalizationConfig,
    SyntheticModel,
    SyntheticModelConfig,
    ZipfianFiringProbabilityConfig,
)

# %%


cfg = SyntheticModelConfig(
    num_features=1024 * 16,
    hidden_dim=768,
    firing_probability=ZipfianFiringProbabilityConfig(
        max_prob=0.4, min_prob=5e-4, exponent=0.5
    ),
    hierarchy=HierarchyConfig(
        total_root_nodes=128,
        branching_factor=4,
        mutually_exclusive_portion=1.0,
        mutually_exclusive_min_depth=0,
        compensate_probabilities=True,
        scale_children_by_parent=True,
        max_depth=3,
    ),
    mean_firing_magnitudes=LinearMagnitudeConfig(start=5.0, end=4.0),
    std_firing_magnitudes=FoldedNormalMagnitudeConfig(
        mean=0.5,
        std=0.5,
    ),
    correlation=LowRankCorrelationConfig(correlation_scale=0.1, rank=25),
    orthogonalization=OrthogonalizationConfig(num_steps=100, lr=3e-4),
    bias=0.5,
    seed=42,
)

model = SyntheticModel(cfg, device="cuda")

# %%

model.save("synth-sae-bench-16k")
