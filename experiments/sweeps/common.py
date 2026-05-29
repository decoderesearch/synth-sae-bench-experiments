"""Shared configurations for synthetic data experiments."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LoggingConfig,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    TrainingSAEConfig,
)
from sae_lens.synthetic.firing_magnitudes import (
    FoldedNormalMagnitudeConfig,
    LinearMagnitudeConfig,
)
from sae_lens.synthetic.stats import compute_superposition_stats
from sae_lens.synthetic.synthetic_model import (
    HierarchyConfig,
    LowRankCorrelationConfig,
    OrthogonalizationConfig,
    SyntheticModelConfig,
    ZipfianFiringProbabilityConfig,
)
from sae_lens.synthetic.synthetic_sae_runner import (
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
    SyntheticSAERunnerResult,
)

from saes.jumprelu_sae import XJumpReLUTrainingSAEConfig
from saes.standard_sae import XStandardTrainingSAEConfig

# Training parameters
TRAINING_SAMPLES = 200_000_000
BATCH_SIZE = 1024
LR = 3e-4


# Default model dimensions
DEFAULT_D_SAE = 4096
DEFAULT_HIDDEN_DIM = 768
DEFAULT_NUM_FEATURES = 1024 * 16  # 16384

# Default L0 (k value for BTK/Matryoshka, max_iterations for MP)
DEFAULT_L0 = 25

# Default correlation scale
DEFAULT_CORRELATION_SCALE = 0.1


def get_base_model_config(
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    correlation_scale: float = DEFAULT_CORRELATION_SCALE,
    include_hierarchy: bool = True,
) -> SyntheticModelConfig:
    """Get the base synthetic model configuration."""
    return SyntheticModelConfig(
        num_features=1024 * 16,
        hidden_dim=hidden_dim,
        firing_probability=ZipfianFiringProbabilityConfig(
            max_prob=0.4, min_prob=5e-4, exponent=0.5
        ),
        hierarchy=(
            HierarchyConfig(
                total_root_nodes=128,
                branching_factor=4,
                mutually_exclusive_portion=1.0,
                mutually_exclusive_min_depth=0,
                compensate_probabilities=True,
                scale_children_by_parent=True,
                max_depth=3,
            )
            if include_hierarchy
            else None
        ),
        mean_firing_magnitudes=LinearMagnitudeConfig(start=5.0, end=4.0),
        std_firing_magnitudes=FoldedNormalMagnitudeConfig(
            mean=0.5,
            std=0.5,
        ),
        correlation=LowRankCorrelationConfig(
            correlation_scale=correlation_scale, rank=25
        ),
        orthogonalization=OrthogonalizationConfig(num_steps=100, lr=3e-4),
        bias=0.5,
        seed=42,
    )


def get_btk_config(d_in: int, d_sae: int, k: int) -> BatchTopKTrainingSAEConfig:
    """Get BatchTopK SAE configuration."""
    return BatchTopKTrainingSAEConfig(k=k, d_in=d_in, d_sae=d_sae)


def get_matryoshka_config(
    d_in: int, d_sae: int, k: int, use_matryoshka_aux_loss: bool = True
) -> MatryoshkaBatchTopKTrainingSAEConfig:
    """Get Matryoshka BatchTopK SAE configuration.

    Inner widths are fixed at [128, 512, 2048], outer width = d_sae.
    """
    widths = [width for width in [128, 512, 2048] if width < d_sae]
    return MatryoshkaBatchTopKTrainingSAEConfig(
        k=k,
        d_in=d_in,
        d_sae=d_sae,
        matryoshka_widths=widths,
        use_matryoshka_aux_loss=use_matryoshka_aux_loss,
    )


def get_mp_config(
    d_in: int, d_sae: int, max_iterations: int
) -> MatchingPursuitTrainingSAEConfig:
    """Get Matching Pursuit SAE configuration."""
    return MatchingPursuitTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        residual_threshold=0,
        max_iterations=max_iterations,  # L0 ceiling
        decoder_init_norm=1.0,
        stop_on_duplicate_support=False,
    )


def get_standard_config(
    d_in: int, d_sae: int, target_l0: float, training_samples: int = TRAINING_SAMPLES
) -> XStandardTrainingSAEConfig:
    """Get Standard SAE configuration."""
    total_steps = training_samples // BATCH_SIZE
    return XStandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        l1_coefficient=2.0,
        l1_warm_up_steps=total_steps // 3,
        autotune_target_l0=target_l0,
        autotune_start_step=total_steps // 3,
        autotune_gain_scale=10.0,
        autotune_integral_gain=3e-4,
        autotune_convergence_gain=1e-2,
    )


def get_jumprelu_config(
    d_in: int,
    d_sae: int,
    target_l0: float,
) -> XJumpReLUTrainingSAEConfig:
    """Get JumpReLU SAE configuration."""
    return XJumpReLUTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        normalize_activations="expected_average_only_in",
        jumprelu_sparsity_loss_mode="step",
        jumprelu_bandwidth=1.0,
        jumprelu_init_threshold=1.0,
        jumprelu_tanh_scale=4.0,  # unused for step mode
        pre_act_loss_coefficient=None,
        decoder_init_norm=0.5,
        # autotuning
        l0_coefficient=1.0,
        l0_warm_up_steps=0,
        autotune_target_l0=target_l0,
        autotune_start_step=0,
        autotune_gain_scale=10.0,
        autotune_integral_gain=3e-4,
        autotune_convergence_gain=1e-2,
    )


def create_runner_config(
    synthetic_model: SyntheticModelConfig | str,
    sae_config: TrainingSAEConfig,
    output_path: str,
    run_name: str,
    training_samples: int = TRAINING_SAMPLES,
) -> SyntheticSAERunnerConfig[TrainingSAEConfig]:
    """Create a SyntheticSAERunnerConfig with standard training parameters."""
    return SyntheticSAERunnerConfig(
        synthetic_model=synthetic_model,
        sae=sae_config,
        lr=LR,
        training_samples=training_samples,
        batch_size=BATCH_SIZE,
        device="cuda",
        eval_frequency=10_000,
        eval_samples=1_000_000,
        output_path=output_path,
        # You may want to edit the following depending on your wandb settings
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_log_frequency=100,
            wandb_project="synthetic",
            run_name=run_name,
        ),
        autocast_sae=True,
        autocast_data=True,
    )


@dataclass
class SyntheticSAEStats:
    """Stats from training an SAE on synthetic data."""

    l0: float
    mcc: float
    uniqueness: float
    shrinkage: float
    dead_latents: int
    true_l0: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    explained_variance: float
    width: int
    training_samples: int
    mean_max_abs_cos_sim: float
    mean_percentile_abs_cos_sim: dict[int, float]
    mean_abs_cos_sim: float

    @classmethod
    def from_runner_result(
        cls,
        runner_result: SyntheticSAERunnerResult,
        width: int,
        training_samples: int,
    ) -> "SyntheticSAEStats":
        eval_result = runner_result.final_eval
        assert eval_result is not None

        superposition_stats = compute_superposition_stats(
            runner_result.synthetic_model.feature_dict,
            batch_size=1024,
            percentiles=[95, 99],
        )

        return cls(
            l0=eval_result.sae_l0,
            mcc=eval_result.mcc,
            uniqueness=eval_result.uniqueness,
            shrinkage=eval_result.shrinkage,
            dead_latents=eval_result.dead_latents,
            true_l0=eval_result.true_l0,
            precision=eval_result.classification.precision,
            recall=eval_result.classification.recall,
            f1_score=eval_result.classification.f1_score,
            accuracy=eval_result.classification.accuracy,
            explained_variance=eval_result.explained_variance,
            width=width,
            training_samples=training_samples,
            mean_max_abs_cos_sim=superposition_stats.mean_max_abs_cos_sim,
            mean_percentile_abs_cos_sim=superposition_stats.mean_percentile_abs_cos_sim,
            mean_abs_cos_sim=superposition_stats.mean_abs_cos_sim,
        )


def train_and_eval(cfg: SyntheticSAERunnerConfig[TrainingSAEConfig]) -> None:
    """Train and evaluate an SAE on synthetic data."""
    if cfg.output_path is None:
        raise ValueError("output_path is required")

    runner = SyntheticSAERunner(cfg)
    res = runner.run()
    stats = SyntheticSAEStats.from_runner_result(
        res, cfg.sae.d_sae, cfg.training_samples
    )

    # Save stats
    with open(Path(cfg.output_path) / "stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)

    # Save runner config
    with open(Path(cfg.output_path) / "runner_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
