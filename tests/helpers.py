from dataclasses import fields
from typing import Any, TypedDict, TypeVar

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    StandardTrainingSAEConfig,
    TrainingSAEConfig,
)

from saes.jumprelu_sae import (
    XJumpReLUTrainingSAE,
    XJumpReLUTrainingSAEConfig,
)
from saes.standard_sae import (
    XStandardTrainingSAE,
    XStandardTrainingSAEConfig,
)


def to_sparse(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a dense tensor to sparse COO format."""
    return tensor.to_sparse_coo()


def to_dense(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to dense format if sparse."""
    return tensor.to_dense() if tensor.is_sparse else tensor


def random_params(model: torch.nn.Module) -> None:
    """
    Fill the parameters of a model with random values.
    """
    for param in model.parameters():
        param.data = torch.rand_like(param)
    for buffer in model.buffers():
        buffer.data = torch.rand_like(buffer)


ALL_TRAINING_SAES_AND_CONFIGS = [
    (XStandardTrainingSAE, XStandardTrainingSAEConfig),
    (XJumpReLUTrainingSAE, XJumpReLUTrainingSAEConfig),
]


@torch.no_grad()
def random_model_params(model: torch.nn.Module):
    for param in model.parameters():
        param.data = torch.randn_like(param)


class BaseSAERunnerConfigDict(TypedDict, total=False):
    model_name: str
    hook_name: str
    hook_head_index: int | None
    dataset_path: str
    dataset_trust_remote_code: bool
    is_dataset_tokenized: bool
    use_cached_activations: bool
    sae: TrainingSAEConfig
    lr: float
    train_batch_size_tokens: int
    context_size: int
    feature_sampling_window: int
    dead_feature_threshold: float
    dead_feature_window: int
    n_batches_in_buffer: int
    training_tokens: int
    store_batch_size_prompts: int
    device: str
    seed: int
    checkpoint_path: str
    dtype: str
    prepend_bos: bool
    logger: LoggingConfig
    save_final_checkpoint: bool
    output_path: str | None
    streaming: bool


T = TypeVar("T", bound=LanguageModelSAERunnerConfig[Any])
K = TypeVar("K", bound=TrainingSAEConfig)


def build_runner_cfg(
    runner_cfg_cls: type[T] = LanguageModelSAERunnerConfig,
    sae_cfg_cls: type[K] = StandardTrainingSAEConfig,
    **kwargs: Any,
) -> T:
    """
    Helper to create a mock instance of BaseSAERunnerConfig.
    """
    # Get SAE config parameters dynamically from the dataclass fields
    sae_config_params = {field.name for field in fields(sae_cfg_cls)}

    # Separate kwargs into SAE config and runner config parameters
    sae_kwargs = {k: v for k, v in kwargs.items() if k in sae_config_params}
    runner_kwargs = {k: v for k, v in kwargs.items() if k not in sae_config_params}

    # Default SAE config parameters
    sae_defaults = {
        "d_in": 64,
        "d_sae": 10,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
    }
    if "k" in sae_config_params:
        sae_defaults["k"] = 2
    if "cutoff_k" in sae_config_params:
        sae_defaults["cutoff_k"] = 2
    if "cutoff_k" in sae_config_params:
        sae_defaults["cutoff_k"] = 2
    sae_defaults.update(sae_kwargs)

    mock_config_dict: BaseSAERunnerConfigDict = {
        "model_name": "test",
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_head_index": None,
        "dataset_path": "test",
        "dataset_trust_remote_code": True,
        "is_dataset_tokenized": False,
        "use_cached_activations": False,
        "sae": sae_cfg_cls(**sae_defaults),  # type: ignore[reportGeneralTypeIssues]
        "lr": 2e-4,
        "train_batch_size_tokens": 4,
        "context_size": 6,
        "feature_sampling_window": 50,
        "dead_feature_threshold": 1e-7,
        "dead_feature_window": 1000,
        "n_batches_in_buffer": 2,
        "training_tokens": 1_000_000,
        "store_batch_size_prompts": 4,
        "logger": LoggingConfig(
            log_to_wandb=False,
            wandb_project="test_project",
            wandb_entity="test_entity",
            wandb_log_frequency=10,
        ),
        "device": "cpu",
        "seed": 24,
        "dtype": "float32",
        "prepend_bos": True,
        "output_path": None,
    }

    # Apply runner config kwargs
    for key, value in runner_kwargs.items():
        mock_config_dict[key] = value

    mock_config = runner_cfg_cls(**mock_config_dict)

    # reset checkpoint path (as we add an id to each each time)
    mock_config.checkpoint_path = kwargs.get("checkpoint_path", "test/checkpoints")

    return mock_config


def build_sae_cfg(
    runner_cfg_cls: type[T] = LanguageModelSAERunnerConfig,
    sae_cfg_cls: type[K] = StandardTrainingSAEConfig,
    **kwargs: Any,
) -> K:
    runner_cfg = build_runner_cfg(
        runner_cfg_cls=runner_cfg_cls, sae_cfg_cls=sae_cfg_cls, **kwargs
    )
    return sae_cfg_cls.from_sae_runner_config(runner_cfg)  # type: ignore


@torch.no_grad()
def random_init_model_params(model: torch.nn.Module):
    for param in model.parameters():
        param.data = torch.randn_like(param)
    for buffer in model.buffers():
        buffer.data = torch.randn_like(buffer)
