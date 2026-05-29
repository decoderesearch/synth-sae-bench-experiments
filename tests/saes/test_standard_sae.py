from pathlib import Path

import torch
from sae_lens import SAE, StandardSAE
from sae_lens.saes.sae import TrainStepInput
from safetensors import safe_open

from saes.standard_sae import (
    XStandardTrainingSAE,
    XStandardTrainingSAEConfig,
)
from tests.helpers import build_sae_cfg


def test_XStandardTrainingSAE_without_autotuning():
    """Test that XStandardTrainingSAE works without autotuning enabled."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
    )
    sae = XStandardTrainingSAE(cfg)

    assert sae.coefficient_autotuner is None

    # Run forward pass
    test_input = torch.randn(32, 64)
    step_input = TrainStepInput(
        sae_in=test_input,
        coefficients={"l1": 1e-3},
        dead_neuron_mask=None,
        n_training_steps=0,
    )
    output = sae.training_forward_pass(step_input)

    assert output.loss is not None
    assert "l1_loss" in output.losses
    # No autotuner metrics should be logged
    assert "autotuner/multiplier" not in output.metrics


def test_XStandardTrainingSAE_with_autotuning():
    """Test that XStandardTrainingSAE initializes autotuner when target_l0 is set."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
    )
    sae = XStandardTrainingSAE(cfg)

    assert sae.coefficient_autotuner is not None
    assert sae.coefficient_autotuner.cfg.target_l0 == 50.0
    assert sae.coefficient_autotuner.multiplier == 1.0  # Starts at 1.0


def test_XStandardTrainingSAE_autotuner_updates_during_forward_pass():
    """Test that autotuner updates multiplier and logs metrics during training."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
        autotune_smoothing_factor=0.5,
        autotune_integral_gain=0.1,
    )
    sae = XStandardTrainingSAE(cfg)

    test_input = torch.randn(32, 64)
    step_input = TrainStepInput(
        sae_in=test_input,
        coefficients={"l1": 1e-3},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(step_input)

    # Check autotuner metrics are logged
    assert "autotuner/multiplier" in output.metrics
    assert "autotuner/effective_l1_coefficient" in output.metrics
    assert "autotuner/smoothed_l0" in output.metrics
    assert "autotuner/batch_l0" in output.metrics


def test_XStandardTrainingSAE_autotuner_multiplier_adjusts():
    """Test that multiplier adjusts based on observed L0."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=10.0,  # Very low target
        l1_coefficient=1e-5,  # Start with low coefficient
        autotune_smoothing_factor=0.0,  # No smoothing
        autotune_integral_gain=0.1,
    )
    sae = XStandardTrainingSAE(cfg)

    assert sae.coefficient_autotuner is not None
    initial_multiplier = sae.coefficient_autotuner.multiplier

    # Run several forward passes - actual L0 will likely be higher than 10
    test_input = torch.randn(32, 64)
    for step in range(10):
        step_input = TrainStepInput(
            sae_in=test_input,
            coefficients={"l1": 1e-5},
            dead_neuron_mask=None,
            n_training_steps=step,
        )
        sae.training_forward_pass(step_input)

    # Multiplier should have changed (trying to adjust L0)
    assert sae.coefficient_autotuner is not None
    assert sae.coefficient_autotuner.multiplier != initial_multiplier


def test_XStandardTrainingSAE_autotuner_state_persists_in_state_dict(tmp_path: Path):
    """Test that autotuner state is saved and loaded with model state dict."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
        autotune_smoothing_factor=0.9,
        autotune_integral_gain=0.05,
    )
    sae = XStandardTrainingSAE(cfg)

    # Run some forward passes to update autotuner state
    test_input = torch.randn(32, 64)
    for step in range(20):
        step_input = TrainStepInput(
            sae_in=test_input,
            coefficients={"l1": 1e-3},
            dead_neuron_mask=None,
            n_training_steps=step,
        )
        sae.training_forward_pass(step_input)

    assert sae.coefficient_autotuner is not None
    original_multiplier = sae.coefficient_autotuner.multiplier
    original_smoothed_l0 = sae.coefficient_autotuner.smoothed_l0

    # Save state dict
    state_dict = sae.state_dict()
    torch.save(state_dict, tmp_path / "sae.pt")

    # Create new SAE and load state
    new_sae = XStandardTrainingSAE(cfg)
    loaded_state = torch.load(tmp_path / "sae.pt", weights_only=True)
    new_sae.load_state_dict(loaded_state)

    assert new_sae.coefficient_autotuner is not None
    assert abs(new_sae.coefficient_autotuner.multiplier - original_multiplier) < 1e-10
    assert abs(new_sae.coefficient_autotuner.smoothed_l0 - original_smoothed_l0) < 1e-10


def test_XStandardTrainingSAE_save_inference_model_excludes_autotuner(
    tmp_path: Path,
):
    """Saved inference weights must not contain coefficient_autotuner keys, so
    they load cleanly into a vanilla SAELens StandardSAE."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
    )
    sae = XStandardTrainingSAE(cfg)
    assert sae.coefficient_autotuner is not None

    sae.save_inference_model(tmp_path)

    with safe_open(tmp_path / "sae_weights.safetensors", framework="pt") as f:
        keys = list(f.keys())
    assert not any(k.startswith("coefficient_autotuner") for k in keys), keys

    # Should load into a plain SAELens inference SAE without error.
    StandardSAE.load_from_disk(tmp_path)


def test_XStandardTrainingSAE_saved_inference_model_loads_as_standard_sae(
    tmp_path: Path,
):
    """The saved inference model loads via the base SAE class as a plain
    SAELens StandardSAE (not the X training variant) and produces matching
    output."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
    )
    sae = XStandardTrainingSAE(cfg)

    sae.save_inference_model(tmp_path)

    loaded = SAE.load_from_disk(tmp_path)

    # Loads as the standard SAELens inference class, not the X training variant.
    assert type(loaded) is StandardSAE
    assert not isinstance(loaded, XStandardTrainingSAE)
    assert loaded.cfg.architecture() == "standard"

    # Inference output should match the training SAE's encode/decode.
    test_input = torch.randn(8, 64)
    expected = sae.decode(sae.encode(test_input))
    actual = loaded.decode(loaded.encode(test_input))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_XStandardTrainingSAE_save_model_keeps_autotuner(tmp_path: Path):
    """save_model (training checkpoint) must keep autotuner keys so training
    can resume from the checkpoint."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1e-3,
    )
    sae = XStandardTrainingSAE(cfg)

    sae.save_model(tmp_path)

    with safe_open(tmp_path / "sae_weights.safetensors", framework="pt") as f:
        keys = list(f.keys())
    assert any(k.startswith("coefficient_autotuner") for k in keys), keys


def test_XStandardTrainingSAE_config_get_autotuner_config():
    """Test that get_autotuner_config returns correct config."""
    cfg = XStandardTrainingSAEConfig(
        d_in=64,
        d_sae=128,
        dtype="float32",
        device="cpu",
        autotune_target_l0=50.0,
        autotune_start_step=100,
        autotune_smoothing_factor=0.95,
        autotune_integral_gain=0.02,
        autotune_min_multiplier=0.01,
        autotune_max_multiplier=50.0,
        autotune_deadband=2.0,
        autotune_gain_scale=1.5,
    )

    autotuner_cfg = cfg.get_autotuner_config()

    assert autotuner_cfg is not None
    assert autotuner_cfg.target_l0 == 50.0
    assert autotuner_cfg.start_step == 100
    assert autotuner_cfg.smoothing_factor == 0.95
    assert autotuner_cfg.integral_gain == 0.02
    assert autotuner_cfg.min_multiplier == 0.01
    assert autotuner_cfg.max_multiplier == 50.0
    assert autotuner_cfg.deadband == 2.0
    assert autotuner_cfg.gain_scale == 1.5


def test_XStandardTrainingSAE_config_get_autotuner_config_returns_none():
    """Test that get_autotuner_config returns None when not enabled."""
    cfg = XStandardTrainingSAEConfig(
        d_in=64,
        d_sae=128,
        dtype="float32",
        device="cpu",
        autotune_target_l0=None,  # Not enabled
    )

    autotuner_cfg = cfg.get_autotuner_config()
    assert autotuner_cfg is None


def test_XStandardTrainingSAE_multiplier_applied_to_step_input_coefficient():
    """Test that the autotuner multiplier is applied to step_input.coefficients['l1']."""
    cfg = build_sae_cfg(
        sae_cfg_cls=XStandardTrainingSAEConfig,
        d_in=64,
        d_sae=128,
        autotune_target_l0=50.0,
        l1_coefficient=1.0,
        autotune_smoothing_factor=0.0,
        autotune_integral_gain=0.0,  # No adjustment - multiplier stays at 1.0
    )
    sae = XStandardTrainingSAE(cfg)

    # Run forward pass with specific step_input coefficient
    test_input = torch.randn(32, 64)
    base_coefficient = 0.5
    step_input = TrainStepInput(
        sae_in=test_input,
        coefficients={"l1": base_coefficient},
        dead_neuron_mask=None,
        n_training_steps=0,
    )

    output = sae.training_forward_pass(step_input)

    # With multiplier=1.0, effective coefficient should equal base coefficient
    assert output.metrics["autotuner/multiplier"] == 1.0
    assert output.metrics["autotuner/effective_l1_coefficient"] == base_coefficient


def test_XStandardTrainingSAE_architecture_name():
    """Test that architecture returns correct name."""
    assert XStandardTrainingSAEConfig.architecture() == "xstandard"
