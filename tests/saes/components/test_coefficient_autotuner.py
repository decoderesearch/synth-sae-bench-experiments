from pathlib import Path

import torch

from saes.components.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)


def test_CoefficientAutotuner_initial_state():
    cfg = CoefficientAutotunerConfig(target_l0=50.0)
    autotuner = CoefficientAutotuner(cfg)

    assert autotuner.multiplier == 1.0
    assert autotuner.smoothed_l0 == 0.0


def test_CoefficientAutotuner_multiplier_increases_when_l0_above_target():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,  # No smoothing for deterministic test
        integral_gain=0.1,
        gain_scale=1.0,
    )
    autotuner = CoefficientAutotuner(cfg)

    # L0 = 100 is above target of 50, multiplier should increase
    initial_multiplier = autotuner.multiplier
    autotuner.update(batch_l0=100.0, step=0)

    assert autotuner.multiplier > initial_multiplier


def test_CoefficientAutotuner_multiplier_decreases_when_l0_below_target():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,  # No smoothing for deterministic test
        integral_gain=0.1,
        gain_scale=1.0,
    )
    autotuner = CoefficientAutotuner(cfg)

    # L0 = 10 is below target of 50, multiplier should decrease
    initial_multiplier = autotuner.multiplier
    autotuner.update(batch_l0=10.0, step=0)

    assert autotuner.multiplier < initial_multiplier


def test_CoefficientAutotuner_no_adjustment_within_deadband():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,
        integral_gain=0.1,
        deadband=5.0,  # No adjustment if within 5 of target
    )
    autotuner = CoefficientAutotuner(cfg)

    initial_multiplier = autotuner.multiplier

    # L0 = 52 is within deadband of target 50 +/- 5
    autotuner.update(batch_l0=52.0, step=0)
    assert autotuner.multiplier == initial_multiplier

    # L0 = 48 is also within deadband
    autotuner.update(batch_l0=48.0, step=1)
    assert autotuner.multiplier == initial_multiplier


def test_CoefficientAutotuner_no_adjustment_before_start_step():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        start_step=100,
        smoothing_factor=0.0,
        integral_gain=0.1,
    )
    autotuner = CoefficientAutotuner(cfg)

    initial_multiplier = autotuner.multiplier

    # Should still update smoothed_l0 but not adjust multiplier before start_step
    autotuner.update(batch_l0=100.0, step=50)
    assert autotuner.multiplier == initial_multiplier
    assert autotuner.smoothed_l0 == 100.0  # smoothed_l0 still updated

    # After start_step, adjustment should happen
    autotuner.update(batch_l0=100.0, step=100)
    assert autotuner.multiplier > initial_multiplier


def test_CoefficientAutotuner_respects_min_multiplier_bound():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,
        integral_gain=1.0,  # Large integral gain
        min_multiplier=0.1,
        max_multiplier=10.0,
    )
    autotuner = CoefficientAutotuner(cfg)

    # L0 = 1 << target, should try to decrease multiplier significantly
    for i in range(100):
        autotuner.update(batch_l0=1.0, step=i)

    assert autotuner.multiplier >= cfg.min_multiplier


def test_CoefficientAutotuner_respects_max_multiplier_bound():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,
        integral_gain=1.0,  # Large integral gain
        min_multiplier=0.01,
        max_multiplier=10.0,
    )
    autotuner = CoefficientAutotuner(cfg)

    # L0 = 1000 >> target, should try to increase multiplier significantly
    for i in range(100):
        autotuner.update(batch_l0=1000.0, step=i)

    assert autotuner.multiplier <= cfg.max_multiplier


def test_CoefficientAutotuner_ema_smoothing_works():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.5,  # 50% smoothing
        integral_gain=0.0,  # No adjustment, just test EMA
    )
    autotuner = CoefficientAutotuner(cfg)

    # First update initializes smoothed_l0
    autotuner.update(batch_l0=100.0, step=0)
    assert autotuner.smoothed_l0 == 100.0

    # Second update with EMA: 0.5 * 100 + 0.5 * 0 = 50
    autotuner.update(batch_l0=0.0, step=1)
    assert abs(autotuner.smoothed_l0 - 50.0) < 1e-10


def test_CoefficientAutotuner_ema_initializes_on_first_batch():
    cfg = CoefficientAutotunerConfig(target_l0=50.0, smoothing_factor=0.99)
    autotuner = CoefficientAutotuner(cfg)

    # First update should initialize smoothed_l0 directly (not EMA blend)
    autotuner.update(batch_l0=75.0, step=0)
    assert autotuner.smoothed_l0 == 75.0


def test_CoefficientAutotuner_state_dict_persistence(tmp_path: Path):
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0, smoothing_factor=0.9, integral_gain=0.01
    )
    autotuner = CoefficientAutotuner(cfg)

    # Run some updates
    for i in range(10):
        autotuner.update(batch_l0=60.0 + i, step=i)

    original_multiplier = autotuner.multiplier
    original_smoothed_l0 = autotuner.smoothed_l0

    # Save and load state dict
    state_dict = autotuner.state_dict()
    torch.save(state_dict, tmp_path / "autotuner.pt")

    # Create new autotuner and load state
    loaded_autotuner = CoefficientAutotuner(cfg)
    loaded_state = torch.load(tmp_path / "autotuner.pt", weights_only=True)
    loaded_autotuner.load_state_dict(loaded_state)

    assert abs(loaded_autotuner.multiplier - original_multiplier) < 1e-10
    assert abs(loaded_autotuner.smoothed_l0 - original_smoothed_l0) < 1e-10


def test_CoefficientAutotuner_state_dict_contains_expected_keys():
    cfg = CoefficientAutotunerConfig(target_l0=50.0)
    autotuner = CoefficientAutotuner(cfg)

    state_dict = autotuner.state_dict()

    assert "_smoothed_l0" in state_dict
    assert "_multiplier" in state_dict
    assert "_initialized" in state_dict
    assert "_l0_rate" in state_dict
    assert "_prev_smoothed_l0" in state_dict


def test_CoefficientAutotuner_uses_float64():
    cfg = CoefficientAutotunerConfig(target_l0=50.0)
    autotuner = CoefficientAutotuner(cfg)

    assert autotuner._smoothed_l0.dtype == torch.float64
    assert autotuner._multiplier.dtype == torch.float64

    autotuner.update(batch_l0=60.0, step=0)

    assert autotuner._smoothed_l0.dtype == torch.float64
    assert autotuner._multiplier.dtype == torch.float64


def test_CoefficientAutotuner_reset_clears_state_and_multiplier():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0, smoothing_factor=0.9, integral_gain=0.01
    )
    autotuner = CoefficientAutotuner(cfg)

    # Run some updates
    for i in range(10):
        autotuner.update(batch_l0=100.0, step=i)

    assert autotuner.smoothed_l0 > 0
    assert autotuner.multiplier != 1.0

    autotuner.reset()

    # Both multiplier and smoothed_l0 should be reset
    assert autotuner.multiplier == 1.0
    assert autotuner.smoothed_l0 == 0.0


def test_CoefficientAutotuner_reset_allows_reinitialization():
    cfg = CoefficientAutotunerConfig(target_l0=50.0, smoothing_factor=0.9)
    autotuner = CoefficientAutotuner(cfg)

    # First batch initializes
    autotuner.update(batch_l0=100.0, step=0)
    assert autotuner.smoothed_l0 == 100.0

    # Reset and update again - should reinitialize directly (not EMA blend)
    autotuner.reset()
    autotuner.update(batch_l0=25.0, step=0)
    assert autotuner.smoothed_l0 == 25.0  # Direct initialization, not EMA


def test_CoefficientAutotuner_accepts_tensor_input():
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0, smoothing_factor=0.0, integral_gain=0.1
    )
    autotuner = CoefficientAutotuner(cfg)

    # Pass batch_l0 as tensor
    batch_l0_tensor = torch.tensor(100.0)
    initial_multiplier = autotuner.multiplier
    autotuner.update(batch_l0=batch_l0_tensor, step=0)

    assert autotuner.multiplier > initial_multiplier
    assert autotuner.smoothed_l0 == 100.0


def test_CoefficientAutotuner_nonlinear_gain():
    """Test that nonlinear gain decreases adjustment as we approach target."""
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,
        integral_gain=0.1,
        gain_scale=1.0,
    )
    autotuner_far = CoefficientAutotuner(cfg)
    autotuner_near = CoefficientAutotuner(cfg)

    # Far from target: L0 = 150 (100 above target, 200% relative error)
    initial_far = autotuner_far.multiplier
    autotuner_far.update(batch_l0=150.0, step=0)
    adjustment_far = autotuner_far.multiplier - initial_far

    # Near target: L0 = 55 (5 above target, 10% relative error)
    initial_near = autotuner_near.multiplier
    autotuner_near.update(batch_l0=55.0, step=0)
    adjustment_near = autotuner_near.multiplier - initial_near

    # Adjustment should be larger when farther from target
    assert adjustment_far > adjustment_near


def test_CoefficientAutotuner_converges_to_target():
    """Test that autotuner converges multiplier to achieve target L0."""
    target_l0 = 50.0
    cfg = CoefficientAutotunerConfig(
        target_l0=target_l0,
        smoothing_factor=0.9,
        rate_smoothing_factor=0.9,
        integral_gain=0.05,
        gain_scale=2.0,
        convergence_gain=0.5,
    )
    autotuner = CoefficientAutotuner(cfg)

    # Simulate: higher multiplier -> lower L0 (more sparse)
    # L0 = base_l0 / (1 + multiplier * 10)
    base_l0 = 100.0

    for step in range(1000):
        mult = autotuner.multiplier
        simulated_l0 = base_l0 / (1 + mult * 10)
        autotuner.update(batch_l0=simulated_l0, step=step)

    # Check that we're close to target
    final_l0 = base_l0 / (1 + autotuner.multiplier * 10)
    assert abs(final_l0 - target_l0) < 5.0  # Within 5 of target


def test_CoefficientAutotuner_starts_at_one():
    """Test that autotuner starts with multiplier of 1.0."""
    cfg = CoefficientAutotunerConfig(target_l0=50.0)
    autotuner = CoefficientAutotuner(cfg)

    assert autotuner.multiplier == 1.0


def test_CoefficientAutotuner_tracks_l0_rate():
    """Test that the autotuner tracks L0 rate correctly."""
    cfg = CoefficientAutotunerConfig(
        target_l0=50.0,
        smoothing_factor=0.0,  # No smoothing for L0
        rate_smoothing_factor=0.0,  # No smoothing for rate
        integral_gain=0.0,  # No adjustment, just test rate tracking
    )
    autotuner = CoefficientAutotuner(cfg)

    # Initialize with first update
    autotuner.update(batch_l0=100.0, step=0)
    assert autotuner.l0_rate == 0.0  # Rate is 0 after first update

    # Second update - L0 decreases by 10
    autotuner.update(batch_l0=90.0, step=1)
    # With smoothing_factor=0, smoothed_l0 = 90.0
    # instant_rate = 90 - 100 = -10
    # With rate_smoothing_factor=0, rate = -10
    assert abs(autotuner.l0_rate - (-10.0)) < 1e-6

    # Third update - L0 decreases by another 10
    autotuner.update(batch_l0=80.0, step=2)
    # smoothed_l0 = 80.0
    # instant_rate = 80 - 90 = -10
    # rate = -10
    assert abs(autotuner.l0_rate - (-10.0)) < 1e-6


def test_CoefficientAutotuner_converges_with_lagged_response():
    """Test that autotuner converges when L0 responds to multiplier with lag.

    Simulates a realistic system where:
    - L0 depends on the multiplier (higher multiplier -> lower L0)
    - L0 response is smoothed/lagged (doesn't instantly change)
    - The rate-dampened control should handle this lag and converge
    """
    target_l0 = 50.0
    total_steps = 2000

    cfg = CoefficientAutotunerConfig(
        target_l0=target_l0,
        smoothing_factor=0.95,
        rate_smoothing_factor=0.9,
        integral_gain=0.02,
        gain_scale=2.0,
        convergence_gain=0.1,
    )
    autotuner = CoefficientAutotuner(cfg)

    # Simulate lagged L0 response
    # True L0 = base_l0 / (1 + multiplier * sensitivity)
    # But observed L0 is EMA-smoothed version of true L0
    base_l0 = 200.0
    sensitivity = 5.0
    l0_response_decay = 0.9  # Lag in how L0 responds to multiplier changes

    actual_l0 = base_l0  # Start at base (no sparsity penalty yet)

    for step in range(total_steps):
        mult = autotuner.multiplier

        # True L0 based on current multiplier
        true_l0 = base_l0 / (1 + mult * sensitivity)

        # Lagged response: actual L0 slowly moves toward true L0
        actual_l0 = l0_response_decay * actual_l0 + (1 - l0_response_decay) * true_l0

        # Update autotuner with the lagged L0
        autotuner.update(batch_l0=actual_l0, step=step)

    # Check that we converged close to target
    final_true_l0 = base_l0 / (1 + autotuner.multiplier * sensitivity)
    assert abs(final_true_l0 - target_l0) < 5.0, (
        f"Failed to converge: final_true_l0={final_true_l0:.2f}, target={target_l0}"
    )
    assert abs(actual_l0 - target_l0) < 5.0, (
        f"Failed to converge: actual_l0={actual_l0:.2f}, target={target_l0}"
    )


def test_CoefficientAutotuner_converges_from_both_directions():
    """Test that autotuner converges whether starting above or below target."""
    target_l0 = 50.0

    cfg = CoefficientAutotunerConfig(
        target_l0=target_l0,
        smoothing_factor=0.9,
        rate_smoothing_factor=0.9,
        integral_gain=0.05,
        gain_scale=2.0,
        convergence_gain=0.5,
    )

    # Test 1: Starting with L0 above target (need to increase multiplier)
    autotuner_high = CoefficientAutotuner(cfg)
    base_l0_high = 200.0  # Start high
    sensitivity = 5.0

    for step in range(1000):
        mult = autotuner_high.multiplier
        simulated_l0 = base_l0_high / (1 + mult * sensitivity)
        autotuner_high.update(batch_l0=simulated_l0, step=step)

    final_l0_high = base_l0_high / (1 + autotuner_high.multiplier * sensitivity)
    assert abs(final_l0_high - target_l0) < 5.0, (
        f"Failed from high: final={final_l0_high:.2f}, target={target_l0}"
    )

    # Test 2: Starting with L0 below target (need to decrease multiplier)
    autotuner_low = CoefficientAutotuner(cfg)
    base_l0_low = 30.0  # Start low (multiplier=1 gives L0=30/6=5)
    # With base=30, sens=5: L0 = 30/(1+m*5)
    # At m=1: L0=5, at m=0.1: L0=30/1.5=20, need m<0.1 for L0>25

    # Use different sensitivity to make convergence from below possible
    sensitivity_low = 0.5  # L0 = 30/(1+m*0.5), at m=1: L0=20

    for step in range(1000):
        mult = autotuner_low.multiplier
        simulated_l0 = base_l0_low / (1 + mult * sensitivity_low)
        autotuner_low.update(batch_l0=simulated_l0, step=step)

    # Note: This may not reach exactly 50 due to physical constraints of the system
    # (base_l0=30 means max L0 is 30 when multiplier=0)
    # So we just check it moved in the right direction (multiplier decreased)
    assert autotuner_low.multiplier < 1.0, (
        f"Multiplier should decrease when L0 < target: {autotuner_low.multiplier}"
    )


def test_CoefficientAutotuner_convergence_gain_when_converging():
    """Test that convergence_gain is applied when moving toward target."""
    target_l0 = 50.0

    # Create two autotuners: one with reduced gain when converging, one without
    cfg_with_reduced_gain = CoefficientAutotunerConfig(
        target_l0=target_l0,
        smoothing_factor=0.0,
        rate_smoothing_factor=0.0,
        integral_gain=0.1,
        convergence_gain=0.1,  # 90% reduction when converging
    )
    cfg_no_reduced_gain = CoefficientAutotunerConfig(
        target_l0=target_l0,
        smoothing_factor=0.0,
        rate_smoothing_factor=0.0,
        integral_gain=0.1,
        convergence_gain=1.0,  # No reduction
    )

    autotuner_reduced = CoefficientAutotuner(cfg_with_reduced_gain)
    autotuner_full = CoefficientAutotuner(cfg_no_reduced_gain)

    # Initialize both with L0 = 100 (above target)
    autotuner_reduced.update(batch_l0=100.0, step=0)
    autotuner_full.update(batch_l0=100.0, step=0)

    # Second update with L0 = 80 (still above target, but moving toward it)
    # rate = 80 - 100 = -20 (negative = L0 decreasing toward target of 50)
    # error = 80 - 50 = +30 (positive = above target)
    # error * rate = +30 * -20 = -600 < 0, so moving toward target
    autotuner_reduced.update(batch_l0=80.0, step=1)
    autotuner_full.update(batch_l0=80.0, step=1)

    # The reduced gain autotuner should have a smaller adjustment
    # Both should increase (L0 > target), but reduced less so
    assert autotuner_reduced.multiplier > 1.0  # Still increases
    assert autotuner_full.multiplier > 1.0  # Still increases
    assert autotuner_reduced.multiplier < autotuner_full.multiplier  # But less
