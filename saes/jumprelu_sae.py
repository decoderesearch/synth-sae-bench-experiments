from dataclasses import dataclass

import torch
from sae_lens import (
    JumpReLUSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
)
from sae_lens.saes.jumprelu_sae import Step, calculate_pre_act_loss
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from typing_extensions import override

from saes.components.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)


@dataclass
class XJumpReLUTrainingSAEConfig(JumpReLUTrainingSAEConfig):
    """
    Extended JumpReLU Training SAE config that supports autotuning L0 coefficient.

    When autotune_target_l0 is not None:
    - A multiplier is autotuned during training to reach the target L0.
    - The effective coefficient is: l0_coefficient * multiplier
    """

    # autotuning
    autotune_target_l0: float | None = None
    autotune_start_step: int = 0
    autotune_smoothing_factor: float = 0.99
    autotune_rate_smoothing_factor: float = 0.95
    autotune_integral_gain: float = 3e-4
    autotune_min_multiplier: float = 1e-2
    autotune_max_multiplier: float = 100.0
    autotune_deadband: float = 0.0
    autotune_gain_scale: float = 10.0
    autotune_convergence_gain: float = 0.01

    @override
    @classmethod
    def architecture(cls) -> str:
        return "xjumprelu_training"

    @override
    def get_inference_config_class(self) -> type[JumpReLUSAEConfig]:
        return JumpReLUSAEConfig

    def get_autotuner_config(self) -> CoefficientAutotunerConfig | None:
        """Returns autotuner config if autotuning is enabled, else None."""
        if self.autotune_target_l0 is None:
            return None
        return CoefficientAutotunerConfig(
            target_l0=self.autotune_target_l0,
            start_step=self.autotune_start_step,
            smoothing_factor=self.autotune_smoothing_factor,
            rate_smoothing_factor=self.autotune_rate_smoothing_factor,
            integral_gain=self.autotune_integral_gain,
            min_multiplier=self.autotune_min_multiplier,
            max_multiplier=self.autotune_max_multiplier,
            deadband=self.autotune_deadband,
            gain_scale=self.autotune_gain_scale,
            convergence_gain=self.autotune_convergence_gain,
        )


class XJumpReLUTrainingSAE(JumpReLUTrainingSAE):
    """
    Extended JumpReLU Training SAE that supports autotuning L0 coefficient.
    """

    cfg: XJumpReLUTrainingSAEConfig  # type: ignore
    coefficient_autotuner: CoefficientAutotuner | None

    def __init__(self, cfg: XJumpReLUTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term=use_error_term)

        autotuner_cfg = cfg.get_autotuner_config()
        if autotuner_cfg is not None:
            self.coefficient_autotuner = CoefficientAutotuner(
                cfg=autotuner_cfg,
                device=cfg.device,
            )
        else:
            self.coefficient_autotuner = None

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)

        if self.coefficient_autotuner is not None:
            # Calculate batch L0 from feature activations
            batch_l0 = (output.feature_acts != 0).float().sum(dim=-1).mean()

            # Update autotuner and get new multiplier
            new_multiplier = self.coefficient_autotuner.update(
                batch_l0, step_input.n_training_steps
            )

            # Log autotuner metrics
            output.metrics["autotuner/multiplier"] = new_multiplier
            output.metrics["autotuner/effective_l0_coefficient"] = (
                step_input.coefficients["l0"] * new_multiplier
            )
            output.metrics["autotuner/smoothed_l0"] = (
                self.coefficient_autotuner.smoothed_l0
            )
            output.metrics["autotuner/batch_l0"] = batch_l0.item()

        return output

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Get base coefficient from step_input, apply multiplier if autotuning
        l0_coefficient = step_input.coefficients["l0"]
        if self.coefficient_autotuner is not None:
            l0_coefficient = l0_coefficient * self.coefficient_autotuner.multiplier

        threshold = self.threshold
        W_dec_norm = self.W_dec.norm(dim=1)

        if self.cfg.jumprelu_sparsity_loss_mode == "step":
            l0 = torch.sum(
                Step.apply(hidden_pre, threshold, self.bandwidth),  # type: ignore
                dim=-1,
            )
            l0_loss = (l0_coefficient * l0).mean()
        elif self.cfg.jumprelu_sparsity_loss_mode == "tanh":
            per_item_l0_loss = torch.tanh(
                self.cfg.jumprelu_tanh_scale * feature_acts * W_dec_norm
            ).sum(dim=-1)
            l0_loss = (l0_coefficient * per_item_l0_loss).mean()
        else:
            raise ValueError(
                f"Invalid sparsity loss mode: {self.cfg.jumprelu_sparsity_loss_mode}"
            )

        losses: dict[str, torch.Tensor] = {"l0_loss": l0_loss}

        if self.cfg.pre_act_loss_coefficient is not None:
            losses["pre_act_loss"] = calculate_pre_act_loss(
                self.cfg.pre_act_loss_coefficient,
                threshold,
                hidden_pre,
                step_input.dead_neuron_mask,
                W_dec_norm,
            )

        return losses
