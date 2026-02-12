"""
Effect of superposition (d_in/hidden_dim) experiment.

Varies hidden_dim values (256, 384, 512, 768, 1024, 1536) with fixed L0=25
"""

from pathlib import Path

from experiments.sweeps.common import (
    DEFAULT_CORRELATION_SCALE,
    DEFAULT_D_SAE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_L0,
    TRAINING_SAMPLES,
    create_runner_config,
    get_base_model_config,
    get_btk_config,
    get_jumprelu_config,
    get_matryoshka_config,
    get_mp_config,
    get_standard_config,
    train_and_eval,
)

EXPERIMENT_NAME = Path(__file__).name
HIDDEN_DIM_VALUES = [256, 384, 512, 768, 1024, 1536]
SEEDS = [0]

# change this as desired
BASE_OUTPUT_PATH = "results/synth-sae-bench-sweep-superposition"


def run() -> None:
    for seed in SEEDS:
        for hidden_dim in HIDDEN_DIM_VALUES:
            # Scale training samples using the power law from OpenAI's "Scaling and evaluating
            # sparse autoencoders" paper, which finds tokens to convergence scales as ~n^0.6
            training_samples = int(
                TRAINING_SAMPLES * (hidden_dim / DEFAULT_HIDDEN_DIM) ** 0.6
            )
            # Get model config with varied hidden_dim
            model_config = get_base_model_config(
                hidden_dim=hidden_dim,
                correlation_scale=DEFAULT_CORRELATION_SCALE,
            )

            # # BTK (BatchTopK)
            btk_suffix = f"btk/d-{hidden_dim}/seed-{seed}"
            btk_output_path = f"{BASE_OUTPUT_PATH}/{btk_suffix}"
            btk_sae_config = get_btk_config(
                d_in=hidden_dim, d_sae=DEFAULT_D_SAE, k=DEFAULT_L0
            )
            btk_runner_config = create_runner_config(
                synthetic_model=model_config,
                sae_config=btk_sae_config,
                output_path=btk_output_path,
                run_name=btk_suffix.replace("/", "-"),
                training_samples=training_samples,
            )
            train_and_eval(btk_runner_config)

            # # Matryoshka
            mat_suffix = f"mat/d-{hidden_dim}/seed-{seed}"
            mat_output_path = f"{BASE_OUTPUT_PATH}/{mat_suffix}"
            mat_sae_config = get_matryoshka_config(
                d_in=hidden_dim, d_sae=DEFAULT_D_SAE, k=DEFAULT_L0
            )
            mat_runner_config = create_runner_config(
                synthetic_model=model_config,
                sae_config=mat_sae_config,
                output_path=mat_output_path,
                run_name=mat_suffix.replace("/", "-"),
                training_samples=training_samples,
            )
            train_and_eval(mat_runner_config)

            # # MP (Matching Pursuit)
            mp_suffix = f"mp/d-{hidden_dim}/seed-{seed}"
            mp_output_path = f"{BASE_OUTPUT_PATH}/{mp_suffix}"
            mp_sae_config = get_mp_config(
                d_in=hidden_dim, d_sae=DEFAULT_D_SAE, max_iterations=DEFAULT_L0
            )
            mp_runner_config = create_runner_config(
                synthetic_model=model_config,
                sae_config=mp_sae_config,
                output_path=mp_output_path,
                run_name=mp_suffix.replace("/", "-"),
                training_samples=training_samples,
            )
            train_and_eval(mp_runner_config)

            # Standard
            standard_suffix = f"standard/d-{hidden_dim}/seed-{seed}"
            standard_output_path = f"{BASE_OUTPUT_PATH}/{standard_suffix}"
            standard_sae_config = get_standard_config(
                d_in=hidden_dim,
                d_sae=DEFAULT_D_SAE,
                target_l0=DEFAULT_L0,
                training_samples=training_samples,
            )
            standard_runner_config = create_runner_config(
                synthetic_model=model_config,
                sae_config=standard_sae_config,
                output_path=standard_output_path,
                run_name=standard_suffix.replace("/", "-"),
                training_samples=training_samples,
            )
            train_and_eval(standard_runner_config)

            # JumpReLU
            jumprelu_suffix = f"jumprelu/d-{hidden_dim}/seed-{seed}"

            jumprelu_output_path = f"{BASE_OUTPUT_PATH}/{jumprelu_suffix}"
            jumprelu_sae_config = get_jumprelu_config(
                d_in=hidden_dim,
                d_sae=DEFAULT_D_SAE,
                target_l0=DEFAULT_L0,
            )
            jumprelu_runner_config = create_runner_config(
                synthetic_model=model_config,
                sae_config=jumprelu_sae_config,
                output_path=jumprelu_output_path,
                run_name=jumprelu_suffix.replace("/", "-"),
                training_samples=training_samples,
            )
            train_and_eval(jumprelu_runner_config)


if __name__ == "__main__":
    run()
