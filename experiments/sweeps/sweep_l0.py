"""
Main synthetic SAE experiment.

Varies L0 values (10, 20, 25, 30, 40, 50) on the SynthSAEBench-16k model.
"""

from pathlib import Path

from experiments.sweeps.common import (
    DEFAULT_D_SAE,
    DEFAULT_HIDDEN_DIM,
    create_runner_config,
    get_btk_config,
    get_jumprelu_config,
    get_matryoshka_config,
    get_mp_config,
    get_standard_config,
    train_and_eval,
)

EXPERIMENT_NAME = Path(__file__).name
L0_VALUES = [15, 20, 25, 30, 35, 40, 45]
SEEDS = [0, 1, 2, 3, 4]

# change this as desired
BASE_OUTPUT_PATH = "results/synth-sae-bench-sweep-l0"


def run() -> None:
    """Build training configurations for all L0 values and SAE types."""

    # Load the synth-sae-bench-16k model

    MODEL_PATH = str(Path(__file__).parent.parent.parent / "synth-sae-bench-16k")

    for seed in SEEDS:
        for l0 in L0_VALUES:
            # BTK (BatchTopK)
            btk_suffix = f"btk/l0-{l0}/seed-{seed}"
            btk_output_path = f"{BASE_OUTPUT_PATH}/{btk_suffix}"
            btk_sae_config = get_btk_config(
                d_in=DEFAULT_HIDDEN_DIM, d_sae=DEFAULT_D_SAE, k=l0
            )
            btk_runner_config = create_runner_config(
                synthetic_model=MODEL_PATH,
                sae_config=btk_sae_config,
                output_path=btk_output_path,
                run_name=btk_suffix.replace("/", "-"),
            )
            train_and_eval(btk_runner_config)

            # Matryoshka
            mat_suffix = f"mat/l0-{l0}/seed-{seed}"
            mat_output_path = f"{BASE_OUTPUT_PATH}/{mat_suffix}"
            mat_sae_config = get_matryoshka_config(
                d_in=DEFAULT_HIDDEN_DIM, d_sae=DEFAULT_D_SAE, k=l0
            )
            mat_runner_config = create_runner_config(
                synthetic_model=MODEL_PATH,
                sae_config=mat_sae_config,
                output_path=mat_output_path,
                run_name=mat_suffix.replace("/", "-"),
            )
            train_and_eval(mat_runner_config)

            # MP (Matching Pursuit)
            mp_suffix = f"mp/l0-{l0}/seed-{seed}"
            mp_output_path = f"{BASE_OUTPUT_PATH}/{mp_suffix}"
            mp_sae_config = get_mp_config(
                d_in=DEFAULT_HIDDEN_DIM, d_sae=DEFAULT_D_SAE, max_iterations=l0
            )
            mp_runner_config = create_runner_config(
                synthetic_model=MODEL_PATH,
                sae_config=mp_sae_config,
                output_path=mp_output_path,
                run_name=mp_suffix.replace("/", "-"),
            )
            train_and_eval(mp_runner_config)

            # Standard
            standard_suffix = f"standard/l0-{l0}/seed-{seed}"
            standard_output_path = f"{BASE_OUTPUT_PATH}/{standard_suffix}"
            standard_sae_config = get_standard_config(
                d_in=DEFAULT_HIDDEN_DIM, d_sae=DEFAULT_D_SAE, target_l0=l0
            )
            standard_runner_config = create_runner_config(
                synthetic_model=MODEL_PATH,
                sae_config=standard_sae_config,
                output_path=standard_output_path,
                run_name=standard_suffix.replace("/", "-"),
            )
            train_and_eval(standard_runner_config)

            # JumpReLU
            jumprelu_suffix = f"jumprelu/l0-{l0}/seed-{seed}"
            jumprelu_output_path = f"{BASE_OUTPUT_PATH}/{jumprelu_suffix}"
            jumprelu_sae_config = get_jumprelu_config(
                d_in=DEFAULT_HIDDEN_DIM, d_sae=DEFAULT_D_SAE, target_l0=l0
            )
            jumprelu_runner_config = create_runner_config(
                synthetic_model=MODEL_PATH,
                sae_config=jumprelu_sae_config,
                output_path=jumprelu_output_path,
                run_name=jumprelu_suffix.replace("/", "-"),
            )
            train_and_eval(jumprelu_runner_config)


if __name__ == "__main__":
    run()
