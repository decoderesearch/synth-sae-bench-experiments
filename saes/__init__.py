from sae_lens.registry import register_sae_training_class

from saes.jumprelu_sae import (
    XJumpReLUTrainingSAE,
    XJumpReLUTrainingSAEConfig,
)
from saes.standard_sae import (
    XStandardTrainingSAE,
    XStandardTrainingSAEConfig,
)

# ------------------------------------------------------------
# Register all SAE classes and configs
# ------------------------------------------------------------


# xjumprelu saes
register_sae_training_class(
    XJumpReLUTrainingSAEConfig.architecture(),
    XJumpReLUTrainingSAE,
    XJumpReLUTrainingSAEConfig,
)

# xstandard saes (extended standard SAE with L1 coefficient autotuning)
register_sae_training_class(
    XStandardTrainingSAEConfig.architecture(),
    XStandardTrainingSAE,
    XStandardTrainingSAEConfig,
)
