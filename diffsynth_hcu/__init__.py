"""Common HCU/DTK training helpers.

Model code may import this package. The package never imports model modules
or patches upstream functions; third-party instance adaptation is explicit.
"""
from .config import add_hcu_config
from .rng import (
    get_training_rng_context,
    sample_noise_like,
    sample_timestep_id,
)

__all__ = ["add_hcu_config", "get_training_rng_context", "sample_noise_like", "sample_timestep_id"]
