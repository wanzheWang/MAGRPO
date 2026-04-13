"""Register the FAS Gymnasium environment."""

from gymnasium.envs.registration import register, registry

ENV_ID = "fas_env-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="fas_env.envs:FASEnv",
    )
