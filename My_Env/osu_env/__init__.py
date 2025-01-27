from gymnasium.envs.registration import register

register(
    id="osu_env/osu-v0",
    entry_point="osu_env.envs:OSU_Env",
)
