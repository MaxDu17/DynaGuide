from gym.envs.registration import register # registeres this so it can be loaded 

# this is how we register a gym environment 
register(
    id='PlayTableSimEnv',
    entry_point='calvin_env.envs.play_table_env:PlayTableSimEnv',
    # max_episode_steps=300,
)

# this is how we register a gym environment 
register(
    id='PushT',
    entry_point='robomimic.envs.env_push_t:PushTImageEnv',
    # max_episode_steps=300,
)

# this is how we register a gym environment 
register(
    id='TouchCube',
    entry_point='robomimic.envs.env_flat_cube:TouchCubeImageEnv',
    # max_episode_steps=300,
)