from types import SimpleNamespace


HYPERPARAMS = {
        'HPC_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':  "hpc-v0",
        'cuda':             False,
        'load_model':       False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000,
        'time_steps':       15000,
        'start_timesteps':  2100,
        'w_step_size':      0.01,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':        3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    1,
        'eval_freq':        200,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      200,
        'seed':             1
    }),
        
}
