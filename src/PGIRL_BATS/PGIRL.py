
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_rl import *

reward_features_num = 4

def create_batch_trajectories(env, batch_size, len_trajectories, param, variance, render=False)

    state_dim = np.prod(env.obsevation_space.shape)
    action_dim = env.action_space.shape[0]

    states = np.zeros((batch_size, len_trajectories, state_dim))
    actions = np.zeros((batch_size, len_trajectories, action_dim))
    rewards = np.zeros((batch_size, len_trajectories))
    mask = np.ones((batch_size, len_trajectories))
    reward_features = np.zeros((batch_size, len_trajectories, reward_features_num))

    for batch in range(batch_size):
        state = env.reset()

        # if render:
        #     env._render()
        
        for t in range(len_trajectories):
            action



def estimate_reward_weight(N_traj_irl, n_samples_irl, len_seqs, fail_prob, save_gradients, trained_model, var_policy, save_path):

    np.random.seed(1)
    env_irl = environments.bat_flying_env.BatFlyingEnv()
    states, actions, _, reward_features, mask = create_batch_trajectories(env_irl, batch_size=N_traj_irl,
                                                                          len_trajectories=len_seqs, param=trained_model,
                                                                          variance=var_policy)
    # エキスパートの報酬関数の因子の重みを乱数で初期化
    expert_reward_weights = torch.randn(4, requires_grad=True)
