
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def estimate_reward_weight(N_traj_irl, n_samples_irl, len_seqs, fail_prob, save_gradients, trained_model, var_policy, save_path):

    np.random.seed(1)
    env_irl = 
    # エキスパートの報酬関数の因子の重みを乱数で初期化
    expert_reward_weights = torch.randn(4, requires_grad=True)
