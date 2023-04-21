
import argparse
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import PGIRL
from utilities import *
from vrnn.models import load_model

fs = 50
with open("params.json") as f:
    params = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--train_data", type=str, required=False)
parser.add_argument("--test_data", type=str, required=False)
parser.add_argument('--val_devide', type=int, default=10)
parser.add_argument("--fps", type=int, required=False)
args = parser.parse_args()


def load_trained_model():
    model = load_model(args.model, params, parser)
    state_dict = torch.load('{}_best.pth'.format(
        params["init_pthname0"], params["model"]), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--len_seqs', type=int, default=356,
                        help='length of the episodes')
    parser.add_argument('--gamma', type=float,
                        default=0.99, help='discount factor')
    parser.add_argument('--var_policy', type=float,
                        default=0.1, help='variance of the policy')
    parser.add_argument('--shape', type=int, nargs='+',
                        default=[], help='shape of gird')
    parser.add_argument('--n_basis', type=int, nargs='+', default=[], help='number of rbf basis for the state '
                                                                           'representation')
    parser.add_argument('--fail_prob', type=float, default=0.1,
                        help='stochasticity of the environment')
    parser.add_argument('--load_policy', action='store_true',
                        help='load a pretrained policy')
    parser.add_argument('--load_path', type=str,
                        default='data/gridworld_single', help='path to model to load')
    parser.add_argument('--save_policy', action='store_true',
                        help='save the trained policy')
    parser.add_argument('--save_path', type=str,
                        default='data/gridworld_single', help='path to save the model')
    parser.add_argument('--save_gradients', action='store_true',
                        help='save the computed gradients')
    parser.add_argument('--train_after_irl', action='store_true',
                        help='train with the computed rewards')
    parser.add_argument('--n_experiments', type=int, default=10,
                        help='number of experiments to perform')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='number of parallel jobs')
    parser.add_argument('--render_policy', action='store_true',
                        help='render the interaction with the environment')
    parser.add_argument('--plot_results', action='store_true',
                        help='plot the results')
    args = parser.parse_args()
    X_train_all = read_data.read_data(args.train_data)
    X_test_all = read_data.read_data(args.test_data)

    X_train_all, ans_train_data = modify_data.make_fix_data(X_train_all)
    X_test_all, ans_test_data = modify_data.make_fix_data(X_test_all)

    len_seqs = args.len_seqs
    if len_seqs == len(X_train_all[0]):
        raise ValueError("must match episode length")

    X_ind = np.arange(len_seqs)
    ind_train, ind_val, _, _ = train_test_split(
        X_ind, X_ind, test_size=1 / args.val_devide, random_state=42)

    subsample_factor = fs / args.fps

    trained_model = load_trained_model()

    n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    N_traj_irl = max(n_samples_irl)
    fail_prob = args.fail_prob
    save_gradients = args.save_gradients
    var_policy = args.var_policy
    all_args = N_traj_irl, n_samples_irl, len_seqs, fail_prob, save_gradients, trained_model, var_policy, args.save_path
    reward_weights = PGIRL.estimate_reward_weight(all_args)
