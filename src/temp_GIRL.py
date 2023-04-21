import torch
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


def inverse_rl(num_inputs, num_outputs, expert_data, learning_rate=1e-3, max_iterations=1000, convergence_threshold=1e-5):
    # ポリシーの初期化
    policy = Policy(num_inputs, num_outputs, 128)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # エキスパートの報酬関数の因子の重みを乱数で初期化
    expert_reward_weights = torch.randn(4, requires_grad=True)

    # 収束するか最大反復回数に到達するまで反復
    for i in range(max_iterations):
        # エキスパートの行動データから、ポリシーの実行と対応する行動を取得
        expert_actions = torch.tensor(
            [sample['action'] for sample in expert_data]).unsqueeze(1)

        # エキスパートの報酬を計算
        expert_states = torch.tensor([sample['state']
                                      for sample in expert_data])
        expert_rewards = torch.sum(
            expert_states * expert_reward_weights, dim=1)

        # ポリシーの実行と対応する行動を取得
        policy_actions = policy(expert_states)
        policy_actions = torch.tanh(policy_actions)

        # ポリシーの報酬を計算
        policy_rewards = torch.sum(
            expert_states * expert_reward_weights, dim=1)

        # 目的関数を最小化するようにポリシーを改善
        loss = torch.mean((expert_rewards - policy_rewards)
                          * torch.log(policy_actions))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # エキスパートの報酬関数の因子の重みを更新
        expert_reward_weights = expert_reward_weights.detach() - learning_rate * \
            expert_reward_weights.grad

        # 勾配をクリア
        expert_reward_weights.grad.zero_()

        # 収束判定
        if torch.abs(loss) < convergence_threshold:
            break

    return expert_reward_weights.detach().numpy()
