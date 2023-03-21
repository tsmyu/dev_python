import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

# Load expert data
data = np.load('expert_data.npz')
expert_states = data['states']
expert_actions = data['actions']

# Define the reward function


def reward_function(states, actions, weights):
    acceleration = actions[:, 2:4]
    pulse = actions[:, -1:]
    distance = torch.norm(states[:, :2], dim=1)
    return -torch.sum(weights[:3] * (acceleration + pulse)[:, :3], dim=1) + weights[3] * distance

# Define the policy function


class PolicyFunction(nn.Module):
    def __init__(self):
        super(PolicyFunction, self).__init__()
        self.fc1 = nn.Linear(200, 6)
        self.fc2 = nn.Linear(200, 6)

    def forward(self, states):
        mean = torch.tanh(self.fc1(states))
        stddev = torch.nn.functional.softplus(self.fc2(states))
        return distributions.MultivariateNormal(mean, torch.diag_embed(stddev))


# Initialize the weights
weights = torch.tensor([1., 1., 1., 1.])

# Define the optimizer
optimizer = optim.Adam([weights], lr=0.01)

# Define the training loop
num_episodes = 1000
batch_size = 32
for episode in range(num_episodes):
    # Sample trajectories using the current policy
    states = expert_states[np.random.choice(
        len(expert_states), size=batch_size)]
    policy = PolicyFunction()
    actions = policy(states).sample()
    rewards = reward_function(states, actions, weights)

    # Compute the surrogate loss
    expert_policy = PolicyFunction()
    expert_log_probs = expert_policy(states).log_prob(actions)
    log_probs = policy(states).log_prob(actions)
    ratios = torch.exp(log_probs - expert_log_probs)
    loss = -torch.mean(ratios * rewards)

    # Compute the gradients and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get the final weights
final_weights = weights.detach().numpy()
