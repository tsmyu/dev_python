import numpy as np
import torch
import torch.optim as optim


def girl(env, expert_trajs, num_iterations, learning_rate):
    # Initialize weights randomly
    w = torch.randn(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = optim.Adam([w], lr=learning_rate)

    # Compute expert feature expectations
    expert_fe = np.zeros(env.observation_space.shape[0])
    for traj in expert_trajs:
        for obs in traj:
            expert_fe += env.feature(obs)
    expert_fe /= len(expert_trajs)

    # GIRL algorithm
    for i in range(num_iterations):
        # Sample trajectories with current policy
        trajs = []
        for j in range(num_samples):
            traj = []
            obs = env.reset()
            done = False
            while not done:
                action = env.sample_action(w, obs)
                next_obs, reward, done, _ = env.step(action)
                traj.append((obs, action, reward))
                obs = next_obs
            trajs.append(traj)

        # Compute feature expectations
        fe = np.zeros(env.observation_space.shape[0])
        for traj in trajs:
            for obs in traj:
                fe += env.feature(obs[0])
        fe /= len(trajs)

        # Compute gradient of objective
        grad = expert_fe - fe
        for traj in trajs:
            for obs, action, _ in traj:
                grad += env.feature(obs) - env.feature(obs, action) @ w

        # Update policy
        optimizer.zero_grad()
        loss = torch.norm(grad)**2
        loss.backward()
        optimizer.step()

    return w
