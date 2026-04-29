"""
Unit 4: Reinforce (MCPG) on CartPole-v1 and Pixelcopter-PLE-v0.
Uploads trained models to HuggingFace Hub.
"""
import os
import json
import datetime
import tempfile
import imageio
import numpy as np
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

HF_USERNAME = "mhpromit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyDeep(nn.Module):
    """Three-layer policy for harder envs like PixelCopter."""
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every, env):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        for _ in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
    return scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action, _ = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)


def record_video(env, policy, out_directory, fps=30):
    images = []
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    done = False
    while not done:
        action, _ = policy.act(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)


def push_to_hub(repo_id, model, hyperparameters, eval_env, video_fps=30):
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        torch.save(model.state_dict(), tmpdir / "model.pt")

        mean_reward, std_reward = evaluate_agent(
            eval_env, hyperparameters["max_t"], 10, model
        )
        print(f"  Eval: {mean_reward:.2f} +/- {std_reward:.2f}")

        results = {
            "env_id": hyperparameters["env_id"],
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": 10,
            "eval_datetime": datetime.datetime.now().isoformat(),
        }
        with open(tmpdir / "results.json", "w") as f:
            json.dump(results, f)

        record_video(eval_env, model, str(tmpdir / "replay.mp4"), fps=video_fps)

        metadata = {
            "tags": [hyperparameters["env_id"], "reinforce", "reinforcement-learning", "custom-implementation"],
        }
        eval_meta = metadata_eval_result(
            model_pretty_name="Reinforce",
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=hyperparameters["env_id"],
            dataset_id=hyperparameters["env_id"],
        )
        metadata = {**metadata, **eval_meta}
        readme_path = tmpdir / "README.md"
        readme_path.write_text(f"# Reinforce on {hyperparameters['env_id']}\n\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        metadata_save(readme_path, metadata)

        upload_folder(folder_path=str(tmpdir), repo_id=repo_id, token=HF_TOKEN)
        print(f"  Pushed to: {repo_id}")


# ============================================================
# CartPole-v1
# ============================================================
print("=== CartPole-v1 ===")
env_id = "CartPole-v1"
env = gym.make(env_id)
eval_env = gym.make(env_id, render_mode="rgb_array")
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

cartpole_policy = Policy(s_size, a_size, cartpole_hyperparameters["h_size"]).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

print("Training Reinforce on CartPole ...")
reinforce(
    cartpole_policy, cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100, env,
)

mean_reward, std_reward = evaluate_agent(eval_env, cartpole_hyperparameters["max_t"], 10, cartpole_policy)
print(f"CartPole mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

print("Uploading CartPole model ...")
push_to_hub(
    repo_id=f"{HF_USERNAME}/Reinforce-CartPole-v1",
    model=cartpole_policy,
    hyperparameters=cartpole_hyperparameters,
    eval_env=eval_env,
    video_fps=30,
)


# ============================================================
# PixelCopter (requires gym-pygame / pygame-learning-environment)
# ============================================================
print("\n=== Pixelcopter-PLE-v0 ===")
try:
    import gym_pygame  # noqa: F401
    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)
    eval_env = gym.make(env_id, render_mode="rgb_array")
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    pixelcopter_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 50000,
        "n_evaluation_episodes": 10,
        "max_t": 10000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    pixelcopter_policy = PolicyDeep(s_size, a_size, pixelcopter_hyperparameters["h_size"]).to(device)
    pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

    print("Training Reinforce on PixelCopter ...")
    reinforce(
        pixelcopter_policy, pixelcopter_optimizer,
        pixelcopter_hyperparameters["n_training_episodes"],
        pixelcopter_hyperparameters["max_t"],
        pixelcopter_hyperparameters["gamma"],
        1000, env,
    )

    print("Uploading PixelCopter model ...")
    push_to_hub(
        repo_id=f"{HF_USERNAME}/Reinforce-Pixelcopter-PLE-v0",
        model=pixelcopter_policy,
        hyperparameters=pixelcopter_hyperparameters,
        eval_env=eval_env,
        video_fps=30,
    )
except ImportError:
    print("gym-pygame not installed. Skipping PixelCopter.")
    print("Install with: pip install gym-pygame")

print("Done! Unit 4 complete.")
