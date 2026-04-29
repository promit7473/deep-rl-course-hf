"""Unit 4 PixelCopter only — skips CartPole."""
import os, sys, json, datetime, tempfile, imageio
import numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save
import pixelcopter_env

HF_USERNAME = "mhpromit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyDeep(nn.Module):
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

def reinforce(policy, optimizer, n_episodes, max_t, gamma, print_every, env):
    from collections import deque
    scores_deque = deque(maxlen=100)
    for i in range(1, n_episodes + 1):
        saved_log_probs, rewards = [], []
        state, _ = env.reset()
        for _ in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated: break
        scores_deque.append(sum(rewards))
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = torch.cat([-lp * R for lp, R in zip(saved_log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            print(f"Episode {i}\tAverage Score: {np.mean(scores_deque):.2f}")

def push_to_hub(repo_id, model, hp, eval_env):
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        torch.save(model.state_dict(), tmpdir / "model.pt")
        # eval
        rewards = []
        for _ in range(10):
            s, _ = eval_env.reset()
            total = 0
            for _ in range(10000):
                a, _ = model.act(s)
                s, r, te, tr, _ = eval_env.step(a)
                total += r
                if te or tr: break
            rewards.append(total)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        print(f"  Eval: {mean_r:.2f} +/- {std_r:.2f}")
        # video
        images = []
        s, _ = eval_env.reset()
        done = False
        while not done:
            a, _ = model.act(s)
            s, _, te, tr, _ = eval_env.step(a)
            done = te or tr
            img = eval_env.render()
            if img is not None: images.append(np.array(img))
        if images: imageio.mimsave(str(tmpdir / "replay.mp4"), images, fps=30)
        # metadata
        meta = {"tags": [hp["env_id"], "reinforce", "reinforcement-learning"]}
        em = metadata_eval_result(
            model_pretty_name="Reinforce",
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_r:.2f} +/- {std_r:.2f}",
            dataset_pretty_name=hp["env_id"],
            dataset_id=hp["env_id"],
        )
        meta = {**meta, **em}
        rp = tmpdir / "README.md"
        rp.write_text(f"# Reinforce on {hp['env_id']}\nMean reward: {mean_r:.2f}")
        metadata_save(rp, meta)
        upload_folder(folder_path=str(tmpdir), repo_id=repo_id, token=HF_TOKEN)
        print(f"  Pushed to: {repo_id}")

env_id = "Pixelcopter-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id, render_mode="rgb_array")
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

hp = {"h_size":64,"n_training_episodes":50000,"n_evaluation_episodes":10,
      "max_t":10000,"gamma":0.99,"lr":1e-4,"env_id":env_id,
      "state_space":s_size,"action_space":a_size}

policy = PolicyDeep(s_size, a_size, hp["h_size"]).to(device)
optimizer = optim.Adam(policy.parameters(), lr=hp["lr"])

print("Training Reinforce on PixelCopter (50000 episodes)...")
reinforce(policy, optimizer, hp["n_training_episodes"], hp["max_t"], hp["gamma"], 1000, env)

print("Uploading PixelCopter model...")
push_to_hub(f"{HF_USERNAME}/Reinforce-Pixelcopter-PLE-v0", policy, hp, eval_env)
print("Done!")
