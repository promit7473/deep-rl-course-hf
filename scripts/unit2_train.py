"""
Unit 2: Q-Learning on FrozenLake-v1 and Taxi-v3, upload to HuggingFace Hub.
"""
import os
import random
import numpy as np
import gymnasium as gym
import imageio
import pickle
from pathlib import Path
import datetime
import json

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

HF_USERNAME = "promit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")


def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))


def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state][:])


def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(Qtable, state)
    return env.action_space.sample()


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, info = env.reset()
        terminated = False
        truncated = False
        for _ in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            Qtable[state][action] += learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            if terminated or truncated:
                break
            state = new_state
        if episode % 1000 == 0:
            print(f"  Episode {episode}/{n_training_episodes}")
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        total_rewards_ep = 0
        for _ in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated and not truncated:
        action = greedy_policy(Qtable, state)
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)


def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True)

    local_repo_path = Path(local_repo_path)
    local_repo_path.mkdir(parents=True, exist_ok=True)

    mean_reward, std_reward = evaluate_agent(
        env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )

    model_file = local_repo_path / "q-learning.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    video_path = local_repo_path / "replay.mp4"
    record_env = gym.make(model["env_id"], render_mode="rgb_array")
    record_video(record_env, model["qtable"], str(video_path), fps=video_fps)

    readme_path = local_repo_path / "README.md"
    metadata = {}
    metadata["tags"] = [model["env_id"], "q-learning", "reinforcement-learning", "custom-implementation"]
    eval_result = metadata_eval_result(
        model_pretty_name="Q-Learning",
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=model["env_id"],
        dataset_id=model["env_id"],
    )
    metadata = {**metadata, **eval_result}

    readme_content = f"# Q-Learning Agent on {model['env_id']}\n\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    metadata_save(readme_path, metadata)

    api.upload_folder(folder_path=str(local_repo_path), repo_id=repo_id, token=HF_TOKEN)
    print(f"Model pushed to: {repo_url}")


# ============================================================
# FrozenLake
# ============================================================
print("=== FrozenLake-v1 ===")
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
env_id = "FrozenLake-v1"
state_space = env.observation_space.n
action_space = env.action_space.n
Qtable_frozenlake = initialize_q_table(state_space, action_space)

n_training_episodes = 10000
learning_rate = 0.7
n_eval_episodes = 100
max_steps = 99
gamma = 0.95
eval_seed = []
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

print("Training Q-Learning on FrozenLake ...")
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"FrozenLake mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

model_frozenlake = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "qtable": Qtable_frozenlake,
}

print("Uploading FrozenLake model ...")
push_to_hub(
    repo_id=f"{HF_USERNAME}/q-FrozenLake-v1-4x4-noSlippery",
    model=model_frozenlake,
    env=env,
)

# ============================================================
# Taxi-v3
# ============================================================
print("\n=== Taxi-v3 ===")
env = gym.make("Taxi-v3", render_mode="rgb_array")
env_id = "Taxi-v3"
state_space = env.observation_space.n
action_space = env.action_space.n
Qtable_taxi = initialize_q_table(state_space, action_space)

n_training_episodes = 25000
learning_rate = 0.7
n_eval_episodes = 100
max_steps = 99
gamma = 0.95
eval_seed = [
    16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172,
    100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55, 161,
    131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59,
    95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82,
    145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168, 123, 97, 21, 83, 158, 26, 80,
    63, 5, 81, 32, 11, 28, 148,
]
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.005

print("Training Q-Learning on Taxi ...")
Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi)
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
print(f"Taxi mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

model_taxi = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "qtable": Qtable_taxi,
}

print("Uploading Taxi model ...")
push_to_hub(
    repo_id=f"{HF_USERNAME}/q-Taxi-v3",
    model=model_taxi,
    env=env,
)

print("Done! Unit 2 complete.")
