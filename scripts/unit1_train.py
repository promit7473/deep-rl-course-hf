"""
Unit 1: Train PPO agent on LunarLander-v3, upload to HuggingFace Hub.
"""
import os
import gymnasium as gym
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

HF_USERNAME = "mhpromit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")

env = make_vec_env("LunarLander-v3", n_envs=16)

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    device="cuda",
)

print("Training PPO on LunarLander-v3 ...")
model.learn(total_timesteps=1_000_000)
model.save("ppo-LunarLander-v3")

eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

print("Uploading to HuggingFace Hub ...")
package_to_hub(
    model=model,
    model_name="ppo-LunarLander-v3",
    model_architecture="PPO",
    env_id="LunarLander-v3",
    eval_env=eval_env,
    repo_id=f"{HF_USERNAME}/ppo-LunarLander-v3",
    commit_message="Trained PPO on LunarLander-v3 - Unit 1",
    token=HF_TOKEN,
)
print("Done! Unit 1 complete.")
