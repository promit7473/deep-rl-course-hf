"""
Unit 6: A2C on PandaReachDense-v3 (and PandaPickAndPlace-v3), upload to HuggingFace Hub.
"""
import os
import gymnasium as gym
import panda_gym

from huggingface_sb3 import package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

HF_USERNAME = "promit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")


def train_and_upload(env_id, model_name, repo_id, total_timesteps=1_000_000):
    print(f"\n=== Training A2C on {env_id} ===")
    env = make_vec_env(env_id, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        device="cuda",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_name)
    env.save("vec_normalize.pkl")

    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{env_id} mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    print(f"Uploading {env_id} model ...")
    package_to_hub(
        model=model,
        model_name=model_name,
        model_architecture="A2C",
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message=f"Trained A2C on {env_id} - Unit 6",
        token=HF_TOKEN,
    )
    print(f"Done: {repo_id}")


train_and_upload(
    env_id="PandaReachDense-v3",
    model_name="a2c-PandaReachDense-v3",
    repo_id=f"{HF_USERNAME}/a2c-PandaReachDense-v3",
)

train_and_upload(
    env_id="PandaPickAndPlace-v3",
    model_name="a2c-PandaPickAndPlace-v3",
    repo_id=f"{HF_USERNAME}/a2c-PandaPickAndPlace-v3",
)

print("Done! Unit 6 complete.")
