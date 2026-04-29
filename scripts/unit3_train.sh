#!/bin/bash
# Unit 3: DQN on SpaceInvadersNoFrameskip-v4 using RL Baselines3 Zoo

set -e

HF_USERNAME="mhpromit7473"

echo "=== Training DQN on SpaceInvaders ==="
python3 -m rl_zoo3.train \
    --algo dqn \
    --env SpaceInvadersNoFrameskip-v4 \
    -f logs/ \
    --hyperparams \
        buffer_size:100000 \
        learning_rate:0.0001 \
        batch_size:32 \
        learning_starts:100000 \
        target_update_interval:1000 \
        train_freq:4 \
        exploration_fraction:0.1 \
        exploration_final_eps:0.01 \
        optimize_memory_usage:False \
    --eval-freq 10000 \
    --eval-episodes 5 \
    --n-timesteps 1000000 \
    --device cuda \
    --verbose 1

echo "=== Pushing to HuggingFace Hub ==="
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
python3 -m rl_zoo3.push_to_hub \
    --algo dqn \
    --env SpaceInvadersNoFrameskip-v4 \
    -f logs/ \
    --repo-name "dqn-SpaceInvadersNoFrameskip-v4" \
    -orga "${HF_USERNAME}"

echo "Done! Unit 3 complete."
