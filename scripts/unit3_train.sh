#!/bin/bash
# Unit 3: DQN on SpaceInvadersNoFrameskip-v4 using RL Baselines3 Zoo
# Trains and pushes to HuggingFace Hub

set -e

HF_USERNAME="promit7473"

echo "=== Unit 3: Installing RL Zoo + Atari ==="
pip install rl_zoo3 gymnasium[atari] gymnasium[accept-rom-license] --quiet

echo "=== Training DQN on SpaceInvaders ==="
python -m rl_zoo3.train \
    --algo dqn \
    --env SpaceInvadersNoFrameskip-v4 \
    -f logs/ \
    -c /dev/null \
    --hyperparams \
        buffer_size:100000 \
        learning_rate:0.0001 \
        batch_size:32 \
        learning_starts:100000 \
        target_update_interval:1000 \
        train_freq:4 \
        exploration_fraction:0.1 \
        exploration_final_eps:0.01 \
        optimize_memory_usage:True \
    --eval-freq 10000 \
    --eval-episodes 5 \
    --n-timesteps 1000000 \
    --device cuda \
    --verbose 1

echo "=== Pushing to HuggingFace Hub ==="
python -m rl_zoo3.push_to_hub \
    --algo dqn \
    --env SpaceInvadersNoFrameskip-v4 \
    -f logs/ \
    --repo-name "${HF_USERNAME}/dqn-SpaceInvadersNoFrameskip-v4" \
    -orga "${HF_USERNAME}" \
    --token "${HF_TOKEN}"

echo "Done! Unit 3 complete."
