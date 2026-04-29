#!/bin/bash
# Install all dependencies for the HuggingFace Deep RL Course
# Assumes CUDA-capable GPU (RTX 5070 Ti) and Python 3.10+

set -e

echo "=== Installing core dependencies ==="
pip install gymnasium[box2d,atari,accept-rom-license] --quiet
pip install stable-baselines3[extra] --quiet
pip install huggingface_sb3 huggingface_hub --quiet
pip install shimmy --quiet

echo "=== Unit 2: Q-Learning ==="
pip install pickle5 --quiet || true  # May fail on Python 3.8+, that's OK

echo "=== Unit 3: DQN SpaceInvaders (RL Zoo) ==="
pip install rl_zoo3 --quiet

echo "=== Unit 4: Reinforce + PixelCopter ==="
pip install pygame --quiet
pip install gym-pygame --quiet || pip install pygame-learning-environment --quiet || true

echo "=== Unit 6: A2C + Panda-Gym ==="
pip install panda_gym --quiet

echo "=== Unit 8 Part 1: CleanRL PPO ==="
pip install tensorboard wasabi --quiet

echo "=== Unit 8 Part 2: Sample Factory + ViZDoom ==="
# System deps (requires sudo)
sudo apt-get install -y \
    build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git \
    libfluidsynth-dev libgme-dev libopenal-dev \
    timidity libwildmidi-dev unzip \
    libboost-all-dev || echo "Some system deps may need manual install"

pip install vizdoom --quiet || echo "ViZDoom install may need manual steps"
pip install sample-factory==2.1.1 --quiet

echo "=== Done! ==="
echo "Set your HuggingFace token: export HF_TOKEN=your_token_here"
