#!/bin/bash
set -e

HF_USERNAME="mhpromit7473"
BASE="$HOME/deep-rl-course-hf"

echo "=== Unit 5: ML-Agents ==="

# Login to HF
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

echo "--- Training SnowballTarget (1M steps) ---"
cd "$BASE"
mlagents-learn config/ppo/SnowballTarget.yaml \
    --env=training-envs-executables/linux/SnowballTarget/SnowballTarget/SnowballTarget.x86_64 \
    --run-id="SnowballTarget1" \
    --no-graphics \
    --force

echo "--- Pushing SnowballTarget to Hub ---"
mlagents-push-to-hf \
    --run-id="SnowballTarget1" \
    --local-dir="results/SnowballTarget1" \
    --repo-id="${HF_USERNAME}/ppo-SnowballTarget" \
    --commit-message="Trained PPO SnowballTarget - Unit 5"

echo "--- Training Pyramids (1M steps) ---"
mlagents-learn config/ppo/PyramidsRND.yaml \
    --env=training-envs-executables/linux/Pyramids/Pyramids/Pyramids \
    --run-id="PyramidsTraining" \
    --no-graphics \
    --force

echo "--- Pushing Pyramids to Hub ---"
mlagents-push-to-hf \
    --run-id="PyramidsTraining" \
    --local-dir="results/PyramidsTraining" \
    --repo-id="${HF_USERNAME}/ppo-PyramidRND" \
    --commit-message="Trained PPO PyramidsRND - Unit 5"

echo "Done! Unit 5 complete."
