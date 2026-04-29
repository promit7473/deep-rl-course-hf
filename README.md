# 🤖 Hugging Face Deep Reinforcement Learning Course

**Completed by Meraj Hossain Promit** — [Certificate of Completion](certificate.pdf)

> Full hands-on solutions for the [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — from beginner to expert, covering 8 units + bonus units with local GPU training scripts.

---

## Certificate

<p align="center">
  <img src="https://raw.githubusercontent.com/promit7473/deep-rl-course-hf/main/certificate_preview.png" width="600"/>
</p>

📄 [Download Certificate PDF](certificate.pdf)

---

## 📺 Demo — DQN Playing Space Invaders

https://github.com/promit7473/deep-rl-course-hf/assets/replay.mp4

*(Local replay: [`hub/replay.mp4`](hub/replay.mp4))*

---

## 📚 Course Coverage

| Unit | Topic | Algorithm | Environment |
|------|-------|-----------|-------------|
| Unit 1 | Intro to Deep RL | PPO | LunarLander-v2 |
| Unit 2 | Q-Learning | Q-Learning | Taxi-v3, FrozenLake |
| Unit 3 | Deep Q-Learning | DQN | Space Invaders |
| Unit 4 | Policy Gradients | REINFORCE | CartPole, PixelCopter |
| Unit 5 | Unity ML-Agents | PPO | SnowballTarget, Pyramids |
| Unit 6 | Actor-Critic | A2C | Panda Reach/Pick & Place |
| Unit 7 | Multi-Agent RL | Self-Play | SoccerTwos |
| Unit 8 | PPO (CleanRL + Doom) | PPO | LunarLander, ViZDoom |
| Bonus 1 | Huggy the Dog | PPO | Huggy |
| Bonus 2 | Hyperparameter Tuning | Optuna | — |
| Bonus 3 | Advanced Topics | RLHF, Offline RL, Decision Transformers | — |

---

## 📁 Repository Structure

```
deep-rl-course-hf/
├── notebooks/          # Solved Jupyter notebooks for each unit
│   ├── unit1/          # LunarLander with PPO
│   ├── unit2/          # Q-Learning (Taxi, FrozenLake)
│   ├── unit3/          # DQN (Space Invaders)
│   ├── unit4/          # Policy Gradients
│   ├── unit5/          # Unity ML-Agents
│   ├── unit6/          # A2C (Panda robotics)
│   ├── unit8/          # PPO with CleanRL + ViZDoom
│   └── bonus-unit1/    # Huggy the Dog
├── scripts/            # Standalone GPU training scripts
│   ├── unit1_train.py  # LunarLander PPO
│   ├── unit2_train.py  # Q-Learning
│   ├── unit3_train.sh  # DQN Atari
│   ├── unit4_train.py  # REINFORCE CartPole
│   ├── unit4_pixelcopter.py
│   ├── unit5_train.sh  # Unity ML-Agents
│   ├── unit6_train.py  # A2C Panda
│   ├── unit8_part1_ppo.py
│   ├── unit8_part2_train.py
│   └── unit8_part2_upload.py
├── hub/
│   ├── replay.mp4      # DQN Space Invaders gameplay
│   └── q-learning.pkl  # Trained Q-table (Taxi-v3, mean reward: 7.38)
├── run_all.py          # Run all training scripts sequentially
├── deep_rl_course_to_pdf.py  # Scrapes & compiles course → PDF
├── deep_rl_course.pdf  # Full course PDF (114 sections, 16 units)
└── certificate.pdf     # Course completion certificate
```

---

## 🚀 Running the Training Scripts

```bash
# Install dependencies
pip install stable-baselines3 gymnasium huggingface_hub

# Train a specific unit
python scripts/unit1_train.py      # LunarLander with PPO
python scripts/unit4_train.py      # CartPole with REINFORCE
python scripts/unit6_train.py      # Panda Reach with A2C

# Run all units sequentially
python run_all.py
```

---

## 📖 Course PDF

The full course (114 sections) is compiled into a single PDF:

```bash
pip install weasyprint beautifulsoup4 requests lxml
python deep_rl_course_to_pdf.py
```

Output: `~/deep_rl_course.pdf` (~69 MB)

---

## 🤗 Models on Hugging Face Hub

Trained models are uploaded to the HF Hub under **[mhpromit7473](https://huggingface.co/mhpromit7473)**:

- `mhpromit7473/ppo-LunarLander-v2`
- `mhpromit7473/dqn-SpaceInvadersNoFrameskip-v4`
- `mhpromit7473/a2c-PandaReachDense-v3`
- `mhpromit7473/a2c-PandaPickAndPlace-v3`
- `mhpromit7473/q-Taxi-v3`

---

## 📜 License

Course content © Hugging Face. Code in this repo is MIT licensed.
