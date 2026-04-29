"""
Master runner for all HuggingFace Deep RL Course notebooks.
Runs each unit sequentially on your local GPU (RTX 5070 Ti).

Usage:
    export HF_TOKEN=your_huggingface_token
    python run_all.py [--units 1 2 3 4 6 8a 8b]

By default runs all units. Units 5 and bonus-unit1 require Unity ML-Agents
executables and must be run manually (see README notes below).
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

UNITS = {
    "1":  ("python", str(SCRIPTS_DIR / "unit1_train.py")),
    "2":  ("python", str(SCRIPTS_DIR / "unit2_train.py")),
    "3":  ("bash",   str(SCRIPTS_DIR / "unit3_train.sh")),
    "4":  ("python", str(SCRIPTS_DIR / "unit4_train.py")),
    "6":  ("python", str(SCRIPTS_DIR / "unit6_train.py")),
    "8a": ("python", str(SCRIPTS_DIR / "unit8_part1_ppo.py"),
           "--env-id", "LunarLander-v2",
           "--repo-id", "promit7473/ppo-LunarLander-v2-cleanrl",
           "--total-timesteps", "500000"),
    "8b": ("python", str(SCRIPTS_DIR / "unit8_part2_train.py")),
}

SKIPPED = {
    "5": "Unit 5 (ML-Agents SnowballTarget/Pyramids) requires Unity executables. "
         "Run manually via mlagents-learn inside the notebooks/unit5/ folder.",
    "bonus1": "Bonus Unit 1 (Huggy) requires the Huggy Unity executable. "
              "Run manually via mlagents-learn inside the notebooks/bonus-unit1/ folder.",
}

ESTIMATED_TIMES = {
    "1":  "~20 min  (PPO LunarLander-v2, 1M steps)",
    "2":  "~2 min   (Q-Learning FrozenLake + Taxi)",
    "3":  "~60 min  (DQN SpaceInvaders, 1M steps)",
    "4":  "~10 min  (Reinforce CartPole) + ~2 hrs (PixelCopter 50k eps)",
    "6":  "~40 min  (A2C PandaReach + PandaPickAndPlace, 1M steps each)",
    "8a": "~15 min  (PPO CleanRL LunarLander, 500k steps)",
    "8b": "~30 min  (Sample Factory ViZDoom, 4M steps)",
}


def run_unit(unit_id, cmd_parts):
    print(f"\n{'='*60}")
    print(f"  UNIT {unit_id}  |  {ESTIMATED_TIMES.get(unit_id, '')}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd_parts)}\n")

    start = time.time()
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN

    result = subprocess.run(cmd_parts, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] Unit {unit_id} exited with code {result.returncode} after {elapsed/60:.1f} min")
        return False
    else:
        print(f"\n[OK] Unit {unit_id} completed in {elapsed/60:.1f} min")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run all Deep RL Course training scripts")
    parser.add_argument("--units", nargs="+", default=list(UNITS.keys()),
                        help=f"Which units to run. Choices: {list(UNITS.keys())}")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Hub uploads will fail.")
        print("Set it with: export HF_TOKEN=your_token\n")

    print("\nHuggingFace Deep RL Course - Local GPU Runner")
    print("GPU: RTX 5070 Ti")
    print(f"Running units: {args.units}\n")

    for unit_id, note in SKIPPED.items():
        if unit_id in args.units or f"unit{unit_id}" in args.units:
            print(f"[SKIP] Unit {unit_id}: {note}\n")

    results = {}
    for unit_id in args.units:
        if unit_id not in UNITS:
            print(f"[SKIP] Unknown unit: {unit_id}")
            continue
        cmd_parts = list(UNITS[unit_id])
        success = run_unit(unit_id, cmd_parts)
        results[unit_id] = "OK" if success else "FAILED"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for unit_id, status in results.items():
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark} Unit {unit_id}: {status}")

    for unit_id, note in SKIPPED.items():
        print(f"  - Unit {unit_id}: SKIPPED (manual) — {note[:60]}...")

    failed = [u for u, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\nFailed units: {failed}")
        sys.exit(1)
    print("\nAll units complete! Upload models to HuggingFace and submit.")


if __name__ == "__main__":
    main()
