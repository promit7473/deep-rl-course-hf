"""
Unit 8 Part 2: Sample Factory PPO on ViZDoom Health Gathering Supreme.
Uploads to HuggingFace Hub.

Prerequisites:
    pip install vizdoom sample-factory==2.1.1
    sudo apt-get install -y libsdl2-dev libopenal-dev
"""
import os
import sys

HF_USERNAME = "mhpromit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")

try:
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.enjoy import enjoy
    from sample_factory.train import run_rl
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
except ImportError as e:
    print(f"Sample Factory or ViZDoom not installed: {e}")
    print("Install with: pip install vizdoom sample-factory==2.1.1")
    sys.exit(1)

def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


if __name__ == "__main__":
    env = "doom_health_gathering_supreme"

    print("=== Training Sample Factory PPO on ViZDoom ===")
    register_vizdoom_components()

    cfg = parse_vizdoom_cfg(argv=[
        f"--env={env}",
        "--num_workers=8",
        "--num_envs_per_worker=4",
        "--train_for_env_steps=8000000",
        "--algo=APPO",
        "--use_rnn=True",
        "--num_epochs=1",
        "--rollout=32",
        "--recurrence=32",
        "--batch_size=2048",
        "--wide_aspect_ratio=False",
        "--save_every_sec=120",
        "--experiment=doom_health_gathering_supreme",
    ])

    status = run_rl(cfg)

    print("\n=== Evaluating and recording video ===")
    cfg_eval = parse_vizdoom_cfg(argv=[
        f"--env={env}",
        "--num_workers=1",
        "--save_video",
        "--no_render",
        "--max_num_episodes=10",
        "--experiment=doom_health_gathering_supreme",
    ], evaluation=True)
    status = enjoy(cfg_eval)

    print("\n=== Uploading to HuggingFace Hub ===")
    cfg_hub = parse_vizdoom_cfg(argv=[
        f"--env={env}",
        "--num_workers=1",
        "--save_video",
        "--no_render",
        "--max_num_episodes=10",
        "--max_num_frames=100000",
        "--push_to_hub",
        f"--hf_repository={HF_USERNAME}/rl_course_vizdoom_health_gathering_supreme",
        "--experiment=doom_health_gathering_supreme",
    ], evaluation=True)
    status = enjoy(cfg_hub)

    print("Done! Unit 8 Part 2 complete.")
