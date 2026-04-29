"""Upload already-trained Doom model to HuggingFace Hub."""
import os, sys

HF_USERNAME = "mhpromit7473"
HF_TOKEN = os.environ.get("HF_TOKEN")

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults

def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    return parse_full_cfg(parser, argv)

if __name__ == "__main__":
    env = "doom_health_gathering_supreme"
    register_vizdoom_components()

    print("=== Uploading Doom model to HuggingFace Hub ===")
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
    enjoy(cfg_hub)
    print("Done!")
