"""
TDMPC2 training entry point. Calls train_vectorized with TDMPC2 curriculum config.
Config paths and settings are set in this file (no environment variables).
"""

from typing import Optional

from rl_hockey.common.training.train import train_vectorized

# Config: paths and settings for cluster and local runs
CONFIG_PATH = "./configs/curriculum_tdmpc2_bonus_decay.json"
RESULT_DIR = "./results/tdmpc2_runs"
ARCHIVE_DIR = "./archive"
NUM_ENVS = 1


def main(
    config_path: str = CONFIG_PATH,
    result_dir: str = RESULT_DIR,
    archive_dir: str = ARCHIVE_DIR,
    num_envs: int = NUM_ENVS,
    verbose: bool = True,
    resume_from: Optional[str] = None,
):
    """Run TDMPC2 curriculum training.

    To continue from an existing run, pass the path to that run's directory, e.g.:
        resume_from="./results/tdmpc2_runs/2026-02-01_09-55-22"
    The latest checkpoint in models/ is loaded and training continues from that episode.
    """
    train_vectorized(
        config_path=config_path,
        result_dir=result_dir,
        archive_dir=archive_dir,
        num_envs=num_envs,
        verbose=verbose,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    # To continue from an existing run, pass the run directory path:
    # main(resume_from="./results/tdmpc2_runs/2026-02-01_09-55-22")
    main()
