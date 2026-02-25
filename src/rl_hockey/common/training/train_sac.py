"""
SAC training entry point. Calls train_vectorized with SAC curriculum config.
Config paths and settings are set in this file (no environment variables).
"""

from rl_hockey.common.training.train import train_vectorized

# Config: paths and settings for cluster and local runs
CONFIG_PATH = "./configs/curriculum_sac_archive2.json"
RESULT_DIR = "./results/sac_runs"
ARCHIVE_DIR = "./archive"
NUM_ENVS = 16


def main(
    config_path: str = CONFIG_PATH,
    result_dir: str = RESULT_DIR,
    archive_dir: str = ARCHIVE_DIR,
    num_envs: int = NUM_ENVS,
    verbose: bool = True,
):
    train_vectorized(
        config_path=config_path,
        result_dir=result_dir,
        archive_dir=archive_dir,
        num_envs=num_envs,
        verbose=verbose,
        update_archive=True
    )


if __name__ == "__main__":
    main()
