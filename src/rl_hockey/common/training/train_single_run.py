"""
Single training run with curriculum learning support.
Can use either JSON config file or dict config.
"""

import json
import tempfile
import os
from rl_hockey.common.training.train_run import train_run


def train_single_run(
    config_path: str = None,
    config_dict: dict = None,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = 10000,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
):
    if config_path is not None:
        return train_run(
            config_path,
            base_output_dir,
            run_name,
            verbose,
            eval_freq_steps=eval_freq_steps,
            eval_num_games=eval_num_games,
            eval_weak_opponent=eval_weak_opponent,
        )
    else:
        raise ValueError("config_path must be provided")


if __name__ == "__main__":
    # path_to_config = "configs/curriculum_sac.json"
    path_to_config = "configs/curriculum_td3.json"

    train_single_run(path_to_config, base_output_dir="results/hyperparameter_runs")

    # nohup python src/rl_hockey/common/training/train_single_run.py > train_single_run.log 2>&1 &
