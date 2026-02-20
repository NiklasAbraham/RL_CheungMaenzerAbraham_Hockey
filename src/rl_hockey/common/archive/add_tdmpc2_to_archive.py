"""
Script to add TDMPC2 agents to the archive.
"""

import logging
from pathlib import Path

from rl_hockey.common.archive.archive import Archive, Rating

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config: provide specific checkpoint .pt file(s). Config is taken from the run folder.
CONFIG = {
    "checkpoint_paths": [
        Path(
            "results/self_play/2026-02-01_09-55-22/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep019488.pt"
        ),
    ],
    "archive_dir": Path("archive"),
    "agent_name": "TDMPC2",
    "add_baselines": True,
}


def get_checkpoint_info(checkpoint_path):
    """Get checkpoint info: config from run_dir/configs/."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    run_dir = checkpoint_path.parent.parent
    configs_dir = run_dir / "configs"
    config_files = sorted(configs_dir.glob("*.json")) if configs_dir.exists() else []
    config_file = config_files[0] if config_files else None
    return {
        "checkpoint": checkpoint_path,
        "config": config_file,
        "run_dir": run_dir,
    }


def add_checkpoint_to_archive(
    archive, checkpoint_info, agent_name="TDMPC2", tags=None, rating=None
):
    """
    Add a checkpoint to the archive.

    Args:
        archive: Archive instance
        checkpoint_info: Dictionary with checkpoint, config, and run_dir
        agent_name: Name of the agent algorithm
        tags: List of tags for the agent
        rating: Rating dictionary or None for default

    Returns:
        Agent ID if successful, None otherwise
    """
    checkpoint_path = checkpoint_info["checkpoint"]
    config_path = checkpoint_info["config"]

    if config_path is None:
        logger.warning(f"No config found for {checkpoint_path.name}, skipping")
        return None

    if tags is None:
        tags = ["tdmpc2", "needs_calibration"]

    # Extract episode number from checkpoint name if available
    step = None
    try:
        parts = checkpoint_path.stem.split("_ep")
        if len(parts) > 1:
            step_str = parts[-1].split("_")[0]
            step = int(step_str)
    except (ValueError, IndexError):
        pass

    try:
        agent_id = archive.add_agent(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            agent_name=agent_name,
            tags=tags,
            rating=rating,
            step=step,
            metadata={
                "training_run": checkpoint_info["run_dir"].name,
                "checkpoint_name": checkpoint_path.name,
            },
        )
        logger.info(f"Added agent {agent_id}")
        logger.info(f"  Checkpoint: {checkpoint_path.name}")
        logger.info(f"  Step: {step}")
        logger.info(f"  Tags: {', '.join(tags)}")
        return agent_id
    except Exception as e:
        logger.error(f"Failed to add {checkpoint_path.name}: {e}")
        return None


def add_baseline_agents(archive):
    """Add baseline opponents to the archive if not already present."""
    logger.info("Checking baseline agents...")

    baselines = [
        ("basic_weak", Rating(24.13, 0.78)),
        ("basic_strong", Rating(26.07, 0.83)),
    ]

    for baseline_name, rating in baselines:
        if baseline_name not in archive.registry:
            archive.add_baseline(baseline_name, rating)
            logger.info(f"Added baseline: {baseline_name}")
        else:
            logger.info(f"Baseline {baseline_name} already exists")


def main():
    cfg = CONFIG
    checkpoint_paths = cfg["checkpoint_paths"]
    archive_dir = cfg["archive_dir"]
    agent_name = cfg["agent_name"]
    add_baselines = cfg.get("add_baselines", True)

    if isinstance(checkpoint_paths, (str, Path)):
        checkpoint_paths = [checkpoint_paths]

    logger.info("=" * 70)
    logger.info("Adding TDMPC2 Agents to Archive")
    logger.info("=" * 70)

    archive = Archive(base_dir=str(archive_dir))
    logger.info(f"Archive directory: {archive.base_dir}")
    logger.info(f"Current agents in archive: {len(archive.registry)}")

    if add_baselines:
        add_baseline_agents(archive)

    checkpoints = []
    for cp in checkpoint_paths:
        info = get_checkpoint_info(cp)
        if info:
            checkpoints.append(info)
    logger.info(f"Adding {len(checkpoints)} checkpoint(s)")

    logger.info("\nAdding agents to archive...")
    added_count = 0
    for checkpoint_info in checkpoints:
        agent_id = add_checkpoint_to_archive(
            archive, checkpoint_info, agent_name=agent_name
        )
        if agent_id:
            added_count += 1

    logger.info("\n" + "=" * 70)
    logger.info(f"Successfully added {added_count}/{len(checkpoints)} agent(s)")
    logger.info(f"Total agents in archive: {len(archive.registry)}")
    logger.info("=" * 70)

    # List agents
    logger.info("\nAgents in archive:")
    for agent_meta in archive.get_agents(sort_by="archived_at"):
        logger.info(f"  {agent_meta.agent_id}")
        if agent_meta.step:
            logger.info(f"    Step: {agent_meta.step}")
        logger.info(f"    Tags: {', '.join(agent_meta.tags)}")
        logger.info(f"    Rating: {agent_meta.rating.rating:.2f}")


if __name__ == "__main__":
    main()
