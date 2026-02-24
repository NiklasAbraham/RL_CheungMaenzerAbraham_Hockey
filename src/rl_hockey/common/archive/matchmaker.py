import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common.agent import Agent
from rl_hockey.common.archive import Archive
from rl_hockey.common.archive.archive import AgentMetadata, Rating
from rl_hockey.common.archive.rating_system import RatingSystem
from rl_hockey.common.training.agent_factory import create_agent
from rl_hockey.common.training.curriculum_manager import (
    AgentConfig,
    OpponentConfig,
    load_curriculum,
)
from rl_hockey.common.training.opponent_manager import load_agent_checkpoint

logger = logging.getLogger(__name__)

Opponent = Agent | h_env.BasicOpponent


class Matchmaker:
    def __init__(
        self,
        archive: Optional[Archive] = None,
        rating_system: Optional[RatingSystem] = None,
        run_models_dir: Optional[str] = None,
    ):
        """Initializes the Matchmaker.

        Args:
            archive: Archive of saved agents for archive-based sampling.
            rating_system: Rating system for skill-based matching.
            run_models_dir: Path to the current run's models directory. Required
                when using the "run_checkpoints" opponent type so the matchmaker
                can discover checkpoints saved during the active training run.
        """
        self.archive = archive
        self.rating_system = rating_system
        self.run_models_dir = run_models_dir

        self.loaded_agents: dict[str, Agent] = {}

    def get_opponent(
        self, config: OpponentConfig, rating: Optional[float] = None
    ) -> Tuple[Opponent, str]:
        """
        Get an opponent based on the provided configuration.
        Args:
            config (OpponentConfig): Configuration for selecting the opponent.
            rating (Optional[float]): The rating of the current agent (used for archive sampling).
        Returns:
            Tuple[Opponent, str]: A tuple containing the opponent agent and its id.
        """
        match config.type:
            case "basic_weak":
                rating = Rating(24.13, 0.78)
                if self.rating_system:
                    rating = self.rating_system.get_rating("basic_weak")
                return h_env.BasicOpponent(weak=True), "basic_weak"
            case "basic_strong":
                rating = Rating(26.07, 0.83)
                if self.rating_system:
                    rating = self.rating_system.get_rating("basic_strong")
                return h_env.BasicOpponent(weak=False), "basic_strong"
            case "archive":
                opponent, agent_id, rating = self.sample_archive_opponent(
                    rating,
                    config.distribution,
                    config.skill_range,
                    config.deterministic,
                )
                msg = (
                    f"Archive opponent chosen: agent_id={agent_id} "
                    f"rating={rating.rating:.2f} (mu={rating.mu:.2f} sigma={rating.sigma:.2f})"
                )
                print(msg, flush=True)
                logging.info("%s", msg)
                return opponent, agent_id
            case "self_play":
                opponent = self._load_self_play_opponent(config)
                return opponent, ""
            case "run_checkpoints":
                opponent = self._sample_run_checkpoint_opponent(config)
                return opponent, ""
            case "weighted_mixture":
                pass  # Handled below
            case _:
                raise ValueError(f"Unknown opponent type: {config.type}")

        # Normalize weights
        weights = [opp.get("weight", 1.0) for opp in config.opponents]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Recursively sample opponent
        index = np.random.choice(len(config.opponents), p=normalized_weights)
        opponent_dict = config.opponents[index]

        config = OpponentConfig.from_dict(opponent_dict)

        return self.get_opponent(config, rating)

    def sample_archive_opponent(
        self,
        rating: float = None,
        distribution: dict[str, float] = None,
        skill_range: float = 3,
        deterministic: bool = True,
    ) -> Tuple[Opponent, str, Rating]:
        """ "Sample an opponent from the archive based on skill distribution.

        Args:
            rating: Current agent's rating
            distribution: Probability distribution for sampling strategies
            skill_range: Skill range for skill-based sampling
            deterministic: Whether to load the opponent in deterministic mode
        Returns:
            A tuple of (opponent agent, opponent agent ID, opponent rating)
        """
        if not self.archive:
            raise ValueError("Archive is not set for Matchmaker.")

        distribution = distribution or {"pfsp": 0.5, "random": 0.3, "baseline": 0.2}

        total_weight = sum(distribution.values())
        distribution = {k: v / total_weight for k, v in distribution.items()}

        if rating is None:
            distribution["random"] += distribution.get("skill", 0)
            distribution["skill"] = 0.0

        rand_value = random.random()
        cumulative = 0.0
        for strategy, prob in distribution.items():
            cumulative += prob
            if rand_value < cumulative:
                selected_strategy = strategy
                break
        else:
            selected_strategy = "random"  # Fallback

        if selected_strategy == "pfsp":
            agents = self.archive.get_agents()

            wrs = [self.rating_system.get_winrate(agent.agent_id) for agent in agents]
            wrs = np.array(wrs)
            priorities = wrs * (1 - wrs)  # Max at 0.5 winrate, 0 at 0 or 1
            if priorities.sum() == 0:
                probs = np.ones(len(agents)) / len(agents)
            else:
                probs = priorities / priorities.sum()

            agent = np.random.choice(agents, p=probs)
        elif selected_strategy == "skill":
            agents = self.archive.get_agents(sort_by="rating")

            suitable_agents = []
            for agent in agents:
                agent_rating = agent.rating.rating
                if agent_rating > rating + skill_range:
                    continue
                if agent_rating < rating - skill_range:
                    break
                suitable_agents.append(agent)

            agent = random.choice(suitable_agents) if suitable_agents else None
            if agent is None:
                logger.warning(
                    "No suitable agents found for skill-based sampling (rating=%.2f, range=%.2f). Falling back to random sampling.",
                    rating,
                    skill_range
                )
        elif selected_strategy == "random":
            agents = self.archive.get_agents()
            agent = random.choice(agents) if agents else None
        elif selected_strategy == "baseline":
            agents = self.archive.get_agents(tags=["baseline"])
            agent = random.choice(agents) if agents else None
        else:
            agents = self.archive.get_agents(tags=[selected_strategy])
            agent = random.choice(agents) if agents else None

        if agent is None:
            agents = self.archive.get_agents()
            agent = random.choice(agents) if agents else None

        return self.load_opponent(agent, deterministic), agent.agent_id, agent.rating

    def _load_self_play_opponent(self, config: OpponentConfig) -> Opponent:
        """Load a self-play opponent from a checkpoint path (used by weighted_mixture with type self_play)."""
        if config.checkpoint is None:
            raise ValueError("self_play opponent requires a checkpoint path in config")
        cache_key = f"self_play:{config.checkpoint}:{config.agent_type or 'SAC'}"
        if cache_key in self.loaded_agents:
            return self.loaded_agents[cache_key]
        env = h_env.HockeyEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        env.close()
        agent_type = config.agent_type or "SAC"
        dummy_config = AgentConfig(
            type=agent_type, hyperparameters={}, checkpoint_path=None
        )
        opponent = load_agent_checkpoint(
            checkpoint_path=config.checkpoint,
            agent_config=dummy_config,
            state_dim=state_dim,
            action_dim=action_dim,
            is_discrete=False,
            opponent_agent_type=agent_type,
        )
        self.loaded_agents[cache_key] = opponent
        return opponent

    def _sample_run_checkpoint_opponent(self, config: OpponentConfig) -> Opponent:
        """Pick a random checkpoint from the current run's models directory and load it.

        The directory is re-scanned on every call (no cached file list), so new
        checkpoints saved during a long run are included the next time an opponent
        is sampled. When no checkpoint exists yet (very beginning of training) it
        falls back to basic_strong.

        Args:
            config: Opponent config; uses agent_type (default "TDMPC2") and
                    deterministic flag.
        Returns:
            Loaded opponent agent, or BasicOpponent(weak=False) as fallback.
        """
        if not self.run_models_dir:
            raise ValueError(
                "run_models_dir is not set on Matchmaker but opponent type "
                "'run_checkpoints' was requested. Pass run_models_dir= when "
                "constructing Matchmaker."
            )

        models_path = Path(self.run_models_dir)
        # Collect checkpoint files; exclude metadata JSON sidecars
        pt_files = [
            f for f in models_path.glob("*.pt") if "_metadata" not in f.name
        ]

        if not pt_files:
            logger.warning(
                "No checkpoints found in %s yet – falling back to basic_strong.",
                self.run_models_dir,
            )
            return h_env.BasicOpponent(weak=False)

        chosen = random.choice(pt_files)
        agent_type = config.agent_type or "TDMPC2"

        msg = f"run_checkpoints opponent chosen: {chosen.name} (type={agent_type})"
        logger.info("%s", msg)

        # Reuse _load_self_play_opponent so the loaded agent is cached by path
        proxy_config = OpponentConfig(
            type="self_play",
            checkpoint=str(chosen),
            agent_type=agent_type,
            deterministic=config.deterministic,
        )
        return self._load_self_play_opponent(proxy_config)

    def load_opponent(
        self, metadata: AgentMetadata, deterministic: bool = True
    ) -> Opponent:
        """Load an agent from metadata checkpoint."""
        if "baseline" in metadata.tags:
            if metadata.agent_id == "basic_weak":
                return h_env.BasicOpponent(weak=True)
            elif metadata.agent_id == "basic_strong":
                return h_env.BasicOpponent(weak=False)
            else:
                raise ValueError(f"Unknown baseline: {metadata.agent_id}")

        if metadata.agent_id in self.loaded_agents:
            return self.loaded_agents[metadata.agent_id]

        env = h_env.HockeyEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        env.close()

        config = load_curriculum(metadata.config_path)

        agent = create_agent(
            agent_config=config.agent,
            state_dim=state_dim,
            action_dim=action_dim,
            common_hyperparams=config.hyperparameters,
            checkpoint_path=metadata.checkpoint_path,
            deterministic=deterministic,
        )

        self.loaded_agents[metadata.agent_id] = agent

        return agent
