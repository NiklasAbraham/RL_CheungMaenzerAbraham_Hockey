import json
import random
from pathlib import Path
from typing import Optional, Tuple
import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common.agent import Agent
from rl_hockey.common.archive import Archive
from rl_hockey.common.archive.archive import AgentMetadata, Rating
from rl_hockey.common.archive.rating_system import RatingSystem
from rl_hockey.common.training.curriculum_manager import OpponentConfig
from rl_hockey.sac.sac import SAC
from rl_hockey.common.training.curriculum_manager import load_curriculum
from rl_hockey.common.training.agent_factory import create_agent


Opponent = Agent | h_env.BasicOpponent


class Matchmaker:
    def __init__(self, archive: Optional[Archive] = None, rating_system: Optional[RatingSystem] = None):
        """Initializes the Matchmaker."""
        self.archive = archive
        self.rating_system = rating_system

        self.loaded_agents: dict[str, Agent] = {}

    def get_opponent(self, config: OpponentConfig, rating: Optional[float] = None) -> Tuple[Opponent, Rating]:
        """
        Get an opponent based on the provided configuration.
        Args:
            config (OpponentConfig): Configuration for selecting the opponent.
            rating (Optional[float]): The rating of the current agent (used for archive sampling).
        Returns:
            Tuple[Opponent, float]: A tuple containing the opponent agent and its rating.
        """
        match config.type:
            case "basic_weak":
                rating = Rating(24.13, 0.78)
                if self.rating_system:
                    rating = self.rating_system.get_rating("basic_weak")
                return h_env.BasicOpponent(weak=True), rating
            case "basic_strong":
                rating = Rating(26.07, 0.83)
                if self.rating_system:
                    rating = self.rating_system.get_rating("basic_strong")
                return h_env.BasicOpponent(weak=False), rating
            case "archive":
                opponent, _, rating = self.sample_archive_opponent(rating, config.distribution, config.skill_range, config.deterministic)
                return opponent, rating
            case "weighted_mixture":
                pass  # Handled below
            case _:
                raise ValueError(f"Unknown opponent type: {config.type}")

        # Normalize weights
        weights = [opp.get('weight', 1.0) for opp in config.opponents]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Recursively sample opponent
        index = np.random.choice(len(config.opponents), p=normalized_weights)
        opponent_dict = config.opponents[index]

        config = OpponentConfig.from_dict(opponent_dict)

        return self.get_opponent(config, rating)


    def sample_archive_opponent(self, rating: float = None, distribution: dict[str, float] = None, skill_range: float = 50, deterministic: bool = True) -> Tuple[Opponent, str, Rating]:
        """"Sample an opponent from the archive based on skill distribution.

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

        distribution = distribution or {
            "skill": 0.8,
            "random": 0.1,
            "baseline": 0.1
        }

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

        if selected_strategy == "skill":
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

    def load_opponent(self, metadata: AgentMetadata, deterministic: bool = True) -> Opponent:
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
            deterministic=deterministic
        )

        self.loaded_agents[metadata.agent_id] = agent

        return agent
