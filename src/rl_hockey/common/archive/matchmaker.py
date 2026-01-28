import random
from typing import Optional, Tuple
import hockey.hockey_env as h_env

from rl_hockey.common.agent import Agent
from rl_hockey.common.archive import Archive
from rl_hockey.common.training.curriculum_manager import OpponentConfig


class Matchmaker:
    def __init__(self, archive: Optional[Archive] = None):
        self.archive = archive

    def get_opponent(self, config: OpponentConfig, rating: Optional[float] = None) -> Tuple[Agent, float]:
        """
        Get an opponent based on the provided configuration.
        Args:
            config (OpponentConfig): Configuration for selecting the opponent.
            rating (Optional[float]): The rating of the current agent (used for archive sampling).
        Returns:
            Tuple[Agent, float]: A tuple containing the opponent agent and its rating.
        """
        match config.type:
            case "basic_weak":
                return h_env.BasicOpponent(weak=True), 0.0
            case "basic_strong":
                return h_env.BasicOpponent(weak=False), 0.0
            case "archive":
                return self._sample_archive_opponent(rating, config.distribution, config.skill_range)

        # TODO mixture

    def _sample_archive_opponent(self, rating: float,  distribution: dict[str, float] = None, skill_range: float = 50) -> dict:
        if not self.archive:
            raise ValueError("Archive is not set for Matchmaker.")

        distribution = distribution or {
            "skill": 0.8,
            "random": 0.1,
            "baseline": 0.1
        }

        rand_value = random.random()
        cumulative = 0.0
        for strategy, prob in distribution.items():
            cumulative += prob
            if rand_value < cumulative:
                selected_strategy = strategy
                break
        else:
            selected_strategy = "skill"  # Fallback

        if selected_strategy == "skill":
            agents = self.archive.get_agents(sort_by="rating")

            suitable_agents = []
            for agent in agents:
                agent_rating = agent.get("rating", 0).get("rating", 0)
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
            raise ValueError("No suitable opponent found based on the selected strategy.")
        
        return self.load_opponent(agent), agent.rating.rating
    
    def load_opponent(self, agent_dict: dict):
        # TODO (see opponent_manager.py, agent_factory.py)
        pass
