import random

from rl_hockey.common.archive import Archive


class Matchmaker:
    def __init__(self, archive: Archive):
        self.archive = archive

    def sample_opponent(self, rating: float,  distribution: dict[str, float] = None, skill_range: float = 50) -> dict:
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
        
        return self.load_opponent(agent)
    
    def load_opponent(self, agent_dict: dict):
        # TODO (see opponent_manager.py, agent_factory.py)
        pass
