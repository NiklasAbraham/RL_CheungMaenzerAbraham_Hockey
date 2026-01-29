"""
Agent Rating System using TrueSkill algorithm.

Manages ratings for archived agents and updates them based on match results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import trueskill

from rl_hockey.common.archive.archive import Archive, Rating


class RatingSystem:
    """Manages ratings for all agents using TrueSkill."""
    
    def __init__(
        self,
        archive: Archive,
        mu: float = 25.0,
        sigma: float = 8.333,
        beta: float = 4.166,
        tau: float = 0.083,
        draw_probability: float = 0.15,
    ):
        """
        Initialize rating system.
        
        Args:
            archive: Archive instance to manage agents
            mu: Initial mean skill level
            sigma: Initial uncertainty
            beta: Skill class width (distance guaranteeing ~76% win probability)
            tau: Dynamics factor (prevents sigma from getting too small)
            draw_probability: Probability of a draw (0.0 to 1.0)
        """
        self.archive = archive
        self.match_history_file = self.archive.base_dir / "match_history.json"
        
        self.env = trueskill.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
        )
        
        # Load or initialize ratings
        self.ratings: Dict[str, Rating] = {}
        self.match_history: List[Dict] = []
        self._load_ratings()
        self._load_match_history()
    
    def _load_ratings(self):
        """Load ratings from disk."""
        agents = self.archive.get_agents()
        for agent in agents:
            agent_id = agent["agent_id"]
            rating_data = agent["rating"]
            if rating_data:
                self.ratings[agent_id] = Rating.from_dict(rating_data)
            else:
                self.ratings[agent_id] = Rating()
    
    def _save_ratings(self):
        """Save ratings to disk."""
        for agent_id, rating in self.ratings.items():
            self.archive.update_agent_rating(agent_id, rating.to_dict())
    
    def _load_match_history(self):
        """Load match history from disk."""
        if self.match_history_file.exists():
            with open(self.match_history_file, 'r') as f:
                self.match_history = json.load(f)
    
    def _save_match_history(self):
        """Save match history to disk."""
        with open(self.match_history_file, 'w') as f:
            json.dump(self.match_history, f, indent=2)
    
    def initialize_agent(self, agent_id: str, mu: float = None, sigma: float = None):
        """
        Initialize rating for a new agent.
        
        Args:
            agent_id: Agent identifier
            mu: Initial mean (if None, uses environment default)
            sigma: Initial uncertainty (if None, uses environment default)
        """
        if agent_id in self.ratings:
            return  # Already initialized
        
        if mu is None:
            mu = self.env.mu
        
        if sigma is None:
            sigma = self.env.sigma
        
        self.ratings[agent_id] = Rating(mu=mu, sigma=sigma)
        self._save_ratings()

    def estimate_rating(self, rating: dict[str, float], opponent_id: str, result: int) -> float:
        """
        Estimate the expected rating after a match against an opponent.
        
        Args:
            rating: Current rating dict of the agent
            opponent_id: Opponent agent identifier
            result: Match result from agent's perspective:
                    1 = win, -1 = loss, 0 = draw
        Returns:
            Estimated rating after the match
        """
        temp_rating = Rating.from_dict(rating)

        self.initialize_agent(opponent_id)
        opponent_rating = self.ratings[opponent_id]
        
        ts_rating = self.env.create_rating(mu=temp_rating.mu, sigma=temp_rating.sigma)
        ts_opponent = self.env.create_rating(mu=opponent_rating.mu, sigma=opponent_rating.sigma)
        
        # Simulate rating update
        if result == 1:  # Win
            (new_ts,), _ = self.env.rate_1vs1(ts_rating, ts_opponent)
        elif result == -1:  # Loss
            _, (new_ts,) = self.env.rate_1vs1(ts_opponent, ts_rating)
        else:  # Draw
            (new_ts,), _ = self.env.rate_1vs1(ts_rating, ts_opponent, drawn=True)
        
        estimated_rating = new_ts.mu - 3*new_ts.sigma
        return estimated_rating
    
    def update_ratings(
        self,
        agent1_id: str,
        agent2_id: str,
        result: int,
        save: bool = False
    ) -> Tuple[Rating, Rating]:
        """
        Update ratings after a match.
        
        Args:
            agent1_id: First agent identifier
            agent2_id: Second agent identifier
            result: Match result from agent1's perspective:
                    1 = agent1 won, -1 = agent2 won, 0 = draw
        
        Returns:
            Tuple of (agent1_new_rating, agent2_new_rating)
        """
        self.initialize_agent(agent1_id)
        self.initialize_agent(agent2_id)
        
        rating1 = self.ratings[agent1_id]
        rating2 = self.ratings[agent2_id]
        
        # Use trueskill.Rating instead
        ts_rating1 = self.env.create_rating(mu=rating1.mu, sigma=rating1.sigma)
        ts_rating2 = self.env.create_rating(mu=rating2.mu, sigma=rating2.sigma)
        
        # Update ratings
        if result == 1:  # Agent1 won
            (new_ts1,), (new_ts2,) = self.env.rate_1vs1(ts_rating1, ts_rating2)
        elif result == -1:  # Agent2 won
            (new_ts2,), (new_ts1,) = self.env.rate_1vs1(ts_rating2, ts_rating1)
        else:  # Draw
            (new_ts1,), (new_ts2,) = self.env.rate_1vs1(ts_rating1, ts_rating2, drawn=True)
        
        rating1.mu = new_ts1.mu
        rating1.sigma = new_ts1.sigma
        rating1.matches_played += 1
        
        rating2.mu = new_ts2.mu
        rating2.sigma = new_ts2.sigma
        rating2.matches_played += 1
        
        if save:
            match_record = {
                "timestamp": datetime.now().isoformat(),
                "agent1_id": agent1_id,
                "agent2_id": agent2_id,
                "result": result,
                "agent1_rating_before": rating1.rating - (new_ts1.mu - ts_rating1.mu - 3*(new_ts1.sigma - ts_rating1.sigma)),
                "agent2_rating_before": rating2.rating - (new_ts2.mu - ts_rating2.mu - 3*(new_ts2.sigma - ts_rating2.sigma)),
                "agent1_rating_after": rating1.rating,
                "agent2_rating_after": rating2.rating,
            }
            self.match_history.append(match_record)
            
            self._save_ratings()
            self._save_match_history()
            
        return rating1, rating2
    
    def get_rating(self, agent_id: str) -> Optional[Rating]:
        """
        Get current rating for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Rating object or None if not found
        """
        return self.ratings.get(agent_id)
    
    def get_leaderboard(self, min_matches: int = 0) -> List[Tuple[str, Rating]]:
        """
        Get leaderboard sorted by rating.
        
        Args:
            min_matches: Minimum number of matches to be included
            
        Returns:
            List of (agent_id, rating) tuples sorted by rating (descending)
        """
        leaderboard = [
            (agent_id, rating)
            for agent_id, rating in self.ratings.items()
            if rating.matches_played >= min_matches
        ]
        leaderboard.sort(key=lambda x: x[1].rating, reverse=True)
        return leaderboard
    
    def get_match_history(
        self,
        agent_id: str = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Get match history, optionally filtered by agent.
        
        Args:
            agent_id: Filter by agent ID (optional)
            limit: Maximum number of matches to return (optional)
            
        Returns:
            List of match records
        """
        history = self.match_history
        
        if agent_id:
            history = [
                match for match in history
                if match["agent1_id"] == agent_id or match["agent2_id"] == agent_id
            ]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def predict_win_probability(self, agent1_id: str, agent2_id: str) -> float:
        """
        Predict probability that agent1 beats agent2.
        
        Args:
            agent1_id: First agent identifier
            agent2_id: Second agent identifier
            
        Returns:
            Probability of agent1 winning (0.0 to 1.0)
        """        
        rating1 = self.ratings.get(agent1_id)
        rating2 = self.ratings.get(agent2_id)
        
        if not rating1 or not rating2:
            return 0.5
        
        # Calculate using TrueSkill's quality function
        # This gives us the probability of a draw, but we can approximate win probability
        delta_mu = rating1.mu - rating2.mu
        sum_sigma = (rating1.sigma ** 2 + rating2.sigma ** 2) ** 0.5
        
        # Using normal CDF approximation
        import math
        win_prob = 0.5 * (1 + math.erf(delta_mu / (sum_sigma * math.sqrt(2))))
        
        return win_prob
