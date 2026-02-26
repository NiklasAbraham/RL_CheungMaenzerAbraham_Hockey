"""
Archive Manager, handles storage and retrieval of trained agents.

This module provides a centralized archive for storing trained agents with metadata,
enabling easy retrieval for self-play training and evaluation.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Rating:
    """Represents a single agent's rating with uncertainty."""

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 8.333,
        matches_played: int = 0,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
    ):
        """
        Initialize agent rating.

        Args:
            mu: Mean skill level (default: 25.0)
            sigma: Uncertainty/standard deviation (default: 8.333)
            matches_played: Number of matches played (default: 0)
            wins: Number of wins (default: 0)
            losses: Number of losses (default: 0)
            draws: Number of draws (default: 0)
        """
        self.mu = mu
        self.sigma = sigma
        self.matches_played = matches_played
        self.wins = wins
        self.losses = losses
        self.draws = draws

    @property
    def rating(self) -> float:
        """
        Conservative skill estimate (mu - 3*sigma).
        """
        return self.mu - 3 * self.sigma

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "rating": self.rating,
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Rating":
        """Create from dictionary."""
        rating = cls(
            mu=data["mu"],
            sigma=data["sigma"],
            matches_played=data.get("matches_played", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
        )
        return rating

    def __str__(self):
        return f"Rating(mu={self.mu:.2f}, sigma={self.sigma:.2f}, rating={self.rating:.2f}, matches_played={self.matches_played})"


@dataclass
class AgentMetadata:
    """Complete metadata for an archived agent."""

    agent_id: str
    archived_at: str
    tags: List[str]
    rating: Rating
    step: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["rating"] = self.rating.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetadata":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            archived_at=data["archived_at"],
            tags=data.get("tags", []),
            rating=Rating.from_dict(data.get("rating", {})),
            step=data.get("step"),
            checkpoint_path=data.get("checkpoint_path"),
            config_path=data.get("config_path"),
        )


@dataclass
class RegistryEntry:
    """Registry entry for an agent."""

    agent_id: str
    archived_at: str
    tags: List[str]
    rating: Rating
    directory: Optional[str] = None
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["rating"] = self.rating.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryEntry":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            archived_at=data["archived_at"],
            tags=data.get("tags", []),
            rating=Rating.from_dict(data.get("rating", {})),
            directory=data.get("directory", ""),
            step=data.get("step"),
        )


class Archive:
    """Manages the agent archive directory and metadata."""

    def __init__(self, base_dir: str = "archive"):
        """
        Initialize the archive.

        Args:
            base_dir: Root directory for the agent archive
        """
        self.base_dir = Path(base_dir)
        self.agents_dir = self.base_dir / "agents"
        self.registry_file = self.base_dir / "registry.json"

        # Create directory structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(exist_ok=True)

        # Initialize or load registry
        self._load_registry()

    def _load_registry(self):
        """Load existing registry or create a new one."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                registry_data = json.load(f)
                self.registry = {
                    agent_id: RegistryEntry.from_dict(entry)
                    for agent_id, entry in registry_data.items()
                }
        else:
            self.registry = {}
            self._save_registry()

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            registry_data = {
                agent_id: entry.to_dict() for agent_id, entry in self.registry.items()
            }
            json.dump(registry_data, f, indent=2)

    def _update_metadata_file(self, agent_id: str):
        """Update metadata file with current rating and tags from registry."""
        if agent_id not in self.registry:
            return

        registry_entry = self.registry[agent_id]

        # Skip baselines (no metadata file)
        if "baseline" in registry_entry.tags:
            return

        raw_dir = registry_entry.directory or ""
        if not raw_dir:
            return
        raw_path = Path(raw_dir.replace("\\", "/"))
        agent_dir = raw_path if raw_path.is_absolute() else (self.base_dir.parent / raw_path).resolve()
        if not agent_dir.exists():
            return

        metadata_path = agent_dir / "metadata.json"
        if not metadata_path.exists():
            return

        # Update metadata file
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        metadata_dict["rating"] = registry_entry.rating.to_dict()
        metadata_dict["tags"] = registry_entry.tags

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _generate_agent_id(self, agent_name: str) -> str:
        """
        Generate a unique agent ID.

        Args:
            agent_name: Optional agent name to include in ID

        Returns:
            Unique agent ID string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_count = len(self.registry) - 1  # Discard baseline agents

        return f"{agent_count:04d}_{agent_name}_{timestamp}"

    def add_agent(
        self,
        checkpoint_path: str,
        config_path: str,
        agent_name: str,
        tags: Optional[List[str]] = None,
        rating: Optional[Rating] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an agent to the archive.

        Args:
            checkpoint_path: Path to agent checkpoint (.pt file)
            config_path: Path to training configuration
            agent_name: Name of the agent algorithm
            tags: List of tags for the agent (optional)
            rating: Rating dictionary with "mu" and "sigma" (optional)
            step: Training step number (optional)
            metadata: Additional metadata to store (optional)

        Returns:
            Generated agent ID
        """
        if tags is None:
            tags = []

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Generate agent ID and create directory
        agent_id = self._generate_agent_id(agent_name)
        agent_dir = self.agents_dir / agent_id
        agent_dir.mkdir(exist_ok=True)

        # Copy checkpoint
        checkpoint_dest = agent_dir / "checkpoint.pt"
        shutil.copy2(checkpoint_path, checkpoint_dest)

        # Copy config if provided
        if config_path and Path(config_path).exists():
            config_dest = agent_dir / "config.json"
            shutil.copy2(config_path, config_dest)

        # Use default rating if not provided
        if not rating:
            rating = Rating(25, 8.333)
            tags.append("needs_calibration")

        # Prepare metadata
        agent_metadata = AgentMetadata(
            agent_id=agent_id,
            archived_at=datetime.now().isoformat(),
            tags=tags,
            step=step,
            rating=rating,
            checkpoint_path=str(checkpoint_dest),
            config_path=str(config_dest) if config_dest else None,
        )

        # Add any additional metadata fields
        metadata_dict = agent_metadata.to_dict()

        if metadata:
            for key, value in metadata.items():
                metadata_dict[key] = value

        # Save metadata
        metadata_path = agent_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Update registry
        registry_entry = RegistryEntry(
            agent_id=agent_id,
            archived_at=agent_metadata.archived_at,
            tags=tags,
            step=step,
            rating=rating,
            directory=str(agent_dir),
        )
        self.registry[agent_id] = registry_entry
        self._save_registry()

        return agent_id

    def add_baseline(self, agent_name: str, rating: Dict[str, float]) -> str:
        """
        Add a baseline agent to the archive.

        Args:
            agent_name: Name of the baseline agent
            rating: Rating dictionary with "mu" and "sigma"

        Returns:
            Generated agent ID
        """
        registry_entry = RegistryEntry(
            agent_id=agent_name,
            archived_at=datetime.now().isoformat(),
            tags=["baseline"],
            rating=rating,
        )
        self.registry[agent_name] = registry_entry
        self._save_registry()

        return agent_name

    def get_agents(
        self, sort_by: str = "archived_at", tags: Optional[List[str]] = None
    ) -> List[AgentMetadata]:
        """
        List all archived agents.

        Args:
            sort_by: Field to sort by ("archived_at", "agent_id", "rating")
            tags: List of tags to filter agents by (optional)
        Returns:
            List of agent metadata objects
        """
        agents = []

        for agent_id, registry_entry in self.registry.items():
            if tags:
                if not all(tag in registry_entry.tags for tag in tags):
                    continue

            metadata = self.get_agent_metadata(agent_id)
            if metadata:
                agents.append(metadata)

        # Sort agents
        if sort_by == "archived_at":
            agents.sort(key=lambda x: x.archived_at, reverse=True)
        elif sort_by == "agent_id":
            agents.sort(key=lambda x: x.agent_id)
        elif sort_by == "rating":
            agents.sort(key=lambda x: x.rating.rating, reverse=True)

        return agents

    def get_agent_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """
        Get metadata for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent metadata object or None if not found
        """
        if agent_id not in self.registry:
            return None

        registry_entry = self.registry[agent_id]

        if "baseline" in registry_entry.tags:
            metadata = AgentMetadata.from_dict(registry_entry.to_dict())
            return metadata

        raw_dir = registry_entry.directory or ""
        if not raw_dir:
            return None
        # Resolve relative to archive base so this works regardless of cwd
        raw_path = Path(raw_dir.replace("\\", "/"))
        if raw_path.is_absolute():
            agent_dir = raw_path
        else:
            agent_dir = (self.base_dir.parent / raw_path).resolve()

        metadata_path = agent_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            if "agent_id" in metadata_dict:
                metadata = AgentMetadata.from_dict(metadata_dict)
                return metadata
            # else: file is a curriculum/config, not archive metadata; use fallback

        # No valid metadata.json: build from registry if checkpoint and config exist
        if not agent_dir.exists():
            return None
        checkpoint_path = agent_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            return None
        config_path = agent_dir / "config.json"
        if not config_path.exists():
            for p in agent_dir.glob("*.json"):
                if p.name != "metadata.json":
                    config_path = p
                    break
            else:
                return None  # need a config to load the agent
        metadata = AgentMetadata(
            agent_id=registry_entry.agent_id,
            archived_at=registry_entry.archived_at,
            tags=list(registry_entry.tags),
            rating=registry_entry.rating,
            step=registry_entry.step,
            checkpoint_path=str(checkpoint_path.resolve()),
            config_path=str(config_path.resolve()),
        )
        return metadata

    def get_agent_checkpoint_path(self, agent_id: str) -> Optional[str]:
        """
        Get the checkpoint path for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Path to checkpoint file or None if not found
        """
        metadata = self.get_agent_metadata(agent_id)
        if metadata and metadata.checkpoint_path:
            checkpoint_path = Path(metadata.checkpoint_path.replace("\\", "/"))
            if checkpoint_path.exists():
                return str(checkpoint_path)
        return None

    def get_agent_config_path(self, agent_id: str) -> Optional[str]:
        """
        Get the config path for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Path to config file or None if not found
        """
        metadata = self.get_agent_metadata(agent_id)
        if metadata and metadata.config_path:
            config_path = Path(metadata.config_path.replace("\\", "/"))
            if config_path.exists():
                return str(config_path)
        return None

    def add_agent_tag(self, agent_id: str, tag: str) -> bool:
        if agent_id not in self.registry:
            return False

        registry_entry = self.registry[agent_id]
        if tag not in registry_entry.tags:
            registry_entry.tags.append(tag)
            self._save_registry()
            self._update_metadata_file(agent_id)
        return True

    def remove_agent_tag(self, agent_id: str, tag: str) -> bool:
        if agent_id not in self.registry:
            return False

        registry_entry = self.registry[agent_id]
        if tag in registry_entry.tags:
            registry_entry.tags.remove(tag)
            self._save_registry()
            self._update_metadata_file(agent_id)
        return True

    def update_agent_rating(self, agent_id: str, rating: Rating) -> bool:
        if agent_id not in self.registry:
            return False

        self.registry[agent_id].rating = rating
        self._save_registry()
        self._update_metadata_file(agent_id)
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the archive.

        Args:
            agent_id: Agent identifier

        Returns:
            True if removed, False if not found
        """
        if agent_id not in self.registry:
            return False

        # Remove agent directory
        registry_entry = self.registry[agent_id]
        raw_dir = registry_entry.directory or ""
        if raw_dir:
            raw_path = Path(raw_dir.replace("\\", "/"))
            agent_dir = raw_path if raw_path.is_absolute() else (self.base_dir.parent / raw_path).resolve()
            if agent_dir.exists():
                shutil.rmtree(agent_dir)

        # Remove from registry
        del self.registry[agent_id]
        self._save_registry()

        return True
