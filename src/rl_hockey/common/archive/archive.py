"""
Archive Manager, handles storage and retrieval of trained agents.

This module provides a centralized archive for storing trained agents with metadata,
enabling easy retrieval for self-play training and evaluation.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Archive:
    """Manages the agent archive directory and metadata."""
    
    def __init__(self, base_dir: str = "results/archive"):
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
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"agents": {}}
            self._save_registry()
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _generate_agent_id(self, agent_name: str) -> str:
        """
        Generate a unique agent ID.
        
        Args:
            agent_name: Optional agent name to include in ID
            
        Returns:
            Unique agent ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_count = len(self.registry["agents"]) + 1
        
        return f"{agent_count:04d}_{agent_name}_{timestamp}"

    
    def add_agent(
        self,
        checkpoint_path: str,
        config_path: str,
        agent_name: str,
        tags: List[str] = [],
        rating: Dict[str, float] = None,
        step: int = None,
        metadata: Dict[str, Any] = None,
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
            rating = {
                "rating": 0.0,
                "mu": 25.0,
                "sigma": 8.333,
                "games_played": 0
            }
            tags.append("needs_calibration")
        
        # Prepare metadata
        agent_metadata = {
            "agent_id": agent_id,
            "archived_at": datetime.now().isoformat(),
            "tags": tags,
            "step": step,
            "rating": rating,
            "checkpoint_path": str(checkpoint_dest),
            "config_path": str(config_dest) if config_path else None,
        }
        
        # Add any additional metadata fields
        if metadata:
            for key, value in metadata.items():
                    agent_metadata[key] = value
        
        # Save metadata
        metadata_path = agent_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(agent_metadata, f, indent=2)
        
        # Update registry
        self.registry["agents"][agent_id] = {
            "agent_id": agent_id,
            "archived_at": agent_metadata["archived_at"],
            "tags": tags,
            "step": step,
            "rating": rating,
            "directory": str(agent_dir),
        }
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
        self.registry["agents"][agent_name] = {
            "agent_id": agent_name,
            "archived_at": datetime.now().isoformat(),
            "tags": ["baseline"],
            "rating": rating,
        }
        self._save_registry()
        
        return agent_name
    
    def get_agents(self, sort_by: str = "archived_at", tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List all archived agents.
        
        Args:
            sort_by: Field to sort by ("archived_at", "agent_id", "rating")
            tags: List of tags to filter agents by (optional)
        Returns:
            List of agent metadata dictionaries
        """
        agents = []
        
        for agent_id, registry_entry in self.registry["agents"].items():
            if tags:
                if not all(tag in registry_entry["tags"] for tag in tags):
                    continue

            agent_dir = Path(registry_entry["directory"])
            metadata_path = agent_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    agents.append(metadata)
        
        # Sort agents
        if sort_by == "archived_at":
            agents.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
        elif sort_by == "agent_id":
            agents.sort(key=lambda x: x.get("agent_id", ""))
        elif sort_by == "rating":
            agents.sort(key=lambda x: x.get("rating", {}).get("rating", 0), reverse=True)
        
        return agents
    
    def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent metadata dictionary or None if not found
        """
        if agent_id not in self.registry["agents"]:
            return None
        
        agent_dir = Path(self.registry["agents"][agent_id]["directory"])
        metadata_path = agent_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_agent_checkpoint_path(self, agent_id: str) -> Optional[str]:
        """
        Get the checkpoint path for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Path to checkpoint file or None if not found
        """
        metadata = self.get_agent_metadata(agent_id)
        if metadata:
            checkpoint_path = Path(metadata["checkpoint_path"])
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
        if metadata and metadata.get("config_path"):
            config_path = Path(metadata["config_path"])
            if config_path.exists():
                return str(config_path)
        return None
    
    def add_agent_tag(self, agent_id: str, tag: str) -> bool:
        if agent_id not in self.registry["agents"]:
            return False
        
        if tag not in self.registry["agents"][agent_id]["tags"]:
            self.registry["agents"][agent_id]["tags"].append(tag)
            self._save_registry()
        return True
    
    def remove_agent_tag(self, agent_id: str, tag: str) -> bool:
        if agent_id not in self.registry["agents"]:
            return False
        
        if tag in self.registry["agents"][agent_id]["tags"]:
            self.registry["agents"][agent_id]["tags"].remove(tag)
            self._save_registry()
        return True
    
    def update_agent_rating(self, agent_id: str, rating: Dict[str, float]) -> bool:
        if agent_id not in self.registry["agents"]:
            return False
        
        self.registry["agents"][agent_id]["rating"] = rating
        self._save_registry()
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the archive.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if removed, False if not found
        """
        if agent_id not in self.registry["agents"]:
            return False
        
        # Remove agent directory
        agent_dir = Path(self.registry["agents"][agent_id]["directory"])
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
        
        # Remove from registry
        del self.registry["agents"][agent_id]
        self._save_registry()
        
        return True
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the archive.
        
        Returns:
            Dictionary with archive statistics
        """
        return {
            "total_agents": len(self.registry["agents"]),
            "archive_directory": str(self.base_dir),
        }
