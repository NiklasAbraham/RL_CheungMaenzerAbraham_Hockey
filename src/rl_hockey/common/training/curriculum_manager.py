import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings."""
    mode: str  # "NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"
    keep_mode: bool = True


@dataclass
class OpponentConfig:
    """Configuration for opponent settings."""
    type: str  # "none", "basic_weak", "basic_strong", "self_play", "weighted_mixture"
    weight: float = 1.0
    checkpoint: Optional[str] = None
    deterministic: bool = True
    opponents: Optional[List[Dict[str, Any]]] = None  # For weighted_mixture


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping parameters."""
    N: int = 600
    K: int = 400
    CLOSENESS_START: float = 20.0
    TOUCH_START: float = 15.0
    CLOSENESS_FINAL: float = 1.5
    TOUCH_FINAL: float = 1.0
    DIRECTION_FINAL: float = 2.0


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    episodes: int
    environment: EnvironmentConfig
    opponent: OpponentConfig
    reward_shaping: Optional[RewardShapingConfig] = None


@dataclass
class AgentConfig:
    """Configuration for agent algorithm and hyperparameters."""
    type: str  # "DDDQN", "SAC", "TD3"
    hyperparameters: Dict[str, Any]


@dataclass
class CurriculumConfig:
    """Complete curriculum configuration."""
    phases: List[PhaseConfig]
    hyperparameters: Dict[str, Any]  # Common hyperparameters (learning_rate, batch_size)
    training: Dict[str, Any]
    agent: AgentConfig


def load_curriculum(config_path: str) -> CurriculumConfig:
    """Load curriculum configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return _parse_config(config_dict)


def _parse_config(config_dict: Dict[str, Any]) -> CurriculumConfig:
    """Parse configuration dictionary into dataclasses."""
    # Parse phases
    phases = []
    for phase_dict in config_dict['curriculum']['phases']:
        env_config = EnvironmentConfig(**phase_dict['environment'])
        
        opponent_dict = phase_dict['opponent']
        opponent_config = OpponentConfig(
            type=opponent_dict['type'],
            weight=opponent_dict.get('weight', 1.0),
            checkpoint=opponent_dict.get('checkpoint'),
            deterministic=opponent_dict.get('deterministic', True),
            opponents=opponent_dict.get('opponents')
        )
        
        # Handle optional reward shaping
        reward_shaping_dict = phase_dict.get('reward_shaping')
        if reward_shaping_dict is None:
            reward_shaping = None
        else:
            reward_shaping = RewardShapingConfig(**reward_shaping_dict)
        
        phase = PhaseConfig(
            name=phase_dict['name'],
            episodes=phase_dict['episodes'],
            environment=env_config,
            opponent=opponent_config,
            reward_shaping=reward_shaping
        )
        phases.append(phase)
    
    # Parse agent config
    agent_dict = config_dict['agent']
    agent_config = AgentConfig(
        type=agent_dict['type'],
        hyperparameters=agent_dict.get('hyperparameters', {})
    )
    
    # Create curriculum config
    curriculum_config = CurriculumConfig(
        phases=phases,
        hyperparameters=config_dict.get('hyperparameters', {}),
        training=config_dict.get('training', {}),
        agent=agent_config
    )
    
    return curriculum_config


def get_phase_for_episode(curriculum: CurriculumConfig, global_episode: int) -> Tuple[int, int, PhaseConfig]:
    """Get the current phase for a given global episode number."""
    episode_count = 0
    for phase_idx, phase in enumerate(curriculum.phases):
        if global_episode < episode_count + phase.episodes:
            phase_local_episode = global_episode - episode_count
            return phase_idx, phase_local_episode, phase
        episode_count += phase.episodes
    
    # If beyond all phases, use last phase
    last_phase = curriculum.phases[-1]
    return len(curriculum.phases) - 1, last_phase.episodes - 1, last_phase


def get_total_episodes(curriculum: CurriculumConfig) -> int:
    """Get total number of episodes across all phases."""
    return sum(phase.episodes for phase in curriculum.phases)


def get_current_phase(curriculum: CurriculumConfig, global_episode: int) -> PhaseConfig:
    """Get current phase configuration for given episode."""
    _, _, phase = get_phase_for_episode(curriculum, global_episode)
    return phase
