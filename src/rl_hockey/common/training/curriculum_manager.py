import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class EnvironmentModeConfig:
    mode: str  # "NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"
    weight: float = 1.0  # Weight/probability for this mode in the mixture


@dataclass
class EnvironmentConfig:
    mode: Optional[str] = None  # "NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE" (for single mode)
    mixture: Optional[List[EnvironmentModeConfig]] = None  # List of modes with weights (for mixture)
    keep_mode: bool = True
    
    def get_mode_for_episode(self, episode: int, random_state=None) -> str:
        
        if self.mixture is not None:
            import numpy as np
            if random_state is None:
                random_state = np.random
            
            modes = [m.mode for m in self.mixture]
            weights = [m.weight for m in self.mixture]
            
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("Environment mixture weights sum to zero")
            normalized_weights = [w / total_weight for w in weights]
            
            return random_state.choice(modes, p=normalized_weights)
        else:
            if self.mode is None:
                raise ValueError("Either 'mode' or 'mixture' must be specified in environment config")
            return self.mode


@dataclass
class OpponentConfig:
    type: str  # "none", "basic_weak", "basic_strong", "self_play", "weighted_mixture", "archive", "run_checkpoints"
    weight: float = 1.0
    checkpoint: Optional[str] = None
    deterministic: bool = True
    opponents: Optional[List[Dict[str, Any]]] = None  # For weighted_mixture
    skill_range: float = 50.0  # For archive sampling
    distribution: Optional[Dict[str, float]] = None  # For archive sampling
    agent_type: Optional[str] = None  # Agent type for loading checkpoints: "SAC", "TD3", "TDMPC2", "DECOYPOLICY", etc.

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpponentConfig':
        return cls(
            type=data['type'],
            weight=data.get('weight', 1.0),
            checkpoint=data.get('checkpoint'),
            deterministic=data.get('deterministic', True),
            opponents=data.get('opponents'),
            skill_range=data.get('skill_range'),
            distribution=data.get('distribution'),
            agent_type=data.get('agent_type')
        )


@dataclass
class RewardShapingConfig:
    N: int = 600
    K: int = 400
    CLOSENESS_START: float = 20.0
    TOUCH_START: float = 15.0
    CLOSENESS_FINAL: float = 1.5
    TOUCH_FINAL: float = 1.0
    DIRECTION_FINAL: float = 2.0


@dataclass
class RewardBonusConfig:
    """Configuration for reward bonus parameters that can change over time within a phase.
    
    Similar to RewardShapingConfig, this allows transitioning from START to FINAL values
    over N + K episodes:
    - Episodes 0 to N-1: Use START values
    - Episodes N to N+K-1: Linearly interpolate from START to FINAL
    - Episodes N+K onwards: Use FINAL values
    """
    N: int = 0  # Number of episodes to use START values
    K: int = 2000  # Number of episodes to transition from START to FINAL
    WIN_BONUS_START: float = 10.0  # Initial win reward bonus
    WIN_BONUS_FINAL: float = 1.0  # Final win reward bonus
    WIN_DISCOUNT_START: float = 0.92  # Initial win reward discount
    WIN_DISCOUNT_FINAL: float = 0.92  # Final win reward discount (usually kept same)


@dataclass
class PhaseConfig:
    name: str
    episodes: int
    environment: EnvironmentConfig
    opponent: OpponentConfig
    reward_shaping: Optional[RewardShapingConfig] = None
    clear_buffer: bool = True
    reward_bonus: Optional[RewardBonusConfig] = None


@dataclass
class AgentConfig:
    type: str  # "DDDQN", "SAC", "TD3"
    hyperparameters: Dict[str, Any]
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Single training config used by curriculum, train_run_refactored, and train.py."""
    warmup_steps: int = 10_000
    updates_per_step: int = 1
    eval_frequency: int = 100_000
    checkpoint_frequency: int = 100_000
    reward_scale: float = 1.0
    max_episode_steps: int = 500
    checkpoint_save_freq: int = 100
    train_freq: int = 1
    resource_log_freq: int = 10
    episode_resource_window: int = 10
    episode_resource_samples: int = 5


@dataclass
class CurriculumConfig:
    phases: List[PhaseConfig]
    hyperparameters: Dict[str, Any]
    training: TrainingConfig
    agent: AgentConfig


def load_curriculum(config_path: str) -> CurriculumConfig:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return _parse_config(config_dict)


def _parse_agent_only_config(config_dict: Dict[str, Any]) -> CurriculumConfig:
    """Parse a minimal config that only has 'agent' and 'hyperparameters' (e.g. archive agent configs)."""
    agent_dict = config_dict['agent']
    agent_config = AgentConfig(
        type=agent_dict['type'],
        hyperparameters=agent_dict.get('hyperparameters', {}),
        checkpoint_path=agent_dict.get('checkpoint_path')
    )
    return CurriculumConfig(
        phases=[],
        hyperparameters=config_dict.get('hyperparameters', {}),
        training=TrainingConfig(),
        agent=agent_config
    )


def _parse_config(config_dict: Dict[str, Any]) -> CurriculumConfig:
    phases = []
    if 'curriculum' not in config_dict:
        return _parse_agent_only_config(config_dict)
    for phase_dict in config_dict['curriculum']['phases']:
        env_dict = phase_dict['environment']
        
        if 'mixture' in env_dict:
            mixture_list = []
            for mix_item in env_dict['mixture']:
                if isinstance(mix_item, dict):
                    mixture_list.append(EnvironmentModeConfig(
                        mode=mix_item['mode'],
                        weight=mix_item.get('weight', 1.0)
                    ))
                else:
                    raise ValueError("Mixture items must be dictionaries with 'mode' and optional 'weight'")
            
            env_config = EnvironmentConfig(
                mode=None,
                mixture=mixture_list,
                keep_mode=env_dict.get('keep_mode', True)
            )
        else:
            env_config = EnvironmentConfig(
                mode=env_dict.get('mode'),
                mixture=None,
                keep_mode=env_dict.get('keep_mode', True)
            )
        
        opponent_dict = phase_dict['opponent']
        opponent_config = OpponentConfig(
            type=opponent_dict['type'],
            weight=opponent_dict.get('weight', 1.0),
            checkpoint=opponent_dict.get('checkpoint'),
            deterministic=opponent_dict.get('deterministic', True),
            opponents=opponent_dict.get('opponents'),
            skill_range=opponent_dict.get('skill_range', 50),
            distribution=opponent_dict.get('distribution'),
            agent_type=opponent_dict.get('agent_type')
        )
        
        reward_shaping_dict = phase_dict.get('reward_shaping')
        if reward_shaping_dict is None:
            reward_shaping = None
        else:
            reward_shaping = RewardShapingConfig(**reward_shaping_dict)
        
        reward_bonus_dict = phase_dict.get('reward_bonus')
        if reward_bonus_dict is None:
            reward_bonus = None
        else:
            reward_bonus = RewardBonusConfig(**reward_bonus_dict)
        
        phase = PhaseConfig(
            name=phase_dict['name'],
            episodes=phase_dict['episodes'],
            environment=env_config,
            opponent=opponent_config,
            reward_shaping=reward_shaping,
            clear_buffer=phase_dict.get('clear_buffer', True),
            reward_bonus=reward_bonus
        )
        phases.append(phase)
    
    agent_dict = config_dict['agent']
    agent_config = AgentConfig(
        type=agent_dict['type'],
        hyperparameters=agent_dict.get('hyperparameters', {}),
        checkpoint_path=agent_dict.get('checkpoint_path')
    )

    training_raw = config_dict.get('training', {})
    training_dict = training_raw if isinstance(training_raw, dict) else {}
    training_config = TrainingConfig(
        warmup_steps=training_dict.get('warmup_steps', 10_000),
        updates_per_step=training_dict.get('updates_per_step', 1),
        eval_frequency=training_dict.get('eval_frequency', 100_000),
        checkpoint_frequency=training_dict.get('checkpoint_frequency', 100_000),
        reward_scale=training_dict.get('reward_scale', 1.0),
        max_episode_steps=training_dict.get('max_episode_steps', 500),
        checkpoint_save_freq=training_dict.get('checkpoint_save_freq', 100),
        train_freq=training_dict.get('train_freq', 1),
        resource_log_freq=training_dict.get('resource_log_freq', 10),
        episode_resource_window=training_dict.get('episode_resource_window', 10),
        episode_resource_samples=training_dict.get('episode_resource_samples', 5),
    )
    
    curriculum_config = CurriculumConfig(
        phases=phases,
        hyperparameters=config_dict.get('hyperparameters', {}),
        training=training_config,
        agent=agent_config
    )
    
    return curriculum_config


def get_phase_for_episode(curriculum: CurriculumConfig, global_episode: int) -> Tuple[int, int, PhaseConfig]:
    episode_count = 0
    for phase_idx, phase in enumerate(curriculum.phases):
        if global_episode < episode_count + phase.episodes:
            phase_local_episode = global_episode - episode_count
            return phase_idx, phase_local_episode, phase
        episode_count += phase.episodes
    
    last_phase = curriculum.phases[-1]
    return len(curriculum.phases) - 1, last_phase.episodes - 1, last_phase


def get_total_episodes(curriculum: CurriculumConfig) -> int:
    return sum(phase.episodes for phase in curriculum.phases)


def get_current_phase(curriculum: CurriculumConfig, global_episode: int) -> PhaseConfig:
    _, _, phase = get_phase_for_episode(curriculum, global_episode)
    return phase
