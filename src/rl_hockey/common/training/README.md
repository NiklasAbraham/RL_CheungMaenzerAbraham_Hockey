# Training Module Documentation

This module implements a flexible curriculum learning system for RL hockey training with support for multiple algorithms, opponents, and hyperparameter tuning.

---

## Interaction Diagrams

### Scenario 1: Single Training Run

```
┌─────────────────────────────────────────────────────────────┐
│                    train_run.py                             │
│  (Main Entry Point)                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► config_validator.py
                     │   └─► Validates JSON config
                     │
                     ├─► curriculum_manager.py
                     │   └─► load_curriculum()
                     │       └─► Returns CurriculumConfig
                     │
                     ├─► agent_factory.py
                     │   ├─► get_action_space_info()
                     │   └─► create_agent()
                     │       └─► Returns Agent (DDDQN/SAC)
                     │
                     │   Training Loop:
                     │   ├─► For each episode:
                     │   │   ├─► get_phase_for_episode()
                     │   │   │   └─► Returns current PhaseConfig
                     │   │   │
                     │   │   ├─► If phase changed:
                     │   │   │   ├─► Recreate environment
                     │   │   │   └─► opponent_manager.py
                     │   │   │       └─► sample_opponent()
                     │   │   │           └─► Returns Opponent
                     │   │   │
                     │   │   ├─► For each step:
                     │   │   │   ├─► agent.act() → action_p1
                     │   │   │   ├─► opponent_manager.get_opponent_action()
                     │   │   │   │   └─► Returns action_p2
                     │   │   │   ├─► env.step([action_p1, action_p2])
                     │   │   │   ├─► Apply reward shaping
                     │   │   │   └─► agent.store_transition()
                     │   │   │
                     │   │   └─► agent.train()
                     │   │
                     │   └─► Every N episodes:
                     │       └─► run_manager.save_checkpoint()
                     │
                     └─► run_manager.py
                         ├─► save_config()
                         ├─► save_rewards_csv(phases=...)
                         ├─► save_losses_csv()
                         ├─► save_plots()
                         └─► agent.save()
```

### Scenario 2: Hyperparameter Tuning

```
┌─────────────────────────────────────────────────────────────┐
│              hyperparameter_tuning.py                       │
│  (Main Entry Point)                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► load_base_config()
                     │   └─► Loads JSON with hyperparameter ranges
                     │
                     ├─► create_hyperparameter_grid()
                     │   └─► Generates all combinations
                     │       └─► Returns List[Dict] (configs)
                     │
                     │   For each config (parallel):
                     │   ├─► Create temp JSON file
                     │   │
                     │   └─► train_run.py
                     │       └─► (Same flow as Scenario 1)
                     │           └─► Each run uses different hyperparameters
                     │               but same curriculum phases
                     │
                     └─► Collect and summarize results
```

### Scenario 3: Self-Play Training

```
┌─────────────────────────────────────────────────────────────┐
│                    train_run.py                             │
│  (Phase with self-play opponent)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► get_phase_for_episode()
                     │   └─► Phase with opponent.type = "self_play"
                     │
                     ├─► opponent_manager.py
                     │   └─► sample_opponent()
                     │       │
                     │       └─► create_self_play_opponent()
                     │           │
                     │           ├─► If checkpoint == null:
                     │           │   └─► copy.deepcopy(agent)
                     │           │       └─► Fresh copy every episode
                     │           │
                     │           ├─► If checkpoint == "latest":
                     │           │   └─► Find latest in models_dir
                     │           │       └─► load_agent_checkpoint()
                     │           │
                     │           └─► If checkpoint == "path":
                     │               └─► load_agent_checkpoint(path)
                     │
                     │   Training Loop:
                     │   ├─► Each episode:
                     │   │   ├─► Create fresh opponent copy (if checkpoint==null)
                     │   │   ├─► opponent.act(obs_agent_two(), deterministic)
                     │   │   └─► Use opponent action as action_p2
                     │   │
                     │   └─► Every N episodes:
                     │       └─► run_manager.save_checkpoint()
                     │           └─► Saves for future "latest" usage
```

### Scenario 4: Weighted Mixture Opponents

```
┌─────────────────────────────────────────────────────────────┐
│                    train_run.py                             │
│  (Phase with weighted_mixture opponent)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► get_phase_for_episode()
                     │   └─► Phase with opponent.type = "weighted_mixture"
                     │
                     │   For each episode:
                     │   ├─► opponent_manager.py
                     │   │   └─► sample_opponent()
                     │   │       │
                     │   │       ├─► Normalize weights
                     │   │       ├─► np.random.choice(opponents, p=weights)
                     │   │       └─► create_opponent(sampled_config)
                     │   │           │
                     │   │           └─► Can be:
                     │   │               ├─► basic_weak
                     │   │               ├─► basic_strong
                     │   │               ├─► self_play
                     │   │               └─► none
                     │   │
                     │   └─► Use sampled opponent for episode
```

## Data Flow

### Configuration Loading Flow

```
JSON Config File
    │
    ├─► config_validator.py
    │   └─► validate_config()
    │       └─► Returns [] if valid, [errors] if invalid
    │
    └─► curriculum_manager.py
        └─► load_curriculum()
            └─► _parse_config()
                └─► Returns CurriculumConfig
                    ├─► phases: List[PhaseConfig]
                    ├─► hyperparameters: Dict
                    ├─► training: Dict
                    └─► agent: AgentConfig
```

### Agent Creation Flow

```
CurriculumConfig.agent
    │
    └─► agent_factory.py
        ├─► get_action_space_info(env, agent.type)
        │   └─► Returns (state_dim, action_dim, is_discrete)
        │
        └─► create_agent(agent_config, state_dim, action_dim, is_discrete, hyperparams)
            │
            ├─► If DDDQN:
            │   └─► DDDQN(state_dim, action_dim, **hyperparams)
            │
            └─► If SAC:
                └─► SAC(state_dim, action_dim, **hyperparams)
```

### Opponent Creation Flow

```
PhaseConfig.opponent
    │
    └─► opponent_manager.py
        │
        ├─► If type == "weighted_mixture":
        │   └─► sample_opponent()
        │       └─► Randomly sample based on weights
        │           └─► Recursively call create_opponent()
        │
        └─► create_opponent()
            │
            ├─► If type == "none":
            │   └─► Return None
            │
            ├─► If type == "basic_weak":
            │   └─► Return BasicOpponent(weak=True)
            │
            ├─► If type == "basic_strong":
            │   └─► Return BasicOpponent(weak=False)
            │
            └─► If type == "self_play":
                └─► create_self_play_opponent()
                    │
                    ├─► If checkpoint == null:
                    │   └─► copy.deepcopy(agent)
                    │
                    ├─► If checkpoint == "latest":
                    │   └─► _find_latest_checkpoint()
                    │       └─► load_agent_checkpoint()
                    │
                    └─► If checkpoint == "path":
                        └─► load_agent_checkpoint(path)
```