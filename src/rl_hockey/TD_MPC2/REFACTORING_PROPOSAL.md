# TD_MPC2 Module Refactoring Proposal

## Current Problem

The `tdmpc2.py` file is over 1800 lines long (currently ~1808), making it difficult to:
- Navigate and understand the codebase
- Maintain and debug specific components
- Test individual functionalities
- Collaborate on different parts of the algorithm

## Proposed New Structure

```
src/rl_hockey/TD_MPC2/
├── __init__.py                              (export main classes)
│
├── core/
│   ├── __init__.py
│   ├── agent.py                             (main TDMPC2 agent class, initialization)
│   ├── training.py                          (training loop and loss computation)
│   ├── inference.py                         (act, act_batch, evaluate methods)
│   └── utils.py                             (helper functions, load_state_dict_compat)
│
├── models/
│   ├── __init__.py
│   ├── model_encoder.py                     (existing - Encoder)
│   ├── model_policy.py                      (existing - Policy)
│   ├── model_q_function.py                  (existing - Q function)
│   ├── model_q_ensemble.py                  (existing - Q ensemble)
│   ├── model_reward.py                      (existing - Reward)
│   ├── model_termination.py                 (existing - Termination)
│   ├── model_dynamics_simple.py             (existing - DynamicsSimple)
│   ├── model_dynamics_opponent.py           (existing - DynamicsOpponent)
│   ├── model_opponent_cloning.py            (existing - OpponentCloning)
│   ├── model_init.py                        (existing - initialization functions)
│   └── dynamics_wrapper.py                  (DynamicsWithOpponentWrapper)
│
├── training/
│   ├── __init__.py
│   ├── td_targets.py                        (TD target computation)
│   ├── policy_update.py                     (policy update logic)
│   ├── value_update.py                      (Q-function, reward, termination updates)
│   ├── model_update.py                      (encoder, dynamics updates)
│   └── optimizer_manager.py                 (optimizer setup and management)
│
├── planning/
│   ├── __init__.py
│   ├── mppi_planner_simple.py               (existing - MPPI planner)
│   └── rollout.py                           (dynamics rollout methods)
│
├── opponent/
│   ├── __init__.py
│   ├── opponent_manager.py                  (opponent simulation orchestration)
│   ├── cloning_trainer.py                   (opponent cloning training logic)
│   └── agent_loader.py                      (load opponent agents)
│
├── persistence/
│   ├── __init__.py
│   ├── checkpoint.py                        (save/load functionality)
│   └── model_io.py                          (model state dict operations)
│
├── utils/
│   ├── __init__.py
│   ├── util.py                              (existing - RunningScale, soft_ce, etc.)
│   └── logging.py                           (architecture logging, statistics)
│
└── legacy/
    └── tdmpc2_original.py                   (backup of original monolithic file)
```

## Buffer architecture (current codebase)

Refactoring must preserve the existing buffer setup:

- **Main replay buffer**: `TDMPC2ReplayBuffer` (from `rl_hockey.common.buffer`). Episode-based; stores transitions via `store(transition, winner=winner)`. When `done=True`, the trajectory is closed and optional win-reward shaping is applied. Training samples contiguous subsequences via `sample_sequences(batch_size, horizon)`. Configured with `buffer_device` (storage) and `device` (where batches are placed for training).
- **Opponent cloning buffers**: One `OpponentCloningBuffer` per opponent (dict `opponent_cloning_buffers`). Store (obs, opponent_action) pairs from `store_opponent_action` / `collect_opponent_demonstrations`. Used only in `_train_opponent_cloning`; no sampling from the main replay buffer for cloning.

## Opponent and agent considerations

Refactoring must preserve two distinct concepts and their integration points:

**Two opponent concepts**

1. **Curriculum/training opponent** (outside TD_MPC2): Who the agent plays against in the environment each episode. Handled by `rl_hockey.common.training.opponent_manager` (OpponentConfig, create_opponent, sample_opponent, self_play, weighted_mixture, basic_weak/strong). The training loop in `train_run.py` chooses this opponent per episode. This is not part of the TD_MPC2 refactor; it stays in common.

2. **TDMPC2 internal opponent simulation**: Reference opponents loaded from checkpoints (by type and path), used only for model-based rollouts and cloning. Configured via `opponent_simulation.opponent_agents` in the agent config. The agent loads these, maintains one cloning network and one cloning buffer per reference opponent, and uses them inside `DynamicsWithOpponentWrapper` during planning. The curriculum opponent (who we play in the env) can be different from these reference opponents; cloning buffers are filled by simulating what each *reference* opponent would do at each step (see below).

**Public API to preserve**

The refactored agent must keep these methods on the main TDMPC2 class so that `train_run.py` and any other callers continue to work:

- `collect_opponent_demonstrations(obs_agent2)`: Called every environment step (single-env and vectorized). Receives the opponent’s observation (player 2 view). The agent runs each loaded reference opponent on this observation and stores (obs, action) in that opponent’s cloning buffer. Refactoring may move implementation into `opponent/` but the public method must remain on the agent.
- `store_opponent_action(obs, opponent_action, opponent_id)`: Optional; for storing a single (obs, opponent_action) pair when the acting opponent is one of the loaded reference opponents. Kept for compatibility.
- `set_current_opponent(opponent_id)`: Forces which reference opponent is used in non-planning rollouts; planning still samples opponents randomly. Must remain on the agent.

**Agent loader (opponent/agent_loader.py)**

`_load_opponent_agent(opponent_type, opponent_path)` must continue to support all current types: TDMPC2, SAC, TD3, DECOYPOLICY. For TDMPC2 opponents, checkpoint hyperparameters (latent_dim, hidden_dim, etc.) are read from the checkpoint so the loaded agent matches the saved architecture. New opponent types should be added here (or via a small registry); the refactor should not hard-code only one type.

**Checkpoint and compatibility**

Opponent simulation state (opponent_agents list, opponent_cloning_networks, cloning optimizers, optional cloning buffers) is saved and loaded in the main checkpoint. Loading must handle: (1) checkpoint had opponent simulation enabled but current config does not, (2) current config enables it but checkpoint did not, (3) both enabled and same opponent list. This logic stays in persistence (checkpoint.py) but must be kept consistent with the opponent module.

**Summary**

- Do not merge or replace `common.training.opponent_manager` with TD_MPC2 opponent code; they serve different roles.
- Keep `collect_opponent_demonstrations`, `store_opponent_action`, and `set_current_opponent` on the TDMPC2 agent and preserve the observation convention (obs_agent2 for demonstrations).
- Ensure agent_loader supports all current opponent types and remains extensible.

## Detailed Breakdown

### 1. core/agent.py (200-300 lines)
**Purpose**: Main TDMPC2 agent class structure and initialization

**Content**:
- TDMPC2 class definition
- `__init__` method (model instantiation, optimizer setup, parameter initialization)
- `store_transition` method (calls `buffer.store(transition, winner=winner)` on TDMPC2ReplayBuffer)
- Basic agent interface methods
- Device and configuration management

**Dependencies**: All model modules, optimizer manager, checkpoint module, TDMPC2ReplayBuffer (main replay buffer), OpponentCloningBuffer (per-opponent cloning buffers in `opponent_cloning_buffers`)

### 2. core/training.py (300-400 lines)
**Purpose**: Main training loop and coordination

**Content**:
- `train` method (main training loop)
- Training step orchestration
- Loss aggregation and backward pass
- Gradient clipping and optimizer steps
- Training metrics and logging
- AMP (Automatic Mixed Precision) handling

**Dependencies**: All update modules (policy_update, value_update, model_update), td_targets

### 3. core/inference.py (200-300 lines)
**Purpose**: Action selection and evaluation

**Content**:
- `act` method (single observation)
- `act_with_stats` method (with detailed statistics)
- `act_batch` method (batch processing)
- `evaluate` method
- Exploration vs exploitation logic
- Planning integration

**Dependencies**: Planning module, models, dynamics wrapper

### 4. core/utils.py (50-100 lines)
**Purpose**: Core utility functions

**Content**:
- `_load_state_dict_compat` function
- Helper functions for batch processing
- Common utility functions used across core modules

### 5. models/dynamics_wrapper.py (50-100 lines)
**Purpose**: Dynamics model wrapper with opponent integration

**Content**:
- `DynamicsWithOpponentWrapper` class (currently in tdmpc2.py)
- Opponent action prediction integration
- Opponent sampling logic

**Dependencies**: model_dynamics_opponent, model_opponent_cloning

### 6. training/td_targets.py (50-100 lines)
**Purpose**: TD target computation

**Content**:
- `_compute_td_targets` method
- TD-lambda calculation
- Bootstrapping logic
- Two-hot encoding/decoding for distributional RL

**Dependencies**: models (Q-ensemble), util

### 7. training/policy_update.py (100-150 lines)
**Purpose**: Policy network updates

**Content**:
- `_update_policy` method
- Policy gradient computation
- Entropy regularization
- Policy loss calculation

**Dependencies**: models (policy, Q-ensemble), planning (rollout)

### 8. training/value_update.py (150-200 lines)
**Purpose**: Value function updates

**Content**:
- Q-function update logic
- Reward model update logic
- Termination model update logic
- Consistency loss computation
- Two-hot target preparation

**Dependencies**: models (Q-ensemble, reward, termination), td_targets

### 9. training/model_update.py (150-200 lines)
**Purpose**: World model updates

**Content**:
- Encoder update logic
- Dynamics model update logic
- Latent space consistency
- Model-based loss computation

**Dependencies**: models (encoder, dynamics)

### 10. training/optimizer_manager.py (100-150 lines)
**Purpose**: Optimizer setup and management

**Content**:
- Optimizer initialization for all models
- Learning rate scheduling
- Parameter grouping
- Gradient scaling for different model components

**Dependencies**: None (pure PyTorch optimizers)

### 11. planning/rollout.py (150-200 lines)
**Purpose**: Multi-step dynamics rollouts

**Content**:
- `rollout_dynamics_multi_step` method
- Trajectory generation
- Latent state evolution
- Reward and termination prediction along trajectories

**Dependencies**: models (encoder, dynamics, reward, termination, policy)

### 12. opponent/opponent_manager.py (100-150 lines)
**Purpose**: High-level internal opponent simulation orchestration (for rollouts and cloning; not the curriculum opponent used in the env)

**Content**:
- `_initialize_opponent_simulation` method (load opponents, create cloning networks and buffers)
- Opponent switching logic
- `set_current_opponent` (force which reference opponent is used in non-planning rollouts)
- Opponent state management

**Dependencies**: cloning_trainer, agent_loader

### 13. opponent/cloning_trainer.py (150-200 lines)
**Purpose**: Training opponent cloning networks

**Content**:
- `_train_opponent_cloning` method
- Cloning network training loop
- Loss computation for behavior cloning
- Training scheduling

**Dependencies**: model_opponent_cloning, OpponentCloningBuffer (per-opponent; one buffer per opponent in `opponent_cloning_buffers`, not the main replay buffer). Cloning training samples from these buffers only.

### 14. opponent/agent_loader.py (100-150 lines)
**Purpose**: Loading and managing opponent agents for internal simulation

**Content**:
- `_load_opponent_agent(opponent_type, opponent_path)` method
- Agent instantiation from checkpoints for types: TDMPC2, SAC, TD3, DECOYPOLICY (and any future types)
- For TDMPC2: read latent_dim, hidden_dim, etc. from checkpoint so architecture matches
- Compatibility checks and eval mode setup after load

**Dependencies**: TDMPC2, SAC, TD3, DecoyPolicy (lazy imports), checkpoint loading; not the common agent_factory (that is for curriculum/env opponents)

### 15. persistence/checkpoint.py (150-200 lines)
**Purpose**: Model saving and loading

**Content**:
- `save` method
- `load` method
- State dict management
- Checkpoint versioning
- Backward compatibility

**Dependencies**: All models, optimizer_manager

### 16. persistence/model_io.py (50-100 lines)
**Purpose**: Low-level model I/O operations

**Content**:
- State dict serialization
- Model parameter transfer
- Torch.compile compatibility handling

**Dependencies**: PyTorch

### 17. utils/logging.py (100-150 lines)
**Purpose**: Logging and diagnostics

**Content**:
- `log_architecture` method
- Parameter counting
- Training statistics collection
- Model visualization helpers

**Dependencies**: All models

## Migration Strategy

### Phase 1: Preparation
1. Create backup of original `tdmpc2.py` in `legacy/` folder
2. Create new directory structure
3. Set up `__init__.py` files with proper imports

### Phase 2: Extract Independent Components (Low Risk)
1. Move `DynamicsWithOpponentWrapper` to `models/dynamics_wrapper.py`
2. Move `_load_state_dict_compat` to `core/utils.py`
3. Create `training/optimizer_manager.py`
4. Create `utils/logging.py` with logging methods

### Phase 3: Extract Functional Components (Medium Risk)
1. Create `planning/rollout.py` with rollout methods
2. Create `training/td_targets.py` with TD target computation
3. Create `opponent/agent_loader.py` with loading logic
4. Create `persistence/model_io.py` with I/O helpers

### Phase 4: Split Training Logic (High Risk)
1. Create `training/policy_update.py`
2. Create `training/value_update.py`
3. Create `training/model_update.py`
4. Refactor `train` method to use these modules

### Phase 5: Extract Core Components (Highest Risk)
1. Create `core/inference.py` with act methods
2. Create `opponent/opponent_manager.py` and `opponent/cloning_trainer.py`
3. Create `persistence/checkpoint.py` with save/load
4. Create `core/training.py` with main training loop
5. Create `core/agent.py` with initialization and agent class

### Phase 6: Testing and Validation
1. Test each module independently
2. Run integration tests
3. Verify checkpoint compatibility
4. Performance benchmarking
5. Update documentation

## Benefits

1. **Improved Maintainability**: Each file has a clear, single responsibility
2. **Better Testing**: Can unit test individual components
3. **Easier Collaboration**: Multiple developers can work on different modules
4. **Enhanced Readability**: Smaller files are easier to understand
5. **Cleaner Imports**: Clear module boundaries
6. **Future Extensions**: Easier to add new features or variants

## Backward Compatibility

- Keep `tdmpc2.py` as a facade that imports and re-exports the refactored components
- Ensure checkpoint loading works with both old and new structure
- Provide migration guide for existing code using the old structure
- **Training entry points unchanged**: `train_single_run.py` and `train_run.py` continue to work as-is. They obtain the agent via `create_agent()` from `agent_factory.py`; no changes required there.
- **Agent base class unchanged**: TDMPC2 continues to inherit from `rl_hockey.common.agent.Agent` (in `src/rl_hockey/common/agent.py`). The refactor only splits the TD_MPC2 module; the base Agent interface (act, act_batch, store_transition, train, save, load, evaluate, hooks) and parent class stay in common.
- **Opponent-related public API unchanged**: `collect_opponent_demonstrations(obs_agent2)`, `store_opponent_action(obs, opponent_action, opponent_id)`, and `set_current_opponent(opponent_id)` remain on the TDMPC2 class so that `train_run.py` and any evaluation scripts keep working without change.

## Estimated Effort

- **Phase 1**: 1-2 hours
- **Phase 2**: 3-4 hours
- **Phase 3**: 4-6 hours
- **Phase 4**: 6-8 hours
- **Phase 5**: 8-10 hours
- **Phase 6**: 4-6 hours

**Total**: 26-36 hours of focused work

## Risk Mitigation

1. **Version Control**: Create a feature branch for refactoring
2. **Incremental Testing**: Test after each phase
3. **Backup**: Keep original file in legacy folder
4. **Documentation**: Update docs as we refactor
5. **Code Review**: Review each phase before proceeding

## Alternative: Minimal Refactoring

If full refactoring is too risky, a minimal approach could be:

```
src/rl_hockey/TD_MPC2/
├── tdmpc2.py                    (reduce to ~500 lines: class definition + coordination)
├── tdmpc2_training.py           (training loop and updates ~400 lines)
├── tdmpc2_inference.py          (act methods ~300 lines)
├── tdmpc2_opponent.py           (opponent simulation ~300 lines)
├── tdmpc2_persistence.py        (save/load ~200 lines)
└── [existing model files...]
```

This would reduce `tdmpc2.py` from ~1808 to ~500 lines while keeping changes minimal.
