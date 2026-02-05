# TD-MPC2 with Opponent Simulation

## Overview

This document explains how the TD-MPC2 agent implementation integrates opponent simulation to learn robust policies in the hockey environment. The opponent simulation feature allows the agent to model and predict opponent behavior during planning and training.

## Architecture

### Refactored Structure

The TD-MPC2 implementation has been refactored into modular components:

```
src/rl_hockey/TD_MPC2/
├── core/                      # Main agent class and core functionality
│   ├── agent.py              # TDMPC2 main class
│   ├── training.py           # Training loop
│   └── inference.py          # Action selection and planning
├── model_definition/          # Neural network architectures
│   ├── dynamics_wrapper.py  # Dynamics with opponent integration
│   └── model_dynamics_opponent.py
├── opponent/                  # Opponent simulation system
│   ├── opponent_manager.py  # High-level orchestration
│   ├── agent_loader.py      # Loading opponent agents
│   └── cloning_trainer.py   # Training opponent cloning networks
├── planning/                  # Planning and rollout methods
│   └── rollout.py
├── training/                  # Training update logic
└── persistence/              # Checkpoint save/load
```

The main `tdmpc2.py` file acts as a facade for backward compatibility, re-exporting the `TDMPC2` class.

## Key Components

### 1. Opponent Simulation Configuration

Opponent simulation is enabled through initialization parameters:

```python
agent = TDMPC2(
    opponent_simulation_enabled=True,
    opponent_cloning_frequency=5000,      # Train cloning networks every N steps
    opponent_cloning_steps=20,            # Gradient steps per cloning training
    opponent_cloning_samples=512,         # Samples for cloning training
    opponent_agents=[                     # List of opponent agents to simulate
        {
            "type": "SAC",
            "path": "path/to/sac_checkpoint.pt"
        },
        {
            "type": "TDMPC2",
            "path": "path/to/tdmpc2_checkpoint.pt"
        }
    ]
)
```

Supported opponent types: `TDMPC2`, `SAC`, `TD3`, `DECOYPOLICY`

### 2. Dynamics Model Selection

The agent automatically selects the appropriate dynamics model based on whether opponent simulation is enabled:

**Without Opponent Simulation:**
```python
self.dynamics = DynamicsSimple(latent_dim, action_dim, ...)
# Predicts: z_{t+1} = f(z_t, a_t)
```

**With Opponent Simulation:**
```python
self.dynamics = DynamicsOpponent(latent_dim, action_dim, action_opponent_dim, ...)
# Predicts: z_{t+1} = f(z_t, a_t, a_opponent_t)
```

The `DynamicsOpponent` model takes both the agent's action and the opponent's action as input to predict the next latent state.

### 3. Opponent Loading and Cloning Networks

#### Opponent Loading Process

When the agent is initialized with opponent simulation enabled, the system performs the following steps:

1. **Load Reference Agents** (`opponent/agent_loader.py`):
   - Each opponent checkpoint is loaded from disk
   - Opponent hyperparameters are extracted from checkpoint
   - Agents are created with matching architectures
   - All networks are set to evaluation mode

2. **Create Cloning Networks** (`opponent/opponent_manager.py`):
   - For each opponent, a dedicated `OpponentCloning` network is created
   - These are lightweight policy networks that map latent states to actions
   - Each network has its own optimizer
   - Architecture: latent_dim → hidden_layers → action_dim (Gaussian policy)

3. **Create Demonstration Buffers**:
   - Each opponent gets an `OpponentCloningBuffer`
   - Stores (observation, action) pairs collected during episodes
   - Used to train cloning networks via behavioral cloning

### 4. Demonstration Collection

During environment interaction, demonstrations are collected from reference opponents:

```python
# Called every environment step with opponent's observation
agent.collect_opponent_demonstrations(obs_agent2)
```

**Collection Process** (`opponent/opponent_manager.py:collect_opponent_demonstrations`):

1. Takes the current observation from the opponent's perspective
2. For each loaded reference opponent:
   - Runs the opponent's policy on the observation (deterministic mode)
   - Extracts the action the opponent would take
   - Stores (obs, action) pair in that opponent's cloning buffer
3. This happens in parallel for all reference opponents, regardless of who the actual training opponent is

**Key Insight:** Even if the agent is playing against opponent A, it collects demonstrations from all loaded opponents (A, B, C, etc.) by simulating what each would do in the current state.

### 5. Cloning Network Training

Periodically during training (every `opponent_cloning_frequency` steps), the agent trains its cloning networks:

**Training Process** (`opponent/cloning_trainer.py:train_opponent_cloning`):

1. **Sample Demonstrations**: Sample N observations and corresponding actions from each opponent's buffer
2. **Encode Observations**: Use the agent's frozen encoder to convert observations to latent states
   ```python
   with torch.no_grad():
       latent_all = agent.encoder(obs_all)
   ```
3. **Train via Behavioral Cloning**: For each opponent's cloning network:
   - Run multiple gradient steps (default: 20)
   - Minimize MSE between cloned actions and reference actions
   ```python
   cloned_actions = cloning_network.mean_action(latent_mb)
   loss = F.mse_loss(cloned_actions, target_mb)
   ```
4. **Preserve Encoder Gradients**: Temporarily disable encoder gradients during cloning training

**Result:** Each cloning network learns to approximate its corresponding reference opponent's policy in latent space.

### 6. Dynamics Wrapper

The `DynamicsWithOpponentWrapper` integrates cloning networks with the dynamics model:

**Core Mechanism** (`model_definition/dynamics_wrapper.py`):

```python
def forward(self, latent, action):
    # Select opponent (randomly or forced)
    if self.force_opponent_id is not None:
        opponent_id = self.force_opponent_id
    else:
        # Random sampling during planning
        opponent_id = random.choice(self.opponent_ids)
    
    # Predict opponent action using cloning network
    cloning_network = self.opponent_cloning_networks[opponent_id]["network"]
    with torch.no_grad():
        opponent_action = cloning_network.mean_action(latent)
    
    # Forward through dynamics model
    return self.dynamics_opponent(latent, action, opponent_action)
```

**Two Modes:**

1. **Planning Mode** (during MPC): Randomly samples opponents for each trajectory
   - Provides robustness by exposing the planner to diverse opponent behaviors
   - Each planned trajectory may face a different opponent

2. **Forced Mode** (during rollouts): Uses a specific opponent when set
   - Used for debugging or evaluating against specific opponents
   - Activated via `agent.set_current_opponent(opponent_id)`

### 7. Training Integration

During world model training, opponent actions are incorporated:

**Consistency Loss Computation** (`core/training.py`):

```python
for t in range(horizon):
    a_t = actions_seq[:, t]  # Agent's action from replay buffer
    z_target = z_seq[:, t + 1].detach()  # Target latent from encoder
    
    if agent.opponent_simulation_enabled:
        # Predict opponent action
        cloning_network = agent.opponent_cloning_networks[opponent_id]["network"]
        with torch.no_grad():
            opponent_action_t = cloning_network.mean_action(z_pred)
        
        # Predict next latent with opponent action
        z_next_pred = agent.dynamics(z_pred, a_t, opponent_action_t)
    else:
        z_next_pred = agent.dynamics(z_pred, a_t)
    
    # Compute consistency loss
    consistency_loss += lambda_weights[t] * F.mse_loss(z_next_pred, z_target)
```

**Key Points:**

- The agent's actions come from the replay buffer (what was actually executed)
- Opponent actions are predicted from the current latent state using cloning networks
- The dynamics model learns to predict state transitions given both actions
- TD-λ weighting emphasizes near-term predictions

### 8. Planning with Opponent Simulation

During action selection, the planner uses the dynamics wrapper:

**MPPI Planning** (`model_definition/model_planner.py`):

1. **Initialize Trajectories**: Sample action sequences from policy + noise
2. **Rollout Dynamics**: For each trajectory:
   ```python
   z = initial_latent
   for t in range(horizon):
       # Dynamics wrapper randomly selects an opponent
       z_next = dynamics_wrapper(z, action[t])
       # Evaluate reward and value
       reward = reward_model(z_next, action[t])
       value = q_ensemble(z_next, action[t])
   ```
3. **Compute Returns**: Aggregate rewards and terminal values
4. **Reweight Trajectories**: Higher-return trajectories get higher weight
5. **Update Action Distribution**: Move mean toward high-performing actions
6. **Iterate**: Repeat refinement for N iterations

**Opponent Diversity During Planning:**
- Each rollout may face a different opponent (random sampling)
- Forces the planner to find robust actions that work against multiple opponent styles
- Reduces overfitting to a single opponent's behavior

### 9. Multi-Step Rollouts

The agent can perform explicit multi-step rollouts for evaluation or debugging:

```python
future_latents = agent.rollout_dynamics_multi_step(
    z0=initial_latent,
    action_sequence=[action_1, action_2, action_3],
    max_horizon=3,
    opponent_id=None  # Optional: force specific opponent
)
# Returns: {1: z_1, 2: z_2, 3: z_3}
```

This function is useful for:
- Visualizing predicted trajectories
- Debugging world model accuracy
- Evaluating predictions against specific opponents

## Workflow Summary

### Initialization
1. Load reference opponent agents from checkpoints
2. Create cloning networks (one per opponent)
3. Create demonstration buffers (one per opponent)
4. Wrap dynamics model with `DynamicsWithOpponentWrapper`

### During Episode Collection
1. Agent observes state and selects action via MPC
2. Environment executes action (agent + opponent)
3. **Collect demonstrations**: Run all reference opponents on current obs, store (obs, action) in their buffers
4. Store transition in replay buffer

### During Training
1. Sample trajectory sequences from replay buffer
2. Encode observations to latent states
3. **Rollout dynamics** with opponent actions:
   - Use cloning networks to predict opponent actions
   - Predict next latents: z_{t+1} = f(z_t, a_agent_t, a_opponent_t)
4. Compute losses (consistency, reward, value, termination)
5. Update world model and policy
6. **Periodically**: Train cloning networks on demonstrations (every N steps)
7. Update target networks

### During Planning (MPC)
1. Initialize from current observation's latent
2. Sample action sequences
3. **Rollout with diverse opponents**:
   - For each trajectory, randomly sample opponent per step
   - Predict opponent actions via cloning networks
   - Predict state transitions
4. Evaluate trajectories and select best action

## Benefits of Opponent Simulation

1. **Robust Planning**: Agent considers diverse opponent behaviors during planning
2. **Accurate World Model**: Dynamics model learns to account for opponent influence
3. **Adaptability**: Agent can generalize to unseen opponents that behave similarly to reference opponents
4. **Curriculum Learning**: Can progressively introduce stronger opponents
5. **Reduced Overfitting**: Random opponent sampling during planning prevents exploitation of single opponent weaknesses

## Hyperparameters

Key hyperparameters for opponent simulation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opponent_simulation_enabled` | `False` | Enable opponent simulation |
| `opponent_cloning_frequency` | `5000` | Train cloning networks every N training steps |
| `opponent_cloning_steps` | `20` | Number of gradient steps per cloning training |
| `opponent_cloning_samples` | `512` | Number of demonstrations to sample for cloning |
| `opponent_agents` | `[]` | List of opponent configurations (type, path) |

## Limitations and Future Work

**Current Limitations:**

1. **Cloning Accuracy**: Cloning networks may not perfectly capture opponent behavior
2. **Computational Cost**: Running multiple opponents for demonstration collection adds overhead
3. **Buffer Size**: Opponent buffers have fixed capacity (50,000 transitions)
4. **Opponent Selection**: Currently uses uniform random sampling during planning

**Potential Improvements:**

1. **Adaptive Opponent Sampling**: Weight opponents by difficulty or uncertainty
2. **Online Opponent Discovery**: Automatically detect and model new opponent strategies
3. **Hierarchical Modeling**: Model opponent types/clusters rather than individuals
4. **Uncertainty Quantification**: Estimate confidence in opponent predictions
5. **Multi-Agent Dynamics**: Extend to more than two agents

## Example Configuration

```json
{
  "opponent_simulation_enabled": true,
  "opponent_cloning_frequency": 5000,
  "opponent_cloning_steps": 20,
  "opponent_cloning_samples": 512,
  "opponent_agents": [
    {
      "type": "SAC",
      "path": "checkpoints/sac_baseline.pt"
    },
    {
      "type": "TDMPC2",
      "path": "checkpoints/tdmpc2_intermediate.pt"
    },
    {
      "type": "TD3",
      "path": "checkpoints/td3_strong.pt"
    }
  ]
}
```

## Code References

### Key Files

- **Main Agent**: `src/rl_hockey/TD_MPC2/core/agent.py`
- **Opponent Manager**: `src/rl_hockey/TD_MPC2/opponent/opponent_manager.py`
- **Cloning Trainer**: `src/rl_hockey/TD_MPC2/opponent/cloning_trainer.py`
- **Dynamics Wrapper**: `src/rl_hockey/TD_MPC2/model_definition/dynamics_wrapper.py`
- **Training Loop**: `src/rl_hockey/TD_MPC2/core/training.py`
- **Agent Loader**: `src/rl_hockey/TD_MPC2/opponent/agent_loader.py`

### Key Functions

- `initialize_opponent_simulation()`: Sets up opponents and cloning networks
- `collect_opponent_demonstrations()`: Collects (obs, action) pairs during episodes
- `train_opponent_cloning()`: Trains cloning networks via behavioral cloning
- `DynamicsWithOpponentWrapper.forward()`: Predicts state transitions with opponent actions
- `rollout_dynamics_multi_step()`: Explicit multi-step rollouts with opponent simulation

## Conclusion

The opponent simulation feature transforms TD-MPC2 from a single-agent model-based RL algorithm into a robust multi-agent learning system. By explicitly modeling opponent behavior through cloning networks and integrating these predictions into the world model and planning process, the agent learns policies that are effective against diverse opponents rather than overfitting to a single opponent's weaknesses.

The modular design allows easy extension to new opponent types, alternative modeling approaches, and different sampling strategies during planning. The system maintains computational efficiency by using lightweight cloning networks in latent space and parallelizing demonstration collection across multiple reference opponents.
