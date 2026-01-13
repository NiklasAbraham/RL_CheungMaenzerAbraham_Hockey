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

### Scenario 5: Vectorized Environment Training (num_envs > 1)

```
┌─────────────────────────────────────────────────────────────┐
│              _train_run_vectorized()                        │
│  (Vectorized Training with Parallel Environments)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─► Create VectorizedHockeyEnv
                     │   └─► num_envs parallel processes
                     │       ├─► Env0 (Process 1)
                     │       ├─► Env1 (Process 2)
                     │       ├─► Env2 (Process 3)
                     │       └─► Env3 (Process 4)
                     │
                     │   Training Loop (Synchronized Steps):
                     │   │
                     │   ┌─────────────────────────────────────────┐
                     │   │  Step 1: Collect Observations (Batched) │
                     │   └─────────────────────────────────────────┘
                     │           │
                     │           ├─► Get states from all envs
                     │           │   └─► states = [s0, s1, s2, s3]
                     │           │
                     │           ├─► GPU: agent.act_batch(states)
                     │           │   └─► Process all 4 observations
                     │           │       simultaneously on GPU
                     │           │       └─► Returns [a0, a1, a2, a3]
                     │           │
                     │           ├─► Get opponent actions (per env)
                     │           │   └─► [opp_a0, opp_a1, opp_a2, opp_a3]
                     │           │
                     │   ┌─────────────────────────────────────────┐
                     │   │  Step 2: Step All Environments (PARALLEL)│
                     │   └─────────────────────────────────────────┘
                     │           │
                     │           ├─► Send actions to all envs
                     │           │   ├─► Env0.step([a0, opp_a0])
                     │           │   ├─► Env1.step([a1, opp_a1])
                     │           │   ├─► Env2.step([a2, opp_a2])
                     │           │   └─► Env3.step([a3, opp_a3])
                     │           │
                     │           ├─► ⏸️  WAIT for ALL to finish
                     │           │   │
                     │           │   ├─► Env0: Done in 5ms ✅
                     │           │   ├─► Env1: Done in 8ms ✅
                     │           │   ├─► Env2: Done in 3ms ✅
                     │           │   └─► Env3: Done in 10ms ✅
                     │           │
                     │           └─► Collect results (blocking)
                     │               └─► [obs0, obs1, obs2, obs3]
                     │                   [rew0, rew1, rew2, rew3]
                     │                   [done0, done1, done2, done3]
                     │
                     │   ┌─────────────────────────────────────────┐
                     │   │  Step 3: Store Transitions (Sequential) │
                     │   └─────────────────────────────────────────┘
                     │           │
                     │           ├─► For each environment i:
                     │           │   ├─► Apply reward shaping
                     │           │   └─► agent.store_transition(
                     │           │       (states[i], actions[i], 
                     │           │        rewards[i], next_states[i], 
                     │           │        dones[i])
                     │           │   )
                     │           │
                     │           └─► Buffer now contains:
                     │               [Env0-transition, Env1-transition,
                     │                Env2-transition, Env3-transition]
                     │               (Interleaved, not sequential)
                     │
                     │   ┌─────────────────────────────────────────┐
                     │   │  Step 4: Train Agent (From Buffer)     │
                     │   └─────────────────────────────────────────┘
                     │           │
                     │           ├─► if steps >= warmup_steps:
                     │           │   └─► agent.train(updates_per_step)
                     │           │       └─► Sample random batch
                     │           │           from replay buffer
                     │           │           └─► Update network
                     │           │
                     │           └─► Continue to next iteration
                     │
                     │   Episode Completion Handling:
                     │   ├─► When env finishes (done=True):
                     │   │   ├─► Auto-reset in worker process
                     │   │   ├─► Track episode stats
                     │   │   └─► Continue (other envs keep running)
                     │   │
                     │   └─► Environments run independently:
                     │       ├─► Env0: Episode 1, Step 100
                     │       ├─► Env1: Episode 1, Step 50
                     │       ├─► Env2: Episode 2, Step 25 (finished Ep1)
                     │       └─► Env3: Episode 1, Step 150
                     │
                     └─► Key Benefits:
                         ├─► 1.4-2.4x faster training
                         ├─► Better GPU utilization (40-60% vs 12-15%)
                         ├─► Batched GPU operations (3-4x more efficient)
                         └─► Mathematically equivalent to parallel single envs
```

#### Vectorized Environment Timeline Example

```
Time T: Training Iteration
─────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│ GPU: Process Batch of Observations                          │
│ Input:  [obs0, obs1, obs2, obs3]  (shape: 4 × obs_dim)     │
│ Output: [act0, act1, act2, act3]  (shape: 4 × act_dim)     │
│ Time:   ~2ms (batched is 3-4x faster than 4 sequential)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ CPU: Step All Environments in Parallel                       │
│                                                              │
│  Env0 Process ──► [action0] ──► Step ──► 5ms ──► Done ✅     │
│  Env1 Process ──► [action1] ──► Step ──► 8ms ──► Done ✅    │
│  Env2 Process ──► [action2] ──► Step ──► 3ms ──► Done ✅    │
│  Env3 Process ──► [action3] ──► Step ──► 10ms ──► Done ✅   │
│                                                              │
│  ⏸️  Main process WAITS for slowest (Env3: 10ms)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Store Transitions in Replay Buffer                           │
│                                                              │
│  Buffer.append(Env0: s0, a0, r0, s0', done0)                │
│  Buffer.append(Env1: s1, a1, r1, s1', done1)                │
│  Buffer.append(Env2: s2, a2, r2, s2', done2)                │
│  Buffer.append(Env3: s3, a3, r3, s3', done3)                │
│                                                              │
│  Buffer order: [..., Env0, Env1, Env2, Env3, ...]          │
│  (Interleaved from different episodes)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Train Agent (if warmup complete)                             │
│                                                              │
│  agent.train()                                               │
│    ├─► Sample random batch from buffer                      │
│    ├─► Compute loss                                          │
│    └─► Update network weights                               │
│                                                              │
│  Note: Order in buffer doesn't matter (random sampling)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Continue to T+1
```

#### Key Synchronization Points

1. **Environment Stepping**: All environments must finish before proceeding
   - `vectorized_env.step()` is blocking
   - Waits for slowest environment to complete
   - Returns batched results: `(obs, rewards, dones, truncs, infos)`

2. **Transition Storage**: Sequential storage after all steps complete
   - Each environment's transition stored independently
   - Transitions from different episodes get interleaved in buffer
   - This is mathematically correct (off-policy algorithms sample randomly)

3. **Training**: Happens after all transitions stored
   - Agent trains from randomly sampled batch
   - Buffer order doesn't matter for learning
   - Same distribution of experiences as parallel single environments

#### Mathematical Equivalence

- **Same transitions**: Each (s, a, r, s', done) is identical to parallel single envs
- **Same distribution**: Experience distribution is equivalent
- **Different order**: Transitions interleaved in buffer (doesn't affect learning)
- **Faster collection**: 1.4-2.4x speedup through batching and parallelism

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