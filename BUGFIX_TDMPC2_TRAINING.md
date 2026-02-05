# TDMPC2 Training Bugs Fixed - February 5, 2026

## Summary
The new `train.py` had 3 critical bugs that broke TDMPC2 training in vectorized mode. All bugs are now fixed.

## Bug #1: Missing `env_id` Parameter (CRITICAL)
**Impact:** Broke TDMPC2's episode-based buffer in vectorized training

### The Problem
TDMPC2ReplayBuffer has two storage modes:
1. **Single-env mode**: Uses `_current_obs`, `_current_actions` (backward compatible)
2. **Vectorized mode**: Uses `_env_episodes[env_id]` to track per-environment episodes

When `env_id` is NOT passed to `store_transition()`:
- Buffer falls back to single-env mode
- All parallel environments write to the SAME episode storage
- Episodes get mixed up (transitions from env 0, 1, 2, etc. all in one episode)
- Buffer stores corrupted episodes with wrong length and structure
- Training fails because episode sequences are broken

### The Fix
**Before (BROKEN):**
```python
agent.store_transition(
    (states[i], actions[i], scaled_reward, next_state_for_buffer, done),
    winner=winner,
)
```

**After (FIXED):**
```python
agent.store_transition(
    (states[i], actions[i], scaled_reward, next_state_for_buffer, done),
    winner=winner,
    env_id=i,  # Critical for TDMPC2 episodic buffer in vectorized mode
)
```

**Location:** Line 777-781 in train.py

---

## Bug #2: Wrong Episode Callback Timing
**Impact:** TDMPC2 agent callbacks received mismatched episode numbers

### The Problem
The new train.py had incorrect timing for `on_episode_start()`:

```python
# Episode completion handling
if dones[i] or truncs[i]:
    if hasattr(agent, "on_episode_end"):
        agent.on_episode_end(training_state.episode)  # Episode N (correct)
    
    training_state.episode += 1  # Now episode N+1
    
    # ... later in code ...
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start(training_state.episode)  # Episode N+1 (wrong!)
```

This meant:
- `on_episode_end(N)` called for episode N (correct)
- Episode counter incremented to N+1
- `on_episode_start(N+1)` called (wrong - should be called immediately after increment)

### The Fix
**After (FIXED):**
```python
# Episode completion handling
if dones[i] or truncs[i]:
    if hasattr(agent, "on_episode_end"):
        agent.on_episode_end(training_state.episode)  # Episode N
    
    training_state.episode += 1  # Now episode N+1
    
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start(training_state.episode)  # Episode N+1 (correct!)
```

Now callbacks are called in the right order with correct episode numbers.

**Location:** Lines 789-800 in train.py

---

## Bug #3: Redundant Initial Episode Callback
**Impact:** Minor - caused duplicate `on_episode_start(0)` call

### The Problem
The code called `on_episode_start(training_state.episode)` before the training loop:

```python
# Before the loop
if hasattr(agent, "on_episode_start"):
    agent.on_episode_start(training_state.episode)  # Episode 0

while training_state.episode < total_episodes:
    # ... training loop ...
    # When first episode completes, episode becomes 1
    # Then on_episode_start(1) is called
```

This was redundant because:
- Episodes start automatically when the environment resets
- The callback should be called when transitioning TO a new episode
- The first on_episode_start should happen naturally in the loop

### The Fix
Removed the redundant call before the loop. The callback is now only called when transitioning between episodes inside the loop.

**Location:** Lines 716-718 (removed) in train.py

---

## How the Bugs Manifested

### Symptoms
- Training appeared to run (no crashes)
- Buffer size grew slowly or incorrectly
- Agent didn't improve (or improved very slowly)
- Episode sequences were corrupted
- Metrics looked strange (inconsistent buffer size, wrong episode counts)

### Why It Seemed to Work
- The code didn't crash because buffer silently fell back to single-env mode
- Episodes were still stored, just with wrong structure
- Training continued, but with corrupted experience sequences
- Logs showed episodes completing and buffer growing

### Why It Didn't Actually Work
- TDMPC2 needs proper episode sequences for world model training
- Mixed-up episodes break the temporal consistency required for:
  - Dynamics model learning (predicts next state from current state+action)
  - Reward model learning (predicts reward from state+action)
  - Value learning (needs proper bootstrap targets)
- Without proper episodes, the agent can't learn the environment dynamics

---

## Verification

To verify the fix is working:

1. **Buffer growth**: Should grow by exactly episode length after each episode
   ```
   Episode 1: buffer=250 (first episode was 250 steps)
   Episode 2: buffer=520 (second episode was 270 steps, total 520)
   ```

2. **First episode log**: Should show clean episode completion
   ```
   First episode complete: length=250, buffer_size=250, done=True, trunc=False, winner=1
   ```

3. **Episode tracking**: Each environment should track episodes independently in vectorized mode

4. **Training metrics**: Should show consistent loss values after warmup

---

## Comparison with Working Version

The old `train_run_refactored.py` (which worked) had:
- ✅ Correct `env_id=i` parameter (line 1211)
- ✅ Proper callback timing (calls `on_episode_start` after completion handler)
- ✅ No redundant initial callback

The new `train.py` (which was broken) had:
- ❌ Missing `env_id` parameter
- ❌ Wrong callback timing (called after increment but in wrong place)
- ❌ Redundant initial callback

Now the new `train.py` has been fixed to match the working version's logic.

---

## Related Files
- `/src/rl_hockey/common/training/train.py` - Main training loop (FIXED)
- `/src/rl_hockey/common/buffer.py` - TDMPC2ReplayBuffer (unchanged - was correct)
- `/src/rl_hockey/TD_MPC2/tdmpc2.py` - TDMPC2 agent (unchanged - was correct)
- `/src/rl_hockey/common/training/a/train_run_refactored.py` - Old working version (reference)
