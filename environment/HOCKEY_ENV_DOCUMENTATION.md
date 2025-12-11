# Hockey Environment Documentation

## Overview

The Hockey Environment is a two-player air hockey simulation built on top of Gymnasium (formerly OpenAI Gym) and Box2D physics engine. It is designed for Reinforcement Learning research and was developed for the RL course at the University of Tübingen.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Game Rules and Mechanics](#game-rules-and-mechanics)
4. [Environment Modes](#environment-modes)
5. [Observation Space](#observation-space)
6. [Action Space](#action-space)
7. [Reward Structure](#reward-structure)
8. [Built-in Opponents](#built-in-opponents)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)
11. [Tips and Best Practices](#tips-and-best-practices)

## Installation

### Dependencies

The environment requires the following packages:

- gymnasium
- numpy
- box2d-py
- pygame

### Installation Methods

**Method 1: Install from GitHub**

```bash
python3 -m pip install git+https://github.com/martius-lab/hockey-env.git
```

**Method 2: Install from local directory**

If you have cloned the repository locally:

```bash
cd hockey-env
pip install -e .
```

**Method 3: Add to Pipfile**

```
hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}
```

### Verify Installation

```python
import hockey.hockey_env as h_env
import gymnasium as gym
import numpy as np

env = h_env.HockeyEnv()
print("Installation successful!")
env.close()
```

## Quick Start

### Basic Usage

```python
import numpy as np
import hockey.hockey_env as h_env

# Create the environment
env = h_env.HockeyEnv()

# Reset the environment
obs, info = env.reset()

# Run one episode
for _ in range(250):
    # Render the environment
    env.render()
    
    # Sample random actions for both players
    a1 = np.random.uniform(-1, 1, 4)
    a2 = np.random.uniform(-1, 1, 4)
    
    # Step the environment
    obs, reward, done, truncated, info = env.step(np.hstack([a1, a2]))
    
    if done or truncated:
        break

env.close()
```

### Using Gymnasium Registry

The environment can also be created via the Gymnasium registry:

```python
import gymnasium as gym

# Normal two-player mode
env = gym.make("Hockey-v0")

# One player vs. basic opponent
env = gym.make("Hockey-One-v0", mode=0, weak_opponent=True)
```

## Game Rules and Mechanics

Understanding the physics and rules of the hockey environment is crucial for developing effective agents.

### Playing Field Layout

The hockey rink is a rectangular playing field with the following structure:

**Dimensions:**
- Width: 10 units (VIEWPORT_W / SCALE = 600 / 60 = 10)
- Height: 8 units (VIEWPORT_H / SCALE = 480 / 60 = 8)
- Center position: (5.0, 4.0)

**Field Structure:**
```
┌─────────────────────────────────────────────────┐
│                  Top Wall                        │
├──┐                                          ┌───┤
│G │          Center Circle                   │ G │
│O │                 ●                        │ O │
│A │                                          │ A │
│L │         Player 1    ●    Player 2       │ L │
│1 │                                          │ 2 │
│  │                                          │   │
├──┘                                          └───┤
│                 Bottom Wall                      │
└─────────────────────────────────────────────────┘
```

### Goals

**Goal 1 (Left side - Player 1 defends):**
- Position: x ≈ 0.92 units from left edge
- Center: approximately (0.92, 4.0)
- Size: 75 pixels radius (1.25 units)
- Vertical range: approximately y ∈ [2.75, 5.25]

**Goal 2 (Right side - Player 2 defends):**
- Position: x ≈ 9.08 units from left edge
- Center: approximately (9.08, 4.0)
- Size: 75 pixels radius (1.25 units)
- Vertical range: approximately y ∈ [2.75, 5.25]

The goals are sensor bodies that detect when the puck enters them, triggering a win/loss condition.

### Players (Rackets)

Each player is represented by a racket-shaped polygon:

**Physical Properties:**
- Shape: Asymmetric polygon (front is pointed, back is wider)
- Mass: 200.0 / RACKETFACTOR (where RACKETFACTOR = 1.2)
- Friction: 1.0
- Maximum angle: ±60 degrees (π/3 radians)
- Maximum linear speed: 10 units/timestep (enforced by action application)

**Colors:**
- Player 1 (left): Red (RGB: 235, 98, 53)
- Player 2 (right): Blue (RGB: 93, 158, 199)

### The Puck

**Physical Properties:**
- Shape: Circle
- Radius: 13 pixels / SCALE ≈ 0.217 units
- Mass: 7.0
- Friction: 0.1
- Linear damping: 0.05 (gradually slows down)
- Restitution: 0.95 (very bouncy)
- Maximum speed: 25 units/timestep (enforced)
- Color: Black

The puck bounces off walls and players, and its speed is automatically limited to prevent unrealistic velocities.

### Movement Boundaries

Players have restricted movement areas to ensure fair play:

**Player 1 (Left player) Boundaries:**
- **Horizontal limits:** 
  - Cannot go left of x = 0.92 (approximately, left goal area)
  - Cannot cross center line: x ≤ 5.0 (CENTER_X)
  - If tries to cross center, gets pushed back with dampened velocity
- **Vertical limits:** 
  - Cannot go below y = 1.2
  - Cannot go above y = 6.8 (H - 1.2)
- **Playing zone:** Left half of the field

**Player 2 (Right player) Boundaries:**
- **Horizontal limits:** 
  - Cannot go right of x = 9.08 (approximately, right goal area)
  - Cannot cross center line: x ≥ 5.0 (CENTER_X)
  - If tries to cross center, gets pushed back with dampened velocity
- **Vertical limits:** 
  - Cannot go below y = 1.2
  - Cannot go above y = 6.8 (H - 1.2)
- **Playing zone:** Right half of the field

**Boundary Enforcement:**
When a player tries to move outside these boundaries:
1. The velocity in that direction is set to 0
2. A counter-force is applied to prevent further movement
3. Linear damping increases to 20.0 (normally 5.0) to slow the player down quickly

### Center Line Rule

The center line at x = 5.0 is a hard boundary:
- Neither player can cross into the opponent's half
- Attempting to cross results in:
  - Immediate velocity cancellation in x-direction
  - Strong opposing force proportional to velocity
  - Increased damping (20.0) to stop movement quickly
- This creates a defensive zone for each player

### Game Start and Reset

**Initial Positions:**

**NORMAL Mode:**
- Player 1: (W/5, H/2) ≈ (2.0, 4.0)
- Player 2: (4*W/5, H/2) ≈ (8.0, 4.0)
- Puck: Randomized near the starting player's side
  - If Player 1 starts: x ∈ [3.0, 4.0], y ∈ [3.0, 5.0] (random)
  - If Player 2 starts: x ∈ [6.0, 7.0], y ∈ [3.0, 5.0] (random)
- Starting player alternates each reset

**TRAIN_SHOOTING Mode:**
- Player 1: (W/5, H/2) ≈ (2.0, 4.0)
- Player 2: Random position (4*W/5 + random, H/2 + random)
- Puck: Left side, x ∈ [3.0, 4.0], random y
- Player 1 always starts

**TRAIN_DEFENSE Mode:**
- Player 1: (W/5, H/2) ≈ (2.0, 4.0)
- Player 2: Random position far right
- Puck: Right side with velocity toward Player 1's goal
- Puck starts moving toward Player 1

### Puck Holding Mechanism (Keep Mode)

When `keep_mode=True`, players can "catch" and hold the puck:

**How It Works:**

1. **Catching the Puck:**
   - Occurs when player contacts puck AND puck's velocity is low
   - Player 1 catches if: player1 touches puck AND puck velocity x < 0.1
   - Player 2 catches if: player2 touches puck AND puck velocity x > -0.1
   - A timer starts: `player_has_puck = MAX_TIME_KEEP_PUCK` (15 timesteps)

2. **While Holding:**
   - Puck position = player position (stuck to player)
   - Puck velocity = player velocity (moves with player)
   - Timer counts down each timestep
   - Player can move freely while holding the puck
   - Observation indices 16-17 show time remaining

3. **Shooting (Releasing):**
   - **Automatic release:** Timer reaches 1 (after 14 timesteps)
   - **Manual release:** Action[3] for Player 1 or Action[7] for Player 2 > 0.5
   - Shooting applies force in the direction player is facing:
     - Force magnitude: `SHOOTFORCEMULTIPLIER * puck.mass / timeStep = 60 * 7.0 / 0.02`
     - Direction: Based on player's angle (rotation)
     - Player 1: shoots in direction of angle
     - Player 2: shoots in opposite direction (mirrored)

4. **Strategic Implications:**
   - Allows controlled possession
   - Enables aimed shots by rotating before shooting
   - Limited holding time forces action
   - Creates passing and shooting opportunities

**Without Keep Mode (`keep_mode=False`):**
- Puck always bounces off players
- No catching or holding mechanism
- Action space is smaller (6 dimensions instead of 8)
- More reactive, less strategic gameplay

### Episode Duration and Termination

**Time Limits:**
- **NORMAL Mode:** 250 timesteps maximum
- **TRAIN_SHOOTING Mode:** 80 timesteps maximum
- **TRAIN_DEFENSE Mode:** 80 timesteps maximum
- Timestep duration: 1/50 second (FPS = 50)

**Episode Ends When:**
1. **Goal scored:** Puck enters either goal
   - `info['winner'] = 1`: Player 1 scored
   - `info['winner'] = -1`: Player 2 scored
   - `done = True`

2. **Time runs out:**
   - `info['winner'] = 0`: Draw (no one scored)
   - `truncated = True` or `done = True`

3. **Manual termination:** `env.close()` called

### Physics and Forces

**Force Application:**
- Player movement: `FORCEMULTIPLIER = 6000`
- Torque for rotation: `TORQUEMULTIPLIER = 400`
- Shooting force: `SHOOTFORCEMULTIPLIER = 60`

**Damping:**
- Normal linear damping: 5.0
- Boundary collision damping: 20.0
- Angular damping: 2.0 (normal), 10.0 (at angle limits)
- Puck linear damping: 0.05 (normal), 10.0 (when speeding)

**Speed Limits:**
- Players: 10 units/timestep (soft limit via force control)
- Puck: 25 units/timestep (hard limit, enforced with damping)

### Angle Constraints

Players can rotate but are limited:
- **Maximum angle:** ±π/3 (±60 degrees)
- At limit:
  - Rotation torque becomes 0
  - Counter-torque applied to stop rotation
  - Torque applied to push angle back toward 0
  - Angular damping increases to 10.0

This prevents players from spinning uncontrollably.

### Collision Behavior

**Puck-Player Collision:**
- Puck bounces off player (restitution = 0.95)
- Bounce direction depends on:
  - Player's velocity
  - Puck's incoming velocity
  - Player's angle (rotation)
- In keep_mode: Can catch instead of bouncing

**Puck-Wall Collision:**
- Puck bounces off walls (restitution = 0.95)
- Slightly loses energy (0.95, not 1.0)

**Puck-Goal Collision:**
- Goal is a sensor (no physical collision)
- Triggers end of episode
- No bouncing

**Player-Wall Collision:**
- Prevented by boundary enforcement
- Velocity zeroed before actual collision

**Player-Player Collision:**
- Cannot occur (center line prevents players from meeting)

### Scoring and Winning

**How to Score:**
1. Get puck past center line
2. Direct puck toward opponent's goal
3. Puck enters goal circle
4. Episode ends, scorer gets +10 reward

**Winner Determination:**
- `winner = 1`: Player 1 (left, red) scored in right goal
- `winner = -1`: Player 2 (right, blue) scored in left goal  
- `winner = 0`: Time ran out, no goal (draw)

**From Player Perspective:**
- Your goal: Defend your side's goal, score in opponent's goal
- Player 1: Defend left, attack right
- Player 2: Defend right, attack left (but sees field as mirrored)

### Symmetry and Fairness

The environment is designed to be symmetric:
- Both players have identical capabilities
- Playing field is symmetric
- `obs_agent_two()` provides mirrored observation
- Rewards are negated for player 2
- Starting side alternates in NORMAL mode

This ensures neither player has an inherent advantage.

### Key Constants Reference

```python
FPS = 50                      # Physics updates per second
SCALE = 60.0                  # Pixels to physics units
W = 10.0                      # Field width
H = 8.0                       # Field height
CENTER_X = 5.0                # Center line x-coordinate
CENTER_Y = 4.0                # Center line y-coordinate
MAX_ANGLE = π/3               # Maximum racket rotation
MAX_TIME_KEEP_PUCK = 15       # Timesteps can hold puck
MAX_PUCK_SPEED = 25           # Maximum puck velocity
GOAL_SIZE = 75                # Goal radius in pixels
```

## Environment Modes

The environment supports three different modes for various training scenarios:

### 1. NORMAL Mode (Default)

Full two-player game where both players start from their respective sides.

```python
env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
```

- **Maximum timesteps**: 250
- **Starting position**: Alternates between players
- **Puck position**: Randomized near the starting player

### 2. TRAIN_SHOOTING Mode

Specialized mode for training shooting and offense.

```python
env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
```

- **Maximum timesteps**: 80
- **Starting position**: Player 1 always starts
- **Puck position**: Randomized on the left side
- **Use case**: Practice offensive strategies

### 3. TRAIN_DEFENSE Mode

Specialized mode for training defensive play.

```python
env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
```

- **Maximum timesteps**: 80
- **Puck position**: Right side with velocity toward player 1's goal
- **Use case**: Practice defensive strategies

### Keep Mode

The `keep_mode` parameter controls whether players can "catch" and hold the puck:

```python
# With keep mode (default)
env = h_env.HockeyEnv(keep_mode=True)  # 4 actions per player

# Without keep mode
env = h_env.HockeyEnv(keep_mode=False)  # 3 actions per player
```

## Observation Space

The observation space is a **18-dimensional continuous vector** (`Box(-inf, inf, shape=(18,))`).

### Observation Vector Structure

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Player 1 x position (relative to center) | [-∞, ∞] |
| 1 | Player 1 y position (relative to center) | [-∞, ∞] |
| 2 | Player 1 angle | [-π/3, π/3] |
| 3 | Player 1 x velocity | [-∞, ∞] |
| 4 | Player 1 y velocity | [-∞, ∞] |
| 5 | Player 1 angular velocity | [-∞, ∞] |
| 6 | Player 2 x position (relative to center) | [-∞, ∞] |
| 7 | Player 2 y position (relative to center) | [-∞, ∞] |
| 8 | Player 2 angle | [-π/3, π/3] |
| 9 | Player 2 x velocity | [-∞, ∞] |
| 10 | Player 2 y velocity | [-∞, ∞] |
| 11 | Player 2 angular velocity | [-∞, ∞] |
| 12 | Puck x position (relative to center) | [-∞, ∞] |
| 13 | Puck y position (relative to center) | [-∞, ∞] |
| 14 | Puck x velocity | [-∞, ∞] |
| 15 | Puck y velocity | [-∞, ∞] |
| 16 | Time left player 1 has puck (keep_mode only) | [0, 15] |
| 17 | Time left player 2 has puck (keep_mode only) | [0, 15] |

### Symmetric Observation for Player 2

Player 2 receives a mirrored observation from their perspective:

```python
obs_agent1 = env.reset()[0]
obs_agent2 = env.obs_agent_two()
```

This ensures that both agents can use the same policy by seeing the game from their own perspective.

### Observation Scaling

For better training stability, consider normalizing observations:

```python
# Suggested scaling factors
scaling = [1.0, 1.0, 0.5, 4.0, 4.0, 4.0,   # Player 1
           1.0, 1.0, 0.5, 4.0, 4.0, 4.0,   # Player 2
           2.0, 2.0, 10.0, 10.0, 4.0, 4.0]  # Puck

scaled_obs = obs * scaling
```

## Action Space

### Continuous Action Space

The action space is continuous with **6 or 8 dimensions** depending on `keep_mode`.

**Without keep_mode (6 dimensions):**
- Actions 0-2: Player 1 (x force, y force, torque)
- Actions 3-5: Player 2 (x force, y force, torque)

**With keep_mode (8 dimensions):**
- Actions 0-3: Player 1 (x force, y force, torque, shoot)
- Actions 4-7: Player 2 (x force, y force, torque, shoot)

All actions are in the range **[-1, 1]**.

```python
action_space = spaces.Box(-1, +1, (6,), dtype=np.float32)  # keep_mode=False
action_space = spaces.Box(-1, +1, (8,), dtype=np.float32)  # keep_mode=True
```

### Action Components

1. **X Force** (action[0] for P1, action[3/4] for P2)
   - Controls horizontal movement
   - -1: move left, +1: move right

2. **Y Force** (action[1] for P1, action[4/5] for P2)
   - Controls vertical movement
   - -1: move down, +1: move up

3. **Torque** (action[2] for P1, action[5/6] for P2)
   - Controls rotation angle
   - -1: rotate counter-clockwise, +1: rotate clockwise

4. **Shoot** (action[3] for P1, action[7] for P2, only in keep_mode)
   - Triggers puck shooting when holding it
   - > 0.5: shoot, ≤ 0.5: keep holding

### Discrete Action Space

For discrete action algorithms, use the built-in conversion:

```python
discrete_action = 2  # Move right
continuous_action = env.discrete_to_continous_action(discrete_action)
```

**Discrete Actions:**
- 0: Do nothing
- 1: Move left (-1 in x)
- 2: Move right (+1 in x)
- 3: Move down (-1 in y)
- 4: Move up (+1 in y)
- 5: Rotate counter-clockwise (-1 in angle)
- 6: Rotate clockwise (+1 in angle)
- 7: Shoot (keep_mode only)

## Reward Structure

### Basic Rewards

The environment provides sparse rewards based on game outcomes:

- **Win**: +10
- **Loss**: -10
- **Draw**: 0

### Proxy Rewards (Reward Shaping)

The `info` dictionary contains additional proxy rewards for easier learning:

```python
obs, reward, done, truncated, info = env.step(action)

# Access proxy rewards
closeness_reward = info['reward_closeness_to_puck']
touch_reward = info['reward_touch_puck']
direction_reward = info['reward_puck_direction']
```

#### 1. Closeness to Puck
- Penalizes distance from puck when it's in own half and moving toward goal
- Maximum penalty: -30
- Encourages defensive positioning

#### 2. Touch Puck
- Reward: +1 when touching the puck (in keep_mode)
- Encourages puck possession

#### 3. Puck Direction
- Rewards when puck moves toward opponent's goal
- Penalizes when puck moves toward own goal
- Maximum reward: ±1

### Custom Reward Functions

You can compute custom rewards using the info dictionary:

```python
def custom_reward(info, base_reward):
    reward = base_reward
    reward += info['reward_closeness_to_puck']
    reward += info['reward_touch_puck'] * 2.0  # Emphasize touching
    reward += info['reward_puck_direction'] * 5.0  # Emphasize direction
    return reward
```

## Built-in Opponents

### BasicOpponent

A hand-crafted rule-based opponent with adjustable difficulty:

```python
# Strong opponent
player = h_env.BasicOpponent(weak=False)

# Weak opponent
player = h_env.BasicOpponent(weak=True)

# Usage in game loop
obs, info = env.reset()
obs_agent2 = env.obs_agent_two()

for _ in range(250):
    env.render()
    a1 = your_agent.act(obs)
    a2 = player.act(obs_agent2)
    obs, r, d, t, info = env.step(np.hstack([a1, a2]))
    obs_agent2 = env.obs_agent_two()
    if d or t:
        break
```

**Behavior:**
- Weak mode: Slower reactions (kp = 0.5)
- Strong mode: Faster reactions (kp = 10)
- Strategy: Defensive positioning, tries to intercept puck
- Shoots automatically when holding puck for 8-15 timesteps

### HumanOpponent

Allows human players to control a player using keyboard:

```python
player1 = h_env.HumanOpponent(env=env, player=1)  # Left player
player2 = h_env.HumanOpponent(env=env, player=2)  # Right player
```

**Controls:**
- Left/Right Arrow: Horizontal movement
- Up/Down Arrow: Vertical movement
- W: Rotate counter-clockwise
- S: Rotate clockwise
- Space: Shoot (in keep_mode)

### HockeyEnv_BasicOpponent

A wrapped environment with built-in opponent:

```python
env = h_env.HockeyEnv_BasicOpponent(
    mode=h_env.Mode.NORMAL,
    weak_opponent=False
)

# Only control player 1, opponent is automatic
obs, info = env.reset()
for _ in range(250):
    a1 = your_agent.act(obs)  # Only player 1 action
    obs, r, d, t, info = env.step(a1)  # Step with single action
    if d or t:
        break
```

## Advanced Usage

### Setting Custom Initial States

You can restore a specific state:

```python
# Save state
saved_state = obs.copy()

# Later, restore it
env.set_state(saved_state)
```

### Accessing Both Players' Info

```python
obs, reward, done, truncated, info1 = env.step(action)

# Get player 2's info
info2 = env.get_info_agent_two()
reward2 = env.get_reward_agent_two(info2)
```

### Custom Rendering

```python
# Human rendering (window display)
env.render(mode='human')

# RGB array for video recording
rgb_array = env.render(mode='rgb_array')
```

### Seeding for Reproducibility

```python
env.seed(42)
obs, info = env.reset()
```

### Changing Mode After Creation

```python
env = h_env.HockeyEnv()

# Switch to shooting mode
obs, info = env.reset(mode=h_env.Mode.TRAIN_SHOOTING)
```

## API Reference

### HockeyEnv Class

#### Constructor

```python
HockeyEnv(keep_mode=True, mode=Mode.NORMAL, verbose=False)
```

**Parameters:**
- `keep_mode` (bool): Enable puck catching and shooting mechanics
- `mode` (Mode | int | str): Game mode (NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE)
- `verbose` (bool): Print debug information

#### Methods

**reset(one_starting=None, mode=None, seed=None, options=None)**

Reset the environment to initial state.

**Returns:** `(observation, info)`

**step(action)**

Execute one timestep with the given action.

**Parameters:**
- `action` (np.ndarray): Combined actions for both players [a1, a2]

**Returns:** `(observation, reward, done, truncated, info)`

**render(mode='human')**

Render the current state.

**Parameters:**
- `mode` (str): 'human' for window display, 'rgb_array' for numpy array

**close()**

Close the environment and clean up resources.

**obs_agent_two()**

Get symmetric observation for player 2.

**Returns:** `np.ndarray` - Mirrored observation

**get_reward(info)**

Calculate reward for player 1 including proxy rewards.

**get_reward_agent_two(info_two)**

Calculate reward for player 2 including proxy rewards.

**get_info_agent_two()**

Get info dictionary for player 2.

**set_state(state)**

Set environment to a specific state.

**discrete_to_continous_action(discrete_action)**

Convert discrete action to continuous.

**seed(seed=None)**

Set random seed for reproducibility.

### Important Constants

```python
FPS = 50                      # Frames per second
SCALE = 60.0                  # Physics scale
VIEWPORT_W = 600              # Window width
VIEWPORT_H = 480              # Window height
MAX_ANGLE = math.pi / 3       # Max racket angle (~60 degrees)
MAX_TIME_KEEP_PUCK = 15       # Max timesteps holding puck
MAX_PUCK_SPEED = 25           # Maximum puck velocity
```

## Tips and Best Practices

### Training Tips

1. **Start with simpler modes**
   - Begin training in TRAIN_SHOOTING or TRAIN_DEFENSE modes
   - Gradually increase to NORMAL mode

2. **Use observation scaling**
   - Normalize observations for better training stability
   - Consider the scaling factors provided above

3. **Leverage proxy rewards**
   - Proxy rewards help with sparse reward problem
   - Experiment with different weightings

4. **Self-play training**
   - Train against past versions of your agent
   - Use BasicOpponent for initial training

5. **Curriculum learning**
   - Start with weak opponent
   - Gradually increase difficulty

### Performance Optimization

1. **Disable rendering during training**
   ```python
   # Don't call render() during training
   obs, r, d, t, info = env.step(action)
   ```

2. **Vectorize environments**
   - Use multiple parallel environments
   - Consider gymnasium's VectorEnv

3. **Limit episode length**
   - Default 250 steps for NORMAL mode is reasonable
   - Consider shorter episodes for faster iteration

### Common Issues

1. **ALSA warnings on Linux**
   - These are harmless audio warnings
   - Can be suppressed or ignored

2. **Rendering issues**
   - Ensure pygame is properly installed
   - Check display settings if running headless

3. **Action clipping**
   - Actions are automatically clipped to [-1, 1]
   - No need to manually clip

4. **Observation symmetry**
   - Always use `obs_agent_two()` for player 2
   - Ensures correct perspective for both agents

### Debugging

Enable verbose mode for detailed information:

```python
env = h_env.HockeyEnv(verbose=True)
```

Monitor info dictionary:

```python
obs, reward, done, truncated, info = env.step(action)
print(f"Winner: {info['winner']}")
print(f"Closeness: {info['reward_closeness_to_puck']}")
print(f"Touch: {info['reward_touch_puck']}")
print(f"Direction: {info['reward_puck_direction']}")
```

## Example Workflows

### Self-Play Training

```python
import numpy as np
import hockey.hockey_env as h_env

env = h_env.HockeyEnv()

# Train two identical agents against each other
for episode in range(1000):
    obs, info = env.reset()
    obs2 = env.obs_agent_two()
    
    for step in range(250):
        a1 = agent.select_action(obs)
        a2 = agent.select_action(obs2)
        
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        obs2 = env.obs_agent_two()
        
        # Training updates here
        
        if d or t:
            break
```

### Evaluation Against BasicOpponent

```python
def evaluate_agent(agent, num_episodes=100):
    env = h_env.HockeyEnv()
    opponent = h_env.BasicOpponent(weak=False)
    
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs2 = env.obs_agent_two()
        
        for step in range(250):
            a1 = agent.select_action(obs)
            a2 = opponent.act(obs2)
            obs, r, d, t, info = env.step(np.hstack([a1, a2]))
            obs2 = env.obs_agent_two()
            
            if d or t:
                if info['winner'] == 1:
                    wins += 1
                elif info['winner'] == -1:
                    losses += 1
                else:
                    draws += 1
                break
    
    return wins, losses, draws
```

### Recording Videos

**Quick Method: Use the provided script**

The repository includes `run_game.py` which automatically saves games as MP4 videos:

```bash
# Save to game.mp4 (default)
python run_game.py

# Save to custom filename
python run_game.py -o my_game.mp4

# Try to show window instead (if display available)
python run_game.py --display

# Run without rendering (fastest)
python run_game.py --no-render
```

This script works perfectly on remote SSH servers - it automatically saves the game as a video file you can download and watch.

**Manual Method: Programmatic recording**

```python
import numpy as np
import imageio

def record_episode(env, agent1, agent2, filename='game.mp4'):
    frames = []
    obs, info = env.reset()
    obs2 = env.obs_agent_two()
    
    for step in range(250):
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        a1 = agent1.act(obs)
        a2 = agent2.act(obs2)
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        obs2 = env.obs_agent_two()
        
        if d or t:
            break
    
    imageio.mimsave(filename, frames, fps=50)
```

**Note:** For video recording, you need `imageio` and `imageio-ffmpeg`:
```bash
pip install imageio imageio-ffmpeg
```

## Creating Custom Training Scenarios

The hockey environment can be extended to create custom training scenarios tailored to your specific needs, such as shooting with obstacles, blocking scenarios, or custom initial positions.

### Method 1: Creating a Custom Environment Wrapper

The recommended approach is to create a wrapper class that inherits from `HockeyEnv`:

```python
import numpy as np
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv, SCALE, CENTER_X, CENTER_Y, W, H

class CustomShootingEnv(HockeyEnv):
    def __init__(self, block_middle=False, obstacle_positions=None, keep_mode=True):
        """
        Custom shooting environment with configurable obstacles.
        
        Args:
            block_middle: Whether to place an obstacle in the middle
            obstacle_positions: List of (x, y) positions for obstacles
            keep_mode: Whether to enable puck catching
        """
        super().__init__(mode=h_env.Mode.TRAIN_SHOOTING, keep_mode=keep_mode)
        self.block_middle = block_middle
        self.obstacle_positions = obstacle_positions or []
        self.obstacles = []
    
    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        # Call parent reset
        obs, info = super().reset(one_starting, mode, seed, options)
        
        # Add custom obstacles after reset
        if self.block_middle:
            self._add_middle_obstacle()
        
        for pos in self.obstacle_positions:
            self._add_custom_obstacle(pos)
        
        return obs, info
    
    def _add_middle_obstacle(self):
        """Add a static obstacle in the middle of the playing field"""
        from Box2D.b2 import fixtureDef, polygonShape
        
        # Create a rectangular obstacle
        obstacle_width = 20 / SCALE
        obstacle_height = 80 / SCALE
        
        obstacle = self.world.CreateStaticBody(
            position=(CENTER_X, CENTER_Y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(obstacle_width, obstacle_height)),
                density=0,
                friction=0.1,
                categoryBits=0x011,
                maskBits=0x0011
            )
        )
        obstacle.color1 = (100, 100, 100)
        obstacle.color2 = (100, 100, 100)
        
        self.obstacles.append(obstacle)
        self.drawlist.append(obstacle)
    
    def _add_custom_obstacle(self, position):
        """Add a circular obstacle at a specific position"""
        from Box2D.b2 import fixtureDef, circleShape
        
        obstacle = self.world.CreateStaticBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=20 / SCALE, pos=(0, 0)),
                density=0,
                friction=0.5,
                categoryBits=0x011,
                maskBits=0x0011
            )
        )
        obstacle.color1 = (150, 150, 150)
        obstacle.color2 = (100, 100, 100)
        
        self.obstacles.append(obstacle)
        self.drawlist.append(obstacle)
    
    def _destroy(self):
        # Clean up custom obstacles
        for obstacle in self.obstacles:
            if obstacle in self.drawlist:
                self.drawlist.remove(obstacle)
            self.world.DestroyBody(obstacle)
        self.obstacles = []
        
        # Call parent destroy
        super()._destroy()


# Usage example
env = CustomShootingEnv(block_middle=True)
obs, info = env.reset()

for _ in range(100):
    env.render()
    a1 = np.random.uniform(-1, 1, 4)
    a2 = [0, 0, 0, 0]
    obs, r, d, t, info = env.step(np.hstack([a1, a2]))
    if d or t:
        break

env.close()
```

### Method 2: Custom Initial Positions

You can create specific starting scenarios by overriding the reset method:

```python
class CustomPositionEnv(HockeyEnv):
    def __init__(self, player1_pos=None, player2_pos=None, puck_pos=None, 
                 puck_velocity=None, mode=h_env.Mode.NORMAL):
        """
        Environment with custom initial positions.
        
        Args:
            player1_pos: Tuple (x, y) for player 1 starting position
            player2_pos: Tuple (x, y) for player 2 starting position
            puck_pos: Tuple (x, y) for puck starting position
            puck_velocity: Tuple (vx, vy) for initial puck velocity
        """
        super().__init__(mode=mode)
        self.custom_player1_pos = player1_pos
        self.custom_player2_pos = player2_pos
        self.custom_puck_pos = puck_pos
        self.custom_puck_velocity = puck_velocity
    
    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        obs, info = super().reset(one_starting, mode, seed, options)
        
        # Override positions if specified
        if self.custom_player1_pos is not None:
            self.player1.position = self.custom_player1_pos
        
        if self.custom_player2_pos is not None:
            self.player2.position = self.custom_player2_pos
        
        if self.custom_puck_pos is not None:
            self.puck.position = self.custom_puck_pos
        
        if self.custom_puck_velocity is not None:
            self.puck.linearVelocity = self.custom_puck_velocity
        
        # Get updated observation
        obs = self._get_obs()
        return obs, info


# Example: Create a defensive scenario
env = CustomPositionEnv(
    player1_pos=(W / 4, H / 2),           # Player 1 near their goal
    player2_pos=(3 * W / 4, H / 2),       # Player 2 on the other side
    puck_pos=(2 * W / 3, H / 2),          # Puck approaching player 1
    puck_velocity=(-5.0, 0.0),            # Puck moving toward player 1's goal
    mode=h_env.Mode.TRAIN_DEFENSE
)
```

### Method 3: Restricted Zone Training

Create zones where the opponent cannot enter:

```python
class RestrictedZoneEnv(HockeyEnv):
    def __init__(self, restricted_zone=None, mode=h_env.Mode.NORMAL):
        """
        Environment where player 2 is restricted to certain zones.
        
        Args:
            restricted_zone: Dict with 'x_min', 'x_max', 'y_min', 'y_max'
        """
        super().__init__(mode=mode)
        self.restricted_zone = restricted_zone or {
            'x_min': CENTER_X,
            'x_max': W,
            'y_min': 0,
            'y_max': H
        }
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Force player 2 back into allowed zone
        if self.player2.position[0] < self.restricted_zone['x_min']:
            self.player2.position = (self.restricted_zone['x_min'] + 0.1, 
                                    self.player2.position[1])
            self.player2.linearVelocity = (0, 0)
        
        if self.player2.position[0] > self.restricted_zone['x_max']:
            self.player2.position = (self.restricted_zone['x_max'] - 0.1, 
                                    self.player2.position[1])
            self.player2.linearVelocity = (0, 0)
        
        return obs, reward, done, truncated, info


# Example: Player 2 can only defend their half
env = RestrictedZoneEnv(
    restricted_zone={'x_min': CENTER_X, 'x_max': W, 'y_min': 0, 'y_max': H},
    mode=h_env.Mode.NORMAL
)
```

### Method 4: Custom Reward Functions

Create specialized reward functions for specific training objectives:

```python
class CustomRewardEnv(HockeyEnv):
    def __init__(self, reward_weights=None, mode=h_env.Mode.NORMAL):
        """
        Environment with custom reward shaping.
        
        Args:
            reward_weights: Dict with weights for different reward components
        """
        super().__init__(mode=mode)
        self.reward_weights = reward_weights or {
            'win': 10.0,
            'loss': -10.0,
            'closeness': 1.0,
            'touch': 2.0,
            'direction': 5.0,
            'shot_on_goal': 3.0
        }
    
    def get_reward(self, info):
        r = 0
        
        # Win/loss rewards
        if self.done:
            if self.winner == 1:
                r += self.reward_weights['win']
            elif self.winner == -1:
                r += self.reward_weights['loss']
        
        # Proxy rewards with custom weights
        r += info['reward_closeness_to_puck'] * self.reward_weights['closeness']
        r += info['reward_touch_puck'] * self.reward_weights['touch']
        r += info['reward_puck_direction'] * self.reward_weights['direction']
        
        # Custom: reward for shooting toward goal
        if self.puck.position[0] > CENTER_X and self.puck.linearVelocity[0] > 0:
            # Puck is past center and moving toward opponent goal
            goal_y = CENTER_Y
            puck_to_goal_y = abs(self.puck.position[1] - goal_y)
            shot_accuracy = max(0, 1.0 - puck_to_goal_y / (H / 2))
            r += shot_accuracy * self.reward_weights['shot_on_goal']
        
        return float(r)


# Example: Emphasize shooting accuracy
env = CustomRewardEnv(
    reward_weights={
        'win': 20.0,
        'loss': -10.0,
        'closeness': 0.5,
        'touch': 3.0,
        'direction': 10.0,
        'shot_on_goal': 5.0
    },
    mode=h_env.Mode.TRAIN_SHOOTING
)
```

### Method 5: Progressive Difficulty Curriculum

Create a curriculum that increases difficulty over time:

```python
class CurriculumEnv(HockeyEnv):
    def __init__(self, difficulty_level=1, mode=h_env.Mode.NORMAL):
        """
        Environment with progressive difficulty.
        
        Args:
            difficulty_level: 1 (easy) to 5 (hard)
        """
        super().__init__(mode=mode)
        self.difficulty_level = difficulty_level
        self.episode_count = 0
    
    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        obs, info = super().reset(one_starting, mode, seed, options)
        self.episode_count += 1
        
        # Adjust difficulty every 100 episodes
        if self.episode_count % 100 == 0 and self.difficulty_level < 5:
            self.difficulty_level += 1
            print(f"Difficulty increased to level {self.difficulty_level}")
        
        return obs, info
    
    def get_opponent(self):
        """Get opponent based on current difficulty level"""
        if self.difficulty_level == 1:
            return h_env.BasicOpponent(weak=True)
        elif self.difficulty_level <= 3:
            return h_env.BasicOpponent(weak=False)
        else:
            # Could return a trained agent at higher levels
            return h_env.BasicOpponent(weak=False)
```

### Complete Example: Shooting with Middle Blocker

Here's a complete example combining multiple customizations:

```python
import numpy as np
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv, SCALE, CENTER_X, CENTER_Y, W, H

class BlockedShootingEnv(HockeyEnv):
    """
    Shooting training environment with a blocker in the middle.
    Player must learn to shoot around the obstacle.
    """
    def __init__(self, blocker_size=(20, 80), blocker_pos=None):
        super().__init__(mode=h_env.Mode.TRAIN_SHOOTING, keep_mode=True)
        self.blocker_size = blocker_size  # (width, height) in pixels
        self.blocker_pos = blocker_pos or (CENTER_X + W / 6, CENTER_Y)
        self.blocker = None
    
    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        obs, info = super().reset(one_starting, mode, seed, options)
        
        # Add blocker obstacle
        from Box2D.b2 import fixtureDef, polygonShape
        
        w, h = self.blocker_size
        self.blocker = self.world.CreateStaticBody(
            position=self.blocker_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(w / SCALE / 2, h / SCALE / 2)),
                density=0,
                friction=0.1,
                categoryBits=0x011,
                maskBits=0x0011,
                restitution=0.5  # Bouncy
            )
        )
        self.blocker.color1 = (200, 100, 100)
        self.blocker.color2 = (150, 50, 50)
        self.drawlist.append(self.blocker)
        
        return obs, info
    
    def get_reward(self, info):
        r = super().get_reward(info)
        
        # Bonus reward for scoring around the blocker
        if self.done and self.winner == 1:
            r += 5.0  # Extra reward for successful shot
        
        return r
    
    def _destroy(self):
        if self.blocker is not None:
            if self.blocker in self.drawlist:
                self.drawlist.remove(self.blocker)
            self.world.DestroyBody(self.blocker)
            self.blocker = None
        super()._destroy()


# Train an agent to shoot around the blocker
env = BlockedShootingEnv(blocker_size=(30, 100))

for episode in range(100):
    obs, info = env.reset()
    
    for step in range(80):
        env.render()
        
        # Your agent's action here
        action = your_agent.select_action(obs)
        
        # Opponent does nothing
        a2 = [0, 0, 0, 0]
        
        obs, reward, done, truncated, info = env.step(np.hstack([action, a2]))
        
        if done or truncated:
            print(f"Episode {episode}: Winner = {info['winner']}")
            break

env.close()
```

### Tips for Custom Scenarios

1. **Start Simple**: Begin with minor modifications before creating complex scenarios

2. **Test Thoroughly**: Always test your custom environment to ensure physics work correctly

3. **Maintain Symmetry**: If creating two-player scenarios, ensure both agents have fair conditions

4. **Gradual Complexity**: Use curriculum learning to gradually increase difficulty

5. **Save Configurations**: Store scenario parameters in config files for reproducibility

```python
# Example configuration file approach
import json

scenario_config = {
    'name': 'blocked_shooting',
    'mode': 'TRAIN_SHOOTING',
    'obstacles': [
        {'type': 'rectangle', 'pos': [5.0, 4.0], 'size': [0.3, 1.3]},
        {'type': 'circle', 'pos': [6.0, 3.0], 'radius': 0.2}
    ],
    'reward_weights': {
        'win': 20.0,
        'shot_accuracy': 5.0
    }
}

# Save configuration
with open('scenarios/blocked_shooting.json', 'w') as f:
    json.dump(scenario_config, f)

# Load and use configuration
with open('scenarios/blocked_shooting.json', 'r') as f:
    config = json.load(f)
```

These customization methods allow you to create training scenarios tailored to specific skills you want your agent to learn, such as shooting around obstacles, defensive positioning, or handling specific game situations.

## License

This environment is released under the MIT License. See LICENSE file for details.

## Citation

If you use this environment in your research, please cite:

```
@misc{hockey-env,
  author = {Georg Martius},
  title = {Hockey Environment},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/martius-lab/hockey-env}},
}
```

## Contact and Support

For issues, questions, or contributions:
- GitHub: https://github.com/martius-lab/hockey-env
- Autonomous Learning Group, University of Tübingen

## Version History

- **v2.6**: Current version with gymnasium support
- Previous versions used gym (deprecated)

