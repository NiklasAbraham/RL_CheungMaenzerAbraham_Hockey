# Design: Opponent Simulation in TD-MPC2

## Problem

The opponent simulation pipeline has several consistency issues:

1. The cloning buffer stores observations from the opponent's perspective (obs_agent2), but
   the cloning network is queried at planning time with latents from the agent's perspective
   (encoder(obs_agent1)). These are fundamentally different distributions because hockey
   observations are player-centric (each player sees itself first, opponent second, puck
   coordinates relative to its own side).

2. During MPPI planning, a different opponent is sampled per dynamics forward call, meaning a
   single imagined trajectory can mix different opponents across timesteps. This is physically
   implausible since the opponent does not change mid-episode.

3. During world model training, a different opponent is sampled per timestep in the consistency
   rollout, creating the same incoherence as in planning.

## Root Cause

The cloning network answers the question: "given the current game state as understood by our
agent, what will the opponent do?" Its training input must therefore match its inference input.

At inference (planning): the cloning network receives latents from encoder(obs_agent1), either
directly at t=0 or from the dynamics model at t>0. The dynamics model is trained via the
consistency loss to output latents close to encoder(obs_agent1), so all latents in the planning
rollout live in the agent-perspective latent space.

At training: the cloning network must also be trained on encoder(obs_agent1) latents, paired
with the corresponding opponent actions.

## Solution

### 1. Cloning Buffer: Store Agent Observations with Opponent Actions

At each environment step, both obs_agent1 and obs_agent2 are available from the environment.
The opponent (reference bot) is run on obs_agent2 to produce its action. The cloning buffer
stores (obs_agent1, action_opponent).

No mirroring. No perspective flags. No encoder changes.

The encoder remains a single network trained only on agent-perspective observations. All latents
in the system (training, planning, cloning) live in the same space.

```
collect_opponent_demonstrations(obs_agent1, obs_agent2):
    for each loaded reference opponent:
        action_opponent = opponent.act(obs_agent2)
        cloning_buffer.add(obs_agent1, action_opponent)
```

### 2. Cloning Network Training

Sample (obs_agent1, action_opponent) from the buffer. Encode with the agent's encoder:
z = encoder(obs_agent1). Train: cloning_net(z) -> action_opponent.

At planning time: cloning_net(z) where z comes from encoder(obs_agent1) at t=0, or from the
dynamics model at t>0. Since the dynamics model is trained to produce encoder(obs_agent1)-like
latents, the distributions match.

### 3. One Opponent per Trajectory During Planning

During MPPI planning, each trajectory (particle) is assigned one opponent at the start and uses
that same opponent for all horizon steps. Different trajectories can use different opponents.

This makes imagined trajectories physically consistent: within one trajectory, the opponent
behaves coherently according to one policy.

Implementation: DynamicsWithOpponentWrapper supports batch opponent assignments. The planner
calls assign_opponents_for_batch(num_samples) before rolling out trajectories. Each row in the
batch gets a fixed opponent_id that persists across all horizon steps.

### 4. One Opponent per Batch Element During Training

During world model training, the consistency loss rollout assigns one opponent per batch element
before the horizon loop. The same opponent is used for all timesteps of that batch element's
imagined rollout.

### 5. No Encoder Changes

The encoder does not need a perspective flag. It only ever sees agent-perspective observations.
The latent space has a single distribution, and all components (dynamics, reward, Q-ensemble,
policy, cloning networks) operate in this unified space.

## Summary Table

| Component              | Before                             | After                                       |
|------------------------|------------------------------------|---------------------------------------------|
| Cloning buffer stores  | (mirrored_obs_agent2, action_opp)  | (obs_agent1, action_opp)                    |
| Cloning training input | encoder(mirrored_obs)              | encoder(obs_agent1)                         |
| collect_opponent_demos | Takes obs_agent2, mirrors it       | Takes obs_agent1 + obs_agent2, stores obs1  |
| Planning opponent      | Random per forward call            | One per trajectory, fixed for all steps     |
| Training opponent      | Random per timestep                | One per batch element, fixed for horizon    |
| Encoder                | Unchanged                          | Unchanged                                   |
| Dynamics               | Unchanged                          | Unchanged                                   |

## Implementation Checklist

1. DynamicsWithOpponentWrapper: add assign_opponents_for_batch() and clear_batch_opponents().
   forward() checks batch assignments first, then force_opponent_id, then random fallback.

2. collect_opponent_demonstrations: change signature to (obs_agent1, obs_agent2). Store
   obs_agent1 with opponent action. Remove _mirror_obs entirely.

3. store_opponent_action: document that obs must be agent-perspective (obs_agent1).

4. train() consistency loop: assign one opponent per batch element before the horizon loop.

5. MPPI planner: call assign_opponents_for_batch(num_samples) before rolling out trajectories.
   Clear after planning.

6. Callers of collect_opponent_demonstrations must be updated to pass both observations.
