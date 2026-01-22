"""
Reward backpropagation utility for TD-MPC2 buffer.

This module provides functions to apply discounted reward bonuses backwards
through episodes when the agent wins, matching the behavior in TDMPC2ReplayBuffer.
"""

import numpy as np
import torch


def apply_win_reward_backprop(
    rewards,
    winner,
    win_reward_bonus=0.0,
    win_reward_discount=0.99,
    use_torch=False,
):
    """
    Apply discounted reward bonus backwards through an episode for winning episodes.

    This function implements the same reward backpropagation logic used in
    TDMPC2ReplayBuffer._flush_episode(). When an episode ends with a win (winner == 1),
    it adds a bonus reward to each step, with the bonus discounted backwards from
    the end of the episode.

    Args:
        rewards: Array of rewards for the episode, shape (T,) or (T, 1)
            Can be numpy array or torch tensor
        winner: Winner information (1 for agent win, -1 for loss, 0 for draw)
        win_reward_bonus: Bonus reward to add to each step in a winning episode
        win_reward_discount: Discount factor for applying win reward bonus
            backwards through the episode (1.0 = no discount, 0.99 = standard)
        use_torch: If True, return torch tensor; else numpy array

    Returns:
        Modified rewards array with bonus applied (same type as input)
        Original rewards array (for comparison)
        Bonus rewards array (for analysis)
    """
    # Convert to numpy for processing if needed
    is_torch_input = isinstance(rewards, torch.Tensor)
    if is_torch_input:
        rewards_np = rewards.detach().cpu().numpy()
    else:
        rewards_np = np.asarray(rewards, dtype=np.float32)

    # Ensure 1D array
    if rewards_np.ndim > 1:
        rewards_np = rewards_np.flatten()

    original_rewards = rewards_np.copy()
    bonus_rewards = np.zeros_like(rewards_np)

    # Apply reward shaping for winning episodes
    # Skip the terminal reward (last step) which already has +10, start from n-1
    if winner == 1 and win_reward_bonus > 0.0 and len(rewards_np) > 0:
        T = len(rewards_np)
        # Only apply bonus to steps 0 to T-2 (skip the last step T-1)
        for t in range(T - 1):
            # Calculate steps from the second-to-last step (n-1)
            # Step T-2 is 0 steps from n-1, step T-3 is 1 step from n-1, etc.
            steps_from_n_minus_1 = (T - 2) - t
            discount_factor = win_reward_discount**steps_from_n_minus_1
            bonus = win_reward_bonus * discount_factor
            bonus_rewards[t] = bonus

        rewards_np = rewards_np + bonus_rewards

    # Convert back to torch if needed
    if use_torch or is_torch_input:
        rewards_out = torch.as_tensor(rewards_np, dtype=torch.float32)
        original_rewards = torch.as_tensor(original_rewards, dtype=torch.float32)
        bonus_rewards = torch.as_tensor(bonus_rewards, dtype=torch.float32)
        if is_torch_input and rewards.device.type != "cpu":
            rewards_out = rewards_out.to(rewards.device)
            original_rewards = original_rewards.to(rewards.device)
            bonus_rewards = bonus_rewards.to(rewards.device)
    else:
        rewards_out = rewards_np

    return rewards_out, original_rewards, bonus_rewards
