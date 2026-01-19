import logging
import os
import sys
import time
import warnings
from datetime import datetime

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common.utils import (
    discrete_to_continuous_action_with_fineness,
    get_discrete_action_dim,
)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("agent_video")


class ALSAFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, message):
        if "ALSA" not in message and "pkg_resources" not in message:
            self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()


if sys.platform == "linux":
    sys.stderr = ALSAFilter(sys.stderr)

# Allow MODEL_PATH to be overridden via environment variable
# If relative path, convert to absolute based on script location or PROJECT_DIR
_DEFAULT_MODEL_PATH = "results/tdmpc2_runs/2026-01-18_12-12-17/models/TDMPC2_run_lr3e04_bs256_h128_256_128_42e91061_20260118_121217_vec16_temp_eval.pt"
_MODEL_PATH_ENV = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)

if os.path.isabs(_MODEL_PATH_ENV):
    MODEL_PATH = _MODEL_PATH_ENV
else:
    # Try to resolve relative path based on PROJECT_DIR or script location
    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir and os.path.exists(project_dir):
        MODEL_PATH = os.path.join(project_dir, _MODEL_PATH_ENV)
    else:
        # Fall back to relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        MODEL_PATH = os.path.join(project_root, _MODEL_PATH_ENV)
NUM_GAMES = 25
OPPONENT_TYPE = "basic_strong"
PAUSE_BETWEEN_GAMES = 1.5
FRAME_DELAY = (
    0.05  # Delay in video playback (seconds per frame). Execution runs at full speed.
)
MAX_STEPS = 250
VIDEO_FPS = 50
ACTION_FINENESS = None  # Set to None to auto-detect, or specify 3, 5, 7, etc.
ENV_MODE = "NORMAL"  # "NORMAL" (250 steps), "TRAIN_SHOOTING" (80 steps), or "TRAIN_DEFENSE" (80 steps)


def infer_fineness_from_action_dim(action_dim, keep_mode=True):
    """
    Infer the fineness parameter from the action dimension.
    Returns None if it doesn't match a known fineness pattern.
    """
    # Try common fineness values: 3, 5, 7, 9, etc.
    for fineness in [3, 5, 7, 9, 11, 13, 15]:
        expected_dim = get_discrete_action_dim(fineness=fineness, keep_mode=keep_mode)
        if expected_dim == action_dim:
            return fineness
    return None


def detect_algorithm_from_filename(model_path):
    """Detect algorithm type from model filename."""
    filename = os.path.basename(model_path)
    if "TDMPC2" in filename or "tdmpc2" in filename.lower():
        return "TDMPC2"
    elif "DDDQN" in filename or "ddqn" in filename.lower():
        return "DDDQN"
    elif "SAC" in filename or "sac" in filename.lower():
        return "SAC"
    return None


def load_agent(model_path, state_dim, action_dim, algorithm=None):
    """Load agent from checkpoint, auto-detecting algorithm if not specified."""
    if algorithm is None:
        algorithm = detect_algorithm_from_filename(model_path)
        if algorithm is None:
            raise ValueError(
                f"Could not detect algorithm from filename: {model_path}. "
                f"Please specify algorithm explicitly or use a filename containing TDMPC2, DDDQN, or SAC."
            )
        logger.info(f"Auto-detected algorithm: {algorithm}")

    logger.info(f"Loading {algorithm} model from: {model_path}")

    if algorithm == "TDMPC2":
        import torch

        from rl_hockey.TD_MPC2.tdmpc2 import TDMPC2

        checkpoint = torch.load(model_path, map_location="cpu")
        obs_dim = checkpoint.get("obs_dim", state_dim)
        action_dim_checkpoint = checkpoint.get("action_dim", action_dim)
        latent_dim = checkpoint.get("latent_dim", 512)
        hidden_dim = checkpoint.get("hidden_dim", [256, 256, 256])
        num_q = checkpoint.get("num_q", 5)

        agent = TDMPC2(
            obs_dim=obs_dim,
            action_dim=action_dim_checkpoint,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
        )
        agent.load(model_path)
        logger.info("TDMPC2 model loaded successfully!")
        return agent, "TDMPC2"

    elif algorithm == "DDDQN":
        from rl_hockey.DDDQN import DDDQN

        agent = DDDQN(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=[256, 256, 256]
        )
        agent.load(model_path)
        logger.info("DDDQN model loaded successfully!")
        return agent, "DDDQN"

    elif algorithm == "SAC":
        from rl_hockey.sac.sac import SAC

        agent = SAC(state_dim=state_dim, action_dim=action_dim)
        agent.load(model_path)
        logger.info("SAC model loaded successfully!")
        return agent, "SAC"

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_blank_frames(frame_shape, duration_seconds, fps=50):
    num_frames = int(duration_seconds * fps)
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    # Use list comprehension with copy to avoid repeated allocations
    return [blank_frame.copy() for _ in range(num_frames)]


def apply_frame_delay(frames, frame_delay, fps=50):
    """
    Duplicate frames to create delay effect in video without slowing down execution.
    If frame_delay > 0, each frame is duplicated to create the delay in playback.
    """
    if frame_delay <= 0:
        return frames

    frames_per_step = int(frame_delay * fps)
    delayed_frames = []
    for frame in frames:
        # Duplicate each frame to create the delay effect
        delayed_frames.extend([frame] * frames_per_step)

    return delayed_frames


def log_stats_summary(game_stats, game_num, step_count):
    """
    Log statistics summary for a game.

    Args:
        game_stats: list of stats dictionaries, one per step
        game_num: game number
        step_count: number of steps in the game
    """
    if not game_stats:
        return

    # Aggregate stats across all steps
    agg_stats = {
        "q_min": [],
        "q_max": [],
        "q_mean": [],
        "q_std": [],
        "q_values_all": [],
        "q_spread": [],
        "q_coefficient_of_variation": [],
        "q_policy_min": [],
        "q_policy_max": [],
        "q_policy_mean": [],
        "dynamics_latent_change_norm": [],
        "dynamics_prediction_error_norm": [],
        "dynamics_prediction_error_mse": [],
        "reward_pred": [],
        "reward_pred_terminal": [],  # Terminal step rewards
        "encoder_latent_norm": [],
        "action_norm": [],
        "action_policy_diff_norm": [],
        "action_smoothness": [],
        "latent_smoothness": [],
        "mppi_elite_return_min": [],
        "mppi_elite_return_max": [],
        "mppi_elite_return_mean": [],
        "mppi_elite_return_std": [],
        "mppi_final_std": [],
        "mppi_std_convergence": [],
    }

    for step_stat in game_stats:
        agg_stats["q_min"].append(step_stat.get("q_min", 0))
        agg_stats["q_max"].append(step_stat.get("q_max", 0))
        agg_stats["q_mean"].append(step_stat.get("q_mean", 0))
        agg_stats["q_std"].append(step_stat.get("q_std", 0))
        if "q_values" in step_stat:
            agg_stats["q_values_all"].extend(step_stat["q_values"])
        agg_stats["q_spread"].append(step_stat.get("q_spread", 0))
        agg_stats["q_coefficient_of_variation"].append(
            step_stat.get("q_coefficient_of_variation", 0)
        )
        agg_stats["q_policy_min"].append(step_stat.get("q_policy_min", 0))
        agg_stats["q_policy_max"].append(step_stat.get("q_policy_max", 0))
        agg_stats["q_policy_mean"].append(step_stat.get("q_policy_mean", 0))
        agg_stats["dynamics_latent_change_norm"].append(
            step_stat.get("dynamics_latent_change_norm", 0)
        )
        # Handle None values for dynamics prediction error (first step)
        dynamics_pred_error_norm = step_stat.get("dynamics_prediction_error_norm")
        if dynamics_pred_error_norm is not None:
            agg_stats["dynamics_prediction_error_norm"].append(dynamics_pred_error_norm)
        dynamics_pred_error_mse = step_stat.get("dynamics_prediction_error_mse")
        if dynamics_pred_error_mse is not None:
            agg_stats["dynamics_prediction_error_mse"].append(dynamics_pred_error_mse)
        agg_stats["reward_pred"].append(step_stat.get("reward_pred", 0))
        # Track terminal reward separately
        if "terminal_reward_actual" in step_stat:
            agg_stats["reward_pred_terminal"].append({
                "predicted": step_stat.get("reward_pred", 0),
                "actual": step_stat.get("terminal_reward_actual", 0),
                "winner": step_stat.get("winner_info", 0)
            })
        agg_stats["encoder_latent_norm"].append(step_stat.get("encoder_latent_norm", 0))
        agg_stats["action_norm"].append(step_stat.get("action_norm", 0))
        agg_stats["action_policy_diff_norm"].append(
            step_stat.get("action_policy_diff_norm", 0)
        )
        # Handle None values for smoothness (first step)
        action_smoothness = step_stat.get("action_smoothness")
        if action_smoothness is not None:
            agg_stats["action_smoothness"].append(action_smoothness)
        latent_smoothness = step_stat.get("latent_smoothness")
        if latent_smoothness is not None:
            agg_stats["latent_smoothness"].append(latent_smoothness)
        # MPC/Planning stats
        if "mppi_elite_return_min" in step_stat:
            agg_stats["mppi_elite_return_min"].append(
                step_stat["mppi_elite_return_min"]
            )
            agg_stats["mppi_elite_return_max"].append(
                step_stat["mppi_elite_return_max"]
            )
            agg_stats["mppi_elite_return_mean"].append(
                step_stat["mppi_elite_return_mean"]
            )
            agg_stats["mppi_elite_return_std"].append(
                step_stat["mppi_elite_return_std"]
            )
        if "mppi_final_std" in step_stat:
            agg_stats["mppi_final_std"].append(step_stat["mppi_final_std"])
        if "mppi_std_convergence" in step_stat:
            agg_stats["mppi_std_convergence"].append(step_stat["mppi_std_convergence"])

    # Compute summary statistics
    logger.info(f"  TDMPC2 Stats Summary (Game {game_num}, {step_count} steps):")
    logger.info(
        f"    Q-values (selected action): min={min(agg_stats['q_min']):.4f}, "
        f"max={max(agg_stats['q_max']):.4f}, mean={np.mean(agg_stats['q_mean']):.4f}"
    )
    if agg_stats["q_values_all"]:
        logger.info(
            f"    Q-values (all networks): min={min(agg_stats['q_values_all']):.4f}, "
            f"max={max(agg_stats['q_values_all']):.4f}, "
            f"mean={np.mean(agg_stats['q_values_all']):.4f}"
        )
    logger.info(
        f"    Q-values (policy action): min={min(agg_stats['q_policy_min']):.4f}, "
        f"max={max(agg_stats['q_policy_max']):.4f}, mean={np.mean(agg_stats['q_policy_mean']):.4f}"
    )
    logger.info(
        f"    Dynamics (latent change norm): min={min(agg_stats['dynamics_latent_change_norm']):.4f}, "
        f"max={max(agg_stats['dynamics_latent_change_norm']):.4f}, "
        f"mean={np.mean(agg_stats['dynamics_latent_change_norm']):.4f}"
    )
    if agg_stats["dynamics_prediction_error_norm"]:
        logger.info(
            f"    Dynamics prediction error (norm): min={min(agg_stats['dynamics_prediction_error_norm']):.4f}, "
            f"max={max(agg_stats['dynamics_prediction_error_norm']):.4f}, "
            f"mean={np.mean(agg_stats['dynamics_prediction_error_norm']):.4f}"
        )
    if agg_stats["dynamics_prediction_error_mse"]:
        logger.info(
            f"    Dynamics prediction error (MSE): min={min(agg_stats['dynamics_prediction_error_mse']):.4f}, "
            f"max={max(agg_stats['dynamics_prediction_error_mse']):.4f}, "
            f"mean={np.mean(agg_stats['dynamics_prediction_error_mse']):.4f}"
        )
    logger.info(
        f"    Reward prediction: min={min(agg_stats['reward_pred']):.4f}, "
        f"max={max(agg_stats['reward_pred']):.4f}, mean={np.mean(agg_stats['reward_pred']):.4f}"
    )
    # Log terminal reward predictions separately
    if agg_stats["reward_pred_terminal"]:
        logger.info("\n  TERMINAL REWARD DIAGNOSTICS (Win/Loss Prediction):")
        for i, term_reward in enumerate(agg_stats["reward_pred_terminal"]):
            logger.info(
                f"    Episode {i}: Predicted={term_reward['predicted']:+.4f}, "
                f"Actual={term_reward['actual']:+.2f}, Winner={term_reward['winner']:+d}"
            )
    logger.info(
        f"    Encoder (latent norm): min={min(agg_stats['encoder_latent_norm']):.4f}, "
        f"max={max(agg_stats['encoder_latent_norm']):.4f}, "
        f"mean={np.mean(agg_stats['encoder_latent_norm']):.4f}"
    )
    logger.info(
        f"    Action norm: min={min(agg_stats['action_norm']):.4f}, "
        f"max={max(agg_stats['action_norm']):.4f}, mean={np.mean(agg_stats['action_norm']):.4f}"
    )
    logger.info("")
    logger.info("  Uncertainty Metrics:")
    if agg_stats["q_spread"]:
        logger.info(
            f"    Q-value spread: min={min(agg_stats['q_spread']):.4f}, "
            f"max={max(agg_stats['q_spread']):.4f}, mean={np.mean(agg_stats['q_spread']):.4f}"
        )
    if agg_stats["q_coefficient_of_variation"]:
        logger.info(
            f"    Q-value CV (std/mean): min={min(agg_stats['q_coefficient_of_variation']):.4f}, "
            f"max={max(agg_stats['q_coefficient_of_variation']):.4f}, "
            f"mean={np.mean(agg_stats['q_coefficient_of_variation']):.4f}"
        )
    logger.info("")
    logger.info("  Action Analysis:")
    if agg_stats["action_policy_diff_norm"]:
        logger.info(
            f"    Action-Policy diff norm: min={min(agg_stats['action_policy_diff_norm']):.4f}, "
            f"max={max(agg_stats['action_policy_diff_norm']):.4f}, "
            f"mean={np.mean(agg_stats['action_policy_diff_norm']):.4f}"
        )
    if agg_stats["action_smoothness"]:
        logger.info(
            f"    Action smoothness: min={min(agg_stats['action_smoothness']):.4f}, "
            f"max={max(agg_stats['action_smoothness']):.4f}, "
            f"mean={np.mean(agg_stats['action_smoothness']):.4f}"
        )
    if agg_stats["latent_smoothness"]:
        logger.info(
            f"    Latent smoothness: min={min(agg_stats['latent_smoothness']):.4f}, "
            f"max={max(agg_stats['latent_smoothness']):.4f}, "
            f"mean={np.mean(agg_stats['latent_smoothness']):.4f}"
        )
    logger.info("")
    logger.info("  MPC Planning Diagnostics:")
    if agg_stats["mppi_elite_return_mean"]:
        logger.info(
            f"    Elite returns: min={min(agg_stats['mppi_elite_return_min']):.4f}, "
            f"max={max(agg_stats['mppi_elite_return_max']):.4f}, "
            f"mean={np.mean(agg_stats['mppi_elite_return_mean']):.4f}, "
            f"std={np.mean(agg_stats['mppi_elite_return_std']):.4f}"
        )
    if agg_stats["mppi_final_std"]:
        logger.info(
            f"    Final sampling std: min={min(agg_stats['mppi_final_std']):.4f}, "
            f"max={max(agg_stats['mppi_final_std']):.4f}, "
            f"mean={np.mean(agg_stats['mppi_final_std']):.4f}"
        )
    if agg_stats["mppi_std_convergence"]:
        logger.info(
            f"    Std convergence (init-final): min={min(agg_stats['mppi_std_convergence']):.4f}, "
            f"max={max(agg_stats['mppi_std_convergence']):.4f}, "
            f"mean={np.mean(agg_stats['mppi_std_convergence']):.4f}"
        )


def run_game(
    env,
    agent,
    opponent,
    game_num,
    max_steps=250,
    action_fineness=None,
    algorithm="DDDQN",
    collect_stats=False,
):
    """
    Run a game at full speed (no delays during execution).
    Frame delays are applied later when creating the video.
    """
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    frames = []
    step_count = 0
    total_reward = 0
    # Pre-compute action dimension for random opponent to avoid repeated calls
    if opponent is None:
        if action_fineness is not None:
            # Use fineness-based action dimension
            action_dim = 4 if env.keep_mode else 3
        else:
            action_dim = len(env.discrete_to_continous_action(0))

    # Collect stats if requested (only for TDMPC2)
    game_stats = []
    prev_action_p1 = None
    prev_latent = None
    prev_predicted_next_latent = None
    episode_reward = 0  # Track actual episode reward

    for step in range(max_steps):
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        # Convert observation to float32 for agent
        obs_float = obs.astype(np.float32) if obs.dtype != np.float32 else obs

        # Handle action selection based on algorithm type
        step_stats = None
        if algorithm == "TDMPC2":
            # Use act_with_stats if collecting stats
            if collect_stats and hasattr(agent, "act_with_stats"):
                action_p1, step_stats = agent.act_with_stats(
                    obs_float,
                    deterministic=True,
                    prev_action=prev_action_p1,
                    prev_latent=prev_latent,
                    prev_predicted_next_latent=prev_predicted_next_latent,
                )
                game_stats.append(step_stats)
                # Extract latent, action, and predicted next latent for next iteration
                if step_stats and "_latent_state" in step_stats:
                    prev_latent = step_stats["_latent_state"]
                if step_stats and "_predicted_next_latent" in step_stats:
                    prev_predicted_next_latent = step_stats["_predicted_next_latent"]
                prev_action_p1 = (
                    action_p1.copy()
                    if hasattr(action_p1, "copy")
                    else np.array(action_p1)
                )
            else:
                action_p1 = agent.act(obs_float, deterministic=True)
                prev_action_p1 = (
                    action_p1.copy() if hasattr(action_p1, "copy") else action_p1
                )
            # Ensure action is properly shaped (should be (action_dim,))
            if isinstance(action_p1, (list, tuple)):
                action_p1 = np.array(action_p1)
            if action_p1.ndim > 1:
                action_p1 = action_p1.flatten()
        elif algorithm == "SAC":
            # Continuous action algorithms return actions directly
            action_p1 = agent.act(obs_float, deterministic=True)
            # Ensure action is properly shaped (should be (action_dim,))
            if isinstance(action_p1, (list, tuple)):
                action_p1 = np.array(action_p1)
            if action_p1.ndim > 1:
                action_p1 = action_p1.flatten()
        else:
            # Discrete action algorithms (DDDQN) need conversion
            discrete_action = agent.act(obs_float, deterministic=True)
            # Use fineness-based conversion if fineness is specified
            if action_fineness is not None:
                action_p1 = discrete_to_continuous_action_with_fineness(
                    discrete_action, fineness=action_fineness, keep_mode=env.keep_mode
                )
            else:
                action_p1 = env.discrete_to_continous_action(discrete_action)

        if opponent is not None:
            action_p2 = opponent.act(obs_agent2)
        else:
            action_p2 = np.random.uniform(-1, 1, action_dim)
        action = np.hstack([action_p1, action_p2])
        obs, reward, done, truncated, info = env.step(action)
        obs_agent2 = env.obs_agent_two()
        total_reward += reward
        episode_reward += reward
        step_count += 1
        
        # Track terminal reward prediction if this is the final step
        if (done or truncated) and step_stats is not None:
            step_stats['terminal_reward_actual'] = episode_reward
            step_stats['winner_info'] = info.get('winner', 0)
        
        if done or truncated:
            break

    # Log stats summary if collected
    if collect_stats and algorithm == "TDMPC2" and game_stats:
        log_stats_summary(game_stats, game_num, step_count)

    winner = info.get("winner", 0)
    return frames, step_count, total_reward, winner, info


def find_available_models(base_dir=None, max_results=10):
    """Find available model files in the results directory."""
    if base_dir is None:
        base_dir = os.environ.get("PROJECT_DIR", os.getcwd())

    model_files = []
    results_dir = os.path.join(base_dir, "results")

    # Search in common model locations
    search_paths = [
        os.path.join(results_dir, "tdmpc2_runs"),
        os.path.join(results_dir, "runs"),
        os.path.join(results_dir, "hyperparameter_runs"),
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".pt"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_dir)
                        model_files.append((full_path, rel_path))

    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return model_files[:max_results]


def get_video_filename(base_folder="videos", base_name="agent_games"):
    now = datetime.now()
    dt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    filename = f"{base_name}_{dt_str}.mp4"
    return os.path.join(base_folder, filename)


def main(
    model_path=MODEL_PATH,
    num_games=NUM_GAMES,
    opponent_type=OPPONENT_TYPE,
    pause_between_games=PAUSE_BETWEEN_GAMES,
    frame_delay=FRAME_DELAY,
    max_steps=MAX_STEPS,
    video_fps=VIDEO_FPS,
    action_fineness=ACTION_FINENESS,
    env_mode=ENV_MODE,
):
    # Validate model path exists
    if not os.path.exists(model_path):
        abs_path = os.path.abspath(model_path)
        error_msg = (
            f"Model file not found: {model_path}\n"
            f"Absolute path: {abs_path}\n"
            f"Current working directory: {os.getcwd()}\n\n"
        )

        # Try to find available models
        try:
            available_models = find_available_models()
            if available_models:
                error_msg += f"Found {len(available_models)} available model files:\n"
                for i, (full_path, rel_path) in enumerate(available_models[:5], 1):
                    mtime = os.path.getmtime(full_path)
                    mtime_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(mtime)
                    )
                    error_msg += f"  {i}. {rel_path} (modified: {mtime_str})\n"
                if len(available_models) > 5:
                    error_msg += f"  ... and {len(available_models) - 5} more\n"
                error_msg += (
                    "\nTo use one of these models, set MODEL_PATH environment variable:\n"
                    f"  export MODEL_PATH='$PROJECT_DIR/{available_models[0][1]}'\n"
                    "Or modify the MODEL_PATH constant in the script.\n"
                )
            else:
                error_msg += (
                    "No model files found in results directories.\n"
                    "Please check if models exist or set MODEL_PATH to the correct path.\n"
                )
        except Exception as e:
            error_msg += f"Could not search for available models: {e}\n"

        raise FileNotFoundError(error_msg)

    output_file = get_video_filename()
    logger.info("=" * 60)
    logger.info("Agent Video Recording")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Model path (absolute): {os.path.abspath(model_path)}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Opponent: {opponent_type}")
    logger.info(f"Environment mode: {env_mode}")
    logger.info(f"Max steps per game: {max_steps}")
    logger.info(f"Frame delay in video: {frame_delay}s per frame")
    logger.info(f"Video FPS: {video_fps}")
    logger.info("=" * 60)
    logger.info("Creating hockey environment...")
    # Map string mode to enum
    mode_map = {
        "NORMAL": h_env.Mode.NORMAL,
        "TRAIN_SHOOTING": h_env.Mode.TRAIN_SHOOTING,
        "TRAIN_DEFENSE": h_env.Mode.TRAIN_DEFENSE,
    }
    if env_mode not in mode_map:
        raise ValueError(
            f"Invalid env_mode: {env_mode}. Must be one of {list(mode_map.keys())}"
        )
    env = h_env.HockeyEnv(mode=mode_map[env_mode])
    # Log actual environment limit after creation
    env_limit = env.max_timesteps if hasattr(env, "max_timesteps") else "unknown"
    logger.info(f"Environment created: mode={env_mode}, max_timesteps={env_limit}")
    if env_limit != "unknown" and max_steps > env_limit:
        logger.warning(
            f"Warning: max_steps ({max_steps}) exceeds environment limit ({env_limit}). Games will end at {env_limit} steps."
        )
    state_dim = env.observation_space.shape[0]

    # Detect algorithm type and load appropriate agent
    algorithm = detect_algorithm_from_filename(model_path)

    # Load checkpoint to get the actual action dimension from the model
    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    actual_action_dim = checkpoint.get("action_dim", None)

    if algorithm == "TDMPC2" or algorithm == "SAC":
        # Continuous action algorithms don't need fineness handling
        action_dim = (
            actual_action_dim
            if actual_action_dim is not None
            else (4 if env.keep_mode else 3)
        )
        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Action dimension: {action_dim}")
        agent, detected_algorithm = load_agent(
            model_path, state_dim, action_dim, algorithm=algorithm
        )
        action_fineness = None
    else:
        # Discrete action algorithms (DDDQN) need fineness handling
        if action_fineness is not None:
            discrete_action_dim = get_discrete_action_dim(
                fineness=action_fineness, keep_mode=env.keep_mode
            )
            logger.info(f"Using specified fineness: {action_fineness}")
            if (
                actual_action_dim is not None
                and discrete_action_dim != actual_action_dim
            ):
                logger.warning(
                    f"Warning: Specified fineness {action_fineness} gives action_dim {discrete_action_dim}, but model has {actual_action_dim}"
                )
                discrete_action_dim = actual_action_dim  # Use model's action_dim
        elif actual_action_dim is not None:
            # Try to infer fineness from the model's action_dim
            inferred_fineness = infer_fineness_from_action_dim(
                actual_action_dim, keep_mode=env.keep_mode
            )
            if inferred_fineness is not None:
                action_fineness = inferred_fineness
                discrete_action_dim = actual_action_dim
                logger.info(
                    f"Auto-detected fineness: {action_fineness} from model action_dim: {actual_action_dim}"
                )
            else:
                # Fall back to using model's action_dim but warn about fineness
                discrete_action_dim = actual_action_dim
                logger.warning(
                    f"Could not infer fineness from action_dim {actual_action_dim}"
                )
                logger.warning(
                    "Assuming default fineness=3. If actions seem incorrect, specify action_fineness parameter"
                )
                action_fineness = None  # Will use env.discrete_to_continous_action
        else:
            # Fall back to default (fineness=3)
            discrete_action_dim = 7 if not env.keep_mode else 8
            logger.info(
                f"Using default action dimension (fineness=3): {discrete_action_dim}"
            )
            logger.info(
                "If your model uses a different fineness, specify action_fineness parameter"
            )
            action_fineness = None  # Will use env.discrete_to_continous_action

        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Action dimension: {discrete_action_dim}")
        if action_fineness is not None:
            logger.info(f"Action fineness: {action_fineness}")
        agent, detected_algorithm = load_agent(
            model_path, state_dim, discrete_action_dim, algorithm=algorithm
        )

    algorithm = detected_algorithm
    opponent = None
    if opponent_type == "basic_weak":
        opponent = h_env.BasicOpponent(weak=True)
        logger.info("Using weak BasicOpponent")
    elif opponent_type == "basic_strong":
        opponent = h_env.BasicOpponent(weak=False)
        logger.info("Using strong BasicOpponent")
    elif opponent_type == "random":
        opponent = None
        logger.info("Using random actions for player 2")
    else:
        raise ValueError(f"Unknown opponent_type: {opponent_type}")
    # Get frame shape without extra reset
    obs_temp, _ = env.reset()
    frame_temp = env.render(mode="rgb_array")
    frame_shape = frame_temp.shape
    logger.info(f"Frame shape: {frame_shape}")
    logger.info(f"Running {num_games} games at full speed...")
    all_frames = []
    game_results = []
    start_time = time.time()
    # Enable stats collection for TDMPC2
    collect_stats = algorithm == "TDMPC2"
    if collect_stats:
        logger.info("Stats collection enabled for TDMPC2")
    for game_num in range(1, num_games + 1):
        logger.info(f"Game {game_num}/{num_games}...")
        frames, steps, reward, winner, info = run_game(
            env,
            agent,
            opponent,
            game_num,
            max_steps=max_steps,
            action_fineness=action_fineness,
            algorithm=algorithm,
            collect_stats=collect_stats,
        )
        all_frames.extend(frames)
        game_results.append(
            {"game": game_num, "steps": steps, "reward": reward, "winner": winner}
        )
        winner_str = ""
        if winner == 1:
            winner_str = "Player 1 (Agent) wins!"
        elif winner == -1:
            winner_str = "Player 2 (Opponent) wins!"
        else:
            winner_str = "Draw"
        logger.info(f"  Steps: {steps}, Reward: {reward:.2f}, {winner_str}")
        if game_num < num_games:
            logger.info(f"  Adding {pause_between_games}s pause (blank screen)...")
            blank_frames = create_blank_frames(
                frame_shape, pause_between_games, fps=video_fps
            )
            all_frames.extend(blank_frames)
    execution_time = time.time() - start_time
    logger.info(f"Games completed in {execution_time:.2f} seconds")
    env.close()
    logger.info("=" * 60)
    logger.info("Game Summary")
    logger.info("=" * 60)
    for result in game_results:
        winner_str = (
            "Agent"
            if result["winner"] == 1
            else ("Opponent" if result["winner"] == -1 else "Draw")
        )
        logger.info(
            f"Game {result['game']}: {result['steps']} steps, "
            f"Reward: {result['reward']:.2f}, Winner: {winner_str}"
        )
    wins = sum(1 for r in game_results if r["winner"] == 1)
    losses = sum(1 for r in game_results if r["winner"] == -1)
    draws = sum(1 for r in game_results if r["winner"] == 0)
    logger.info(f"Overall: {wins} wins, {losses} losses, {draws} draws")
    if all_frames:
        try:
            import imageio

            # Apply frame delay by duplicating frames (creates delay in video without slowing execution)
            logger.info(f"Applying frame delay of {frame_delay}s per frame...")
            original_frame_count = len(all_frames)
            all_frames = apply_frame_delay(all_frames, frame_delay, fps=video_fps)
            logger.info(
                f"Expanded from {original_frame_count} to {len(all_frames)} frames for video"
            )

            logger.info(f"Saving {len(all_frames)} frames as MP4 video...")
            logger.info(
                f"Estimated video duration: {len(all_frames) / video_fps / 60:.1f} minutes"
            )
            logger.info("This may take 15-60 minutes depending on your CPU...")
            encoding_start = time.time()
            # Optimize video encoding for speed: use faster preset
            # Note: imageio automatically handles pixel format, so we don't need to specify it
            imageio.mimsave(
                output_file,
                all_frames,
                fps=video_fps,
                codec="libx264",
                quality=8,
                ffmpeg_params=["-preset", "fast"],
            )
            encoding_time = time.time() - encoding_start
            logger.info(f"Video encoding completed in {encoding_time / 60:.1f} minutes")
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            logger.info(f"Video saved to '{output_file}'")
            logger.info(f"File size: {file_size:.2f} MB")
        except ImportError:
            logger.error("=" * 60)
            logger.error("ERROR: imageio not installed!")
            logger.error("=" * 60)
            logger.error("Please install imageio to save videos:")
            logger.error("  pip install imageio imageio-ffmpeg")
            logger.warning("Saving frames as numpy array instead...")
            np.savez(output_file.replace(".mp4", ".npz"), frames=np.array(all_frames))
            logger.info(f"Frames saved to '{output_file.replace('.mp4', '.npz')}'")
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            logger.warning("Saving frames as numpy array as backup...")
            np.savez(output_file.replace(".mp4", ".npz"), frames=np.array(all_frames))
            logger.info(f"Frames saved to '{output_file.replace('.mp4', '.npz')}'")
    else:
        logger.warning("No frames collected - video not saved.")
    logger.info("Done!")


if __name__ == "__main__":
    main()
