import warnings
import os
import sys
import numpy as np
import hockey.hockey_env as h_env
import time
from datetime import datetime
import logging
from rl_hockey.common.utils import get_discrete_action_dim, discrete_to_continuous_action_with_fineness

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("hockey_video")


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

MODEL_PATH = "results/hyperparameter_runs/2026-01-03_18-24-14/run_lr1e04_bs256_h512_512_512_31fb74b2_0002/2026-01-03_18-24-17/models/run_lr1e04_bs256_h512_512_512_31fb74b2_0002.pt"
NUM_GAMES = 25
OPPONENT_TYPE = "basic_strong"
PAUSE_BETWEEN_GAMES = 1.5
FRAME_DELAY = 0.05  # Delay in video playback (seconds per frame). Execution runs at full speed.
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

def load_ddqn_agent(model_path, state_dim, action_dim):
    from rl_hockey.DDDQN import DDDQN

    agent = DDDQN(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=[256, 256, 256]
    )
    logger.info(f"Loading model from: {model_path}")
    agent.load(model_path)
    logger.info("Model loaded successfully!")
    return agent


def load_td3_agent(model_path, agent_config_path, state_dim, action_dim):
    from rl_hockey.td3.td3 import TD3
    import json

    with open(agent_config_path, "r") as f:
        agent_config = json.load(f)
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config["agent"]["hyperparameters"],
    )
    logger.info(f"Loading model from: {model_path}")
    agent.load(model_path)
    logger.info("Model loaded successfully!")
    return agent


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

def run_game(env, agent, opponent, game_num, max_steps=250, action_fineness=None):
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
    for step in range(max_steps):
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        # Convert observation to float32 for agent
        obs_float = obs.astype(np.float32) if obs.dtype != np.float32 else obs
        discrete_action = agent.act(obs_float, deterministic=True)
        # Use fineness-based conversion if fineness is specified
        if action_fineness is not None:
            action_p1 = discrete_to_continuous_action_with_fineness(
                discrete_action, 
                fineness=action_fineness, 
                keep_mode=env.keep_mode
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
        step_count += 1
        if done or truncated:
            break
    winner = info.get("winner", 0)
    return frames, step_count, total_reward, winner, info


def get_video_filename(base_folder="videos", base_name="td3_games"):
    now = datetime.now()
    dt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    filename = f"{base_name}_{dt_str}.mp4"
    return os.path.join(base_folder, filename)

def main(model_path=MODEL_PATH, num_games=NUM_GAMES, opponent_type=OPPONENT_TYPE, pause_between_games=PAUSE_BETWEEN_GAMES, frame_delay=FRAME_DELAY, max_steps=MAX_STEPS, video_fps=VIDEO_FPS, action_fineness=ACTION_FINENESS, env_mode=ENV_MODE):
    output_file = get_video_filename()
    logger.info("=" * 60)
    logger.info("DDDQN Agent Video Recording")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Opponent: {opponent_type}")
    logger.info(f"Environment mode: {env_mode}")
    logger.info(f"Max steps per game: {max_steps}")
    logger.info(f"Frame delay in video: {frame_delay}s per frame")
    logger.info(f"Video FPS: {video_fps}")
    logger.info("="*60)
    logger.info("Creating hockey environment...")
    # Map string mode to enum
    mode_map = {
        "NORMAL": h_env.Mode.NORMAL,
        "TRAIN_SHOOTING": h_env.Mode.TRAIN_SHOOTING,
        "TRAIN_DEFENSE": h_env.Mode.TRAIN_DEFENSE
    }
    if env_mode not in mode_map:
        raise ValueError(f"Invalid env_mode: {env_mode}. Must be one of {list(mode_map.keys())}")
    env = h_env.HockeyEnv(mode=mode_map[env_mode])
    # Log actual environment limit after creation
    env_limit = env.max_timesteps if hasattr(env, 'max_timesteps') else 'unknown'
    logger.info(f"Environment created: mode={env_mode}, max_timesteps={env_limit}")
    if env_limit != 'unknown' and max_steps > env_limit:
        logger.warning(f"Warning: max_steps ({max_steps}) exceeds environment limit ({env_limit}). Games will end at {env_limit} steps.")
    state_dim = env.observation_space.shape[0]
    
    # Load checkpoint to get the actual action dimension from the model
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    actual_action_dim = checkpoint.get('action_dim', None)
    
    # Determine action dimension and fineness
    if action_fineness is not None:
        discrete_action_dim = get_discrete_action_dim(fineness=action_fineness, keep_mode=env.keep_mode)
        logger.info(f"Using specified fineness: {action_fineness}")
        if actual_action_dim is not None and discrete_action_dim != actual_action_dim:
            logger.warning(f"Warning: Specified fineness {action_fineness} gives action_dim {discrete_action_dim}, but model has {actual_action_dim}")
            discrete_action_dim = actual_action_dim  # Use model's action_dim
    elif actual_action_dim is not None:
        # Try to infer fineness from the model's action_dim
        inferred_fineness = infer_fineness_from_action_dim(actual_action_dim, keep_mode=env.keep_mode)
        if inferred_fineness is not None:
            action_fineness = inferred_fineness
            discrete_action_dim = actual_action_dim
            logger.info(f"Auto-detected fineness: {action_fineness} from model action_dim: {actual_action_dim}")
        else:
            # Fall back to using model's action_dim but warn about fineness
            discrete_action_dim = actual_action_dim
            logger.warning(f"Could not infer fineness from action_dim {actual_action_dim}")
            logger.warning("Assuming default fineness=3. If actions seem incorrect, specify action_fineness parameter")
            action_fineness = None  # Will use env.discrete_to_continous_action
    else:
        # Fall back to default (fineness=3)
        discrete_action_dim = 7 if not env.keep_mode else 8
        logger.info(f"Using default action dimension (fineness=3): {discrete_action_dim}")
        logger.info("If your model uses a different fineness, specify action_fineness parameter")
        action_fineness = None  # Will use env.discrete_to_continous_action
    
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {discrete_action_dim}")
    if action_fineness is not None:
        logger.info(f"Action fineness: {action_fineness}")
    agent = load_ddqn_agent(model_path, state_dim, discrete_action_dim)
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
    for game_num in range(1, num_games + 1):
        logger.info(f"Game {game_num}/{num_games}...")
        frames, steps, reward, winner, info = run_game(env, agent, opponent, game_num, max_steps=max_steps, action_fineness=action_fineness)
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
            blank_frames = create_blank_frames(frame_shape, pause_between_games, fps=video_fps)
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
            logger.info(f"Expanded from {original_frame_count} to {len(all_frames)} frames for video")
            
            logger.info(f"Saving {len(all_frames)} frames as MP4 video...")
            logger.info(f"Estimated video duration: {len(all_frames) / video_fps / 60:.1f} minutes")
            logger.info("This may take 15-60 minutes depending on your CPU...")
            encoding_start = time.time()
            # Optimize video encoding for speed: use faster preset
            # Note: imageio automatically handles pixel format, so we don't need to specify it
            imageio.mimsave(
                output_file, 
                all_frames, 
                fps=video_fps, 
                codec='libx264', 
                quality=8,
                ffmpeg_params=['-preset', 'fast']
            )
            encoding_time = time.time() - encoding_start
            logger.info(f"Video encoding completed in {encoding_time / 60:.1f} minutes")
            file_size = os.path.getsize(output_file) / (1024*1024)
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
