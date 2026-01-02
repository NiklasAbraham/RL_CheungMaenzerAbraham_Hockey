import warnings
import os
import sys
import numpy as np
import hockey.hockey_env as h_env
import time
from datetime import datetime
import logging

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("ddqn_video")

class ALSAFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, message):
        if 'ALSA' not in message and 'pkg_resources' not in message:
            self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()

if sys.platform == 'linux':
    sys.stderr = ALSAFilter(sys.stderr)

MODEL_PATH = "models/dddqn/hockey-shooting-ddqn_25k_2026-01-02_10-05-20.pt"
NUM_GAMES = 10
OPPONENT_TYPE = "basic_weak"
PAUSE_BETWEEN_GAMES = 1.5
FRAME_DELAY = 0.1 # 20 FPS = 0.05 seconds per frame # 50 FPS = 0.02 seconds per frame # 10 FPS = 0.1 seconds per frame

def load_ddqn_agent(model_path, state_dim, action_dim):
    from rl_hockey.DDDQN import DDDQN
    agent = DDDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=[256, 256, 256]
    )
    logger.info(f"Loading model from: {model_path}")
    agent.load(model_path)
    logger.info("Model loaded successfully!")
    return agent

def create_blank_frames(frame_shape, duration_seconds, fps=50):
    num_frames = int(duration_seconds * fps)
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    return [blank_frame.copy() for _ in range(num_frames)]

def run_game(env, agent, opponent, game_num, max_steps=250, frame_delay=FRAME_DELAY):
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    frames = []
    step_count = 0
    total_reward = 0
    for step in range(max_steps):
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        discrete_action = agent.act(obs.astype(np.float32), deterministic=True)
        action_p1 = env.discrete_to_continous_action(discrete_action)
        if opponent is not None:
            action_p2 = opponent.act(obs_agent2)
        else:
            action_p2 = np.random.uniform(-1, 1, len(action_p1))
        action = np.hstack([action_p1, action_p2])
        obs, reward, done, truncated, info = env.step(action)
        obs_agent2 = env.obs_agent_two()
        total_reward += reward
        step_count += 1
        time.sleep(frame_delay)
        if done or truncated:
            break
    winner = info.get('winner', 0)
    return frames, step_count, total_reward, winner, info

def get_video_filename(base_folder="videos", base_name="ddqn_games"):
    now = datetime.now()
    dt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    filename = f"{base_name}_{dt_str}.mp4"
    return os.path.join(base_folder, filename)

def main(model_path=MODEL_PATH, num_games=NUM_GAMES, opponent_type=OPPONENT_TYPE, pause_between_games=PAUSE_BETWEEN_GAMES, frame_delay=FRAME_DELAY):
    output_file = get_video_filename()
    logger.info("="*60)
    logger.info("DDDQN Agent Video Recording")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Opponent: {opponent_type}")
    logger.info("="*60)
    logger.info("Creating hockey environment...")
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = 7 if not env.keep_mode else 8
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {discrete_action_dim}")
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
    obs_temp, _ = env.reset()
    frame_temp = env.render(mode="rgb_array")
    frame_shape = frame_temp.shape
    logger.info(f"Frame shape: {frame_shape}")
    env.reset()
    logger.info(f"Running {num_games} games...")
    all_frames = []
    game_results = []
    for game_num in range(1, num_games + 1):
        logger.info(f"Game {game_num}/{num_games}...")
        frames, steps, reward, winner, info = run_game(env, agent, opponent, game_num, max_steps=250, frame_delay=frame_delay)
        all_frames.extend(frames)
        game_results.append({
            'game': game_num,
            'steps': steps,
            'reward': reward,
            'winner': winner
        })
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
            blank_frames = create_blank_frames(frame_shape, pause_between_games, fps=50)
            all_frames.extend(blank_frames)
    env.close()
    logger.info("="*60)
    logger.info("Game Summary")
    logger.info("="*60)
    for result in game_results:
        winner_str = "Agent" if result['winner'] == 1 else ("Opponent" if result['winner'] == -1 else "Draw")
        logger.info(f"Game {result['game']}: {result['steps']} steps, "
                    f"Reward: {result['reward']:.2f}, Winner: {winner_str}")
    wins = sum(1 for r in game_results if r['winner'] == 1)
    losses = sum(1 for r in game_results if r['winner'] == -1)
    draws = sum(1 for r in game_results if r['winner'] == 0)
    logger.info(f"Overall: {wins} wins, {losses} losses, {draws} draws")
    if all_frames:
        try:
            import imageio
            logger.info(f"Saving {len(all_frames)} frames as MP4 video...")
            imageio.mimsave(output_file, all_frames, fps=50, codec='libx264', quality=8)
            file_size = os.path.getsize(output_file) / (1024*1024)
            logger.info(f"Video saved to '{output_file}'")
            logger.info(f"File size: {file_size:.2f} MB")
        except ImportError:
            logger.error("="*60)
            logger.error("ERROR: imageio not installed!")
            logger.error("="*60)
            logger.error("Please install imageio to save videos:")
            logger.error("  pip install imageio imageio-ffmpeg")
            logger.warning("Saving frames as numpy array instead...")
            np.savez(output_file.replace('.mp4', '.npz'), frames=np.array(all_frames))
            logger.info(f"Frames saved to '{output_file.replace('.mp4', '.npz')}'")
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            logger.warning("Saving frames as numpy array as backup...")
            np.savez(output_file.replace('.mp4', '.npz'), frames=np.array(all_frames))
            logger.info(f"Frames saved to '{output_file.replace('.mp4', '.npz')}'")
    else:
        logger.warning("No frames collected - video not saved.")
    logger.info("Done!")

if __name__ == "__main__":
    main()
