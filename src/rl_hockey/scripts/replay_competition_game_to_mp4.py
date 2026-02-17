"""
Replay a competition server game from a pickle file and save it as an MP4 video.

The pickle file is expected to contain:
  - user_ids: list of 2 ids
  - user_names: list of 2 names (used as P1/P2 labels)
  - rounds: list of round dicts, each with:
    - actions: list of joint actions (8-dim per step)
    - observations: list of observations (18-dim, length = len(actions)+1)
    - rewards: list of rewards per step
    - score: (score_p1, score_p2)

Replay is done by resetting the hockey env and stepping with the stored actions.
"""

import logging
import os
import pickle
import sys
import warnings
from datetime import datetime

import hockey.hockey_env as h_env
import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("replay_competition")


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


def _project_root():
    """Project root from script location (script is at PROJECT_ROOT/src/rl_hockey/scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))


def _resolve_path(path, root=None):
    """If path is relative, join with root (default: project root). Otherwise return as is."""
    if path is None or path == "":
        return path
    if os.path.isabs(path):
        return path
    root = root or _project_root()
    return os.path.normpath(os.path.join(root, path))


def _default_font(size=16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size
            )
        except OSError:
            return ImageFont.load_default()


def add_agent_labels(frame, label_p1, label_p2):
    """Draw agent labels on frame (mutates frame in place)."""
    if not label_p1 and not label_p2:
        return
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    font = _default_font(18)
    pad = 6
    if label_p1:
        text_p1 = f"P1: {label_p1}"
        bbox = draw.textbbox((0, 0), text_p1, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            [0, 0, w + 2 * pad, h + 2 * pad], fill=(0, 0, 0), outline=(255, 255, 255)
        )
        draw.text((pad, pad), text_p1, fill=(255, 255, 255), font=font)
    if label_p2:
        text_p2 = f"P2: {label_p2}"
        bbox = draw.textbbox((0, 0), text_p2, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = frame.shape[1] - w - 2 * pad
        draw.rectangle(
            [x, 0, x + w + 2 * pad, h + 2 * pad],
            fill=(0, 0, 0),
            outline=(255, 255, 255),
        )
        draw.text((x + pad, pad), text_p2, fill=(255, 255, 255), font=font)
    frame[:] = np.array(pil)


def create_blank_frames(
    frame_shape, duration_seconds, fps=50, label_p1=None, label_p2=None
):
    num_frames = int(duration_seconds * fps)
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    out = []
    for _ in range(num_frames):
        f = blank_frame.copy()
        if label_p1 or label_p2:
            add_agent_labels(f, label_p1, label_p2)
        out.append(f)
    return out


def apply_frame_delay(frames, frame_delay, fps=50):
    """Duplicate frames to create delay effect in video."""
    if frame_delay <= 0:
        return frames
    frames_per_step = int(frame_delay * fps)
    delayed_frames = []
    for frame in frames:
        delayed_frames.extend([frame] * frames_per_step)
    return delayed_frames


def load_competition_game(pkl_path):
    """Load competition game from pickle. Returns dict with user_ids, user_names, rounds."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "rounds" not in data or "user_names" not in data:
        raise ValueError(
            f"Invalid competition pkl: expected 'rounds' and 'user_names'. Got keys: {list(data.keys())}"
        )
    return data


def replay_round(env, round_data, label_p1=None, label_p2=None):
    """
    Replay one round: reset env, then step with stored actions and collect frames.
    round_data: dict with 'actions' (list of 8-dim arrays) and optionally 'observations'.
    Returns list of frames (numpy rgb arrays).
    """
    actions = round_data["actions"]
    if not actions:
        return []

    obs, info = env.reset()
    frames = []
    frame = env.render(mode="rgb_array")
    frames.append(frame.copy())
    if label_p1 or label_p2:
        add_agent_labels(frames[-1], label_p1, label_p2)

    for i, a in enumerate(actions):
        action = np.array(a, dtype=np.float64)
        if action.shape != (8,):
            action = action.flatten()
        if action.size != 8:
            raise ValueError(
                f"Expected action size 8 (joint action), got {action.size} at step {i}"
            )
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render(mode="rgb_array")
        frames.append(frame.copy())
        if label_p1 or label_p2:
            add_agent_labels(frames[-1], label_p1, label_p2)
        if done or truncated:
            break

    return frames


def get_video_filename(base_folder, base_name="competition_replay"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    now = datetime.now()
    dt_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_name}_{dt_str}.mp4"
    return os.path.join(base_folder, filename)


def main(
    pkl_path,
    round_indices=None,
    video_output_dir=None,
    output_path=None,
    video_fps=50,
    frame_delay=0.05,
    pause_between_rounds=1.5,
    env_mode="NORMAL",
):
    """
    Replay competition game(s) from pkl and save as MP4.

    Parameters
    ----------
    pkl_path : str
        Path to the competition game pickle file.
    round_indices : list of int or None
        Which rounds to include (0-based). None = all rounds.
    video_output_dir : str or None
        Directory for output video. None = project root / videos.
    output_path : str or None
        Explicit output file path. If set, overrides video_output_dir + default name.
    video_fps : int
        Frames per second of the output video.
    frame_delay : float
        Seconds per env step in the video (frames are duplicated to achieve this).
    pause_between_rounds : float
        Seconds of blank screen between rounds.
    env_mode : str
        Hockey env mode: "NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE".
    """
    pkl_path = _resolve_path(pkl_path)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Competition game not found: {pkl_path}")

    data = load_competition_game(pkl_path)
    rounds = data["rounds"]
    user_names = data.get("user_names", ["P1", "P2"])
    label_p1 = str(user_names[0]) if len(user_names) > 0 else None
    label_p2 = str(user_names[1]) if len(user_names) > 1 else None

    if round_indices is None:
        round_indices = list(range(len(rounds)))
    else:
        for i in round_indices:
            if i < 0 or i >= len(rounds):
                raise ValueError(
                    f"round_indices contains {i}; valid range is 0..{len(rounds) - 1}"
                )

    mode_map = {
        "NORMAL": h_env.Mode.NORMAL,
        "TRAIN_SHOOTING": h_env.Mode.TRAIN_SHOOTING,
        "TRAIN_DEFENSE": h_env.Mode.TRAIN_DEFENSE,
    }
    if env_mode not in mode_map:
        raise ValueError(
            f"Invalid env_mode: {env_mode}. Must be one of {list(mode_map.keys())}"
        )

    if output_path:
        output_file = _resolve_path(output_path)
        base_folder = os.path.dirname(output_file)
    else:
        base_folder = video_output_dir or os.path.join(_project_root(), "videos")
        if not os.path.isabs(base_folder):
            base_folder = _resolve_path(base_folder)
        base_name = "competition_replay"
        output_file = get_video_filename(base_folder, base_name)

    logger.info("=" * 60)
    logger.info("Competition game replay")
    logger.info("=" * 60)
    logger.info(f"Input: {pkl_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Rounds: {round_indices} (of {len(rounds)})")
    logger.info(f"Labels: P1={label_p1}, P2={label_p2}")
    logger.info(f"Video FPS: {video_fps}, frame_delay: {frame_delay}s")
    logger.info("=" * 60)

    env = h_env.HockeyEnv(mode=mode_map[env_mode])
    frame_shape = env.render(mode="rgb_array").shape

    all_frames = []
    for idx, r in enumerate(round_indices):
        round_data = rounds[r]
        num_actions = len(round_data["actions"])
        logger.info(f"Replaying round {r} ({num_actions} steps)...")
        frames = replay_round(env, round_data, label_p1=label_p1, label_p2=label_p2)
        all_frames.extend(frames)
        score = round_data.get("score", (0, 0))
        logger.info(f"  Frames: {len(frames)}, score: {score}")

        if idx < len(round_indices) - 1 and pause_between_rounds > 0:
            blank = create_blank_frames(
                frame_shape,
                pause_between_rounds,
                fps=video_fps,
                label_p1=label_p1,
                label_p2=label_p2,
            )
            all_frames.extend(blank)

    env.close()

    if not all_frames:
        logger.warning("No frames collected; video not saved.")
        return

    logger.info(f"Applying frame delay {frame_delay}s per step...")
    all_frames = apply_frame_delay(all_frames, frame_delay, fps=video_fps)
    logger.info(f"Total frames for video: {len(all_frames)}")

    try:
        import imageio

        logger.info(f"Saving MP4 to {output_file}...")
        imageio.mimsave(
            output_file,
            all_frames,
            fps=video_fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-preset", "fast"],
        )
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Video saved. Size: {size_mb:.2f} MB")
    except ImportError:
        logger.error(
            "imageio not installed. Install with: pip install imageio imageio-ffmpeg"
        )
        fallback = output_file.replace(".mp4", ".npz")
        np.savez(fallback, frames=np.array(all_frames))
        logger.info(f"Frames saved as {fallback}")
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        fallback = output_file.replace(".mp4", ".npz")
        np.savez(fallback, frames=np.array(all_frames))
        logger.info(f"Frames saved as {fallback}")

    logger.info("Done.")


if __name__ == "__main__":
    import sys

    pkl_path = "results/competition_games/f0297633-7a84-41a2-bd05-8d3c0f171a9a.pkl"
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]

    main(
        pkl_path=pkl_path,
        round_indices=None,
        video_output_dir=None,
        output_path=None,
        video_fps=50,
        frame_delay=0.05,
        pause_between_rounds=1.5,
        env_mode="NORMAL",
    )
