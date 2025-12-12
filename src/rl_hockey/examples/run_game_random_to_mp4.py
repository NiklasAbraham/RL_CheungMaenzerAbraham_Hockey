"""
Simple script to run one hockey game and save as MP4 video.

This script demonstrates a basic game between two random agents
and saves the game as an MP4 video file.

By default, saves to 'game.mp4' in the current directory.
Works perfectly on remote SSH servers without display.

Configuration:
Edit the CONFIG section below to change settings.
"""

# ============================================================================
# CONFIGURATION - Edit these settings as needed
# ============================================================================

# Output video filename
OUTPUT_FILE = "game.mp4"

# Render mode options:
#   "video"   - Save as MP4 video (default, works on remote servers)
#   "display" - Try to show in window (requires display/X11)
#   "none"    - No rendering (fastest, no video output)
RENDER_MODE = "video"

# ============================================================================

import warnings
import os
import sys
import numpy as np
import hockey.hockey_env as h_env
import time

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
# Suppress pygame support prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Filter out ALSA audio warnings (harmless - just means no audio hardware)
# These warnings don't affect game rendering
class ALSAFilter:
    """Filter to suppress ALSA audio warnings while keeping other errors."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        # Only suppress ALSA-related messages
        if 'ALSA' not in message and 'pkg_resources' not in message:
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Apply filter on Linux systems
if sys.platform == 'linux':
    sys.stderr = ALSAFilter(sys.stderr)


def check_display():
    """Check if a display is available."""
    display = os.environ.get('DISPLAY')
    if display is None:
        return False
    # Try to import pygame to check if display works
    try:
        import pygame
        pygame.init()
        pygame.quit()
        return True
    except:
        return False


def main():
    # Create the environment
    print("Creating hockey environment...")
    env = h_env.HockeyEnv()
    
    # Determine render mode from config
    render_mode = None
    
    if RENDER_MODE == "none":
        render_mode = None
        print("Running without rendering (RENDER_MODE = 'none')")
    elif RENDER_MODE == "display":
        # Try to use display if requested
        has_display = check_display()
        if has_display:
            render_mode = "human"
            print("Display detected - rendering in window")
        else:
            print("WARNING: RENDER_MODE = 'display' but no display available!")
            print("Falling back to saving video instead...")
            render_mode = "rgb_array"
    else:
        # Default: save as video
        render_mode = "rgb_array"
        print(f"Saving game as video: {OUTPUT_FILE}")
    
    # Reset to start a new game
    print("\nStarting new game...")
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    if render_mode == "human":
        print("\nGame started! Watch the window...")
        print("Close the window or press Ctrl+C to stop.\n")
    elif render_mode == "rgb_array":
        print(f"\nGame started! Recording frames for video...\n")
    else:
        print("\nGame started! Running simulation (no rendering)...\n")
    
    # Game loop
    step_count = 0
    total_reward = 0
    frames = []
    
    try:
        for step in range(250):  # Maximum 250 steps in NORMAL mode
            # Render the game
            if render_mode is not None:
                if render_mode == "rgb_array":
                    frame = env.render(mode="rgb_array")
                    frames.append(frame)
                else:
                    env.render(mode=render_mode)
            
            # Sample random actions for both players
            # Action format: [x_force, y_force, torque, shoot] for each player
            a1 = np.random.uniform(-1, 1, 4)  # Player 1 actions
            a2 = np.random.uniform(-1, 1, 4)  # Player 2 actions
            
            # Combine actions and step the environment
            action = np.hstack([a1, a2])
            obs, reward, done, truncated, info = env.step(action)
            obs_agent2 = env.obs_agent_two()
            
            total_reward += reward
            step_count += 1
            
            # Small delay to make the game visible (only if rendering)
            if render_mode == "human":
                time.sleep(0.02)  # ~50 FPS
            
            # Check if game ended
            if done or truncated:
                break
        
        # Print game results
        print("\n" + "="*50)
        print("Game finished!")
        print("="*50)
        print(f"Total steps: {step_count}")
        print(f"Total reward (Player 1): {total_reward:.2f}")
        print(f"Winner: ", end="")
        if info['winner'] == 1:
            print("Player 1 (Red) wins!")
        elif info['winner'] == -1:
            print("Player 2 (Blue) wins!")
        else:
            print("Draw - no winner")
        print(f"\nFinal info: {info}")
        print("="*50)
        
        # Save video if frames were collected
        if render_mode == "rgb_array" and frames:
            try:
                import imageio
                print(f"\nSaving {len(frames)} frames as MP4 video...")
                imageio.mimsave(OUTPUT_FILE, frames, fps=50, codec='libx264', quality=8)
                print(f"✓ Video saved to '{OUTPUT_FILE}'")
                print(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
            except ImportError:
                print("\n" + "="*60)
                print("ERROR: imageio not installed!")
                print("="*60)
                print("Please install imageio to save videos:")
                print("  pip install imageio imageio-ffmpeg")
                print("\nSaving frames as numpy array instead...")
                np.savez(OUTPUT_FILE.replace('.mp4', '.npz'), frames=np.array(frames))
                print(f"Frames saved to '{OUTPUT_FILE.replace('.mp4', '.npz')}'")
                print("You can convert this later using imageio.")
            except Exception as e:
                print(f"\nError saving video: {e}")
                print("Saving frames as numpy array as backup...")
                np.savez(OUTPUT_FILE.replace('.mp4', '.npz'), frames=np.array(frames))
                print(f"Frames saved to '{OUTPUT_FILE.replace('.mp4', '.npz')}'")
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    
    finally:
        # Always close the environment
        env.close()
        print("\nEnvironment closed.")


if __name__ == "__main__":
    main()

