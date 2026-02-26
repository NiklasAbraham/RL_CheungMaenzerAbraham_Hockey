# TD-MPC2: Scripts, Functions, and Figures

This document describes runnable Python scripts, main functions, and report figures related to the TD-MPC2 agent and reward backpropagation in the Hockey RL project.

---

## 1. Runnable scripts

### 1.1 Training

**Script:** `src/rl_hockey/common/training/train_tdmpc2.py`

Entry point for TD-MPC2 curriculum training. It calls `train_vectorized` with a TD-MPC2 curriculum config.

**How to run (from project root, with conda env active):**
```bash
python -m rl_hockey.common.training.train_tdmpc2
# Or with resume:
# Edit __main__ to call main(resume_from="./results/tdmpc2_runs/YYYY-MM-DD_HH-MM-SS")
```

**Main function:**
```python
def main(
    config_path: str = CONFIG_PATH,
    result_dir: str = RESULT_DIR,
    archive_dir: str = ARCHIVE_DIR,
    num_envs: int = NUM_ENVS,
    verbose: bool = True,
    resume_from: Optional[str] = None,
):
```
- `config_path`: Path to curriculum JSON (default `./configs/curriculum_tdmpc2_mixed_opponents.json`).
- `result_dir`: Output directory for runs (default `./results/tdmpc2_runs_horizon`).
- `resume_from`: Optional path to a run directory to continue training (e.g. `./results/tdmpc2_runs/2026-02-01_09-55-22`).

---

### 1.2 Plot episode logs (run folder)

**Script:** `src/rl_hockey/TD_MPC2/plot_episode_logs.py`

Reads episode log CSVs from a TD-MPC2 run folder (including checkpoint CSVs), combines them, and plots reward, shaped reward, loss curves, and reward distribution.

**How to run:**
```bash
python -m rl_hockey.TD_MPC2.plot_episode_logs
# Or from code:
from rl_hockey.TD_MPC2.plot_episode_logs import plot_episode_logs
plot_episode_logs("results/tdmpc2_runs/2026-02-01_09-55-22", window_size=250)
```

**Main function:**
```python
def plot_episode_logs(
    folder_path: str,
    window_size: int = 500,
    save_path: Optional[Path] = None,
):
```
- `folder_path`: Run directory containing `csvs/` (and optionally `plots/`). Expects `csvs/*_episode_logs.csv`.
- `window_size`: Moving average window for smoothing.
- `save_path`: Optional; default is `folder_path/plots/{run_name}_episode_logs.png`.

**Notable helpers in this module:**
- `load_episode_logs_from_csv(csv_path)`: Load one CSV; columns include `episode`, `reward`, `shaped_reward`, `backprop_reward`, and loss columns.
- `find_all_episode_log_files(csvs_dir, run_name)`: Collect main and checkpoint `*_episode_logs.csv` files.
- `combine_episode_logs(log_files)`: Merge logs and deduplicate by episode.

---

### 1.3 Plot episode logs (report figure)

**Script:** `report/figures/plot_episode_logs.py`

Same idea as above but tailored for the report: reads a single episode log CSV, uses tueplots styling, and saves a publication-style figure to `report/figures/`.

**How to run:**
```bash
cd report/figures
# Set EPISODE_LOG_CSV at top of file to your run's CSV, e.g.:
# EPISODE_LOG_CSV = "results/tdmpc2_runs_horizon/2026-02-13_13-49-57/csvs/..._episode_logs.csv"
python plot_episode_logs.py
```

**Main function:**
```python
def plot_episode_logs(
    csv_path: Path = EPISODE_LOG_CSV,
    output_dir: Path = OUTPUT_DIR,
    window_size: int = WINDOW_SIZE,
    save_name: str = "episode_logs",
) -> None:
```
- Output: `output_dir/{save_name}.png` (default `report/figures/episode_logs.png`).

---

### 1.5 Reward backpropagation analysis

**Script:** `src/rl_hockey/scripts/reward_sprase.py`

Analyses sparse rewards and the effect of reward backpropagation: runs random-play episodes, then plots per-step reward before/after backprop and a scatter of episode totals. Saves the figure to `report/figures/reward_backprop_analysis.png`.

**How to run:**
```bash
python -m rl_hockey.scripts.reward_sprase
# Or: python src/rl_hockey/scripts/reward_sprase.py  (from project root with PYTHONPATH=src)
```

**Main logic:**
- `play_random_games(num_games, max_steps)`: Simulates random actions for both players; returns dict with `player1_wins`, `player2_wins`, `draws`, `all`.
- `plot_backprop_analysis(episodes, win_reward_bonus, win_reward_discount, output_dir)`: Builds the two-panel figure (per-step trace for a representative winning episode; scatter of original vs after-backprop episode totals) and saves to `output_dir/reward_backprop_analysis.png`.
- `main()`: Calls `play_random_games()`, `print_summary()`, and `plot_backprop_analysis()`.

Backpropagation uses `rl_hockey.common.reward_backprop.apply_win_reward_backprop` (same logic as in `TDMPC2ReplayBuffer` when flushing a winning episode).

---

## 2. Reward backpropagation

**Module:** `src/rl_hockey/common/reward_backprop.py`

**Function:**
```python
def apply_win_reward_backprop(
    rewards,
    winner,
    win_reward_bonus=0.0,
    win_reward_discount=0.99,
    use_torch=False,
):
```
- Applies a discounted win bonus backwards through the episode when `winner == 1`, matching `TDMPC2ReplayBuffer._flush_episode()`.
- Returns: `(rewards_out, original_rewards, bonus_rewards)`.

During training, `train.py` computes a scalar **backprop_reward** per episode via `_compute_backprop_reward()`: if the agent uses a buffer with `win_reward_bonus` and `win_reward_discount`, it uses `apply_win_reward_backprop()` on the episode rewards and returns the sum of the modified rewards; otherwise it falls back to the episode’s shaped reward. This value is written to the episode log CSV as `backprop_reward` and can be plotted by the episode-log plotting scripts above.
