"""
Extract speed metrics (steps per second) per episode from a training log file (e.g. test.txt).
Uses timestamps and ep_steps from lines like: [INFO] Ep1: reward=... | ep_steps=37 | ...
"""

import re
from datetime import datetime
from pathlib import Path


def parse_episode_line(line: str):
    """Return (timestamp, episode_num, ep_steps) or None if not an episode log line."""
    # [2026-02-13 13:57:29] [INFO] Ep1: reward=-11.40 | ep_steps=37 | ...
    m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+\[INFO\]\s+Ep(\d+):\s+.*?\bep_steps=(\d+)", line)
    if not m:
        return None
    ts_str, ep_num_str, steps_str = m.groups()
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return ts, int(ep_num_str), int(steps_str)
    except ValueError:
        return None


def compute_speed_metrics(
    log_path: str = "test.txt",
    output_path: str | None = None,
):
    """
    Read log file, compute steps/sec per episode (from consecutive timestamps and ep_steps),
    then print and optionally write summary (avg, var, std, min, max).

    Speed for episode N is computed as ep_steps_N / (timestamp_N - timestamp_{N-1}).
    Episode 1 has no previous timestamp so its speed is not computed.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        parsed = parse_episode_line(line)
        if parsed is not None:
            rows.append(parsed)

    if len(rows) < 2:
        print("Need at least 2 episode lines to compute speed. Found:", len(rows))
        return

    # Build per-episode stats: (episode_num, ep_steps, duration_s, steps_per_sec)
    results = []
    for i in range(1, len(rows)):
        t_prev, ep_prev, steps_prev = rows[i - 1]
        t_curr, ep_curr, steps_curr = rows[i]
        if ep_curr != ep_prev + 1:
            continue
        duration_s = (t_curr - t_prev).total_seconds()
        if duration_s <= 0:
            continue
        steps_per_sec = steps_curr / duration_s
        results.append((ep_curr, steps_curr, duration_s, steps_per_sec))

    if not results:
        print("No consecutive episode pairs with positive duration.")
        return

    episodes = [r[0] for r in results]
    steps_per_ep = [r[1] for r in results]
    durations_s = [r[2] for r in results]
    speeds = [r[3] for r in results]

    avg_steps_per_episode = sum(steps_per_ep) / len(steps_per_ep)
    avg_time_per_episode_s = sum(durations_s) / len(durations_s)
    avg_speed = sum(speeds) / len(speeds)
    min_speed = min(speeds)
    max_speed = max(speeds)
    n = len(speeds)
    variance = sum((s - avg_speed) ** 2 for s in speeds) / n
    std_speed = variance ** 0.5

    out_lines = []
    out_lines.append("Speed metrics (steps per second) from training log")
    out_lines.append("=" * 50)
    out_lines.append(f"Episodes: {episodes[0]} to {episodes[-1]}  (N={len(results)})")
    out_lines.append("")
    out_lines.append("  avg steps per episode:   {:.2f}".format(avg_steps_per_episode))
    out_lines.append("  avg time per episode (s): {:.2f}".format(avg_time_per_episode_s))
    out_lines.append("  avg (steps/s):          {:.4f}".format(avg_speed))
    out_lines.append("  var:                    {:.4f}".format(variance))
    out_lines.append("  std (steps/s):          {:.4f}".format(std_speed))
    out_lines.append("  min (steps/s):          {:.4f}".format(min_speed))
    out_lines.append("  max (steps/s):          {:.4f}".format(max_speed))
    out_lines.append("=" * 50)

    text = "\n".join(out_lines)
    print(text)

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Written to {output_path}")


def main(
    log_path: str = "test.txt",
    output_path: str | None = None,
):
    """Entry point: run from project root so default test.txt resolves."""
    compute_speed_metrics(log_path=log_path, output_path=output_path)


if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else "test.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(log_path=log_path, output_path=output_path)
