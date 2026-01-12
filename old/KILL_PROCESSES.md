# How to Kill All Python Processes Using rl-hockey Conda Environment

This guide shows how to find and terminate all Python processes that are running with the `rl-hockey` conda environment.

## Quick Method (Recommended)

```bash
# Find and kill all processes with rl-hockey in the path
pkill -f "rl-hockey.*python"

# If that doesn't work, use force kill
ps aux | grep -i "rl-hockey" | grep -i python | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

## Step-by-Step Method

### 1. Find All rl-hockey Python Processes

First, check what processes are running:

```bash
ps aux | grep -i "rl-hockey" | grep -i python | grep -v grep
```

This will show you:
- Process IDs (PIDs)
- CPU/Memory usage
- Full command paths

### 2. Find Parent Processes

If you see many worker processes, find the parent process that's spawning them:

```bash
ps aux | grep -E "(train_single_run|train_run|hyperparameter)" | grep -i python | grep -v grep
```

Common parent processes:
- `hyperparameter_tuning.py` - Spawns multiple parallel training runs
- `train_single_run.py` - Single training run
- `train_run.py` - Curriculum training

### 3. Kill the Parent Process First

Kill the parent process by PID (replace `PID` with the actual process ID):

```bash
kill -9 <PID>
```

For example:
```bash
kill -9 132696
```

### 4. Kill All Remaining Worker Processes

After killing the parent, kill all remaining worker processes:

```bash
ps aux | grep -i "rl-hockey" | grep -i python | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

This command:
- Finds all rl-hockey Python processes
- Extracts their PIDs (column 2)
- Kills them with `kill -9` (force kill)

### 5. Verify All Processes Are Terminated

Check that everything is stopped:

```bash
ps aux | grep -i "rl-hockey" | grep -i python | grep -v grep || echo "✓ All processes terminated"
```

If you see "✓ All processes terminated", you're done!

## Complete One-Liner

If you want to do everything in one command:

```bash
# Kill parent processes first
ps aux | grep -E "(train_single_run|train_run|hyperparameter)" | grep -i python | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Then kill all remaining rl-hockey Python processes
sleep 1 && ps aux | grep -i "rl-hockey" | grep -i python | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null && echo "All processes killed"
```

## Alternative: Kill by Process Name Pattern

You can also use `pkill` with patterns:

```bash
# Kill all processes matching the pattern
pkill -f "rl-hockey.*python"

# Force kill if needed
pkill -9 -f "rl-hockey.*python"
```

## Why This Happens

When running hyperparameter tuning or parallel training:
- Parent process spawns multiple worker processes
- Each worker runs in a separate process
- If the parent isn't killed, it keeps spawning new workers
- That's why you need to kill the parent first

## Troubleshooting

### Processes Keep Coming Back

If processes keep respawning:
1. Find the parent process (step 2 above)
2. Kill the parent first
3. Then kill all workers

### Permission Denied

If you get "Permission denied":
- You can only kill your own processes
- Check with `ps aux` to see if processes belong to another user
- If needed, use `sudo` (be careful!)

### Process Not Found

If you get "No such process":
- The process may have already terminated
- Run the verification command to confirm

## Safety Notes

⚠️ **Warning**: `kill -9` is a force kill that doesn't allow processes to clean up. Use it when:
- Normal `kill` doesn't work
- Processes are hung/frozen
- You need immediate termination

For graceful shutdown, try `kill` (without `-9`) first:
```bash
kill <PID>  # Graceful shutdown
sleep 2
kill -9 <PID>  # Force kill if still running
```

## Example Output

When you run the find command, you might see:

```
nab  132696  1.5  1.0  python src/rl_hockey/common/training/hyperparameter_tuning.py
nab  132726  0.0  0.0  /home/nab/anaconda3/envs/rl-hockey/bin/python -c from multiprocessing...
nab  132727 98.9  2.0  /home/nab/anaconda3/envs/rl-hockey/bin/python -c from multiprocessing...
```

The first line is the parent process (kill this first), the others are workers.
