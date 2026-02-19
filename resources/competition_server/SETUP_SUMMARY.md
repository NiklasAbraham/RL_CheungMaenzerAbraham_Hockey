# Competition Agent Setup Summary

Setup complete for running archive agent 0010 on the competition server, both locally and on the cluster.

## Files Created/Modified

### Competition Client (Local)

| File | Purpose |
|------|---------|
| `comprl-hockey-agent/run_client.py` | Extended to support `--agent=archive --archive-id=0010`. Loads agent from project archive and wraps it for comprl. |
| `comprl-hockey-agent/niklas_env.env` | Server credentials (URL, port, token). Source this before running. **Added to .gitignore.** |
| `README_archive_agent.md` | Instructions for running archive agents locally and on cluster. |
| `.gitignore` | Added `niklas_env.env` to prevent committing the token. |

### Cluster Setup

| File | Purpose |
|------|---------|
| `resources/container/container_competition.def` | Singularity container definition with comprl, hockey, torch, and all deps. |
| `resources/cluster/niklas/run_competition_agent.sbatch` | SLURM sbatch script to run archive agent 0010 on cluster with GPU. |
| `resources/cluster/niklas/build_competition_container.sh` | Helper script to build the competition container on the cluster. |
| `resources/cluster/niklas/README_competition_cluster.md` | Full instructions for cluster setup and running. |

## Quick Start: Local (CPU or GPU)

```bash
cd /path/to/RL_CheungMaenzerAbraham_Hockey

# Source server credentials
source resources/competition_server/comprl-hockey-agent/niklas_env.env

# Run agent 0010
PYTHONPATH=src python resources/competition_server/comprl-hockey-agent/run_client.py \
  --args --agent=archive --archive-id=0010
```

**Note:** TDMPC2 agents need GPU for fast inference. If your local machine has no GPU, the agent may time out. Use the cluster instead (see below).

## Quick Start: Cluster (GPU)

```bash
# 1. Sync to cluster
bash resources/cluster/niklas/sync_to_cluster.sh

# 2. SSH to cluster
ssh <username>@login.cluster.uni-tuebingen.de

# 3. Build container (once)
cd ~/RL_CheungMaenzerAbraham_Hockey
bash resources/cluster/niklas/build_competition_container.sh

# 4. Submit job
sbatch resources/cluster/niklas/run_competition_agent.sbatch

# 5. Monitor
squeue -u $USER
tail -f logs/comprl_agent_0010_*.out
```

Check the leaderboard: http://comprl.cs.uni-tuebingen.de

## How It Works

### Local

1. `run_client.py` with `--agent=archive --archive-id=0010` finds agent 0010 in `archive/registry.json`.
2. Loads `archive/agents/0010_TDMPC2_.../config.json` and `checkpoint.pt`.
3. Creates TDMPC2 agent via `create_agent()` from `agent_factory.py`.
4. Wraps the agent in `ArchiveAgentWrapper` that implements comprl `Agent.get_step()`.
5. Connects to comprl server and plays matches.

### Cluster

1. Sbatch script loads `niklas_env.env` for server credentials.
2. Runs Singularity container with GPU (`--nv`), binds repo as `/workspace`.
3. Container installs `rl_hockey` package and activates venv.
4. Runs `autorestart.sh` which wraps `run_client.py` for automatic reconnection.
5. Agent connects to competition server and plays until job time limit, auto-restarting on disconnect.

## Customizing

**Change agent:** Edit `ARCHIVE_AGENT_ID="0010"` in `run_competition_agent.sbatch` or pass `--archive-id=<ID>` locally.

**Run longer:** Edit `--time=02:00:00` in the sbatch script (default 2 hours).

**Multiple agents:** Submit multiple sbatch jobs with different agent IDs.

## Troubleshooting

**Agent times out locally**

Your machine doesn't have a GPU or it's too slow. Use the cluster.

**Container not found on cluster**

Build it: `bash resources/cluster/niklas/build_competition_container.sh`

**Wrong token**

Check `niklas_env.env` has the correct `COMPRL_ACCESS_TOKEN`.

**Agent not appearing on leaderboard**

Check job logs: `cat logs/comprl_agent_0010_*.out` and `*.err` for errors.

## For the Final Tournament (24.02.2026)

According to the project PDF, the tournament runs from 10am for 12 hours on 24.02.2026. To participate:

1. Submit sbatch job **before** 10am on 24.02 with `--time=12:00:00` (or longer).
2. Use the cluster so the agent runs reliably with GPU.
3. Monitor via leaderboard and job logs.
4. Run multiple instances (2 recommended per agent) for redundancy.

Example for 2 instances:

```bash
sbatch resources/cluster/niklas/run_competition_agent.sbatch
sbatch resources/cluster/niklas/run_competition_agent.sbatch
```

Each gets its own GPU and connects independently.
