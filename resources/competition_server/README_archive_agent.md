# Running an archive agent on the competition server

The competition client in `comprl-hockey-agent/` can run an agent from the project archive (e.g. agent 0010) on the comprl server.

## Folder: comprl-hockey-agent

| File | Purpose |
|------|--------|
| `run_client.py` | Client entry point; supports `--agent=archive --archive-id=0010`. |
| `autorestart.sh` | Wrapper that restarts the client after drops/server restarts. |
| `requirements.txt` | comprl, hockey, numpy. |
| `README.md` | Upstream comprl client usage. |
| `niklas_env.env` | Exports `COMPRL_SERVER_URL`, `COMPRL_SERVER_PORT`, `COMPRL_ACCESS_TOKEN`. Source this to set server connection. |

## Prerequisites

- Run from the **repository root** so the archive and `src/` are visible.
- Use the project conda environment (with `rl_hockey`, `torch`, `hockey`, `comprl` installed).
- Source `niklas_env.env` to set server URL, port, and token (or set them manually).

## Start archive agent 0010

From the **repository root**:

```bash
# Set server connection (use your env file in comprl-hockey-agent)
source resources/competition_server/comprl-hockey-agent/niklas_env.env

# Run the client with archive agent 0010
PYTHONPATH=src python resources/competition_server/comprl-hockey-agent/run_client.py \
  --args --agent=archive --archive-id=0010
```

Or from inside the client directory:

```bash
cd resources/competition_server/comprl-hockey-agent
source niklas_env.env
PYTHONPATH=../../src python run_client.py --args --agent=archive --archive-id=0010
```

## With auto-restart (recommended)

From the **repository root** (PYTHONPATH must be absolute so it works when autorestart runs the client from inside comprl-hockey-agent):

```bash
source resources/competition_server/comprl-hockey-agent/niklas_env.env
PYTHONPATH="$PWD/src" bash resources/competition_server/comprl-hockey-agent/autorestart.sh \
  --args --agent=archive --archive-id=0010
```

Or from inside the client directory:

```bash
cd resources/competition_server/comprl-hockey-agent
source niklas_env.env
PYTHONPATH=../../src bash autorestart.sh --args --agent=archive --archive-id=0010
```

## Other archive agents

Use any archive id that appears in `archive/registry.json` (e.g. `0001`, `0010`, or the full id like `0010_TDMPC2_2026-02-16_09-46-14`):

```bash
PYTHONPATH=src python resources/competition_server/comprl-hockey-agent/run_client.py \
  --args --agent=archive --archive-id=0010
```

## Running on the cluster (recommended for GPU agents)

If your local machine doesn't have a GPU or the agent times out during load, run the client on the TCML cluster instead:

See: `resources/cluster/niklas/README_competition_cluster.md`

Quick start:
1. Sync project to cluster: `bash resources/cluster/niklas/sync_to_cluster.sh`
2. SSH to cluster and build container: `bash resources/cluster/niklas/build_competition_container.sh`
3. Submit job: `sbatch resources/cluster/niklas/run_competition_agent.sbatch`

The sbatch job runs the competition client with GPU support and keeps running until the job time limit (default 2 hours).

## Server details (from project PDF)

- Host: `comprl.cs.uni-tuebingen.de`
- Port: `65335`
- Leaderboard: http://comprl.cs.uni-tuebingen.de
- Use tmux or screen to keep the client running after closing SSH (for local runs).
