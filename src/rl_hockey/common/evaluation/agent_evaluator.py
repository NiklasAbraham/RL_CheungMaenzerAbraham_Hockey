import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import hockey.hockey_env as h_env
import numpy as np
import torch

from rl_hockey.common.training.agent_factory import create_agent, get_action_space_info
from rl_hockey.common.training.curriculum_manager import AgentConfig, load_curriculum
from rl_hockey.common.utils import (
    discrete_to_continuous_action_with_fineness,
    get_discrete_action_dim,
    set_cuda_device,
)

logger = logging.getLogger(__name__)


def find_config_from_model_path(model_path: str) -> Optional[str]:
    """
    Try to find the config file automatically from the model path.
    Looks for configs in the same directory structure.
    """
    model_path_obj = Path(model_path)

    # Try to find config in the same run directory
    # Model path: .../run_name/timestamp/models/model.pt
    # Config path: .../run_name/timestamp/configs/run_name.json
    if model_path_obj.parent.name == "models":
        run_dir = model_path_obj.parent.parent
        configs_dir = run_dir / "configs"
        if configs_dir.exists():
            # Try to find config file with same name as model (without extension)
            model_name = model_path_obj.stem
            config_file = configs_dir / f"{model_name}.json"
            if config_file.exists():
                return str(config_file)

            # Try to find any config file in the directory
            config_files = list(configs_dir.glob("*.json"))
            if config_files:
                return str(config_files[0])

    return None


def infer_fineness_from_action_dim(
    action_dim: int, keep_mode: bool = True
) -> Optional[int]:
    """
    Infer the action_fineness parameter from the action dimension.
    Returns None if it doesn't match a known fineness pattern.
    """
    # Try common fineness values: 3, 5, 7, 9, 11, 13, 15, etc.
    for fineness in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        expected_dim = get_discrete_action_dim(fineness=fineness, keep_mode=keep_mode)
        if expected_dim == action_dim:
            return fineness
    return None


def is_decoy_policy_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Return True if the checkpoint is from a DecoyPolicy (has decoy-specific keys)."""
    return (
        "network_state_dict" in checkpoint
        and "hidden_layers" in checkpoint
        and "config" not in checkpoint
    )


def load_agent_params_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load agent parameters from checkpoint file.
    Returns a dictionary with available parameters.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        params = dict(checkpoint) if isinstance(checkpoint, dict) else {}

        if "action_dim" in checkpoint:
            params["action_dim"] = checkpoint["action_dim"]
        if "state_dim" in checkpoint:
            params["state_dim"] = checkpoint["state_dim"]
        if "obs_dim" in checkpoint:
            params["obs_dim"] = checkpoint["obs_dim"]
        if "config" in checkpoint:
            params["config"] = checkpoint["config"]
        if "hidden_dim" in checkpoint:
            params["hidden_dim"] = checkpoint["hidden_dim"]
        if "hidden_layers" in checkpoint:
            params["hidden_layers"] = checkpoint["hidden_layers"]
        if "learning_rate" in checkpoint:
            params["learning_rate"] = checkpoint["learning_rate"]
        if "buffer_max_size" in checkpoint:
            params["buffer_max_size"] = checkpoint["buffer_max_size"]

        return params
    except Exception:
        return {}


def _pool_initializer():
    """Initialize worker processes with a safe working directory."""
    # Change to a safe directory that always exists
    # This prevents FileNotFoundError when multiprocessing tries to get cwd
    try:
        os.chdir("/tmp")
    except OSError:
        # If /tmp doesn't work, try home directory
        try:
            os.chdir(os.path.expanduser("~"))
        except OSError:
            pass  # If all else fails, let it use current directory


# Worker-local agent cache (set by _init_eval_worker, used by run_single_game in parallel mode)
_worker_agent = None
_worker_action_fineness = None
_worker_is_discrete = None


def _load_agent_for_eval(
    agent_path: str,
    agent_config_dict: Dict[str, Any],
    device: Optional[Union[str, int]],
) -> Tuple[Any, Optional[int], bool]:
    """
    Create agent, load checkpoint, put in eval mode. Used once per worker or once for sequential eval.
    Returns (agent, action_fineness, is_discrete).
    """
    if not os.path.exists(agent_path):
        current_cwd = "unknown"
        try:
            current_cwd = os.getcwd()
        except (OSError, FileNotFoundError):
            pass
        raise FileNotFoundError(
            f"Agent checkpoint not found: {agent_path}\n"
            f"Current working directory: {current_cwd}\n"
            f"File is absolute: {os.path.isabs(agent_path)}"
        )
    set_cuda_device(device)
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    action_fineness = agent_config_dict.get("hyperparameters", {}).get(
        "action_fineness", None
    )
    state_dim, action_dim, is_discrete = get_action_space_info(
        env, agent_config_dict["type"], fineness=action_fineness
    )
    env.close()
    agent_config = AgentConfig(
        type=agent_config_dict["type"],
        hyperparameters=agent_config_dict["hyperparameters"],
    )
    agent = create_agent(
        agent_config, state_dim, action_dim, {}, inference_mode=True
    )
    if agent_config.type == "TDMPC2":
        agent.load(agent_path, inference_mode=True)
    else:
        agent.load(agent_path)
    if hasattr(agent, "q_network"):
        agent.q_network.eval()
    if hasattr(agent, "q_network_target"):
        agent.q_network_target.eval()
    if hasattr(agent, "actor") and hasattr(agent.actor, "eval"):
        agent.actor.eval()
    if hasattr(agent, "critic1") and hasattr(agent.critic1, "eval"):
        agent.critic1.eval()
    if hasattr(agent, "actor_target") and hasattr(agent.actor_target, "eval"):
        agent.actor_target.eval()
    if hasattr(agent, "critic") and hasattr(agent.critic, "eval"):
        agent.critic.eval()
    if hasattr(agent, "critic_target") and hasattr(agent.critic_target, "eval"):
        agent.critic_target.eval()
    if hasattr(agent, "network") and hasattr(agent.network, "eval"):
        agent.network.eval()
    return agent, action_fineness, is_discrete


def _run_one_game(
    agent: Any,
    action_fineness: Optional[int],
    is_discrete: bool,
    weak_opponent: bool,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single game with an already-loaded agent. Creates env, runs episode, returns result."""
    np.random.seed(seed)
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    opponent = h_env.BasicOpponent(weak=weak_opponent)
    state, _ = env.reset()
    obs_agent2 = env.obs_agent_two()
    total_reward = 0
    step = 0
    for step in range(max_steps):
        if is_discrete:
            discrete_action = agent.act(state.astype(np.float32), deterministic=True)
            if action_fineness is not None:
                action_p1 = discrete_to_continuous_action_with_fineness(
                    discrete_action, fineness=action_fineness, keep_mode=env.keep_mode
                )
            else:
                action_p1 = env.discrete_to_continous_action(discrete_action)
        else:
            action_p1 = agent.act(state.astype(np.float32), deterministic=True)
        action_p2 = opponent.act(obs_agent2)
        action = np.hstack([action_p1, action_p2])
        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        state = next_state
        obs_agent2 = env.obs_agent_two()
        if done or trunc:
            break
    winner = info.get("winner", 0)
    env.close()
    return {"winner": winner, "reward": total_reward, "steps": step + 1}


def _init_eval_worker(
    agent_path: str,
    agent_config_dict: Dict[str, Any],
    device: Optional[Union[str, int]],
) -> None:
    """Load agent once per worker and store in module-level variables."""
    global _worker_agent, _worker_action_fineness, _worker_is_discrete
    _pool_initializer()
    _worker_agent, _worker_action_fineness, _worker_is_discrete = _load_agent_for_eval(
        agent_path, agent_config_dict, device
    )


def run_single_game(args: Tuple) -> Dict[str, Any]:
    """
    Run a single game. In parallel mode, args = (weak_opponent, max_steps, seed) and the agent
    is taken from the worker cache (set by _init_eval_worker). In sequential mode this is not used;
    the caller uses _load_agent_for_eval once and _run_one_game in a loop.
    """
    weak_opponent, max_steps, seed = args
    if _worker_agent is None:
        raise RuntimeError(
            "Worker agent not initialized; use Pool with initializer=_init_eval_worker"
        )
    return _run_one_game(
        _worker_agent,
        _worker_action_fineness,
        _worker_is_discrete,
        weak_opponent,
        max_steps,
        seed,
    )


def evaluate_agent(
    agent_path: str,
    config_path: str = None,
    agent_config_dict: Dict[str, Any] = None,
    num_games: int = 100,
    weak_opponent: bool = True,
    max_steps: int = 250,
    num_parallel: int = None,
    device: Optional[Union[str, int]] = None,
    use_cpu_for_eval: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate an agent against a weak opponent.

    Args:
        agent_path: Path to agent checkpoint
        config_path: Path to curriculum configuration JSON file (if None, will try to auto-detect from model path)
        agent_config_dict: Dictionary containing agent configuration (if None, will try to load from config or checkpoint)
        num_games: Number of games to evaluate
        weak_opponent: Whether to use weak opponent
        max_steps: Maximum number of steps per game
        num_parallel: Number of parallel processes to use
        device: CUDA device to use (None = auto-detect, 'cpu' = CPU, 'cuda' = cuda:0, 'cuda:0' = first GPU, 'cuda:1' = second GPU, etc.). Can also be an integer (0, 1, etc.) for device ID.
        use_cpu_for_eval: If True, force CPU usage for evaluation to avoid GPU memory issues with parallel processes
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Starting agent evaluation: {agent_path}")
    logger.info(
        f"Configuration: num_games={num_games}, weak_opponent={weak_opponent}, max_steps={max_steps}"
    )

    # Save current working directory early so we can resolve relative paths before any chdir
    original_cwd = None
    try:
        original_cwd = os.getcwd()
    except (OSError, FileNotFoundError):
        original_cwd = None

    # Auto-detect config_path if not provided
    if config_path is None:
        detected_config = find_config_from_model_path(agent_path)
        if detected_config is not None:
            config_path = detected_config
            logger.info(f"Auto-detected config path: {config_path}")

    # Load config if available
    if config_path is not None:
        curriculum = load_curriculum(config_path)
        agent_config_dict = {
            "type": curriculum.agent.type,
            "hyperparameters": curriculum.agent.hyperparameters,
        }
        logger.info(f"Loaded agent config: type={agent_config_dict['type']}")

    # If still no config, try to load from checkpoint and infer parameters
    if agent_config_dict is None:
        checkpoint_params = load_agent_params_from_checkpoint(agent_path)

        # Detect DecoyPolicy checkpoints (they have network_state_dict + hidden_layers, no config)
        if is_decoy_policy_checkpoint(checkpoint_params):
            agent_type = "DecoyPolicy"
            hyperparameters = {
                "hidden_layers": checkpoint_params.get("hidden_layers", [256, 256]),
                "learning_rate": checkpoint_params.get("learning_rate", 3e-4),
                "buffer_max_size": checkpoint_params.get("buffer_max_size", 100_000),
            }
            agent_config_dict = {"type": agent_type, "hyperparameters": hyperparameters}
            logger.info("Inferred agent type from checkpoint: DecoyPolicy")
        else:
            # Try to infer agent type from checkpoint or default to DDDQN
            # Check if checkpoint has q_network (DDDQN) or actor (SAC/PPO)
            agent_type = "DDDQN"  # Default fallback
            if "config" in checkpoint_params:
                agent_type = checkpoint_params["config"].get("type", "DDDQN")

            # Try to infer action_fineness from action_dim
            action_fineness = None
            if "action_dim" in checkpoint_params:
                action_dim = checkpoint_params["action_dim"]
                # Try both keep_mode=True and keep_mode=False
                action_fineness = infer_fineness_from_action_dim(
                    action_dim, keep_mode=True
                )
                if action_fineness is None:
                    action_fineness = infer_fineness_from_action_dim(
                        action_dim, keep_mode=False
                    )

            # Build hyperparameters dict
            hyperparameters = {}
            if action_fineness is not None:
                hyperparameters["action_fineness"] = action_fineness

            # Add other hyperparameters from checkpoint config if available
            if "config" in checkpoint_params:
                checkpoint_config = checkpoint_params["config"]
                # Copy relevant hyperparameters
                for key in [
                    "hidden_dim",
                    "target_update_freq",
                    "eps",
                    "eps_min",
                    "eps_decay",
                    "use_huber_loss",
                ]:
                    if key in checkpoint_config:
                        hyperparameters[key] = checkpoint_config[key]

            agent_config_dict = {"type": agent_type, "hyperparameters": hyperparameters}

    # Final check: if action_fineness is still missing, try to infer from checkpoint
    if agent_config_dict.get("hyperparameters", {}).get("action_fineness") is None:
        checkpoint_params = load_agent_params_from_checkpoint(agent_path)
        if "action_dim" in checkpoint_params:
            action_dim = checkpoint_params["action_dim"]
            action_fineness = infer_fineness_from_action_dim(action_dim, keep_mode=True)
            if action_fineness is None:
                action_fineness = infer_fineness_from_action_dim(
                    action_dim, keep_mode=False
                )
            if action_fineness is not None:
                if "hyperparameters" not in agent_config_dict:
                    agent_config_dict["hyperparameters"] = {}
                agent_config_dict["hyperparameters"]["action_fineness"] = (
                    action_fineness
                )

    # Resolve opponent simulation paths to absolute so they work after cwd change or in workers
    hyperparams = agent_config_dict.get("hyperparameters", {})
    opp_sim = hyperparams.get("opponent_simulation")
    if opp_sim and opp_sim.get("enabled") and opp_sim.get("opponent_agents"):
        base_dir = original_cwd if original_cwd is not None else os.path.curdir
        for opp_info in opp_sim["opponent_agents"]:
            if "path" in opp_info and not os.path.isabs(opp_info["path"]):
                opp_info["path"] = os.path.normpath(
                    os.path.join(base_dir, opp_info["path"])
                )
                logger.debug("Resolved opponent path to %s", opp_info["path"])

    # Override device to CPU if requested
    if use_cpu_for_eval:
        device = "cpu"
        logger.info("Forcing CPU usage for evaluation")

    if num_parallel is None:
        # Limit parallel processes to avoid GPU memory exhaustion
        # Each process loads the model, so we cap at a reasonable number
        # Use CPU count for CPU evaluation, but limit to 3 for GPU evaluation
        if (
            device is not None
            and device != "cpu"
            and (
                isinstance(device, str)
                and "cuda" in str(device)
                or isinstance(device, int)
            )
        ):
            # GPU evaluation: limit to 3 parallel processes to avoid OOM
            num_parallel = min(3, mp.cpu_count(), num_games)
        else:
            # CPU evaluation: can use more processes
            num_parallel = min(mp.cpu_count(), num_games)

    logger.info(f"Using device: {device}, parallel processes: {num_parallel}")

    # Convert agent_path to absolute path BEFORE changing directories
    # This ensures worker processes can find the checkpoint file even after we change cwd
    if not os.path.isabs(agent_path):
        if original_cwd is not None:
            # We have a valid cwd, use os.path.join with cwd to make absolute
            agent_path = os.path.normpath(os.path.join(original_cwd, agent_path))
        else:
            # No valid cwd, try Path.resolve() which doesn't require file to exist
            try:
                agent_path = str(Path(agent_path).resolve())
            except (OSError, RuntimeError):
                # If that fails, try to expand user and resolve
                agent_path = os.path.normpath(os.path.expanduser(agent_path))

    # Change to a safe directory that always exists BEFORE creating Pool
    # This prevents FileNotFoundError when multiprocessing spawns workers
    # in SLURM/Singularity environments where the working directory might not be accessible
    safe_dir = "/tmp"
    try:
        home_dir = os.path.expanduser("~")
        if os.path.exists(home_dir):
            safe_dir = home_dir
        else:
            safe_dir = "/tmp"
        os.chdir(safe_dir)
    except (OSError, FileNotFoundError):
        # If we can't change directory, try to continue
        pass

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    seeds = np.random.randint(0, 2**31, size=num_games)
    game_args_list = [(weak_opponent, max_steps, int(seed)) for seed in seeds]

    logger.info(f"Starting {num_games} game(s)...")
    results = []
    try:
        if num_parallel > 1:
            with mp.Pool(
                processes=num_parallel,
                initializer=_init_eval_worker,
                initargs=(agent_path, agent_config_dict, device),
            ) as pool:
                completed = 0
                for result in pool.imap(run_single_game, game_args_list):
                    results.append(result)
                    completed += 1
                    if (
                        completed % max(1, num_games // 10) == 0
                        or completed == num_games
                    ):
                        logger.info(
                            f"Progress: {completed}/{num_games} games completed"
                        )
                        wins_so_far = sum(1 for r in results if r["winner"] == 1)
                        logger.info(
                            f"  Current stats: {wins_so_far} wins, {completed - wins_so_far} non-wins"
                        )
        else:
            agent, action_fineness, is_discrete = _load_agent_for_eval(
                agent_path, agent_config_dict, device
            )
            for i, (_, max_s, seed) in enumerate(game_args_list, 1):
                result = _run_one_game(
                    agent,
                    action_fineness,
                    is_discrete,
                    weak_opponent,
                    max_s,
                    seed,
                )
                results.append(result)
                winner_str = (
                    "win"
                    if result["winner"] == 1
                    else ("loss" if result["winner"] == -1 else "draw")
                )
                logger.info(
                    f"Game {i}/{num_games} completed: {winner_str}, reward={result['reward']:.2f}, steps={result['steps']}"
                )
                wins_so_far = sum(1 for r in results if r["winner"] == 1)
                if i % 10 == 0 or i == num_games:
                    logger.info(f"  Progress: {wins_so_far}/{i} wins so far")
    finally:
        # Restore original working directory if we changed it
        if original_cwd is not None:
            try:
                os.chdir(original_cwd)
            except (OSError, FileNotFoundError):
                pass

    logger.info(f"All {num_games} games completed")

    wins = sum(1 for r in results if r["winner"] == 1)
    losses = sum(1 for r in results if r["winner"] == -1)
    draws = sum(1 for r in results if r["winner"] == 0)
    rewards = [r["reward"] for r in results]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    win_rate = wins / num_games if num_games > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Evaluation Results Summary:")
    logger.info(f"  Wins: {wins}")
    logger.info(f"  Losses: {losses}")
    logger.info(f"  Draws: {draws}")
    logger.info(f"  Win Rate: {win_rate:.2%}")
    logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logger.info("=" * 60)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "num_games": num_games,
        "weak_opponent": weak_opponent,
        "all_rewards": rewards,
        "all_winners": [r["winner"] for r in results],
    }


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    # Now config_path can be None - it will auto-detect from model path
    print(
        evaluate_agent(
            agent_path="results/self_play/2026-02-01_09-55-22/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep025646.pt",
            config_path=None,
            num_games=500,
            weak_opponent=False,
            max_steps=500,
            num_parallel=1,
        )
    )
