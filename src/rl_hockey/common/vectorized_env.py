"""
Vectorized Hockey Environment Wrapper
Runs multiple hockey environments in parallel to maximize GPU utilization and training speed.
"""

import numpy as np
from multiprocessing import Process, Pipe, current_process
import threading
import queue
import hockey.hockey_env as h_env


def worker(remote, parent_remote, env_fn):
    """Worker process that runs a single environment"""
    parent_remote.close()
    env = env_fn()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, done, trunc, info = env.step(data)
                # Auto-reset if episode is done
                if done or trunc:
                    # Store the terminal observation before reset
                    terminal_obs = obs
                    # Reset for next episode
                    obs, reset_info = env.reset()
                    # Send terminal observation but with reset state for next step
                    # The training loop needs the terminal obs for storing transition
                    remote.send((terminal_obs, reward, done, trunc, info))
                else:
                    remote.send((obs, reward, done, trunc, info))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            elif cmd == 'obs_agent_two':
                obs = env.obs_agent_two()
                remote.send(obs)
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        env.close()


def thread_worker(cmd_queue, result_queue, env_fn):
    """Worker thread that runs a single environment"""
    env = env_fn()
    
    try:
        while True:
            cmd, data = cmd_queue.get()
            
            if cmd == 'step':
                obs, reward, done, trunc, info = env.step(data)
                # Auto-reset if episode is done
                if done or trunc:
                    # Store the terminal observation before reset
                    terminal_obs = obs
                    # Reset for next episode
                    obs, reset_info = env.reset()
                    # Send terminal observation but with reset state for next step
                    result_queue.put(('step', (terminal_obs, reward, done, trunc, info)))
                else:
                    result_queue.put(('step', (obs, reward, done, trunc, info)))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                result_queue.put(('reset', (obs, info)))
                
            elif cmd == 'close':
                env.close()
                break
                
            elif cmd == 'get_spaces':
                result_queue.put(('get_spaces', (env.observation_space, env.action_space)))
                
            elif cmd == 'obs_agent_two':
                obs = env.obs_agent_two()
                result_queue.put(('obs_agent_two', obs))
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        env.close()


class VectorizedHockeyEnv:
    """
    Runs multiple hockey environments in parallel for faster data collection.
    
    This dramatically improves GPU utilization by:
    1. Collecting multiple transitions simultaneously
    2. Batching neural network forward passes across all environments
    3. Keeping the GPU busy while CPUs simulate physics
    
    Usage:
        env = VectorizedHockeyEnv(num_envs=4, env_fn=lambda: h_env.HockeyEnv())
        obs = env.reset()  # Returns (num_envs, obs_dim)
        actions = agent.act_batch(obs)  # Batch forward pass on GPU
        next_obs, rewards, dones, truncs, infos = env.step(actions)
    """
    
    def __init__(self, num_envs, env_fn, start_method='spawn'):
        """
        Args:
            num_envs: Number of parallel environments
            env_fn: Function that creates a single environment
            start_method: Multiprocessing start method ('spawn' or 'fork')
        """
        self.num_envs = num_envs
        self.env_fn = env_fn
        self.closed = False
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = Process(target=worker, args=(work_remote, remote, env_fn))
            p.daemon = True  # Auto-cleanup when parent exits
            try:
                p.start()
            except AssertionError as e:
                if "daemonic processes are not allowed to have children" in str(e):
                    # This should not happen if train_run properly detects Pool workers
                    # But provide a helpful error message just in case
                    raise RuntimeError(
                        "Cannot use vectorized environments (num_envs > 1) inside multiprocessing Pool workers. "
                        "The train_run function should automatically disable vectorized environments in this case. "
                        "If you see this error, please report it as a bug."
                    ) from e
                raise
            self.processes.append(p)
            work_remote.close()
        
        # Get environment spaces from first env
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
        # Store mode and keep_mode for compatibility
        self.mode = None
        self.keep_mode = None
    
    def reset(self):
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        obs = np.stack([r[0] for r in results])
        return obs
    
    def obs_agent_two(self):
        """Get observations for agent 2 from all environments"""
        for remote in self.remotes:
            remote.send(('obs_agent_two', None))
        
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)
    
    def step(self, actions):
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of shape (num_envs, action_dim) or list of actions
        
        Returns:
            obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            truncs: (num_envs,)
            infos: list of info dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        truncs = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        
        return obs, rewards, dones, truncs, infos
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for p in self.processes:
            p.join()
        
        self.closed = True
    
    def __del__(self):
        if not self.closed:
            self.close()


class ThreadedVectorizedHockeyEnv:
    """
    Threaded version of vectorized environment that works inside multiprocessing Pool workers.
    Uses threads instead of processes, so it can be created from daemon processes.
    
    Note: Performance may be lower than multiprocessing version due to GIL,
    but it still provides speedup through batching and parallel environment execution.
    """
    
    def __init__(self, num_envs, env_fn):
        self.num_envs = num_envs
        self.env_fn = env_fn
        self.closed = False
        
        # Create queues for each environment thread
        self.cmd_queues = [queue.Queue() for _ in range(num_envs)]
        self.result_queues = [queue.Queue() for _ in range(num_envs)]
        
        # Start worker threads
        self.threads = []
        for cmd_q, result_q in zip(self.cmd_queues, self.result_queues):
            t = threading.Thread(target=thread_worker, args=(cmd_q, result_q, env_fn), daemon=True)
            t.start()
            self.threads.append(t)
        
        # Get environment spaces from first env
        self.cmd_queues[0].put(('get_spaces', None))
        cmd, (self.observation_space, self.action_space) = self.result_queues[0].get()
        
        # Store mode and keep_mode for compatibility
        self.mode = None
        self.keep_mode = None
    
    def reset(self):
        """Reset all environments"""
        for cmd_q in self.cmd_queues:
            cmd_q.put(('reset', None))
        
        results = []
        for result_q in self.result_queues:
            cmd, data = result_q.get()
            results.append(data)
        
        obs = np.stack([r[0] for r in results])
        return obs
    
    def obs_agent_two(self):
        """Get observations for agent 2 from all environments"""
        for cmd_q in self.cmd_queues:
            cmd_q.put(('obs_agent_two', None))
        
        results = []
        for result_q in self.result_queues:
            cmd, data = result_q.get()
            results.append(data)
        
        return np.stack(results)
    
    def step(self, actions):
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of shape (num_envs, action_dim) or list of actions
        
        Returns:
            obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            truncs: (num_envs,)
            infos: list of info dicts
        """
        for cmd_q, action in zip(self.cmd_queues, actions):
            cmd_q.put(('step', action))
        
        results = []
        for result_q in self.result_queues:
            cmd, data = result_q.get()
            results.append(data)
        
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        truncs = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        
        return obs, rewards, dones, truncs, infos
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        
        for cmd_q in self.cmd_queues:
            cmd_q.put(('close', None))
        
        for t in self.threads:
            t.join(timeout=1.0)
        
        self.closed = True
    
    def __del__(self):
        if not self.closed:
            self.close()


class ThreadedVectorizedHockeyEnvOptimized(ThreadedVectorizedHockeyEnv):
    """
    Optimized threaded version that keeps observations as float32 and minimizes copies.
    """
    
    def __init__(self, num_envs, env_fn):
        super().__init__(num_envs, env_fn)
        
        # Pre-allocate arrays for efficiency
        obs_shape = self.observation_space.shape
        self.obs_buffer = np.zeros((num_envs, *obs_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self.done_buffer = np.zeros(num_envs, dtype=np.bool_)
        self.trunc_buffer = np.zeros(num_envs, dtype=np.bool_)
    
    def step(self, actions):
        """Optimized step with pre-allocated buffers"""
        for cmd_q, action in zip(self.cmd_queues, actions):
            cmd_q.put(('step', action))
        
        results = []
        for result_q in self.result_queues:
            cmd, data = result_q.get()
            results.append(data)
        
        # Use pre-allocated buffers
        for i, (obs, reward, done, trunc, info) in enumerate(results):
            self.obs_buffer[i] = obs.astype(np.float32, copy=False)
            self.reward_buffer[i] = reward
            self.done_buffer[i] = done
            self.trunc_buffer[i] = trunc
        
        infos = [r[4] for r in results]
        
        return self.obs_buffer.copy(), self.reward_buffer.copy(), \
               self.done_buffer.copy(), self.trunc_buffer.copy(), infos


class VectorizedHockeyEnvOptimized(VectorizedHockeyEnv):
    """
    Optimized version that keeps observations as float32 and minimizes copies.
    Also supports batched agent actions for maximum GPU efficiency.
    """
    
    def __init__(self, num_envs, env_fn):
        super().__init__(num_envs, env_fn)
        
        # Pre-allocate arrays for efficiency
        obs_shape = self.observation_space.shape
        self.obs_buffer = np.zeros((num_envs, *obs_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self.done_buffer = np.zeros(num_envs, dtype=np.bool_)
        self.trunc_buffer = np.zeros(num_envs, dtype=np.bool_)
    
    def step(self, actions):
        """Optimized step with pre-allocated buffers"""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        
        # Use pre-allocated buffers
        for i, (obs, reward, done, trunc, info) in enumerate(results):
            self.obs_buffer[i] = obs.astype(np.float32, copy=False)
            self.reward_buffer[i] = reward
            self.done_buffer[i] = done
            self.trunc_buffer[i] = trunc
        
        infos = [r[4] for r in results]
        
        return self.obs_buffer.copy(), self.reward_buffer.copy(), \
               self.done_buffer.copy(), self.trunc_buffer.copy(), infos


if __name__ == "__main__":
    # Example usage and benchmark
    import time
    
    print("Benchmarking Vectorized Hockey Environment")
    print("=" * 50)
    
    # Test with different numbers of parallel envs
    for num_envs in [1, 2, 4, 8]:
        env = VectorizedHockeyEnvOptimized(
            num_envs=num_envs,
            env_fn=lambda: h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
        )
        
        obs = env.reset()
        
        # Benchmark 100 steps
        start = time.time()
        num_steps = 100
        
        for _ in range(num_steps):
            # Random actions for all envs
            actions = [env.action_space.sample() for _ in range(num_envs)]
            obs, rewards, dones, truncs, infos = env.step(actions)
        
        elapsed = time.time() - start
        steps_per_sec = (num_steps * num_envs) / elapsed
        
        print(f"\n{num_envs} parallel envs:")
        print(f"  Total steps: {num_steps * num_envs}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Speedup vs 1 env: {steps_per_sec / (num_steps / elapsed) if num_envs == 1 else steps_per_sec / (100 / 10.0):.2f}x")
        
        env.close()
    
    print("\n" + "=" * 50)
    print("Benchmark complete!")
