# TD-MPC2 Profiling Analysis & Optimization Recommendations

## Overview

**Device:** NVIDIA RTX A4000 (16.8 GB VRAM)  
**Configuration:** latent_dim=512, horizon=8, num_samples=256, num_iterations=6

---

## 🚨 THE REAL PROBLEM: CPU IS THE BOTTLENECK, NOT GPU

| Metric | Single Action | Batch Action |
|--------|--------------|--------------|
| CPU Time | **12.54s** | **24.71s** |
| CUDA Time | 0.72s | 1.45s |
| **Ratio** | **17.4x** | **17.1x** |

**Your GPU is sitting idle 94% of the time waiting for the CPU!**

The GPU does 0.72s of actual work but the CPU takes 12.54s to orchestrate it. This is a massive CPU bottleneck caused by:
1. Python overhead launching thousands of CUDA kernels
2. Type conversions between numpy/CPU tensors and CUDA
3. `torch.compile` overhead (CUDAGraphs/Triton compilation)

---

## Key Bottlenecks Identified

### 1. 🔴 Critical: CUDA Out of Memory

Even with `latent_dim=512`, you're running out of memory:
- 15.34 GiB allocated by PyTorch
- Only 1.19 MiB free during planning
- The profiler itself adds memory overhead

### 2. 🔴 Critical: CPU-Bound Execution

The CPU is the bottleneck, NOT the GPU:
- **50.5% CPU time** in `aten::linear` (Python dispatch overhead)
- **18.3% CPU time** in `aten::to`/`aten::_to_copy` (type conversions)
- **28.3% CPU time** in `aten::layer_norm`

### 3. 🟠 High: Type Conversions

- 62,250 copy operations in single action selection
- **18% of CPU time** just moving data between formats
- Each `torch.FloatTensor(x).to(device)` triggers CPU→GPU copy

### 4. 🟡 Medium: Too Many Kernel Launches

- 62,700 linear layer calls (each is a separate CUDA kernel launch)
- 44,200 layer norm calls
- Python overhead to launch each kernel ~100-200μs

---

## Concrete Optimization Tips for TRAINING

### 🎯 Priority 1: Fix Memory Issues (MUST DO)

#### 1.1 Reduce Planning Parameters (Not Model Size)

Your `latent_dim=512` is fine. The OOM is from MPPI planning. Reduce:

```json
{
    "num_samples": 128,       // Was 256 - cuts planning memory in half
    "horizon": 5,             // Was 8 - significant memory reduction
    "num_iterations": 4       // Was 6 - fewer MPPI iterations
}
```

#### 1.2 Set Memory Environment Variable

Add to your sbatch script BEFORE python runs:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 1.3 Clear CUDA Cache Periodically

In your training loop, add:

```python
# Every N episodes
if episode % 100 == 0:
    torch.cuda.empty_cache()
```

---

### 🎯 Priority 2: Reduce CPU Overhead (BIGGEST SPEEDUP)

#### 2.1 Disable torch.compile During Training

`torch.compile` has MASSIVE overhead for dynamic shapes. In `tdmpc2.py`, **comment out or remove** the compile calls:

```python
# REMOVE OR COMMENT THESE LINES (around line 270-310):
# compile_available = hasattr(torch, "compile")
# if compile_available:
#     self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
#     ...etc
```

Why? Each unique tensor shape triggers recompilation. With varying batch sizes during training, this causes constant recompilation overhead.

#### 2.2 Fix Type Conversions in `act()` Method

**Current (slow):**
```python
obs = torch.FloatTensor(obs).to(self.device)  # Creates CPU tensor, then copies
```

**Fixed (fast):**
```python
if isinstance(obs, np.ndarray):
    obs = torch.from_numpy(obs).to(self.device, dtype=torch.float32, non_blocking=True)
else:
    obs = obs.to(self.device, dtype=torch.float32, non_blocking=True)
```

#### 2.3 Use `torch.inference_mode()` Instead of `torch.no_grad()`

In `act()` method:

```python
@torch.no_grad()  # CHANGE TO:
def act(self, obs, deterministic=False, t0=False):
    with torch.inference_mode():  # 10-20% faster than no_grad
        ...
```

#### 2.4 Pre-allocate Tensors in Planner

In `mppi_planner_simple.py`, pre-allocate buffers once instead of creating new tensors each call:

```python
def __init__(self, ...):
    ...
    # Pre-allocate buffers (add after other initializations)
    self._actions_buffer = None
    self._returns_buffer = None

def plan(self, latent, ...):
    # Reuse buffers instead of creating new tensors
    if self._actions_buffer is None or self._actions_buffer.shape[1] != self.num_samples:
        self._actions_buffer = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=latent.device
        )
```

---

### 🎯 Priority 3: Reduce Training Loop Overhead

#### 3.1 Increase Training Steps Per Environment Step

Instead of training every step, train less frequently with more steps:

```python
# Instead of:
for step in range(total_steps):
    action = agent.act(obs)
    ...
    agent.train(steps=1)  # Train every step

# Do this:
for step in range(total_steps):
    action = agent.act(obs)
    ...
    if step % 4 == 0:  # Train every 4 steps
        agent.train(steps=4)  # But do 4 training steps
```

This reduces CPU overhead from environment interaction.

#### 3.2 Use Larger Batch Sizes

Larger batches = fewer kernel launches = less CPU overhead:

```json
{
    "batch_size": 512   // Was 256
}
```

#### 3.3 Reduce Horizon in Training (Not Just Planning)

In your config, the training horizon should match planning:

```json
{
    "horizon": 5  // Shorter sequences = faster training
}
```

---

### 🎯 Priority 4: Simplify Network Architecture

#### 4.1 Fewer, Wider Layers

More layers = more kernel launches. Reduce layers:

```json
{
    "hidden_dim": {
        "encoder": [512, 512],      // Was [256, 256, 256] - fewer layers
        "dynamics": [512, 512],
        "reward": [384, 384],
        "termination": [256, 256],
        "q_function": [512, 512],
        "policy": [512, 512]
    }
}
```

#### 4.2 Reduce Number of Q-Networks

```json
{
    "num_q": 3   // Was 5 - 40% fewer Q-network forward passes
}
```

---

## 📋 CONCRETE ACTION CHECKLIST

### Immediate Changes (Do Now)

1. **[ ] Update config file:**
```json
{
    "num_samples": 128,
    "horizon": 5,
    "num_iterations": 4,
    "batch_size": 512,
    "num_q": 3
}
```

2. **[ ] Add to sbatch script:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

3. **[ ] Comment out torch.compile in tdmpc2.py** (lines ~270-310)

### Code Changes (tdmpc2.py)

4. **[ ] Fix `act()` method** - replace `torch.FloatTensor` with `torch.from_numpy`

5. **[ ] Add `torch.inference_mode()`** to `act()` method

### Training Loop Changes

6. **[ ] Train every 4 steps** instead of every step (if not already)

7. **[ ] Add periodic `torch.cuda.empty_cache()`**

---

## Expected Results

| Change | CPU Speedup | Memory Reduction |
|--------|-------------|------------------|
| Disable torch.compile | **2-5x** | 10% |
| Fix type conversions | 1.3-1.5x | 5% |
| Reduce num_samples 256→128 | 1.5x | 50% planning memory |
| Reduce horizon 8→5 | 1.6x | 40% |
| Reduce num_q 5→3 | 1.4x | 20% |
| Larger batch_size | 1.2x | - |
| **Combined** | **5-10x** | **60-70%** |

---

## Recommended Final Config for RTX A4000

```json
{
    "obs_dim": 18,
    "action_dim": 8,
    "latent_dim": 512,
    "hidden_dim": {
        "encoder": [512, 512],
        "dynamics": [512, 512],
        "reward": [384, 384],
        "termination": [256, 256],
        "q_function": [512, 512],
        "policy": [512, 512]
    },
    "num_q": 3,
    "horizon": 5,
    "num_samples": 128,
    "num_iterations": 4,
    "num_elites": 32,
    "num_pi_trajs": 16,
    "batch_size": 512,
    "use_amp": true
}
```

---

## Why This Happens: Understanding the Profile

### The CPU-GPU Gap Explained

```
CPU Time:  12.54s  ████████████████████████████████████████████████████████████
CUDA Time:  0.72s  ████
```

The GPU finishes in 0.72s but the CPU takes 12.54s because:

1. **Python GIL**: Every tensor operation goes through Python interpreter
2. **Kernel Launch Overhead**: ~100-200μs per CUDA kernel launch × 62,700 launches = ~10s
3. **Type Conversions**: Creating CPU tensors then copying to GPU adds latency
4. **torch.compile Overhead**: Dynamic shape recompilation when batch sizes change

### Why torch.compile Hurts Training

`torch.compile` with `mode="reduce-overhead"` uses CUDA Graphs which:
- Cache kernel sequences for FIXED tensor shapes
- Recompile when shapes change (every different batch size)
- Add significant overhead for the first run of each shape

During training, you have varying batch sizes (from buffer sampling, different episode lengths), causing constant recompilation.

---

## Monitoring After Changes

Profile again with:

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

Use `nvidia-smi` to monitor GPU utilization during training:

```bash
watch -n 1 nvidia-smi
```

If GPU utilization is low (<50%), the CPU is still the bottleneck.
