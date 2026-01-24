# TD-MPC2 Training Optimization Plan (No Hyperparameter Changes)

## Profiling Analysis Summary

From the profiling results (50 action selections):
- **Total time**: 4.9s → **~98ms per action** (way too slow for real-time)
- **Bottleneck #1**: `aten::linear/addmm` - 74% of CUDA time (930ms)
- **Bottleneck #2**: `aten::layer_norm` - 7.9% of CUDA time  
- **Bottleneck #3**: `aten::mish` - 4.4% of CUDA time
- **22,450 linear operations** for 50 actions → **~449 linear ops per action**

### Root Cause
MPPI planning with `512 samples × 6 iterations × horizon 5` = massive number of forward passes through all networks per action.

---

## Optimization Strategies (Without Changing Hyperparameters)

### 🔴 HIGH IMPACT

#### 1. Mixed Precision (AMP) - Automatic Mixed Precision
**Expected speedup: 1.5-2x on RTX 2080 Ti**

The biggest win without changing any hyperparameters. Uses FP16 for most operations while keeping FP32 for sensitive parts.

**Files to modify**: `tdmpc2.py`, `mppi_planner_simple.py`

```python
# In tdmpc2.py __init__:
self.use_amp = True  # Add flag

# In act():
@torch.no_grad()
def act(self, obs, deterministic=False, t0=False):
    obs = torch.FloatTensor(obs).to(self.device)
    with torch.cuda.amp.autocast(enabled=self.use_amp):
        z = self.encoder(obs.unsqueeze(0)).squeeze(0)
        action = self.planner.plan(z, return_mean=deterministic, t0=t0)
    return action.cpu().numpy()

# In train():
with torch.cuda.amp.autocast(enabled=self.use_amp):
    # All forward passes
    ...
# Use GradScaler for backward pass
self.scaler.scale(total_loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

#### 2. Add "Fast Mode" - Policy-Only Inference
**Expected speedup: 10-50x for inference**

Skip MPPI planning entirely and use the learned policy network directly. This is **critical** for:
- Evaluation episodes
- Self-play data collection after warmup
- Any time you don't need the full MPPI planning

```python
@torch.no_grad()
def act(self, obs, deterministic=False, t0=False, fast_mode=False):
    obs = torch.FloatTensor(obs).to(self.device)
    
    with torch.cuda.amp.autocast(enabled=self.use_amp):
        z = self.encoder(obs.unsqueeze(0))
        
        if fast_mode:
            # Direct policy inference: 1 forward pass vs hundreds
            if deterministic:
                action = self.policy.mean_action(z).squeeze(0)
            else:
                action, _, _, _ = self.policy.sample(z)
                action = action.squeeze(0)
            return action.cpu().numpy()
        
        # Full MPPI planning
        action = self.planner.plan(z.squeeze(0), return_mean=deterministic, t0=t0)
    
    return action.cpu().numpy()
```

#### 3. Compile Policy & Q-Ensemble (Currently Missing)
**Expected speedup: 1.2-1.4x**

Current code only compiles encoder, dynamics, reward. Add:

```python
# In tdmpc2.py __init__, after existing compile block:
if compile_available:
    try:
        self.policy = torch.compile(self.policy, mode="reduce-overhead")
        self.q_ensemble = torch.compile(self.q_ensemble, mode="reduce-overhead")
        self.termination = torch.compile(self.termination, mode="reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile failed for additional models: {e}")
```

---

### 🟠 MEDIUM IMPACT

#### 4. Replace Mish with SiLU (Swish)
**Expected speedup: ~5-8% overall**

`aten::mish` takes 4.4% of CUDA time. SiLU is mathematically similar but has optimized CUDA kernels.

**Files to modify**: `model_encoder.py`, `model_dynamics_simple.py`, `model_policy.py`, `model_reward.py`, `model_termination.py`, `model_q_function.py`

```python
# Replace all instances of:
nn.Mish()
# With:
nn.SiLU()  # Swish activation - faster, nearly identical performance
```

#### 5. Pre-allocate Tensors in MPPI Planner
**Expected speedup: ~10-15%**

Avoid repeated tensor allocations in the hot loop:

```python
# In MPPIPlannerSimplePaper.__init__:
def __init__(self, ...):
    ...
    # Pre-allocate buffers (will be resized on first use)
    self._actions_buffer = None
    self._returns_buffer = None
    self._z_buffer = None

# In plan():
def plan(self, latent, ...):
    # Reuse buffers instead of creating new tensors
    if self._actions_buffer is None or self._actions_buffer.shape[1] != self.num_samples:
        self._actions_buffer = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=latent.device
        )
        self._returns_buffer = torch.empty(self.num_samples, device=latent.device)
    
    actions = self._actions_buffer
    # ... use pre-allocated buffers
```

#### 6. Optimize SimNorm with In-Place Operations
**Expected speedup: ~3-5%**

```python
# In util.py SimNorm.forward():
def forward(self, x):
    batch_size = x.shape[0]
    # Use view instead of reshape when possible (avoids copy)
    x = x.view(batch_size, -1, self.simplex_dim)
    # In-place division
    x = x.div_(self.temperature)
    x = F.softmax(x, dim=2)
    return x.view(batch_size, self.dim)
```

#### 7. Fuse Operations in two_hot_inv
**Expected speedup: ~2-3%**

This is called many times during MPPI rollouts:

```python
@torch.jit.script
def two_hot_inv_fused(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """JIT-compiled two_hot_inv for faster execution."""
    x_probs = F.softmax(x, dim=-1)
    dreg_bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x_symlog = torch.sum(x_probs * dreg_bins, dim=-1, keepdim=True)
    return torch.sign(x_symlog) * (torch.exp(torch.abs(x_symlog)) - 1)
```

---

### 🟡 LOWER IMPACT

#### 8. Use torch.inference_mode() Instead of torch.no_grad()
**Expected speedup: ~1-2%**

`inference_mode` is faster than `no_grad` as it also disables view tracking:

```python
# Replace:
@torch.no_grad()
def act(self, obs, ...):

# With:
@torch.inference_mode()
def act(self, obs, ...):
```

#### 9. Pin Memory for Faster CPU→GPU Transfers
**Expected speedup: ~1-2%**

```python
# In act():
obs = torch.FloatTensor(obs).pin_memory().to(self.device, non_blocking=True)
```

#### 10. Use Contiguous Memory Layout
**Expected speedup: ~1-2%**

```python
# In MPPI planner, ensure contiguous tensors:
actions_reshaped = actions.transpose(0, 1).contiguous()
```

#### 11. Batch Elite Selection
**Expected speedup: ~1%**

Use `torch.topk` with `sorted=False` for slightly faster selection:

```python
# In MPPI planner:
elite_idxs = torch.topk(returns, self.num_elites, dim=0, sorted=False).indices
```

---

## Implementation Plan

### Phase 1: Quick Wins (30 minutes - 1 hour)
1. ✅ Add `torch.cuda.amp.autocast()` around forward passes in `act()` and `train()`
2. ✅ Add `fast_mode` parameter to `act()` method
3. ✅ Replace `@torch.no_grad()` with `@torch.inference_mode()`
4. ✅ Compile policy, q_ensemble, termination

### Phase 2: Activation Function Change (30 minutes)
5. ✅ Replace `nn.Mish()` → `nn.SiLU()` in all model files

### Phase 3: Memory Optimizations (1-2 hours)
6. ✅ Pre-allocate tensors in MPPI planner
7. ✅ Optimize SimNorm with in-place ops
8. ✅ JIT compile `two_hot_inv`

---

## Expected Combined Speedup

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Mixed Precision (AMP) | 1.5-2x | 1.5-2x |
| Fast Mode (eval only) | 10-50x | N/A (different use case) |
| Compile all models | 1.2-1.4x | 1.8-2.8x |
| Mish → SiLU | 1.05-1.08x | 1.9-3.0x |
| Pre-allocate + fused ops | 1.1-1.2x | **2.1-3.6x** |

**Target**: Reduce from ~98ms/action to **~30-45ms/action** for full MPPI planning, **<2ms** for fast-mode policy inference.

---

## Files to Modify

1. **`tdmpc2.py`** - AMP, fast_mode, inference_mode, compile additional models
2. **`mppi_planner_simple.py`** - Pre-allocate buffers, optimize operations
3. **`util.py`** - JIT compile two_hot_inv, optimize SimNorm
4. **`model_encoder.py`** - Mish → SiLU
5. **`model_dynamics_simple.py`** - Mish → SiLU
6. **`model_policy.py`** - Mish → SiLU
7. **`model_reward.py`** - Mish → SiLU
8. **`model_termination.py`** - Mish → SiLU
9. **`model_q_function.py`** - Mish → SiLU

---

## Verification

After implementing, re-run the profiler to verify speedup:

```bash
sbatch resources/profile_tdmpc2.sbatch
```

Key metrics to compare:
- Total time for 50 action selections
- CUDA time for `aten::addmm` (should be ~halved with AMP)
- CUDA time for activation functions (should decrease)
