# TD-MPC2 Profiling Analysis & Training Optimization

## Optimization Results Summary

### ✅ Implemented Optimizations (Reference Repo Strategy)

We implemented the same optimization strategy used by the official TD-MPC2 repository. The results show a **~2x speedup** in training:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **CPU time per step** | 138.3ms | 67.4ms | **2.05× faster** |
| **CUDA time per step** | 18.8ms | 9.7ms | **1.94× faster** |
| **Total CPU time (25 steps)** | 3.457s | 1.686s | **2.05× faster** |
| **Total CUDA time (25 steps)** | 469.7ms | 242.7ms | **1.93× faster** |
| **`aten::linear` calls** | 5,450 | 900 | **6× fewer** |
| **`aten::layer_norm` calls** | 3,750 | 600 | **6.25× fewer** |
| **`CompiledFunctionBackward` calls** | 675 | 325 | **2.08× fewer** |

### What Was Changed

#### 1. `capturable=True` in Optimizers
**File**: [tdmpc2.py#L244-L261](../src/rl_hockey/TD_MPC2/tdmpc2.py#L244-L261)

```python
# BEFORE
self.optimizer = torch.optim.Adam([...], lr=self.lr)
self.pi_optimizer = torch.optim.Adam([...], lr=self.lr, eps=1e-5)

# AFTER
self.optimizer = torch.optim.Adam([...], lr=self.lr, capturable=True)
self.pi_optimizer = torch.optim.Adam([...], lr=self.lr, eps=1e-5, capturable=True)
```

**Why**: Allows the optimizer to be captured in CUDAGraphs, enabling the GPU to execute the entire optimization step as a single graph instead of launching individual kernels.

#### 2. `cudagraph_mark_step_begin()` at Training Iteration Start
**File**: [tdmpc2.py#L632-L635](../src/rl_hockey/TD_MPC2/tdmpc2.py#L632-L635)

```python
for _ in range(steps):
    # Mark step boundary for CUDAGraphs (like reference repo)
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
```

**Why**: Explicitly marks iteration boundaries for torch.compile's CUDAGraph capture. This allows the compiler to know where one training step ends and the next begins, enabling better graph optimization.

#### 3. Pre-compute TD Targets Before the Dynamics Loop
**File**: [tdmpc2.py#L671-L678](../src/rl_hockey/TD_MPC2/tdmpc2.py#L671-L678)

```python
# BEFORE: TD targets computed INSIDE the loop for each timestep
for t in range(horizon):
    # ... compute TD target for this timestep
    z_bootstrap = z_seq[:, t + n].detach()
    next_action, _, _, _ = self.policy.sample(z_bootstrap)
    target_q = self.target_q_ensemble.min(z_bootstrap, next_action)
    # ...

# AFTER: TD targets computed ONCE before the loop
with torch.no_grad():
    next_z = z_seq[:, 1:].detach()  # (batch, horizon, latent)
    td_targets = self._compute_td_targets(next_z, rewards_seq, dones_seq, horizon)
```

**Why**: Instead of computing policy actions and target Q-values H times (once per horizon step), we compute them all at once in a single batched operation. This reduces:
- Policy forward passes: 8 → 1
- Target Q-ensemble forward passes: 8 → 1

#### 4. Pre-compute Lambda Weights
**File**: [tdmpc2.py#L692-L695](../src/rl_hockey/TD_MPC2/tdmpc2.py#L692-L695)

```python
# BEFORE: Computed inside loop
for t in range(horizon):
    weight = self.lambda_coef**t  # Recomputed every iteration

# AFTER: Pre-computed tensor
lambda_weights = self.lambda_coef ** torch.arange(
    horizon, device=self.device, dtype=torch.float32
)
# Then use lambda_weights[t] in loop
```

**Why**: Avoids Python overhead of computing `lambda_coef**t` on each iteration. Minor improvement but contributes to overall efficiency.

#### 5. Batched Reward and Q-Value Predictions After Dynamics Rollout
**File**: [tdmpc2.py#L715-L748](../src/rl_hockey/TD_MPC2/tdmpc2.py#L715-L748)

```python
# BEFORE: Predictions INSIDE the loop
for t in range(horizon):
    r_pred_logits = self.reward(z_pred, a_t)  # Called 8 times
    q_preds_logits = self.q_ensemble(z_pred, a_t)  # Called 8 times
    # ... compute losses

# AFTER: Predictions OUTSIDE the loop (single batched call)
_zs = zs[:-1]  # (horizon, batch, latent)
_zs_flat = _zs.reshape(-1, self.latent_dim)  # (horizon*batch, latent)
_actions_flat = actions_seq.permute(1, 0, 2).reshape(-1, self.action_dim)

# SINGLE forward pass for all timesteps
reward_preds = self.reward(_zs_flat, _actions_flat)
q_preds_all = self.q_ensemble(_zs_flat, _actions_flat)
```

**Why**: This is the **most impactful optimization**. Instead of calling `reward()` and `q_ensemble()` H times (8 times for horizon=8), we call them once with a batch size of `horizon * batch_size`. This:
- Reduces CUDA kernel launches from 16 (8 reward + 8 Q) to 2 (1 reward + 1 Q)
- Amortizes kernel launch overhead over more data
- Enables better GPU utilization with larger batch sizes

#### 6. New `_compute_td_targets()` Helper Method
**File**: [tdmpc2.py#L908-L964](../src/rl_hockey/TD_MPC2/tdmpc2.py#L908-L964)

This new method computes all TD targets in a single batched operation:

```python
@torch.no_grad()
def _compute_td_targets(self, next_z, rewards_seq, dones_seq, horizon):
    """Compute TD targets for all timesteps at once (like reference repo)."""
    batch_size = next_z.shape[0]
    
    if self.n_step == 1:
        # Flatten for batched computation
        next_z_flat = next_z.reshape(-1, self.latent_dim)
        
        # Get policy actions for ALL next states at once
        next_actions, _, _, _ = self.policy.sample(next_z_flat)
        
        # Get target Q-values for ALL next states at once
        target_q_flat = self.target_q_ensemble.min(next_z_flat, next_actions)
        target_q = target_q_flat.reshape(batch_size, horizon, 1)
        
        # TD target: r + γ(1-d)Q(s',a')
        td_targets = rewards_seq + self.gamma * (1.0 - dones_seq) * target_q
    # ...
    return td_targets
```

### Why The Dynamics Loop Remains Sequential

The dynamics rollout **must remain sequential** because each prediction depends on the previous one:

```
z₁ = dynamics(z₀, a₀)
z₂ = dynamics(z₁, a₁)  ← depends on z₁!
z₃ = dynamics(z₂, a₂)  ← depends on z₂!
```

The reference TD-MPC2 repo uses the same approach: **sequential dynamics, batched everything else**.

The `zs` tensor (predicted latent states) is needed for **policy training** - the policy learns to maximize Q-values in the predicted future states, not just ground-truth states.

### Profiling Evidence

The kernel call reduction proves the optimization worked:

| Kernel Type | Before | After | Reduction |
|-------------|--------|-------|-----------|
| `aten::linear` | 5,450 calls | 900 calls | **6× fewer** |
| `aten::layer_norm` | 3,750 calls | 600 calls | **6.25× fewer** |
| `CompiledFunctionBackward` | 675 calls | 325 calls | **2.08× fewer** |
| `aten::copy_` | 15,700 calls | 4,350 calls | **3.6× fewer** |

With `horizon=8`, we'd expect roughly a 6-8× reduction in forward pass calls (from inside-loop to batched), which matches the observed ~6× reduction in `aten::linear` calls.

---

## Original Profiling Results (BEFORE Optimization)

```
================================================================================
TD-MPC2 PROFILING SUMMARY
================================================================================

Device: cuda
GPU 0: NVIDIA GeForce RTX 2080 Ti
GPU Memory: 11.3 GB
Config: /home/stud421/RL_CheungMaenzerAbraham_Hockey/configs/curriculum_tdmpc2.json
Iterations: 50

Agent configuration:
  obs_dim: 18
  action_dim: 8
  latent_dim: 256
  horizon: 8
  num_samples: 256
  num_iterations: 6

================================================================================
TRAINING STEP (Most Relevant for Training Speedup)
================================================================================

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                             train_step         0.00%       0.000us         0.00%       0.000us       0.000us        2.718s       578.63%        2.718s     108.714ms            25
                                         training_total         0.01%     447.509us        82.29%        2.844s        2.844s       0.000us         0.00%     233.014ms     233.014ms             1
                                             train_step        33.62%        1.162s        82.27%        2.844s     113.751ms       0.000us         0.00%     233.014ms       9.321ms            25
autograd::engine::evaluate_function: CompiledFunctio...         1.13%      38.935ms        11.02%     380.809ms     564.162us       0.000us         0.00%     208.957ms     309.565us           675
                               CompiledFunctionBackward         4.02%     139.107ms         6.75%     233.469ms     345.880us     185.804ms        39.56%     189.164ms     280.244us           675
                               Optimizer.step#Adam.step         0.00%       0.000us         0.00%       0.000us       0.000us      72.810ms        15.50%      72.810ms       1.456ms            50
                                       CompiledFunction         3.18%     109.849ms         6.10%     210.871ms     312.401us      61.002ms        12.99%      64.184ms      95.088us           675
                                           aten::linear         0.60%      20.731ms        13.44%     464.646ms      85.256us       0.000us         0.00%      49.233ms       9.034us          5450
                                            aten::copy_         2.97%     102.633ms         6.52%     225.223ms      14.345us      37.810ms         8.05%      37.825ms       2.409us         15700
triton_red_fused__to_copy_add_fill_mish_mul_native_l...         0.00%       0.000us         0.00%       0.000us       0.000us      35.259ms         7.51%      35.259ms      25.185us          1400
                                               aten::to         0.35%      12.107ms         7.98%     275.968ms      14.080us       0.000us         0.00%      30.427ms       1.552us         19600
                                         aten::_to_copy         1.05%      36.339ms         7.63%     263.861ms      20.736us       0.000us         0.00%      30.427ms       2.391us         12725
                                       aten::layer_norm         0.26%       8.845ms         5.16%     178.321ms      47.552us       0.000us         0.00%      29.293ms       7.811us          3750
void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us      29.095ms         6.19%      29.095ms       6.064us          4798
                             Torch-Compiled Region: 3/0         0.37%      12.731ms         2.89%      99.912ms     499.562us       0.000us         0.00%      25.218ms     126.092us           200
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.949ms         5.10%      23.949ms       1.797us         13325
                                             aten::add_         2.48%      85.892ms         3.90%     134.794ms       9.509us      22.355ms         4.76%      22.355ms       1.577us         14175
                               Optimizer.step#Adam.step         1.11%      38.319ms         2.46%      85.192ms       1.704ms       0.000us         0.00%      19.183ms     383.663us            50
triton_red_fused__to_copy_add_fill_mish_mul_native_l...         0.00%       0.000us         0.00%       0.000us       0.000us      17.105ms         3.64%      17.105ms      28.509us           600
                                            aten::addmm         2.10%      72.608ms         3.52%     121.652ms      44.643us      16.799ms         3.58%      16.811ms       6.169us          2725
                                              aten::mul         1.86%      64.121ms         3.00%     103.808ms      15.210us      16.795ms         3.58%      16.807ms       2.463us          6825
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      16.354ms         3.48%      16.354ms       1.874us          8725
triton_per_fused__to_copy_add_fill_mish_mul_native_l...         0.00%       0.000us         0.00%       0.000us       0.000us      14.556ms         3.10%      14.556ms       2.426us          6000
                             Torch-Compiled Region: 1/1         0.54%      18.515ms         2.23%      77.059ms     385.294us       0.000us         0.00%      13.550ms      67.750us           200
void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us      12.702ms         2.70%      12.702ms       7.057us          1800
                             Torch-Compiled Region: 2/1         0.47%      16.349ms         1.73%      59.935ms     299.674us       0.000us         0.00%      12.435ms      62.175us           200
void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us      12.277ms         2.61%      12.277ms       6.820us          1800
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      10.974ms         2.34%      10.974ms       4.027us          2725
                                aten::native_layer_norm         0.65%      22.559ms         1.66%      57.312ms      30.567us      10.853ms         2.31%      10.853ms       5.788us          1875
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      10.853ms         2.31%      10.853ms       5.788us          1875
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.457s
Self CUDA time total: 469.702ms
```

---

## Profiling Results AFTER Optimization

```
================================================================================
TRAINING STEP (AFTER OPTIMIZATION)
================================================================================

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             train_step         0.00%       0.000us         0.00%       0.000us       0.000us        1.211s       498.84%        1.211s      48.431ms            25  
                               Optimizer.step#Adam.step         0.00%       0.000us         0.00%       0.000us       0.000us     154.580ms        63.69%     154.580ms       3.092ms            50  
                                         training_total         0.03%     461.066us        79.74%        1.344s        1.344s       0.000us         0.00%     122.513ms     122.513ms             1  
                                             train_step        37.29%     628.626ms        79.71%        1.344s      53.747ms       0.000us         0.00%     122.513ms       4.901ms            25  
autograd::engine::evaluate_function: CompiledFunctio...         0.81%      13.591ms        10.47%     176.418ms     542.825us       0.000us         0.00%      97.689ms     300.580us           325  
                               CompiledFunctionBackward         3.96%      66.682ms         7.05%     118.771ms     365.450us      88.632ms        36.52%      92.305ms     284.014us           325  
                                       CompiledFunction         3.37%      56.744ms         5.36%      90.428ms     278.240us      34.535ms        14.23%      37.291ms     114.742us           325  
                               Optimizer.step#Adam.step         2.27%      38.346ms         9.93%     167.366ms       3.347ms       0.000us         0.00%      29.924ms     598.477us            50  
                                           aten::linear         0.20%       3.431ms         4.04%      68.083ms      75.648us       0.000us         0.00%      18.578ms      20.642us           900  
triton_red_fused__to_copy_add_fill_mish_mul_native_l...         0.00%       0.000us         0.00%       0.000us       0.000us      17.147ms         7.06%      17.147ms      28.579us           600  
                                            aten::copy_         1.56%      26.355ms         3.63%      61.259ms      14.082us      14.859ms         6.12%      14.859ms       3.416us          4350  
turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_sta...         0.00%       0.000us         0.00%       0.000us       0.000us      14.680ms         6.05%      14.680ms      20.333us           722  
                             Torch-Compiled Region: 1/2         1.10%      18.524ms         4.32%      72.845ms     364.223us       0.000us         0.00%      12.984ms      64.919us           200  
                                       aten::layer_norm         0.08%       1.429ms         1.62%      27.333ms      45.555us       0.000us         0.00%      12.539ms      20.899us           600  
                                    aten::_foreach_div_         0.41%       6.836ms         3.49%      58.862ms     130.805us     328.408us         0.14%      10.182ms      22.626us           450  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       9.959ms         4.10%       9.959ms       2.108us          4725  
                                             aten::div_         1.71%      28.896ms         3.05%      51.451ms      10.889us       9.888ms         4.07%       9.892ms       2.094us          4725  
                                               aten::to         0.17%       2.907ms         3.27%      55.062ms       6.403us       0.000us         0.00%       9.576ms       1.114us          8600  
                                         aten::_to_copy         0.51%       8.605ms         3.09%      52.155ms      18.965us       0.000us         0.00%       9.576ms       3.482us          2750  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.247ms         3.40%       8.247ms       1.874us          4400  
                             Torch-Compiled Region: 3/1         0.14%       2.350ms         0.68%      11.496ms     459.847us       0.000us         0.00%       7.994ms     319.747us            25  
turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       7.983ms         3.29%       7.983ms      18.783us           425  
                                             aten::add_         2.32%      39.134ms         3.87%      65.224ms      16.618us       7.727ms         3.18%       7.732ms       1.970us          3925  
                                            aten::addmm         0.75%      12.637ms         1.03%      17.281ms      38.403us       7.571ms         3.12%       7.571ms      16.825us           450  
                             Torch-Compiled Region: 3/0         0.09%       1.554ms         0.56%       9.493ms     379.735us       0.000us         0.00%       7.368ms     294.731us            25  
                                   aten::_foreach_copy_         0.37%       6.294ms         1.06%      17.916ms      27.563us       5.765ms         2.38%       6.298ms       9.690us           650  
                                              aten::mul         2.16%      36.393ms         3.01%      50.689ms      20.480us       5.698ms         2.35%       5.698ms       2.302us          2475  
void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us       5.395ms         2.22%       5.395ms       6.347us           850  
    autograd::engine::evaluate_function: AddmmBackward0         0.09%       1.454ms         0.85%      14.288ms     114.301us       0.000us         0.00%       5.300ms      42.396us           125  
triton_per_fused__to_copy_native_layer_norm_native_l...         0.00%       0.000us         0.00%       0.000us       0.000us       5.180ms         2.13%       5.180ms       2.158us          2400  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.686s
Self CUDA time total: 242.718ms
```

### Side-by-Side Comparison

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Self CPU time total** | 3.457s | 1.686s | **-51%** |
| **Self CUDA time total** | 469.7ms | 242.7ms | **-48%** |
| **train_step CPU avg** | 113.75ms | 53.75ms | **-53%** |
| **train_step CUDA avg** | 9.32ms | 4.90ms | **-47%** |
| **`aten::linear` calls** | 5,450 | 900 | **-83%** |
| **`aten::layer_norm` calls** | 3,750 | 600 | **-84%** |
| **`aten::copy_` calls** | 15,700 | 4,350 | **-72%** |
| **`CompiledFunctionBackward` calls** | 675 | 325 | **-52%** |

---

## Original Single Action Selection (For Reference)

```
================================================================================
SINGLE ACTION SELECTION (For Reference)
================================================================================

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              act_total         0.00%       0.000us         0.00%       0.000us       0.000us        6.455s       888.58%        6.455s        6.455s             1
                                              act_total        20.67%        1.336s       100.00%        6.462s        6.462s       0.000us         0.00%     726.627ms     726.627ms             1
                                           aten::linear         2.78%     179.440ms        51.82%        3.349s      53.409us       0.000us         0.00%     422.392ms       6.737us         62700
                                       aten::layer_norm         1.53%      98.727ms        27.80%        1.797s      40.651us       0.000us         0.00%     305.008ms       6.901us         44200
                                            aten::addmm        13.00%     840.333ms        18.19%        1.175s      34.875us     193.282ms        26.61%     193.379ms       5.738us         33700
                                            aten::copy_         5.25%     339.400ms        10.90%     704.298ms      11.314us     154.239ms        21.23%     154.275ms       2.478us         62250
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 6.462s
Self CUDA time total: 726.444ms

================================================================================
PROFILING COMPLETE
================================================================================
```

---

## Understanding the BEFORE Profiling Results

This section explains the profiling data that motivated our optimizations.

### Training Step Breakdown (Before Optimization)

The training step profile over 25 iterations reveals critical bottlenecks:

| Metric | Value | Meaning |
|--------|-------|---------|
| **Total CPU Time** | 3.457s | Time the CPU spent orchestrating operations |
| **Total CUDA Time** | 469.7ms | Actual GPU compute time |
| **CPU/CUDA Ratio** | **7.4x** | CPU takes 7.4x longer than GPU — severe bottleneck |
| **Time per train step** | 113.75ms CPU / 9.32ms CUDA | Per-step overhead |

#### Top Time Consumers (Training)

1. **`CompiledFunctionBackward` (39.6% CUDA, 6.75% CPU total)**
   - This is the backward pass through torch.compile'd functions
   - 185.8ms of CUDA time across 675 calls
   - torch.compile adds overhead but also provides some fusion benefits
   - The high CPU overhead (233ms total) suggests compilation/dispatch costs

2. **`Optimizer.step#Adam.step` (15.5% CUDA)**
   - Adam optimizer updates take 72.8ms of GPU time across 50 calls (2 optimizers × 25 steps)
   - This is ~1.46ms per optimizer step — relatively efficient

3. **`aten::linear` (13.4% CPU total, but only 49.2ms CUDA)**
   - 5,450 linear layer calls
   - CPU takes 464ms to dispatch these operations, but GPU only computes for 49ms
   - **This is the CPU bottleneck** — Python/PyTorch dispatch overhead dominates

4. **`aten::copy_` / `aten::to` / `aten::_to_copy` (~14.5% CPU total)**
   - 15,700 copy operations consuming 225ms CPU time
   - 19,600 `to()` calls consuming 276ms CPU time  
   - These are dtype conversions and device transfers
   - Only 30-38ms of actual CUDA time — the rest is CPU overhead

5. **`aten::layer_norm` (5.16% CPU total)**
   - 3,750 layer norm calls taking 178ms CPU but only 29ms CUDA
   - 6x overhead ratio due to kernel launch costs

6. **Triton Fused Kernels (positive sign)**
   - `triton_red_fused_*` kernels show torch.compile IS fusing some operations
   - ~52ms of CUDA time in fused operations — this is actually working

### The Core Problem (Solved): CPU Dispatch Overhead

Looking at the BEFORE numbers:
- **CPU time**: 3.457s for 25 training steps = **138ms per step**
- **CUDA time**: 469.7ms for 25 training steps = **18.8ms per step**

The GPU finished in 18.8ms but had to wait **119.2ms** for the CPU to prepare the next batch of operations. This was caused by:

1. **Too many kernel launches**: 8× reward predictions, 8× Q predictions per training step
2. **Python loop overhead**: Sequential Python `for` loop couldn't be optimized by torch.compile
3. **Kernel launch latency**: ~5-10μs per CUDA kernel × thousands of kernels

**After optimization**: CPU time dropped to 67.4ms (2.05× faster) by batching predictions and enabling CUDAGraph optimizations.

### Why torch.compile Alone Wasn't Enough

The BEFORE profile shows `Torch-Compiled Region` entries:
- Region 3/0: 99.9ms CPU, 25.2ms CUDA
- Region 1/1: 77ms CPU, 13.5ms CUDA  
- Region 2/1: 60ms CPU, 12.4ms CUDA

torch.compile fuses operations and generates Triton kernels, but it couldn't fully optimize our code because:
- **Control flow breaks compilation**: The `for t in range(horizon)` loop prevented full graph capture
- **Many small kernel calls**: 8× reward and Q predictions launched separate kernels

**Solution implemented**: We kept the dynamics loop sequential (necessary for autoregressive rollout) but moved reward/Q predictions OUTSIDE the loop into single batched calls. Combined with `capturable=True` and `cudagraph_mark_step_begin()`, this allowed better CUDAGraph capture.

---

## Optimization Status Summary

### ✅ Implemented Optimizations (Achieved ~2x Speedup)

| Optimization | Status | Actual Impact |
|--------------|--------|---------------|
| `capturable=True` in optimizers | ✅ **Done** | Enables CUDAGraph capture |
| `cudagraph_mark_step_begin()` | ✅ **Done** | Better graph optimization |
| Pre-compute TD targets | ✅ **Done** | 8→1 policy/Q-target forward passes |
| Pre-compute lambda weights | ✅ **Done** | Minor CPU overhead reduction |
| Batched reward predictions | ✅ **Done** | 8→1 reward forward passes |
| Batched Q-value predictions | ✅ **Done** | 8→1 Q-ensemble forward passes |

**Result**: ~2x speedup achieved (138ms → 67ms per train step)

### 🔄 Remaining Optimization Opportunities

| Optimization | Status | Expected Impact |
|--------------|--------|-----------------|
| Batch `two_hot_inv` in policy update | ❌ Not done | 1.3-1.5x policy update |
| `torch.inference_mode()` for inference | ❌ Not done | 5-15% inference speedup |
| Optimize gradient norm computation | ❌ Not done | 10-20% gradient time |
| Buffer sampling optimization | ❌ Not done | 1.2-1.5x sampling |
| Fused Adam optimizer | ❌ Not done | 10-20% optimizer step |
| Reduce horizon (8→5) | ⚙️ Config option | ~1.6x (trade-off with planning) |
| Larger batch size | ⚙️ Config option | 1.1-1.3x (memory trade-off) |

---

## Technical Background: Why the Dynamics Loop Remains Sequential

### The Problem: Autoregressive Dependencies

The training loop has an **autoregressive structure** where each dynamics prediction depends on the previous one:

```
Step 0: z₁_pred = dynamics(z₀, a₀)           # z₀ from encoder
Step 1: z₂_pred = dynamics(z₁_pred, a₁)     # depends on z₁_pred!
Step 2: z₃_pred = dynamics(z₂_pred, a₂)     # depends on z₂_pred!
...
Step H-1: zₕ_pred = dynamics(zₕ₋₁_pred, aₕ₋₁)
```

This creates a **sequential dependency chain** that cannot be fully parallelized.

### The Reference Repo Strategy (What We Implemented)

The official TD-MPC2 repository uses a **hybrid strategy**:

1. **Sequential dynamics rollout** (necessary for autoregressive `zs` used in policy update)
2. **Batched predictions AFTER rollout** (Q, reward predictions for all timesteps at once)

```python
# Build zs sequentially (must be autoregressive)
zs = [z_seq[:, 0]]
for t in range(horizon):
    z_pred = self.dynamics(z_pred, actions_seq[:, t])
    zs.append(z_pred)

# THEN batch all predictions (this is what we optimized)
zs_flat = torch.stack(zs[:-1]).reshape(-1, latent_dim)
actions_flat = actions_seq.permute(1, 0, 2).reshape(-1, action_dim)
reward_preds = self.reward(zs_flat, actions_flat)      # 1 call instead of 8
q_preds = self.q_ensemble(zs_flat, actions_flat)       # 1 call instead of 8
```

### Why `zs` Must Be Autoregressive

The predicted latent states `zs` are needed for **policy training**:

```python
def _update_policy(self, zs, ...):
    actions = self.policy(zs)           # Policy actions from PREDICTED states
    q_values = self.q_ensemble(zs, actions)  # Q-values in predicted trajectory
```

The policy learns to maximize Q-values **in the predicted future states**, not just ground-truth states. This is critical for model-based RL.

---

## Future Optimization Opportunities

The following optimizations were identified but not yet implemented. They could provide additional speedups beyond the ~2x already achieved.

### Priority 1: Batch `two_hot_inv` in Policy Update

**Problem**: In the policy update, `two_hot_inv` is called separately for each Q-network:

```python
q_values = torch.stack(
    [two_hot_inv(q_logits, self.num_bins, self.vmin, self.vmax)
     for q_logits in q_logits_all],
    dim=0,
)
```

This creates `num_q` (default 5) separate calls.

**Solution**: Modify `two_hot_inv` to accept batched input:

```python
q_logits_stacked = torch.stack(q_logits_all, dim=0)  # (num_q, batch, num_bins)
q_values = two_hot_inv_batched(q_logits_stacked, self.num_bins, self.vmin, self.vmax)
```

**Expected speedup**: 1.3-1.5x for policy update portion.

---

### Priority 2: Use `torch.inference_mode()` for Non-Training Code

**Problem**: `torch.no_grad()` has more overhead than `torch.inference_mode()`.

**Solution**: Replace in `act()`, `act_batch()`, and target network computation:

```python
@torch.inference_mode()  # Instead of @torch.no_grad()
def act(self, obs, deterministic=False, t0=False):
```

**Expected speedup**: 5-15% for inference paths.

---

### Priority 3: Optimize Gradient Norm Computation

**Problem**: Computing gradient norms 6 times per training step is wasteful when using `max_norm=float("inf")`:

```python
grad_norm_encoder = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=float("inf"))
grad_norm_dynamics = torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), max_norm=float("inf"))
# ... 4 more times
```

**Solution**: Compute norm once for all parameters:

```python
torch.nn.utils.clip_grad_norm_(self._model_params, max_norm=self.grad_clip_norm)
```

**Expected speedup**: 10-20% of gradient computation time.

---

### Priority 4: Buffer Sampling Optimization

**Problem**: `sample()` uses Python loops to build batches.

**Solution**: Pre-allocate tensors and use advanced indexing.

**Expected speedup**: 1.2-1.5x for sampling.

---

### Priority 5: Fused Adam Optimizer

**Problem**: Standard Adam has separate kernels for each parameter update.

**Solution**: Use `torch.optim.AdamW` with `fused=True` (PyTorch 2.0+):

```python
self.optimizer = torch.optim.AdamW(param_groups, lr=self.lr, fused=True)
```

**Expected speedup**: 10-20% faster optimizer step.

---

### Configuration Tuning Options

These don't require code changes:

| Config Change | Trade-off | Expected Impact |
|---------------|-----------|-----------------|
| `horizon: 5` (from 8) | Less planning depth | ~1.6x faster |
| `batch_size: 512` (from 256) | More GPU memory | 1.1-1.3x faster |

---

## Estimated Additional Speedup Potential

| Optimization | Expected Speedup |
|--------------|------------------|
| Batch two_hot_inv | 1.1-1.2x |
| torch.inference_mode() | 1.05-1.1x |
| Gradient norm optimization | 1.05-1.1x |
| Buffer sampling | 1.1-1.2x |
| Fused Adam | 1.05-1.1x |
| **Combined (multiplicative)** | **~1.3-1.5x additional** |

With the current ~2x speedup already achieved, implementing these remaining optimizations could potentially yield **~2.5-3x total speedup** compared to the original baseline.
