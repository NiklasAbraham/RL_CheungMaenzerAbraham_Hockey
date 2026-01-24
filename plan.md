--- /dev/null
+++ /home/stud421/RL_CheungMaenzerAbraham_Hockey/optimization_plan.md
@@ -0,0 +1,108 @@
+# TD-MPC2 Optimization Plan
+
+## Diagnosis
+The profiling results indicate a severe **CPU bottleneck** caused by kernel launch overhead.
+- **Total CPU Time:** 4.90s
+- **Total GPU Time:** 1.25s
+- **Ratio:** The CPU is working 4x longer than the GPU.
+
+TD-MPC2 relies on Model Predictive Control (MPC), which performs thousands of small forward passes (rollouts) using the World Model. In standard PyTorch, each layer in these rollouts requires a separate communication round-trip between CPU and GPU.
+
+## Strategy 1: CUDA Graphs (Critical for 10x Speedup)
+
+This is the single most effective change for MPC-based algorithms. CUDA Graphs record the sequence of kernel launches once and replay them instantly, bypassing the Python/CPU overhead entirely.
+
+### Implementation
+Wrap the planning loop (MPPI/CEM) in a CUDA Graph.
+
+```python
+import torch
+
+class TDMPC2Agent:
+    # ... inside your agent class ...
+
+    def plan(self, obs, ...):
+        # 1. Warmup
+        if self.graph is None:
+            # Run once to initialize caches
+            self._plan_step(obs, ...)
+            
+            # 2. Capture
+            self.graph = torch.cuda.CUDAGraph()
+            with torch.cuda.graph(self.graph):
+                self.output = self._plan_step(obs, ...)
+        
+        # 3. Replay
+        # Copy new inputs into the graph's input tensors
+        self.static_obs.copy_(obs)
+        self.graph.replay()
+        return self.output
+```
+
+*Note: You must ensure tensor shapes are static (fixed batch size) for CUDA Graphs to work.*
+
+## Strategy 2: PyTorch 2.0 Compilation (`torch.compile`)
+
+If manually implementing CUDA Graphs is too complex for the current codebase structure, use `torch.compile`. It automatically fuses kernels and reduces overhead.
+
+**Action:**
+Add this to your model initialization:
+
+```python
+import torch
+
+# Compile the world model and policy
+self.model = torch.compile(self.model, mode="reduce-overhead")
+```
+*Note: `mode="reduce-overhead"` is specifically designed for small-batch, iterative workloads like RL.*
+
+## Strategy 3: Mixed Precision (AMP)
+
+Your profile shows `volta_sgemm` (FP32). The RTX 2080 Ti has Tensor Cores that are significantly faster with FP16 (Half Precision).
+
+**Action:**
+Wrap the forward pass in `autocast`:
+
+```python
+with torch.autocast(device_type='cuda', dtype=torch.float16):
+    action = agent.act(obs)
+```
+
+This reduces memory bandwidth usage and utilizes the specialized hardware on your GPU.
+
+## Strategy 4: Fusing Activations
+
+The profile shows `aten::mish` taking ~5% of compute. Standard Mish is memory-bound (reads memory, applies math, writes memory).
+
+**Action:**
+Ensure you are using `torch.nn.functional.mish` (which is optimized) or JIT script the activation if it is a custom implementation.
+
+```python
+@torch.jit.script
+def mish(x):
+    return x * torch.tanh(torch.nn.functional.softplus(x))
+```
+
+## Strategy 5: Algorithmic Tuning (Trade-offs)
+
+If code optimization doesn't yield the full 10x, adjust the complexity of the MPC planner in `config.json`.
+
+1.  **`num_samples`**: Currently **512**. Reducing to **256** cuts the workload in half linearly.
+2.  **`horizon`**: Currently **5**. Reducing to **3** reduces the depth of the rollout, significantly saving time, though it may hurt long-term planning capability.
+3.  **`num_iterations`**: Currently **6**. Reducing to **3** or **4** is often sufficient for simple tasks like Hopper or Reacher.
+
+## Summary of Expected Gains
+
+| Optimization | Estimated Speedup | Difficulty |
+| :--- | :--- | :--- |
+| **CUDA Graphs** | **3x - 8x** | High (Requires static shapes) |
+| **torch.compile** | **2x - 5x** | Low (One line of code) |
+| **Mixed Precision** | **1.5x** | Low |
+| **Algorithmic Tuning** | **2x - 4x** | Low (Config change) |
+
+**Recommendation:** Start with **`torch.compile(mode="reduce-overhead")`** and **Mixed Precision**. If that is not enough, implement **CUDA Graphs**.
+```

<!--
[PROMPT_SUGGESTION]Can you show me the code changes required to apply torch.compile to the TD-MPC2 agent class?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]How do I implement CUDA Graphs for the MPPI planning loop specifically?[/PROMPT_SUGGESTION]
-->
