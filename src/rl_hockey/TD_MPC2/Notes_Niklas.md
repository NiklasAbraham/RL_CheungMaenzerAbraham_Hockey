

## Notes Niklas


#### Architecture of TDMPC-2 

```
1. ENCODER NETWORK:
   Maps observations to latent state representations
OptimizedModule(
  (_orig_mod): Encoder(
    (net): Sequential(
      (0): Linear(in_features=18, out_features=512, bias=True)
      (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (2): Mish()
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (5): Mish()
      (6): Linear(in_features=512, out_features=512, bias=True)
      (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (8): Mish()
      (9): Linear(in_features=512, out_features=512, bias=True)
      (10): SimNorm()
    )
  )
)
   Trainable Parameters: 800,768

2. DYNAMICS MODEL:
   Predicts next latent state given current state and action
OptimizedModule(
  (_orig_mod): DynamicsSimple(
    (net): Sequential(
      (0): Linear(in_features=516, out_features=512, bias=True)
      (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (2): Mish()
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (5): Mish()
      (6): Linear(in_features=512, out_features=512, bias=True)
      (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (8): Mish()
      (9): Linear(in_features=512, out_features=512, bias=True)
      (10): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (simnorm): SimNorm()
  )
)
   Trainable Parameters: 1,056,768

3. REWARD MODEL:
   Predicts reward given latent state and action
OptimizedModule(
  (_orig_mod): Reward(
    (net): Sequential(
      (0): Linear(in_features=516, out_features=512, bias=True)
      (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (2): Mish()
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (5): Mish()
      (6): Linear(in_features=512, out_features=512, bias=True)
      (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (8): Mish()
      (9): Linear(in_features=512, out_features=101, bias=True)
    )
  )
)
   Trainable Parameters: 844,901

3a. TERMINATION MODEL:
   Predicts termination probability given latent state and action
Termination(
  (mlp): Sequential(
    (0): Linear(in_features=516, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=1, bias=True)
  )
)
   Trainable Parameters: 527,873

4. Q ENSEMBLE:
   5 Q-networks for value estimation
QEnsemble(
  (q_functions): ModuleList(
    (0-4): 5 x QFunction(
      (net): Sequential(
        (0): Linear(in_features=516, out_features=512, bias=True)
        (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (2): Mish()
        (3): Linear(in_features=512, out_features=512, bias=True)
        (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (5): Mish()
        (6): Linear(in_features=512, out_features=512, bias=True)
        (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (8): Mish()
        (9): Linear(in_features=512, out_features=101, bias=True)
      )
    )
  )
)
   Trainable Parameters: 4,224,505

5. TARGET Q ENSEMBLE:
   Target network for stable Q-learning
   Same architecture as Q Ensemble
   Trainable Parameters: 0

6. POLICY NETWORK:
   Learns to mimic the MPC planner for fast inference
Policy(
  (net): Sequential(
    (0): Linear(in_features=512, out_features=512, bias=True)
    (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (2): Mish()
    (3): Linear(in_features=512, out_features=512, bias=True)
    (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (5): Mish()
    (6): Linear(in_features=512, out_features=512, bias=True)
    (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (8): Mish()
  )
  (mean_head): Linear(in_features=512, out_features=4, bias=True)
  (log_std_head): Linear(in_features=512, out_features=4, bias=True)
)
   Trainable Parameters: 795,144

PARAMETER SUMMARY:
  World Model (Encoder + Dynamics + Reward + Termination + Q): 7,454,815
  Policy Network: 795,144
  Target Q Network: 0
  TOTAL TRAINABLE PARAMETERS: 8,249,959

OPTIMIZERS:
  World Model + Q Optimizer: Adam (LR: 0.0003)
  Policy Optimizer: Adam (LR: 0.0003)

MPC PLANNER:
  Type: MPPI (Model Predictive Path Integral)
  Horizon: 10
  Samples per iteration: 512
  Planning iterations: 6
  Temperature: 0.5

================================================================================

```