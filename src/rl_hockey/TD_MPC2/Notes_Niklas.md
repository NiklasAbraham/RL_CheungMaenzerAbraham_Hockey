

## Notes Niklas


#### Jobs


##### 1969941 - 2026-01-22_22-05-49
- simple fucking run of the current repo state nothing exiting
- no backprop reward shaping calssic boring things
- Horizon 18
--> fucked because 18! and config error 


##### 1970116 - 2026-01-22_22-35-14
- run with minimla reward shaping
- curretn state of repo and then with simple reward prop
- Horizon 18
--> fucked because 18! and config error


##### 1971497 - 2026-01-23_08-49-02
- run with minimal reward shaping
- run with backward prop

##### 1971498 - 2026-01-23_08-51-48
- run without shaping and back prop

##### 1971519 - killed
- run with backprop aber ohne reward shaping

##### 1971525
- more loggign , shorter horizion 5

##### 1972025 
- entire poliy change rolled back with the tanh, might be the ciritla fucking issue

#### Architecture of TDMPC-2 

```
MODEL-BASED RL COMPONENTS:

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
      (9): Linear(in_features=512, out_features=1024, bias=True)
      (10): SimNorm()
    )
  )
)
   Trainable Parameters: 1,063,424

2. DYNAMICS MODEL:
   Predicts next latent state given current state and action
OptimizedModule(
  (_orig_mod): DynamicsSimple(
    (net): Sequential(
      (0): Linear(in_features=1028, out_features=512, bias=True)
      (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (2): Mish()
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (5): Mish()
      (6): Linear(in_features=512, out_features=512, bias=True)
      (7): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (8): Mish()
      (9): Linear(in_features=512, out_features=1024, bias=True)
      (10): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (simnorm): SimNorm()
  )
)
   Trainable Parameters: 1,582,592

3. REWARD MODEL:
   Predicts reward given latent state and action
OptimizedModule(
  (_orig_mod): Reward(
    (net): Sequential(
      (0): Linear(in_features=1028, out_features=512, bias=True)
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
   Trainable Parameters: 1,107,045

3a. TERMINATION MODEL:
   Predicts termination probability given latent state and action
Termination(
  (mlp): Sequential(
    (0): Linear(in_features=1024, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=1, bias=True)
  )
)
   Trainable Parameters: 328,449

4. Q ENSEMBLE:
   5 Q-networks for value estimation
QEnsemble(
  (q_functions): ModuleList(
    (0-4): 5 x QFunction(
      (net): Sequential(
        (0): Linear(in_features=1028, out_features=512, bias=True)
        (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (2): Dropout(p=0.01, inplace=False)
        (3): Mish()
        (4): Linear(in_features=512, out_features=512, bias=True)
        (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (6): Mish()
        (7): Linear(in_features=512, out_features=512, bias=True)
        (8): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (9): Mish()
        (10): Linear(in_features=512, out_features=101, bias=True)
      )
    )
  )
)
   Trainable Parameters: 5,535,225

5. TARGET Q ENSEMBLE:
   Target network for stable Q-learning
   Same architecture as Q Ensemble
   Trainable Parameters: 0

6. POLICY NETWORK:
   Learns to mimic the MPC planner for fast inference
Policy(
  (net): Sequential(
    (0): Linear(in_features=1024, out_features=512, bias=True)
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
   Trainable Parameters: 1,057,288

PARAMETER SUMMARY:
  World Model (Encoder + Dynamics + Reward + Termination + Q): 9,616,735
  Policy Network: 1,057,288
  Target Q Network: 0
  TOTAL TRAINABLE PARAMETERS: 10,674,023

OPTIMIZERS:
  World Model + Q Optimizer: Adam (LR: 0.0003)
  Policy Optimizer: Adam (LR: 0.0003)

MPC PLANNER:
  Type: MPPI (Model Predictive Path Integral)
  Horizon: 9
  Samples per iteration: 512
  Planning iterations: 6
  Temperature: 0.5

```