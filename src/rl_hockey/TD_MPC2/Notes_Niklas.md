

## Notes Niklas


#### Jobs


#### Snapshot Niklas 01.02 14:55

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1980432      week hockey_s  stud421  R      11:49      1 tcml-node37   -> jannik SAC test by Niklas
           1980366      week hockey_s  stud421  R    4:59:44      1 tcml-node37   -> new niklas with jannick and the self play starting at episode 3000
           1980103      week hockey_s  stud421  R 1-02:18:05      1 tcml-node37   -> old niklas version with oppponent
           1980094      week hockey_s  stud421  R 1-03:05:05      1 tcml-node37   -> Horizion, without opponent
           1980093      week hockey_s  stud421  R 1-03:05:15      1 tcml-node37   -> Horizion, without opponent
           1980092      week hockey_s  stud421  R 1-03:05:27      1 tcml-node37   -> Horizion, without opponent



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