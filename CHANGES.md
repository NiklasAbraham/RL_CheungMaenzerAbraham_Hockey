# Changes (integrate_tdmpc2_eval)

## TDMPC2 archive integration

- `add_tdmpc2_to_archive.py`: add TDMPC2 checkpoints to archive (config read from run folder)
- `run_archive_tournament.py`: run tournaments between archived agents
- `ARCHIVE_GUIDE.md`: guide for adding agents, calibrating, running tournaments

## Training and curriculum

- SAC run function for single runs
- `curriculum_tdmpc2_mixed_opponents.json`: mixed opponent curriculum
- Opponent manager: archive distribution and skill range
- Run manager and curriculum manager extended for archive opponents

## Plotting

- `plot_episode_logs.py`: more detailed episode log plots

## Resources

- `train_sac_single_run.sbatch`: SAC training on cluster
- `run_archive_calibration.sbatch`, `run_archive_tournament.sbatch`: archive jobs
- Container and .gitignore updates
