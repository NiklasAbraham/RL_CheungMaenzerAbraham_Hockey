# Changes

## How to use

- agents stored in archive:
    - rating, checkpoint, config, tags
    - bots: tagged as `baseline`
- new opponent type: `archive` in curriculum:
    - distribution (skill-based, random, baseline)
    - skill range
- add agents manually to archive via `archive.add_agent()`
    - will have tag `needs_calibration` by default
- calibrate agents in archive using `calibrate.py`
- running rating estimated during training (doesn't affect opponent ratings)
- agent config supports `checkpoint` to continue training from a saved model

## To implement

- agent loading for other agents
- use deterministic flag in agent.act
- combine logging (Niklas)
