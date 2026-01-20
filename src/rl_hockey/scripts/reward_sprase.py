import hockey.hockey_env as h_env

env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
obs = env.reset()

nonzero = 0
T = 5000

for t in range(T):
    a = env.action_space.sample()
    step_result = env.step(a)
    # Handle both 4 and 5 return values for env.step
    if len(step_result) == 4:
        obs, r, done, info = step_result
    elif len(step_result) == 5:
        obs, r, done, truncated, info = step_result
    else:
        raise ValueError(
            f"Unexpected number of values returned from env.step: {len(step_result)}"
        )
    if abs(float(r)) > 1e-12:
        nonzero += 1
    if done:
        obs = env.reset()

print("nonzero steps:", nonzero, "out of", T, "=", nonzero / T)
