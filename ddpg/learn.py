from functools import partial

import numpy as np
import torch
import time

from ddpg.ddpg import DDPG
from ddpg.utils import CosineAnnealingNormalNoise, ReplayBuffer
from utils.run_utils import evaluate_policy, fill_buffer_randomly, scale


def learn(env_fn,
          epochs,
          buffer_size,
          eval_freq,
          eta_init,
          eta_min,
          reward_scale=5000,
          **algo_kwargs):

    # Build env and get related info
    env = env_fn()
    obs_shp = env.observation_space.shape
    act_shp = env.action_space.shape
    act_dtype = env.action_space.dtype
    assert np.all(env.action_space.high == 1) & np.all(
        env.action_space.low == -1)

    # scalers
    obs_scale = partial(scale, x_rng=env.observation_space.high)

    def rwd_scale(rew):
        return rew / reward_scale

    # explorative noise
    noise = CosineAnnealingNormalNoise(mu=np.zeros(act_shp),
                                       sigma=eta_init,
                                       sigma_min=eta_min,
                                       T_max=epochs)

    # construct model and buffer
    model = DDPG(obs_shp[0], act_shp[0], epochs=epochs, **algo_kwargs)

    buffer = ReplayBuffer(obs_shp[0],
                          act_shp[0],
                          maxlen=buffer_size,
                          act_dtype=act_dtype)

    fill_buffer_randomly(env_fn, buffer, obs_scale)
    assert buffer.is_full()

    # init recorder
    q_opt = np.inf
    start = time.time()
    pop_opt = []
    try:
        # main loop
        for epoch in range(epochs):
            ret_ep, act_ep = 0., []
            obs, done = env.reset()
            obs = obs_scale(obs)

            for step in range(env.max_svs):
                act = model.act(torch.as_tensor(obs, dtype=torch.float32))
                act = np.clip(act + noise(), -1, 1)

                # env steps
                obs2, pwc, opc, done = env.step(act)
                rew = pwc + opc
                obs2 = obs_scale(obs2)

                # push experience into the buffer
                buffer.store(obs, act, rew, done, obs2)

                # record
                act_ep.append(act)
                ret_ep += rew

                obs = obs2

                # update the model
                model.update(buffer, rwd_scale)
                if done:
                    # update the global optima
                    if q_opt >= ret_ep:
                        q_opt = ret_ep
                        pop_opt = np.asarray(act_ep)

                    if epoch % 500 == 0:
                        print(f'EP: {epoch}|EpR: {ret_ep:.0f}| Q*: {q_opt:.0f}| T: {time.time()-start:.0f}|N:{noise.sigma:.3f}')

                    noise.step()
                    model.lr_step()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("done")
