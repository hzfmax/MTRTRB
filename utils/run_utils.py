# author : C.S. Ying
import numpy as np
import torch
import numba as nb
from tqdm import tqdm


@nb.njit('f8[:](f8[:], f8[:])', fastmath=True)
def scale(x, x_rng):
    x_embbed = x / x_rng
    return x_embbed


def evaluate_policy(env, policy, eval_epochs=10, scale_func=lambda x: x):
    policy.eval()
    reward = []
    for ep in range(eval_epochs):
        ep_rwd = 0.
        obs, done = env.reset()
        obs = scale_func(obs)
        for step in range(env.max_svs):
            act = policy.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, pwc, opc, done = env.step(act)
            ep_rwd += pwc + opc
            obs = scale_func(obs)
            if done:
                break
        reward.append(ep_rwd)
    reward = np.asarray(reward, dtype=float)
    avg_rwd = np.mean(reward)
    policy.train()
    return avg_rwd, reward


def fill_buffer_randomly(env_fn,
                         buffer,
                         scale_func=lambda x: x):

    env = env_fn()
    pbar = tqdm(total=buffer.size)
    while not buffer.is_full():
        obs, done = env.reset()
        obs = scale_func(obs)
        for step in range(env.max_svs):
            act = np.random.randn(*env.action_space.shape)
            obs2, pwc, opc, done = env.step(act)
            rew = pwc + opc
            obs2 = scale_func(obs2)
            buffer.store(obs, act, rew, done, obs2)
            if done:
                pbar.update(step + 1)
                break
            obs = obs2
