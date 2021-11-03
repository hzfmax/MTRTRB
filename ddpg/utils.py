# author : C.S. Ying
import math

import numpy as np
import torch
import torch.nn as nn


# ====== Run utils =======
def init_params(model, gain=1.0):
    for params in model.parameters():
        if len(params.shape) > 1:
            nn.init.xavier_uniform_(params.data, gain=gain)


# ====== Buffer =======
def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer(object):
    def __init__(self, obs_dim, act_dim, maxlen=int(1e6),
                 act_dtype=np.float32):
        self.size, self.ptr = 0, 0
        self.maxlen = maxlen
        self.act_dtype = act_dtype
        self.obs_buf = np.zeros(combined_shape(maxlen, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(maxlen, act_dim),
                                dtype=self.act_dtype)
        self.obs2_buf = np.zeros(combined_shape(maxlen, obs_dim),
                                 dtype=np.float32)
        self.rew_buf = np.zeros((maxlen, 1), dtype=np.float32)
        self.done_buf = np.zeros((maxlen, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def is_full(self):
        return self.__len__() == self.maxlen

    def store(self, obs, act, rew, done, obs2):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def _encode_sample(self, idxes):
        return (torch.as_tensor(x[idxes], dtype=torch.float32)
                for x in (self.obs_buf, self.act_buf, self.rew_buf,
                          self.done_buf, self.obs2_buf))

    def sample(self, batch_size=32):
        idxes = np.random.randint(0, self.size, size=batch_size)
        return self._encode_sample(idxes)


# ====== Decay & Noise =======
class CosineAnnealingDecay(object):
    def __init__(self, initial_x, T_max, eta_min, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = initial_x
        self.initial_x = initial_x
        self.last_epoch = last_epoch
        self.step()

    def _get_closed_form_x(self):
        if self.last_epoch == 0:
            return self.initial_x
        else:
            return self.eta_min + 1 / 2 * (self.eta_max - self.eta_min) * \
                (1 + math.cos(self.last_epoch / self.T_max * math.pi))

    def step(self):
        self.last_epoch += 1
        self._last_x = self._get_closed_form_x()

    def __call__(self):
        return self._last_x

    def reset(self):
        self._last_x = self.initial_x


class CosineAnnealingNormalNoise(object):
    def __init__(self, mu, sigma, sigma_min, T_max, last_epoch=-1):
        self.mu = mu
        self.sigma = sigma
        self._sigma_decay = CosineAnnealingDecay(sigma, T_max, sigma_min,
                                                 last_epoch)

    def step(self):
        self._sigma_decay.step()
        self.sigma = self._sigma_decay()

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return f'CosineAnnealingNormalNoise(mu={self.mu}, sigma={self.sigma})'
