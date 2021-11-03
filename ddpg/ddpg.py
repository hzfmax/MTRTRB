from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ddpg.utils import init_params


class MLP(nn.Module): 
    def __init__(self, sizes, activ_fn, activ_out):
        super(MLP, self).__init__()
        nl = len(sizes)
        assert nl >= 2, "at least two layers should be specified"

        layers = []
        for i in range(nl - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1])]
            activ = activ_out if i == nl - 2 else activ_fn
            layers += [activ()]
        self.module = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.module(inputs)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dims):
        super().__init__()
        self.net = MLP([obs_dim] + list(hid_dims) + [act_dim],
                       activ_fn=nn.ReLU,
                       activ_out=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dims):
        super().__init__()
        self.net = MLP([obs_dim + act_dim] + list(hid_dims) + [1],
                       activ_fn=nn.ReLU,
                       activ_out=nn.Identity)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class DDPG(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dims, batch_size, tau, gamma,
                 lr_pi, lr_vf, lr_decay, epochs, device, max_grad_norm=2.0, **kwargs):
        super(DDPG, self).__init__()
        self.device = torch.device(device=device)
        self.batch_size = batch_size
        self.gamma, self.tau = gamma, tau
        self.max_grad_norm = max_grad_norm

        # Evaluation networks
        self.pi = Actor(obs_dim, act_dim, hid_dims).to(self.device)
        self.vf = Critic(obs_dim, act_dim, hid_dims).to(self.device)

        # init parameters
        self.apply(init_params)

        # Target networks
        self.pi_targ = deepcopy(self.pi)
        self.vf_targ = deepcopy(self.vf)

        # optimizers
        self.optim_pi = Adam(self.pi.parameters(), lr=lr_pi)
        self.optim_vf = Adam(self.vf.parameters(), lr=lr_vf)

        # learning rate schedule
        self.schedule_pi = CosineAnnealingLR(self.optim_pi,
                                             epochs,
                                             eta_min=lr_decay * lr_pi)
        self.schedule_vf = CosineAnnealingLR(self.optim_vf,
                                             epochs,
                                             eta_min=lr_decay * lr_vf)
        self.train()

    @torch.no_grad()
    def act(self, obs, **kwargs):
        return self.pi(obs.to(self.device)).cpu().numpy()

    def update(self, buffer, rwd_scale=lambda x: x):
        # Sample experiences from the replay buffer
        obs, act, rew, done, obs2 = (x.to(
            self.device) for x in buffer.sample(batch_size=self.batch_size))
        rew = rwd_scale(rew)

        # Compute the target Q estimates
        with torch.no_grad():
            q_targ = self.vf_targ(obs2, self.pi_targ(obs2))
            q_targ = rew + self.gamma * (1 - done) * q_targ

        # Compute Q loss
        q = self.vf(obs, act)
        loss_vf = (q - q_targ).pow(2).mean()

        # Optimize the vf
        self.optim_vf.zero_grad()
        loss_vf.backward()
        clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
        self.optim_vf.step()

        # Compute the pi loss
        loss_pi = self.vf(obs, self.pi(obs)).mean()

        # Optimize the pi
        self.optim_pi.zero_grad()
        loss_pi.backward()
        clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
        self.optim_pi.step()

        # Update the target networks
        self.soft_update(self.pi, self.pi_targ)
        self.soft_update(self.vf, self.vf_targ)

    def soft_update(self, source, target):
        with torch.no_grad():
            for p, p_targ in zip(source.parameters(), target.parameters()):
                p_targ.mul_(1 - self.tau).add_(self.tau * p.data)

    def lr_step(self):
        self.schedule_pi.step()
        self.schedule_vf.step()
