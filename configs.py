import argparse
import os
import random

import numpy as np
import torch

from utils.loader import get_victoria_data


def get_env_kwargs(demo=False, **kwargs):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic information
    parser.add_argument('--t_start', default=6, type=int)
    parser.add_argument('--t_end', default=10, type=int)
    parser.add_argument('--capacity', default=500, type=int)
    parser.add_argument('--alpha',
                        default=kwargs.get('alpha') or 0.5,
                        type=float)
    parser.add_argument('--turn', default=218, type=int)
    parser.add_argument('--stock_size', default=24, type=int)
    parser.add_argument('--num_station', default=8, type=int)
    parser.add_argument('--stochastic',
                        default=kwargs.get('stochastic') or False,
                        action='store_true')
    parser.add_argument('--random_factor',
                        default=kwargs.get('random_factor') or 0.25,
                        type=float)

    # Decision Variable related (unit -- second)
    parser.add_argument('--headway_min', default=100, type=int)
    parser.add_argument('--headway_opt', default=9, type=int)
    parser.add_argument('--headway_int', default=50, type=int)
    parser.add_argument('--headway_safe', default=60, type=int)
    parser.add_argument('--dwell_min', default=25, type=int)
    parser.add_argument('--dwell_int', default=5, type=int)
    parser.add_argument('--dwell_opt', default=6, type=int)
    parser.add_argument('--run_int', default=10, type=int)
    parser.add_argument('--run_opt', default=4, type=int)

    args, unknown = parser.parse_known_args()
    if not args.stochastic:
        args.random_factor = 0.
    if demo:
        args.t_start, args.t_end = 7.5, 9.5
        args.capacity, args.stock_size, args.num_station = 330, 14, 4

    kwargs = vars(args)

    data = get_victoria_data()
    data['demand'] *= 0.8
    data['demand'] = np.moveaxis(data['demand'], 2, 0)

    kwargs['data'] = data
    return kwargs


def get_device(gpu=0):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        return 'cuda'
    else:
        return 'cpu'


def get_ddpg_kwargs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hid_dims', default=[400, 300], type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--tau', default=5e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr_pi', default=5e-4, type=float)
    parser.add_argument('--lr_vf', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    # main variables
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--buffer_size', default=int(5e4), type=int)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--reward_scale', default=5000, type=int)

    # noise decay
    parser.add_argument('--eta_init', default=0.15, type=float)
    parser.add_argument('--eta_min', default=0.075, type=float)

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['device'] = get_device()
    return kwargs


def get_configs(algo='ddpg',
                stochastic=False,
                demo=False,
                seed=123,
                alpha=0.5,
                random_factor=0.25):
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument("--algo", default=algo.upper(), type=str)
    parser.add_argument('--seed', default=seed, type=int)

    args, unknown = parser.parse_known_args()
    env_kwargs = get_env_kwargs(demo=demo,
                                alpha=alpha,
                                stochastic=stochastic,
                                random_factor=random_factor)
    algo_kwargs = get_ddpg_kwargs()

    if args.seed:
        env_kwargs['seed'] = args.seed + 1

    env_kwargs['alpha'] = alpha

    # get exp_name
    from env import TubeEnv
    exp_name = ("Demo" if demo else "Main") + (
        "Sto" if stochastic else "Det") + env_kwargs['data']['name'][:3]
    exp_name += f"{TubeEnv.name}_v{TubeEnv.version}" + f"_A{env_kwargs['alpha'] * 100:.0f}_"
    exp_name += f"R{env_kwargs['random_factor'] * 100:.0f}"

    return args, algo_kwargs, env_kwargs


def set_global_seeds(seed):
    if seed is None:
        return
    elif np.isscalar(seed):
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        raise ValueError(f"Invalid seed: {seed} (type {type(seed)})")
