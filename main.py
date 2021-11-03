from configs import get_configs, set_global_seeds
from ddpg.learn import learn

from env import TubeEnv


def main():
    settings = []

    for seed in range(1):
        stochastic = False
        settings.append((stochastic, 0.5, 0.5, seed))

    for setting in settings:
        stochastic, alpha, random_factor, seed = setting

        args, akwargs, ekwargs = get_configs(algo='ddpg',
                                             stochastic=stochastic,
                                             random_factor=random_factor,
                                             alpha=alpha,
                                             seed=seed)
        set_global_seeds(args.seed)

        def env_fn():
            return TubeEnv(**ekwargs)

        learn(env_fn, seed=args.seed, **akwargs)


if __name__ == '__main__':
    main()
