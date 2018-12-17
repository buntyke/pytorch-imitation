import os
import gym
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from train import get_instance
import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
import data_loader.data_loaders as module_data


def main(config, resume, env):
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # initialize gym env
    num_rollouts = 5
    env = gym.make(env)
    max_steps = env.spec.timestep_limit

    # loop for num_rollouts
    returns = []
    for i in range(num_rollouts):
        print('Iter', i)

        steps = 0
        totalr = 0.0
        done = False
        obs = env.reset()

        while not done:
            obs = torch.from_numpy(obs[None, :].astype(np.float32))
            if torch.cuda.is_available():
                obs = obs.cuda()
            action = model(Variable(obs))

            action = action.data
            if torch.cuda.is_available():
                action = action.cpu()
            obs, r, done, _ = env.step(action.numpy()[0])

            steps += 1
            totalr += r

            env.render()

            if steps >= max_steps:
                break
        returns.append(totalr)

    # output statistics
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-m', '--model', required=True, type=str,
                           help='path to latest checkpoint')
    parser.add_argument('-e', '--env', required=True, type=str,
                           help='openai gym env')
    args = parser.parse_args()

    config = torch.load(args.model)['config']
    main(config, args.model, args.env)
