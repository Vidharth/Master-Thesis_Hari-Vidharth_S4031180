import torch
from oracle.rl_modules.models import actor
from oracle.arguments import get_args
import gym
import numpy as np


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def demonstrations(env, observation, goal):
    args = get_args()

    model_path = 'oracle/' + args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)

    env_params = {'obs': observation.shape[0], 
                  'goal': goal.shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }

    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    obs = observation
    g = goal

    inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
    
    with torch.no_grad():
        pi = actor_network(inputs)
    
    action = pi.detach().numpy().squeeze()

    return action, o_mean, o_std, g_mean, g_std
