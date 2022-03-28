#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:12:19 2022

@author: qiang
"""
#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeTrajectoryEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys
import torch
from rrc_example_package.her.rl_modules.models import actor
import numpy as np
from rrc_example_package import cube_trajectory_env
import time
from rrc_example_package.her.rl_modules.replay_buffer_experience import replay_buffer
from rrc_example_package.her.her_modules.her import her_sampler
from rrc_example_package.her.arguments import get_args

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    clip_obs = 200
    clip_range = 5
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def main():    
    # goal_json = sys.argv[1]
    # goal = json.loads(goal_json)
    # print(goal)
    # some arguments
    load_actor=True
    max_steps=90
    steps_per_goal=100
    env_type='real'
    visualization = False
    args = get_args()
    buffer_size = int(163800)
    save_freq = 200
    # Arguments for 'PINCHING' policy
    
    #############
    step_size=50
    difficulty=3
    obs_type='default'
    model_path = '/userhome/model_no_dr.pt'
    #############
    
    # Make sim environment
    sim_env = cube_trajectory_env.SimtoRealEnv(visualization=visualization, max_steps=max_steps, \
                                               xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type='sim', obs_type=obs_type, env_wrapped=False,\
                                                       # increase_fps=False, goal_trajectory=goal)
                                                       increase_fps=False)
    # get the env params
    observation = sim_env.reset(difficulty=difficulty, init_state='normal')
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': sim_env.action_space.shape[0], 
                  'action_max': sim_env.action_space.high[0],
                  }
    env_params['max_timesteps'] = sim_env._max_episode_steps
    # delete sim env
    del sim_env
    
    if load_actor:
        # load the model param
        print('Loading in model from: {}'.format(model_path))
        o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        actor_network = actor(env_params)
        actor_network.load_state_dict(model)
        actor_network.eval()
    
    # Make real environment
    env = cube_trajectory_env.SimtoRealEnv(visualization=visualization, max_steps=max_steps, \
                                               xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type=env_type, obs_type=obs_type, env_wrapped=False,\
                                                       # increase_fps=False, goal_trajectory=goal)
                                                       increase_fps=False)
        
    her_module = her_sampler(args.replay_strategy, args.replay_k, env.compute_reward, env.steps_per_goal, args.xy_only, args.trajectory_aware)
    buffer = replay_buffer(env_params, buffer_size, her_module.sample_her_transitions)
    
    print('Beginning...')
    done = False
    # items for disrupting policy from stuck states
    xy_fails = 0
    rand_actions = 5
    fails_threshold = 50
    T = env_params['max_timesteps']
    n_epochs = 20
    
    for epoch in range(n_epochs):

        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
        observation = env.reset(difficulty=difficulty, init_state='normal')
        
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        
        t0 = time.time()
        for _ in range(env_params['max_timesteps']):
            if difficulty == 1:
                # Move goal to the floor
                g[2] = 0.0325
            if xy_fails < fails_threshold:
                inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
                with torch.no_grad():
                    pi = actor_network(inputs)
                action = pi.detach().numpy().squeeze()
            else:
                action = env.action_space.sample()
                print('Stuck - taking random action!!!')
                if xy_fails > fails_threshold + rand_actions:
                    xy_fails = 0
            # feed the actions into the environment
            
            observation_new, reward, done, info = env.step(action)
            
            if info['xy_fail']:
                xy_fails += 1
            else:
                xy_fails = 0
                
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            g_new = observation_new['desired_goal']
            # append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            # re-assign the observation
            obs = obs_new
            ag = ag_new
            g = g_new
        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(g.copy())

        # convert them into arrays
        ep_obs = np.array(ep_obs)
        ep_ag = np.array(ep_ag)
        ep_g = np.array(ep_g)
        ep_actions = np.array(ep_actions)
        # store the episodes
        buffer.store_episode([ep_obs, ep_ag, ep_g, ep_actions])
        
        tf = time.time()
        
        print('-'*30)
        print(f'Epoch {epoch}/{n_epochs}')
        print('Time taken for epoch: {:.2f} seconds'.format(tf-t0))
        print('\nRRC reward: {}'.format(info['rrc_reward']))
        
        if epoch % save_freq == 0:
            print('saving...')
            torch.save(buffer.buffers, '/output/experience.pth')

if __name__ == "__main__":
    main()
