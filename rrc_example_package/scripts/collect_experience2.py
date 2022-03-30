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
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.her_modules.her import her_sampler
from rrc_example_package.her.arguments import get_args
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks.move_cube_on_trajectory as task
from rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks import move_cube
import pybullet as p

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
    #############
    step_size=50
    difficulty=3
    obs_type='default'
    model_path = '/userhome/model_dr.pt'
    #############
    task.GOAL_DURATION = 5 * 1000
    task.FIRST_GOAL_DURATION = 5 * 1000
    task.EPISODE_LENGTH = 2 * 60 * 1000
    goal_length = 90
    
    goal = task.sample_goal()
    print(goal)
    print('len(goal)',len(goal))
    
    load_actor=True
    max_steps=int(task.EPISODE_LENGTH / step_size)
    steps_per_goal=100
    env_type='real'
    visualization = 0
    args = get_args()
    buffer_size = len(goal) * goal_length
    # Arguments for 'PINCHING' policy
    assert goal_length <= (task.GOAL_DURATION / step_size)
    #############
    
    # Make sim environment
    sim_env = cube_trajectory_env.SimtoRealEnv(visualization=0, max_steps=max_steps, \
                                               xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                   env_type='sim', obs_type=obs_type, env_wrapped=False,\
                                                        increase_fps=False, goal_trajectory=goal)

    # get the env params
    observation = sim_env.reset(difficulty=difficulty, init_state='normal')
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': sim_env.action_space.shape[0], 
                  'action_max': sim_env.action_space.high[0],
                  }
    env_params['max_timesteps'] = goal_length

    del sim_env
    
    if load_actor:
        # load the model param
        print('Loading in model from: {}'.format(model_path))
        o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        actor_network = actor(env_params)
        actor_network.load_state_dict(model)
        actor_network.eval()
    
    # # Make real environment
    env = cube_trajectory_env.SimtoRealEnv(visualization=visualization, max_steps=max_steps, \
                                                xy_only=False, steps_per_goal=steps_per_goal, step_size=step_size,\
                                                    env_type=env_type, obs_type=obs_type, env_wrapped=False,\
                                                        increase_fps=False, goal_trajectory=goal)
        
    her_module = her_sampler(args.replay_strategy, args.replay_k, env.compute_reward, env.steps_per_goal, args.xy_only, args.trajectory_aware)
    buffer = replay_buffer(env_params, buffer_size, her_module.sample_her_transitions)
    
    print('Beginning...')
    done = False
    # items for disrupting policy from stuck states
    xy_fails = 0
    rand_actions = 5
    fails_threshold = 50
    
    observation = env.reset(difficulty=difficulty, init_state='normal')
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=180, cameraPitch=-30.0, cameraTargetPosition=[0, 0.0, 0.0])
    obs = observation['observation']
    ag = observation['achieved_goal']
    g = observation['desired_goal']

    for t in range(len(goal)):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
        t0 = time.time()
        count = 0
        while True:
            count += 1
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
            
            if count % goal_length == 0:
                break

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(g.copy())
        
        mb_obs.append(ep_obs)
        mb_ag.append(ep_ag)
        mb_g.append(ep_g)
        mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        
        tf = time.time()
        
        print('-'*30)
        print(buffer.current_size)
        print(f'Epoch {t}/{len(goal)}')
        print('Time taken for epoch: {:.2f} seconds'.format(tf-t0))
        print('\nRRC reward: {}'.format(info['rrc_reward']))
        
    torch.save(buffer.buffers, '/output/experience.pth')

if __name__ == "__main__":
    main()
