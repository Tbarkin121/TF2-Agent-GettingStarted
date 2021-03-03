from __future__ import absolute_import, division, print_function
import argparse

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from helperFunctions import create_policy_eval_video, compute_avg_return, plot_performance

# import IPython
# import pyvirtualdisplay

#--------------------------------------------------------------
tf.compat.v1.enable_v2_behavior()
tf.version.VERSION
#--------------------------------------------------------------
# def configure_parser():
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
# parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()
#--------------------------------------------------------------
# def configure_hyperparameters():
num_iterations = 500000 # @param {type:"integer"}

# initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 1  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}
#--------------------------------------------------------------
# def configure_environment():
# env_name = 'CartPole-v0'
env_name = 'Pong-ram-v0'
# env = suite_gym.load(env_name)
# env.reset()
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#--------------------------------------------------------------
# def configure_agent():
fc_layer_params = (100,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.compat.v1.train.get_or_create_global_step()
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step,
    epsilon_greedy=0.99,
    gamma=0.99)

agent.initialize()
print('epsilon_greedy set to : {0}'.format(agent._epsilon_greedy))

eval_policy = agent.policy
collect_policy = agent.collect_policy


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

collect_driver.run()

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

print("dataset")
print(dataset)
print("iterator")
print(iterator)
# print(iterator.next())

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
# global_step = 0
#--------------------------------------------------------------


#------------------Creating Checkpointer Object-----------------------
#----------------This will continue from last checkpoint----------------
checkpoint_dir = 'checkpoint_dir'

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()




#--------------------------------------------------------------
# def run_some_training():
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        create_policy_eval_video(eval_env, eval_py_env, agent.policy, "agent-itr-{}".format(step), num_episodes=1, fps=60)
        agent._epsilon_greedy *= 0.95
        print('epsilon_greedy set to : {0}'.format(agent._epsilon_greedy))


train_checkpointer.save(global_step)

if args.mode == 'train':
    print(args.mode)
    # run_some_training()

    plot_performance(num_iterations, eval_interval, returns)
    create_policy_eval_video(eval_env, eval_py_env, agent.policy, "trained-agent", num_episodes=5, fps=60)
    #-----------------Train A Little--------------------
    # train_n_iteration(num_iterations)
    # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    # returns = [avg_return]

    #-----------------Save Checkpoint--------------------
    # train_checkpointer.save(global_step)

elif args.mode == 'test':
    print(args.mode)
    # tmp_action_spec = tensor_spec.BoundedTensorSpec((),
    #                                         tf.int64,
    #                                         minimum=4,
    #                                         maximum=5)
    # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), tmp_action_spec)
    compute_avg_return(eval_env, agent.policy, 10)
    create_policy_eval_video(eval_env, eval_py_env, agent.policy, "trained-agent")
