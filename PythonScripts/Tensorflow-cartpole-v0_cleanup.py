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
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import imageio
# import IPython
# import pyvirtualdisplay

import io
import os
import shutil
import tempfile
import zipfile

tf.compat.v1.enable_v2_behavior()
tf.version.VERSION

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

# ---------------------Parser----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
# parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# -------------------Environment-------------------------------
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env.reset()

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#-------------------Hyper Parameters---------------------------
num_iterations = 50000 # @param {type:"integer"}

collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

video_interval = 2500

fc_layer_params = (100,)

#-------------------Initilize Agent----------------------------
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.compat.v1.train.get_or_create_global_step()
# train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

agent.initialize()

#-------------------Initilize Policy----------------------------

#--------------------------------------------------------------
#@title
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

# Initial data collection
collect_driver.run()

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

#@title
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

def train_n_iteration(num_iterations):
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        iteration = agent.train_step_counter.numpy()
        if iteration % log_interval == 0:
            print ('iteration: {0} loss: {1}'.format(iteration, train_loss))
        if iteration % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print ('iteration: {0} Average Return: {1}'.format(iteration, avg_return))
            returns.append(avg_return)
        if iteration % video_interval ==0:
            create_policy_eval_video(agent.policy, "trained-agent_itr_{}".format(iteration))


#--------------------------Generate Output Videos--------------------------*Stuff like this should be moved to it's own helper file
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    video.close()

def create_policy_eval_gif(policy, filename, num_episodes=5, _fps=30):
    filename = filename + ".gif"
    frames = []

    for _ in range(num_episodes):
        time_step = eval_env.reset()
        frames.append(eval_py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            frames.append(eval_py_env.render())
    
    imageio.mimsave(filename, frames, format='gif', fps=_fps)

# create_policy_eval_video(agent.policy, "trained-agent")
# create_policy_eval_gif(agent.policy, "untrained-agent")

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# compute_avg_return(eval_env, random_policy, num_eval_episodes)  

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

policy_dir = 'policy_dir'
tf_policy_saver = policy_saver.PolicySaver(agent.policy)


#--------------------------Manage Ziped Checkpoints--------------------------*Stuff like this should be moved to it's own helper file
def create_zip_file(dirname, base_filename):
    shutil.make_archive(base_filename, 'zip', dirname)

def upload_and_unzip_file_to(dirname, base_filename):
    # shutil.rmtree(dirname)
    print(base_filename)
    zip_files = zipfile.ZipFile(base_filename, 'r')
    # zip_files = zipfile.ZipFile('archive\checkpoint_arch.zip', 'r')
    zip_files.extractall(dirname)
    zip_files.close()

# archive_dir = 'archive'
# create_zip_file(checkpoint_dir, os.path.join(archive_dir, 'checkpoint_arch'))
# create_zip_file(policy_dir, os.path.join(archive_dir, 'policy_arch'))
# upload_and_unzip_file_to(archive_dir, os.path.join(archive_dir, 'checkpoint_arch.zip'))
#--------------------------------------------------------------

if args.mode == 'train':
    #-----------------Train A Little--------------------
    train_n_iteration(num_iterations)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    #-----------------Save Checkpoint--------------------
    train_checkpointer.save(global_step)

elif args.mode == 'test':
    create_policy_eval_video(agent.policy, "trained-agent")

