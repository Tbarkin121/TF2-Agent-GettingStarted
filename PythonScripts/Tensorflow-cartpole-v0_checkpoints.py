from __future__ import absolute_import, division, print_function

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

#--------------------------------------------------------------
tf.compat.v1.enable_v2_behavior()
tf.version.VERSION

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

# --------------------------------------------------------------
env_name = 'CartPole-v1'
env = suite_gym.load(env_name)
env.reset()

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#--------------------------------------------------------------
num_iterations = 5000 # @param {type:"integer"}

collect_steps_per_iteration = 100  # @param {type:"integer"}
replay_buffer_capacity = 100000

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 5  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

fc_layer_params = (100,)
#--------------------------------------------------------------
# print('Observation Spec:')
# print(env.time_step_spec().observation)
# print('Reward Spec:')
# print(env.time_step_spec().reward)
# print('Action Spec:')
# print(env.action_spec())
# time_step = env.reset()

# print('Time step:')
# print(time_step)
# action = np.array(1, dtype=np.int32)
# next_time_step = env.step(action)
# print('Next time step:')
# print(next_time_step)
#--------------------------------------------------------------
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
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

#@title
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

def train_one_iteration(n):
    for _ in range(n):
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

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_driver.run()

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  iteration = agent.train_step_counter.numpy()
  print ('iteration: {0} loss: {1}'.format(iteration, train_loss.loss))

#--------------------------------------------------------------
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
create_policy_eval_gif(agent.policy, "untrained-agent")

#--------------------------------------------------------------
checkpoint_dir = 'checkpoint_dir'

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
policy_dir = 'policy_dir'
tf_policy_saver = policy_saver.PolicySaver(agent.policy)


train_one_iteration(num_iterations)

train_checkpointer.save(global_step)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

tf_policy_saver.save(policy_dir)

saved_policy = tf.compat.v2.saved_model.load(policy_dir)
create_policy_eval_gif(agent.policy, "trained-agent")
#--------------------------------------------------------------
def create_zip_file(dirname, base_filename):
    shutil.make_archive(base_filename, 'zip', dirname)

def upload_and_unzip_file_to(dirname, base_filename):
    # shutil.rmtree(dirname)
    print(base_filename)
    zip_files = zipfile.ZipFile(base_filename, 'r')
    # zip_files = zipfile.ZipFile('archive\checkpoint_arch.zip', 'r')
    
    zip_files.extractall(dirname)
    zip_files.close()
#--------------------------------------------------------------
train_checkpointer.save(global_step)
archive_dir = 'archive'
create_zip_file(checkpoint_dir, os.path.join(archive_dir, 'checkpoint_arch'))
create_zip_file(policy_dir, os.path.join(archive_dir, 'policy_arch'))
upload_and_unzip_file_to(archive_dir, os.path.join(archive_dir, 'checkpoint_arch.zip'))



