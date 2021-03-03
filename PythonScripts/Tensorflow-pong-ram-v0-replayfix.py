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

from tf_agents.environments import suite_atari
from tf_agents.environments import atari_preprocessing
from tf_agents.environments import atari_wrappers

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
from tf_agents.train.utils import spec_utils

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

import imageio

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

#-------------------Hyper Parameters---------------------------
num_iterations = 50000 # @param {type:"integer"}

collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000
initial_collect_steps = 200

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 1  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

video_interval = 10000

fc_layer_params = (256,256,)

# -------------------Environment-------------------------------
# env_name = 'Pong-ram-v0'
# env_name = 'Breakout-ram-v0'
# env_name = 'BreakoutNoFrameskip-v4'
# env_name = 'CartPole-v0'
env_name = suite_atari.game('Breakout', 'ram')
# print(env_name)
env = suite_atari.load(env_name,
    max_episode_steps= 1000,
    gym_env_wrappers=[atari_wrappers.FireOnReset])
# print(env)
# env.reset()
# print('Observation Spec:')
# print(env.time_step_spec().observation)
# print('Action Spec:')
# print(env.action_spec())

# # train_py_env = suite_gym.load(env_name)
# # eval_py_env = suite_gym.load(env_name)
train_py_env = suite_atari.load(env_name,
    max_episode_steps= 10000,
    gym_env_wrappers=[atari_wrappers.FireOnReset])
eval_py_env = suite_atari.load(env_name,
    max_episode_steps= 10000,
    gym_env_wrappers=[atari_wrappers.FireOnReset])
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(train_py_env))
print("!!!!!!!!!!!!!!!!!!!")
print('Observation Spec:')
print(observation_spec)
print('Reward Spec:')
print(time_step_spec.reward)
print('Action Spec:')
print(action_spec)
print("Time Step Spec")
print(time_step_spec)
# print("!!!!!!!!!!!!!!!!!!!")



#-------------------Initilize Agent----------------------------
# class AtariQNetwork(q_network.QNetwork):
#   """QNetwork subclass that ..."""

#   def call(self,
#            observation,
#            step_type=None,
#            network_state=(),
#            training=False):
#     state = tf.cast(observation, tf.float32)
#     # We divide the grayscale pixel values by 255 here rather than storing
#     # normalized values beause uint8s are 4x cheaper to store than float32s.
#     state = state / 255
#     return super(AtariQNetwork, self).call(
#         state, step_type=step_type, network_state=network_state,
#         training=training)

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
    # td_errors_loss_fn=common.element_wise_squared_loss,
    # td_errors_loss_fn=tf.math.squared_difference,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=global_step,
    epsilon_greedy=0.01,
    gamma=0.99,
    n_step_update = 1.0,
    target_update_tau = 1.0,
    target_update_period= 2000,
    reward_scale_factor = 1.0,
    gradient_clipping = None,
    debug_summaries = False,
    summarize_grads_and_vars = False) 

agent.initialize()

#-------------------Initilize Policy----------------------------
eval_policy = agent.policy
collect_policy = agent.collect_policy
#--------------------------------------------------------------
#@title
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)
print("agent.collect_data_spec")
print(agent.collect_data_spec)
print("agent.collect_data_spec._fields")
print(agent.collect_data_spec._fields)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=train_env.batch_size,
    num_steps=2).prefetch(3)



# collect_driver = dynamic_step_driver.DynamicStepDriver(
#     train_env,
#     agent.collect_policy,
#     observers=[replay_buffer.add_batch],
#     num_steps=collect_steps_per_iteration)

# print(agent.collect_data_spec._fields)
# # Initial data collection
# collect_driver.run()

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)
iterator = iter(dataset)




print("!!!!!!!!!!!!!!!!!!!")
print("dataset")
print(dataset)
print("iterator")
print(iterator)
# print(iterator.next())
print("!!!!!!!!!!!!!!!!!!!")

#@title
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)


def train_n_iteration(num_iterations):
    agent.train_step_counter.assign(0) #Idk if i need this or not
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        # collect_driver.run()
        collect_step(train_env, agent.collect_policy, replay_buffer)

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
    
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.title('Average Rewards vs Iteration')
    # plt.grid(True)
    # plt.savefig("test.png")
    plt.show()
    create_policy_eval_video(agent.policy, "trained-agent")

#--------------------------Generate Output Videos--------------------------*Stuff like this should be moved to it's own helper file
def create_policy_eval_video(policy, filename, num_episodes=1, fps=30):
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

def create_policy_eval_gif(policy, filename, num_episodes=1, _fps=30):
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
    #   print("time_step.reward")
    #   print(time_step.reward)
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
    # tmp_action_spec = tensor_spec.BoundedTensorSpec((),
    #                                         tf.int64,
    #                                         minimum=4,
    #                                         maximum=5)
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    compute_avg_return(eval_env, agent.policy, 1)
    create_policy_eval_video(random_policy, "trained-agent")