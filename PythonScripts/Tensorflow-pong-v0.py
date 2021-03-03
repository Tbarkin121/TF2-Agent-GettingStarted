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
from tf_agents.environments import suite_gym, suite_atari


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

# -------------------Environment-------------------------------
# env_name = 'BreakoutDeterministic-v4'
env_name = 'Pong-v0'


DEFAULT_ATARI_GYM_WRAPPERS = (atari_preprocessing.AtariPreprocessing,)
DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING = DEFAULT_ATARI_GYM_WRAPPERS + (atari_wrappers.FrameStack4,)
env = suite_atari.load(env_name, gym_env_wrappers = DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)

env.reset()
print("!!!!!!!!!!!!!!!!!!!")
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Reward Spec:')
print(env.time_step_spec().reward)
print('Action Spec:')
print(env.action_spec())
print("!!!!!!!!!!!!!!!!!!!")

# train_py_env = suite_gym.load(env_name)
# eval_py_env = suite_gym.load(env_name)
# train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

