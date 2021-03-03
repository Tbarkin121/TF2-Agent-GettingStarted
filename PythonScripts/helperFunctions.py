import imageio
import matplotlib
import matplotlib.pyplot as plt

import io
import os
import shutil
import tempfile
import zipfile

#--------------------------Generate Output Videos--------------------------
def create_policy_eval_video(env, py_env, policy, filename, num_episodes=1, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = env.reset()
            video.append_data(py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                video.append_data(py_env.render())
    video.close()

def create_policy_eval_gif(env, py_env, policy, filename, num_episodes=1, _fps=30):
    filename = filename + ".gif"
    frames = []

    for _ in range(num_episodes):
        time_step = env.reset()
        frames.append(py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            frames.append(py_env.render())
    
    imageio.mimsave(filename, frames, format='gif', fps=_fps)

#--------------------------Performance--------------------------
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


def plot_performance(num_iterations, eval_interval, returns):
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.title('Average Rewards vs Iteration')
    # plt.grid(True)
    # plt.savefig("test.png")
    plt.show()

#--------------------------Data Storage--------------------------
def create_zip_file(dirname, base_filename):
    shutil.make_archive(base_filename, 'zip', dirname)

def upload_and_unzip_file_to(dirname, base_filename):
    # shutil.rmtree(dirname)
    print(base_filename)
    zip_files = zipfile.ZipFile(base_filename, 'r')
    # zip_files = zipfile.ZipFile('archive\checkpoint_arch.zip', 'r')
    zip_files.extractall(dirname)
    zip_files.close()
