import copy
import gym
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from marlgrid.envs.doorkey import DoorKeyEnv
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import pickle

player_interface_config = {
    "view_size": 6,
    "view_offset": 1,
    "view_tile_size": 10,
    "observation_style": "rich",
    "see_through_walls": False,
    "color": "prestige",
    "observe_position": True,
    "observe_orientation": True,
    "type": "teacher"
}
student_config = copy.deepcopy(player_interface_config)
student_config["color"] = "blue"
student_config["type"] = "student"


def _save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """Takes a list of frames (each frame can be generated with the `env.render()` function from OpenAI gym)
    and converts it into GIF, and saves it to the specified location.
    Code adapted from this gist: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    """
    imageio.mimwrite(os.path.join(path, filename), frames, fps=60)


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {episode_num + 1}', fill=text_color)

    return im


def reset(env):
    obs = env.reset()
    # env.agents[0].set_position([1, 1])
    # env.agents[1].set_position([3, 2])
    # next_obs, reward, done, info = env.step([6, 6])
    return obs


# Add the player/agent config to the environment config
agents = [player_interface_config, student_config]


def sample_env():
    i = 2
    while True:
        env = DoorKeyEnv(
            grid_size=6,
            max_steps=250,
            respawn=True,
            ghost_mode=True,
            reward_decay=False,
            agents=agents,
            seed=i
        )
        i += 1
        teacher_obs = env.agents[0].pos
        student_obs = env.agents[1].pos
        if 0 < teacher_obs[0] < 3 and 0 < teacher_obs[1] < 5 and \
                0 < student_obs[0] < 3 and 0 < student_obs[1] < 5:
            break
    return env


env = sample_env()
teacher_obs = env.agents[0].pos
student_obs = env.agents[1].pos
teacher_dir = env.agents[0].dir
student_dir = env.agents[1].dir
print('selected', teacher_obs, student_obs, teacher_dir, student_dir)


############################################################

# Hyper parameters
alpha = 0.01
gamma = 0.99
epsilon = 0.1

num_episodes = 2000

# For plotting metrics
all_steps = np.zeros(num_episodes)
all_rewards = np.zeros(num_episodes)

#
# frames = []

# Train the teacher

q_table_teacher = np.zeros([36 * 4, 7])

for i in range(num_episodes):
    env = sample_env()
    teacher = env.agents[0].pos
    student = env.agents[1].pos
    teacher_dir = env.agents[0].dir
    student_dir = env.agents[1].dir
    state = teacher[1] * 6 * 4 + teacher[0] * 4 + teacher_dir

    steps_per_episode, reward_per_episode = 0, 0
    done = False
    # print("episode", i, obs[0]["position"], obs[1]["position"])

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space[0].sample()  # Explore action space

        else:
            action = np.random.choice(
                np.where(q_table_teacher[state] == np.max(q_table_teacher[state]))[0])  # Exploit learned values

        next_obs, reward, done, info = env.step([action, 6])
        # teacher = next_obs[0]
        teacher = env.agents[0].pos
        teacher_dir = env.agents[0].dir
        next_state = teacher[1] * 6 * 4 + teacher[0] * 4 + teacher_dir

        # frame = env.render(mode='rgb_array')
        # frames.append(_label_with_episode_number(frame, episode_num=i))

        old_value = q_table_teacher[state, action]
        next_max = np.max(q_table_teacher[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward[0] + gamma * next_max)
        q_table_teacher[state, action] = new_value
        if reward[0] != 0:
            print('episode:', i)
            print('reward:', reward)
            print('new value: ', new_value)
            print('done', done)

        state = next_state
        steps_per_episode += 1
        reward_per_episode += reward[0]
        if reward[0] > 0:
            break

    all_steps[i] = steps_per_episode
    all_rewards[i] = reward_per_episode

print("Training of the teacher finished.\n")



np.save(f'/home/yseult/PycharmProjects/BMMproject/marlgrid-master/q_table_teacher_withkey_{num_episodes}episodes.npy', q_table_teacher)

# #q_table_teacher = np.load('/home/yseult/PycharmProjects/BMMproject/marlgrid-master/q_table_teacher.npy')

# env.close()
# _save_frames_as_gif(frames, path='./videos/', filename='teacher_training.gif')
#
# # print("Video of the teacher's training saved.\n")
#
# #############################################################
#
#############################################################

# Plot the cumulative reward (the sum of all rewards received so far) as a function of the number of episodes

smoothing_window = 10

figname = f'teacher_training_withkey_results_{num_episodes}_episodes.png'

plt.figure(figsize=(10, 5))
rewards_smoothed = pd.Series(all_rewards).rolling(smoothing_window).mean()
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time for the Teacher (Smoothed over window size {})".format(smoothing_window))
plt.savefig(figname)
plt.show(block=False)
plt.close()

# # ###########################################################
# frames = []
"""Evaluate teacher's performance after Q-learning"""

total_epochs, total_reward = 0, 0
episodes = 100

for _ in range(episodes):
    env = sample_env()
    teacher = env.agents[0].pos
    student = env.agents[1].pos
    teacher_dir = env.agents[0].dir
    student_dir = env.agents[1].dir
    state = teacher[1] * 6 * 4 + teacher[0] * 4 + teacher_dir

    epochs, reward = 0, 0

    done = False

    while not done:
        action = np.random.choice(
            np.where(q_table_teacher[state] == np.max(q_table_teacher[state]))[0])  # Exploit learned values

        # frame = env.render(mode='rgb_array')
        # frames.append(_label_with_episode_number(frame, episode_num=i))

        next_obs, reward, done, info = env.step([action, 6])
        teacher = env.agents[0].pos
        teacher_dir = env.agents[0].dir
        state = teacher[1] * 6 * 4 + teacher[0] * 4 + teacher_dir
        epochs += 1

        if reward[0] > 0:
            break

    total_epochs += epochs
    total_reward += reward[0]

print(f"Results after {episodes} episodes:")
print(f"Average time steps per episode: {total_epochs / episodes}")
print(f"Average reward per episode: {total_reward / episodes}")

# # env.close()
# # _save_frames_as_gif(frames, path='./videos/', filename='teacher_test_new.gif')


###################################################################
# q_table_teacher = np.load('/home/yseult/PycharmProjects/BMMproject/marlgrid-master/q_table_teacher2000episodes.npy')
# print(q_table_teacher[0])
#
# ## STUDENT
#
# num_episodes = 2000
# all_epochs = np.zeros(num_episodes)
#
# epsilon = 0.3
#
# """ Train the student """
#
# q_table_student = np.zeros([36 * 4, 7])
#
# for i in range(num_episodes):
#     env = sample_env()
#     teacher = env.agents[0].pos
#     student = env.agents[1].pos
#     teacher_dir = env.agents[0].dir
#     student_dir = env.agents[1].dir
#     state = student[1] * 6 * 4 + student[0] * 4 + student_dir
#
#     steps_per_episode, reward_per_episode = 0, 0
#     done = False
#
#     while not done:
#         if random.uniform(0, 1) < epsilon:
#             action_student = env.action_space[1].sample()  # Explore action space
#         else:
#             action_student = np.random.choice(
#                 np.where(q_table_student[state] == np.max(q_table_student[state]))[0])  # Exploit learned values
#
#         action_teacher = np.random.choice(
#             np.where(q_table_teacher[state] == np.max(q_table_teacher[state]))[0])  # Exploit learned values
#
#         # print('s', action_student)
#         # print('t', action_teacher)
#
#         if action_student != action_teacher:
#             reward = -1
#         else:
#             reward = 1
#
#         # if action_student != action_teacher:
#         #    proba = random.random()
#         #    if proba > 0.3:
#         #        reward = -1
#         #    else:
#         #        reward = 1
#         # else:
#         #    reward = 1
#
#         next_obs, env_reward, done, info = env.step([6, action_student])
#         student = env.agents[1].pos
#         student_dir = env.agents[1].dir
#         next_state = student[1] * 6 * 4 + student[0] * 4 + student_dir
#
#         old_value = q_table_student[state, action_student]
#         next_max = np.max(q_table_student[next_state])
#
#         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#         q_table_student[state, action_student] = new_value
#         if reward != 0:
#             print('episode:', i)
#             print('reward:', reward)
#             print('new value: ', new_value)
#             print('done', done)
#
#         state = next_state
#         steps_per_episode += 1
#         reward_per_episode += reward
#         if reward > 0:
#             break
#
#     all_steps[i] = steps_per_episode
#     all_rewards[i] = reward_per_episode
#
# print("Training of the student finished.\n")
#
# figname = f'epsilon_doornrewardmoved_smooth100_student_training_results_{num_episodes}_episodes.png'
#
# smoothing_window = 100
#
# plt.figure(figsize=(10, 5))
# rewards_smoothed = pd.Series(all_rewards).rolling(smoothing_window).mean()
# plt.plot(rewards_smoothed)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward (Smoothed)")
# plt.title("Episode Reward over Time for the Student (Smoothed over window size {})".format(smoothing_window))
# plt.savefig(figname)
# plt.show(block=False)
# plt.close()
#
# """Evaluate student's performance after Q-learning"""
#
# total_epochs, total_reward = 0, 0
# episodes = 1
#
# for _ in range(episodes):
#     env = sample_env()
#     teacher = env.agents[0].pos
#     student = env.agents[1].pos
#     teacher_dir = env.agents[0].dir
#     student_dir = env.agents[1].dir
#     state = student[1] * 6 * 4 + student[0] * 4 + student_dir
#
#     epochs, reward = 0, 0
#
#     done = False
#     # print(teacher, student)
#
#     while not done:
#         action_student = np.random.choice(
#             np.where(q_table_student[state] == np.max(q_table_student[state]))[0])  # Exploit learned values
#
#         next_obs, reward, done, info = env.step([6, action_student])
#         student = env.agents[1].pos
#         student_dir = env.agents[1].dir
#         state = student[1] * 6 * 4 + student[0] * 4 + student_dir
#         epochs += 1
#
#         if reward[1] > 0:
#             break
#
#     total_epochs += epochs
#     total_reward += reward[1]
#
# print(f"Results after {episodes} episodes:")
# print(f"Average time steps per episode: {total_epochs / episodes}")
# print(f"Average reward per episode: {total_reward / episodes}")













# Student's turn to learn
#
# """ Train the student """
# num_episodes = 10
# all_epochs = np.zeros(num_episodes)
#
# q_table_student = np.zeros([49, 7])
#
# frames = []
#
# for i in range(num_episodes):
#     env = sample_env()
#     obs = reset(env)
#     student = obs[1]
#     state = student['position'][1] * 7 + student['position'][0]
#
#     epochs, reward, = 0, 0
#     done = False
#
#     while not done:
#         if random.uniform(0, 1) < epsilon:
#             action_student = env.action_space[1].sample()  # Explore action space
#         else:
#             action_student = np.random.choice(
#                 np.where(q_table_student[state] == np.max(q_table_student[state]))[0])  # Exploit learned values
#             # np.argmax is biased towards the first value when several values are equal, we want to avoid that
#
#         action_teacher = np.random.choice(
#             np.where(q_table_teacher[state] == np.max(q_table_teacher[state]))[0])  # Exploit learned values
#
#         # print('s', action_student)
#         # print('t', action_teacher)
#
#         if action_student != action_teacher:
#             reward = -1
#         else:
#             reward = 1
#
#         next_obs, env_reward, done, info = env.step([6, action_student])
#         student = next_obs[1]
#         next_state = student['position'][1] * 7 + student['position'][0]
#
#         # frame = env.render(mode='rgb_array')
#         # frames.append(_label_with_episode_number(frame, episode_num=i))
#
#         old_value = q_table_student[state, action_student]
#         next_max = np.max(q_table_student[next_state])
#
#         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#         q_table_student[state, action_student] = new_value
#
#         if reward != 0:
#            print('reward:', reward)
#            print('old value: ', old_value)
#            print('new value: ', new_value)
#
#         state = next_state
#         epochs += 1
#
#         all_epochs[i] += epochs
#         all_rewards[i] += reward
#
#     if i % 10 == 0:
#         print(f"Episode: {i}")
#
# # np.save('/home/yseult/PycharmProjects/BMMproject/marlgrid-master/q_table_student.npy', q_table_student)
#
#
# print("Training of the student finished.\n")

# env.close()
# _save_frames_as_gif(frames, path='./videos/', filename='student_training.gif')
#
# print("Video of the student's training saved.\n")

#####################################################################

# # Plot
# figname = f'student_training_results_{num_episodes}_episodes.png'
#
# smoothing_window = 10
#
# plt.figure(figsize=(10, 5))
# rewards_smoothed = pd.Series(all_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
# plt.plot(rewards_smoothed)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward (Smoothed)")
# plt.title("Episode Reward over Time for the Student (Smoothed over window size {})".format(smoothing_window))
# plt.savefig(figname)
# plt.show(block=False)
# plt.close()

#############################################################################

# Evaluation of the student training

# total_epochs, total_reward = 0, 0
# episodes = 100
#
# for _ in range(episodes):
#     env = sample_env()
#     obs = reset(env)
#     student = obs[1]
#     state = student['position'][1] * 7 + student['position'][0]
#
#     epochs, reward = 0, 0
#
#     done = False
#
#     while not done:
#         action_student = np.random.choice(
#             np.where(q_table_student[state] == np.max(q_table_student[state]))[0])  # Exploit learned values
#
#         # frame = env.render(mode='rgb_array')
#         # frames.append(_label_with_episode_number(frame, episode_num=i))
#
#         obs, reward, done, info = env.step([6, action_student])
#
#         epochs += 1
#
#     total_epochs += epochs
#
# print(f"Results after {episodes} episodes:")
# print(f"Average time steps per episode: {total_epochs / episodes}")

# env.close()
# _save_frames_as_gif(frames, path='./videos/', filename='student_test.gif')
#
# print("Video of the student's test saved.\n")

########################################################################
#
# # Train a second student, which is more exploratory: epsilon is higher and there's flexibility when the student
# # doesn't follow the exact same action as the teacher
#
# """ Train the student """
# num_episodes = 10000
# all_epochs = np.zeros(num_episodes)
# epsilon_explo = 0.3
#
# q_table_student_explo = np.zeros([49, 7])
#
# frames = []
#
# for i in range(num_episodes):
#     env = sample_env()
#     obs = reset(env)
#     student = obs[1]
#     state = student['position'][1] * 7 + student['position'][0]
#
#     epochs, reward, = 0, 0
#     done = False
#
#     while not done:
#         if random.uniform(0, 1) < epsilon_explo:
#             action_student = env.action_space[1].sample()  # Explore action space
#         else:
#             action_student = np.random.choice(
#                 np.where(q_table_student[state] == np.max(q_table_student[state]))[0])  # Exploit learned values
#             # np.argmax is biased towards the first value when several values are equal, we want to avoid that
#
#         action_teacher = np.random.choice(
#             np.where(q_table_teacher[state] == np.max(q_table_teacher[state]))[0])  # Exploit learned values
#
#         # print('s', action_student)
#         # print('t', action_teacher)
#
#         if action_student != action_teacher:
#             proba = random.random()
#             if proba > 0.3:
#                 reward = -1
#             else:
#                 reward = 1
#         else:
#             reward = 1
#
#         next_obs, env_reward, done, info = env.step([6, action_student])
#         student = next_obs[1]
#         next_state = student['position'][1] * 7 + student['position'][0]
#
#         # frame = env.render(mode='rgb_array')
#         # frames.append(_label_with_episode_number(frame, episode_num=i))
#
#         old_value = q_table_student_explo[state, action_student]
#         next_max = np.max(q_table_student_explo[next_state])
#
#         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#         q_table_student_explo[state, action_student] = new_value
#
#         state = next_state
#         epochs += 1
#
#         all_epochs[i] += epochs
#         all_rewards[i] += reward
#
#     if i % 1000 == 0:
#         print(f"Episode: {i}")
#
# np.save('/home/yseult/PycharmProjects/BMMproject/marlgrid-master/q_table_student_explo.npy', q_table_student_explo)
#
#
# print("Training of the second student finished.\n")
#
# #####################################################################
#
# # Plot
# figname = f'exploratory_student_training_results_{num_episodes}_episodes.png'
#
# smoothing_window = 10
#
# plt.figure(figsize=(10, 5))
# rewards_smoothed = pd.Series(all_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
# plt.plot(rewards_smoothed)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward (Smoothed)")
# plt.title("Episode Reward over Time for the exploratory Student (Smoothed over window size {})".format(smoothing_window))
# plt.savefig(figname)
# plt.show(block=False)
# plt.close()
# ###########################################################################
#
# # Evaluation of the second student training
#
# total_epochs, total_reward = 0, 0
# episodes = 100
#
# for _ in range(episodes):
#     env = sample_env()
#     obs = reset(env)
#     student = obs[1]
#     state = student['position'][1] * 7 + student['position'][0]
#
#     epochs, reward = 0, 0
#
#     done = False
#
#     while not done:
#         action_student = np.random.choice(
#             np.where(q_table_student_explo[state] == np.max(q_table_student_explo[state]))[0])  # Exploit learned values
#
#         # frame = env.render(mode='rgb_array')
#         # frames.append(_label_with_episode_number(frame, episode_num=i))
#
#         obs, reward, done, info = env.step([6, action_student])
#
#         epochs += 1
#
#     total_epochs += epochs
#
# print(f"Results after {episodes} episodes:")
# print(f"Average time steps per episode: {total_epochs / episodes}")
