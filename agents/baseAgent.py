import tensorflow as tf
import numpy as np
import cv2
import random
from replayMemory import ReplayMemory
import commonOps
import time

# Abstract agent class that implements a observation
# buffer and handles the replay memory
#
# the interface consists of:
#   step(screen, reward)    returns the action
#   terminal()              call it when the episode is finished
#   self.action_modes       dictionary that allows the agent to use different
#                           action selection alogrithms durning testing.
#                           The base class only has e_greedy (testing_epsilon greedy)
#                           but chilldren can add others.
#
# Notes for the creation of chilldren agents:
#   Must implement the update function
#   self.RM is the replay memory
#   self.game_state is a buffer of screens of size config.buff_size
#   self.game_reward is the latest reward received
#   sefl.game_action is the latest action taken

class BaseAgent:

    # must be implemented by each agent
    def update(self):
        return

    def __init__(self, config, session):
        # build the net
        self.config = config
        self.sess = session
        self.RM = ReplayMemory(config)
        self.step_count = 0
        self.episode = 0
        self.isTesting = False
        self.game_state = np.zeros(
            (1, 84, 84, self.config.buff_size), dtype=np.uint8)
        self.reset_game()
        self.timeout_option = tf.RunOptions(timeout_in_ms=5000)

        # if the new agent needs other action modes define a different dict
        self.action_modes = {
            str(config.testing_epsilon) + "_greedy": self.e_greedy_action}
        self.default_action_mode = self.action_modes.items()[0][0]
        self.action_mode = self.default_action_mode

        self.representations = []

    def step(self, screen, reward):
        # clip the reward
        if not self.isTesting:
            # add the last transition
            self.RM.add(self.game_state[:, :, :, -1],
                        self.game_action, self.game_reward, False)
            self.observe(screen, reward)
            self.game_action = self.e_greedy_action(self.epsilon())
            if self.step_count > self.config.steps_before_training:
                self.update()
            self.step_count += 1
        else:
            # if the agent is testing
            self.observe(screen, reward)
            self.game_action = self.e_greedy_action(0.01)
        return self.game_action

    # Add the final transition to the RM and reset the internal state for the next
    # episode
    def terminal(self):
        if not self.isTesting:
            self.RM.add(
                self.game_state[:, :, :, -1],
                self.game_action, self.game_reward, True)
        self.reset_game()

    def observe(self, screen, reward):
        self.game_reward = max(-1, min(1, reward))
        screen = cv2.resize(screen, (84, 84))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        self.game_state = np.roll(self.game_state, -1, axis=3)
        self.game_state[0, :, :, -1] = screen

    def e_greedy_action(self, epsilon):
        ops = [self.Q]+self.representations
        res = self.sess.run(ops, feed_dict={
                        self.state_ph: self.game_state})[0]

        Q_np = res[0]
        self.representations_np = []
        for rep in res[1:]:
            self.representations_np.append(rep)

        action = np.argmax(Q_np)
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        return action

    def testing(self, t=True):
        self.isTesting = t

    def set_action_mode(self, mode):
        if mode not in self.action_modes:
            raise Exception(str(mode) + " is not a valid action mode")
        self.select_action = self.action_modes[mode]

    def reset_game(self):
        self.game_state.fill(0)
        self.game_action = 0
        self.game_reward = 0
        if not self.isTesting:
            # add initial black screens for next episode
            for i in range(self.config.buff_size - 1):
                self.RM.add(np.zeros((84, 84)), 0, 0, False)

    def epsilon(self):
        if self.step_count < self.config.exploration_steps:
            return self.config.initial_epsilon - \
                ((self.config.initial_epsilon - self.config.final_epsilon) /
                 self.config.exploration_steps) * self.step_count
        else:
            return self.config.final_epsilon
