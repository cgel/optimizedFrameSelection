import tensorflow as tf
import time
import numpy as np
import parseConfig
import utils
import importlib
from dataHandler import dataHandler

config = parseConfig.config

env = utils.create_env(config)

tf.logging.set_verbosity(tf.logging.ERROR)
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

Agent = getattr(importlib.import_module(
    "agents." + config.agent), config.agent)
agent = Agent(config, sess)
agent.testing(True)

saver = tf.train.Saver(max_to_keep=20)

if config.load_checkpoint != "":
    utils.load_checkpoint(sess, saver, config)
else:
    print("Using random agent")
    sess.run(tf.initialize_all_variables())

print("Using agent " + config.agent)
print("On device: " + config.device)

dh = dataHandler()


def generate_dataset():
    global_step = 0
    episode = 0
    while global_step < config.num_steps:
        x, r, done, score = env.reset(), 0, False, 0
        ep_begin_t = time.time()
        ep_begin_step_count = agent.step_count
        while not done:
            action = agent.step(x, r)
            dh.addData(agent.game_state[:,:,:,-1], agent.Q_np, np.array([action]), np.array([r]), np.array([done]),
                       agent.representations_np[0], agent.representations_np[1])
            x, r, done, info = env.step(action)
            score += r
            global_step += 1
        agent.terminal()
        if episode % 5 == 0:
            print("step %i out of %i -- %i%% -- score: %i" % (global_step, config.num_steps, 100*float(global_step)/config.num_steps, score))
        episode += 1

generate_dataset()
