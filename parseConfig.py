import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-device", default="0")
parser.add_argument("-gamma", type=float, default=0.99)
parser.add_argument("-epsilon", type=float, default=0.05)
parser.add_argument("-buff_size", type=float, default=4)
parser.add_argument("-gpu", type=int, default=True)
parser.add_argument("-load_checkpoint", default="")
parser.add_argument("-agent", default="DQN")
parser.add_argument("-env_name", default="Breakout-v0")
parser.add_argument("-num_steps", type=int, default=1000000)

# useless but necessary config
parser.add_argument("-replay_memory_capacity", type=int, default=10)
parser.add_argument("-learning_rate", type=float, default=0.00025)

config = parser.parse_args()
if config.gpu != 0:
  config.device = "/gpu:"+config.device
else:
  config.device = "/cpu:"+config.device
config.logging = False
