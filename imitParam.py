import argparse



"""
  Parser containing all parameters for the imitation learner.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_shape',
                    type=int,
                    default=[None, 784],
                    help='Input shape for the network')

parser.add_argument('--output_shape',
                    type=float,
                    default=[None, 10],
                    help='Output shape for the network.')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='Initial learning rate.')

parser.add_argument('--max_steps',
                    type=int,
                    default=100000,
                    help='Number of steps to run trainer.')

parser.add_argument('--Conv_param',
                    type=int,
                    default=[[32,8,8,4],\
                             [64,4,4,2],\
                             [64,3,3,2]],
                    help='Parameters of convolutional layers: \
                         filters,kernel, kernel, stride')

parser.add_argument('--fully_connected_layer',
                    type=int,
                    default=[[512]],
                    help='Number of units in each FCL.')

parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='Batch size. \
                          Must divide evenly into the dataset sizes.')

parser.add_argument('--input_data_dir',
                    type=str,
                    default='/tmp/tensorflow/mnist/input_data',
                    help='Directory to put the input data.')

parser.add_argument('--log_dir',
                    type=str,
                    default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
                    help='Directory to put the log data.')

imitParam = parser.parse_args()