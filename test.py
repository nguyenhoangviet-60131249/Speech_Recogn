import pathlib

import tensorflow as tf
import numpy as np

data_dir = pathlib.Path('data/mini_speech_commands')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")  # đường dẫn file
filenames = tf.random.shuffle(filenames)  # trộn file
num_samples = len(filenames)
print("Total of samples:", num_samples)
print("Length of yes label:", len(tf.io.gfile.listdir(str(data_dir / commands[0]))))
print("Example of tensor:", filenames[0])  # lấy file âm đầu tiên sau khhi trộn
