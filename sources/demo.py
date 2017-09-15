# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.

import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

TRAINING_INPUT_PATH = os.path.join("..", "data", "training", "input")
TRAINING_OUTPUT_PATH = os.path.join("..", "data", "training", "output")

IMAGE_HEIGHT  = 2448
IMAGE_WIDTH   = 3264
NUM_CHANNELS  = 3
BATCH_SIZE    = 100

# Reading image file paths
train_filepaths = os.listdir(TRAINING_INPUT_PATH)
train_filepaths.sort()
train_filepaths = [os.path.join(TRAINING_INPUT_PATH, fp)
                   for fp in train_filepaths]

# Reading labels
train_labels = pd.read_csv(os.path.join(TRAINING_OUTPUT_PATH,
                                        "labels.csv")).iloc[:,6:].values

# Convert string file paths and integer labels into tensors
train_images = ops.convert_to_tensor(train_filepaths, dtype=tf.string)
train_labels = ops.convert_to_tensor(train_labels, dtype=tf.int16)

# Create input queues
train_input_queue = tf.train.slice_input_producer([train_images, train_labels],
                                                  shuffle=False)

# Process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

# Define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# Collect batches of images before processing
train_image_batch, train_label_batch, train_filename_batch = tf.train.batch(
    [train_image,
     train_label,
     train_input_queue[0]],
    batch_size=BATCH_SIZE,
    num_threads=4
)

with tf.Session() as sess:
  # Initialize the variables
  sess.run(tf.global_variables_initializer())
  # Initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  print("From the train set:")
  for i in range(2):
    b_filenames, b_images, b_labels = sess.run([train_filename_batch,
                                                train_image_batch,
                                                train_label_batch])
    print(b_filenames)
    print(b_labels)
  # Stop our queue threads
  coord.request_stop()
  coord.join(threads)
