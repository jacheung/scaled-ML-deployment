from preprocess import preprocess_mnist_tfds
import argparse
from kfp.components import InputPath, OutputPath
import tensorflow as tf

def preprocess_component(data_file_input: InputPath('tf.data.Dataset'), 
                         pp_data_file_output: OutputPath('tf.data.Dataset')):
    # load data from previous step
    data = tf.data.Dataset.load(data_file_input)

    data = data.map(preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(128)