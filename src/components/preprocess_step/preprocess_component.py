from preprocess import preprocess_mnist_tfds
from kfp.dsl  import InputPath, OutputPath
import tensorflow as tf
import argparse

def preprocess_component(dataset_file_input: InputPath(tf.data.Dataset), 
                         pp_dataset_file_output: OutputPath(tf.data.Dataset)):
    # load data from previous step
    data = tf.data.Dataset.load(dataset_file_input)

    data = data.map(preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(128)
    print('Preprocessing done. Saving output files...')
    data.save(pp_dataset_file_output)

parser = argparse.ArgumentParser(description="Preprocess images for MobileNetV2")
parser.add_argument("--dataset-file-input", type=str, 
                    help="Directory for input file. Should be tensorflow dataset format.")
parser.add_argument("--pp_dataset-file-output", type=str,
                    help="Directory for preprocessed dataset output file")
args = parser.parse_args()
    
preprocess_component(args.dataset_file_input, 
                     args.pp_dataset_file_output)