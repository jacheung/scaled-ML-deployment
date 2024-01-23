import tensorflow as tf
from load import load_tensorflow_dataset
import argparse
from kfp.dsl import OutputPath


# Function doing actual work of loading datasets
def load_tf_dataset_component(dataset_str: str,
                              dataset_output_path: OutputPath(tf.data.Dataset)):
    
    # output is a dictionary with 'test' and 'train' keys. 
    data = load_tensorflow_dataset(dataset_str=dataset_str)
    
    # save output data 
    data.save(dataset_output_path)
    

# Defining the argument parser via the CLI
parser = argparse.ArgumentParser(description="Loads from TensorFlow Datasets, a collection of ready-to-use datasets.")
parser.add_argument("--dataset-str", type=str, 
                    help="Dataset string to download. (e.g. mnist)")
parser.add_argument("--dataset-output-path", type=str,
                    help="Directory for output file")
args = parser.parse_args()
    
load_tf_dataset_component(args.dataset_str, 
                          args.dataset_output_path)
