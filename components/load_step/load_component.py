import tensorflow as tf
from load import load_tensorflow_dataset
import argparse
from kfp.components import OutputPath

def load_tf_dataset_component(dataset_str: str,
                             data_output_path: OutputPath('tf.data.Dataset')):
    
    # output is a dictionary with 'test' and 'train' keys. 
    data = load_tensorflow_dataset(dataset_str=dataset_str)
    
    # save output data 
    data.save(data_output_path)
    

parser = argparse.ArgumentParser(description="Load a cleaned tensorflow dataset from tfds.")
parser.add_argument("--dataset_str", type=str, 
                    help="Dataset string to download. (e.g. mnist)")
parser.add_argument("--data_output_path", type=str,
                    help="Directory for output file")
args = parser.parse_args()
    
load_tf_dataset_component(args.dataset_str, args.train_test_split, 
                          args.data_output_path)
