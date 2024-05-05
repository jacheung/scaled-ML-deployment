from tfds_train_test_split import train_test_split
from kfp.dsl  import InputPath, OutputPath
import tensorflow as tf
import argparse

def train_test_split_component(tfds_input_path: InputPath('tf.data.Dataset'),
                               train_output_path: OutputPath('tf.data.Dataset'),
                               test_output_path: OutputPath('tf.data.Dataset'),
                               train_proportion: float):
    
    # load data from previous step
    data = tf.data.Dataset.load(tfds_input_path)

    # train test split data based on train and test proportion
    train_dataset, test_dataset = train_test_split(data=data,
                                                   train_proportion=float(train_proportion))
    
    # return train and test data output paths
    train_dataset.save(train_output_path)
    test_dataset.save(test_output_path)
    


parser = argparse.ArgumentParser(description="Split tfds file into train and test data")
parser.add_argument("--tfds-input-path", type=str, 
                    help="Directory for input file. Should be tensorflow dataset format.")
parser.add_argument("--train-output-path", type=str,
                    help="Directory for train output file")
parser.add_argument("--test-output-path", type=str,
                    help="Directory for test output file")
parser.add_argument("--train-proportion", type=str,
                    help="Proportion of data to use for training. Acceptable values from [0.0 - 1.0]")
args = parser.parse_args()
    
train_test_split_component(args.tfds_input_path, args.train_output_path, args.test_output_path,
                           args.train_proportion)

    