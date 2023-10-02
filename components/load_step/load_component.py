from load import load_tensorflow_dataset
import argparse
from kfp.components import OutputPath

def load_tf_dataset_component(dataset_str: str, 
                             train_test_split: bool,
                             data_file_output: OutputPath('tf.data.Dataset')):
    
    # output is a dictionary with 'test' and 'train' keys. 
    data = load_tensorflow_dataset(dataset_str=dataset_str, 
                                    train_test_split=train_test_split)
    
    # save output data 
    data['train'].save(data_file_output)


parser = argparse.ArgumentParser(description="Split CSV file into train and test data")
parser.add_argument("--dataset_str", type=str, 
                    help="Dataset string to download. (e.g. mnist)")
parser.add_argument("--train_test_split", type=bool, 
                    help="Boolean to do train-test split. Default ratio is 80/20")
parser.add_argument("--output_path", type=str,
                    help="Directory for output file")
args = parser.parse_args()
    
load_tf_dataset_component(args.dataset_str, args.train_test_split, args.output_path)
