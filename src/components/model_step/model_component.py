import tensorflow as tf
from model import MNIST
import argparse
from kfp.dsl import InputPath
from utils import set_mlflow_experiment

import mlflow

# Function doing actual work of loading datasets
def production_model_component(training_dataset_file_input: InputPath(tf.data.Dataset)):
    
    # mlflow experiment
    experiment_id = set_mlflow_experiment(args.mlflow_experiment_name)

    # Since we'll have done hyperparameter search in training, we'll just load the best params from mlflow
    # let's add in dummy ones for now
    hyperparameters = {'epochs': 10,
                    'l1': 0,
                    'l2': 0, 
                    'num_hidden': 64,
                    'learning_rate': .01}

    # load data from previous step
    xy_train = tf.data.Dataset.load(training_dataset_file_input)

    
    MNIST_CNN = MNIST()
    with mlflow.start_run():
        MNIST_CNN.fit_production(xy_train=xy_train,
                                 hyperparameters=hyperparameters)
        # MLFlow Tracking parameters
        mlflow.log_params(params=hyperparameters)

        # MLFlow Tracking metrics 
        # Logging metrics for each epoch (housed in dictionary)
        training_history = MNIST_CNN._train_history.history
        for epoch in range(0, hyperparameters['epochs']):
            insert = {}
            for metric, value in training_history.items():
                insert[metric] = training_history[metric][epoch]
            mlflow.log_metrics(metrics=insert, step=epoch+1)


parser = argparse.ArgumentParser(description="Train MobileNetV2 for Production")
parser.add_argument("--training-dataset-file-input", type=str, 
                    help="Directory for input file. Should be tensorflow dataset format.")
args = parser.parse_args()
    
production_model_component(args.dataset_file_input)