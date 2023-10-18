import mlflow
import tensorflow as tf
import argparse
# Project imports
from components.load_step import load
from components.preprocess_step import preprocess
from components.train_test_split_step import tfds_train_test_split
from components.model_step import model
from components.utils import set_mlflow_experiment


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_experiment_name", type=str,
                        help="MLflow experiment name")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Number of epochs")
    parser.add_argument("--l1", type=float, default=0,
                         help="Output layer 1 kernel regularizer")
    parser.add_argument("--l2", type=float, default=0,
                         help="Output classification head kernel regularizer")
    parser.add_argument("--num_hidden", type=int, default=64,
                         help="Output layer 1 number of units")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                         help="Learning rate")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # mlflow experiment
    experiment_id = set_mlflow_experiment(args.mlflow_experiment_name)

    # begin data preparation of pipeline
    data = load.mini_load_tensorflow_dataset(dataset_str='mnist',
                                            percent_of_dataset=10)
    data = data.map(preprocess.preprocess_mnist_tfds,
                    num_parallel_calls=tf.data.AUTOTUNE).batch(128)
    train_dataset, test_dataset = tfds_train_test_split.train_test_split(data=data,
                                                                        train_proportion=.75,
                                                                        test_proportion=.25)
    # instantiate hyperparameters from arguments. This'll be changed for katib
    hyperparameters = {'epochs': args.epochs,
                    'l1': args.l1,
                    'l2': args.l2, 
                    'num_hidden': args.num_hidden,
                    'learning_rate': args.learning_rate}

    MNIST = model.MNIST()
    with mlflow.start_run():
        MNIST.fit_hp_search(xy_train=train_dataset,
                            xy_test=test_dataset,
                            hyperparameters=hyperparameters)
        # MLFlow Tracking parameters
        mlflow.log_params(params=hyperparameters)

        # MLFlow Tracking metrics 
        # Logging metrics for each epoch (housed in dictionary)
        training_history = MNIST._train_history.history
        for epoch in range(0, hyperparameters['epochs']):
            insert = {}
            for metric, value in training_history.items():
                insert[metric] = training_history[metric][epoch]
            mlflow.log_metrics(metrics=insert, step=epoch+1)
        

