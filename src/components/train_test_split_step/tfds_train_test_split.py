import tensorflow as tf

def train_test_split(data: tf.data.Dataset, 
                     train_proportion: float, 
                     test_proportion: float):
    
    if train_proportion + test_proportion != 1:
        raise Exception(f"Sum of proportions must equal 1. \
                        Proportions equal {train_proportion + test_proportion}")
    
    # use dataset size to determine data splits
    DATASET_SIZE = len(data)

    # calculate training and test size (technically test size isn't used...)
    train_size = int(train_proportion * DATASET_SIZE)
    test_size = int(test_proportion * DATASET_SIZE)

    # shuffle dataset and split data accordingly
    train_dataset = data.take(train_size)
    test_dataset = data.skip(train_size)

    return train_dataset, test_dataset