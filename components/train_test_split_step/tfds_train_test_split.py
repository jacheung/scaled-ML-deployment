import tensorflow as tf

def train_test_split(data: tf.Data.Dataset, 
                     train_proportion: float, 
                     test_proportion: float):
    
    if train_proportion + test_proportion != 1:
        raise Exception("Sum of proportions must equal 1.")
    
    # use dataset size to determine data splits
    DATASET_SIZE = len(data)

    # calculate training and test size (technically test size isn't used...)
    train_size = int(train_proportion * DATASET_SIZE)
    test_size = int(test_proportion * DATASET_SIZE)

    # shuffle dataset and split data accordingly
    shuffled_dataset = data.shuffle()
    train_dataset = shuffled_dataset.take(train_size)
    test_dataset = shuffled_dataset.skip(train_size)

    return train_dataset, test_dataset