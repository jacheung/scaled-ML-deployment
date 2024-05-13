import tensorflow as tf

def train_test_split(data: tf.data.Dataset, 
                     train_proportion: float):
    
    # use dataset size to determine data splits
    DATASET_SIZE = len(data)

    # calculate training and test size (technically test size isn't used...)
    train_size = int(train_proportion * DATASET_SIZE)

    # shuffle dataset and split data accordingly
    train_dataset = data.take(train_size).cache()
    test_dataset = data.skip(train_size).cache()

    return train_dataset, test_dataset