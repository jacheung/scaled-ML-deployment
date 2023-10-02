import tensorflow as tf

def train_test_split(data: tf.Data.Dataset, 
                     train_proportion: float, 
                     test_proportion: float):
    
    DATASET_SIZE = len(data)

    train_size = int(train_proportion * DATASET_SIZE)
    test_size = int(test_proportion * DATASET_SIZE)

    shuffled_dataset = data.shuffle()
    train_dataset = shuffled_dataset.take(train_size)
    test_dataset = shuffled_dataset.skip(train_size)

    return train_dataset, test_dataset