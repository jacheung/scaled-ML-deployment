import tensorflow_datasets as tfds


def load_tensorflow_dataset(dataset_str: str):
    # load tensorflow dataset
    data, _ = tfds.load(
        dataset_str,
        split='all', shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    return data

def mini_load_tensorflow_dataset(dataset_str: str, percent_of_dataset: int):
    # Error checking
    if (percent_of_dataset <= 0) or (percent_of_dataset > 100):
        raise Exception('Values for percent_of_dataset must be integer between 0 and 100')
    
    # load mini batch of tensorflow dataset
    data, _ = tfds.load(
        dataset_str,
        split=[f'train[:{percent_of_dataset}%]', 'test'], shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    return data[0]



