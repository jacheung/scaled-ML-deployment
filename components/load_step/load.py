import tensorflow_datasets as tfds


def load_tensorflow_dataset(dataset_str: str, train_test_split: bool = True):
    # assign train_test_split param
    if train_test_split is True:
        split = ['train', 'test']
    else:
        split = 'all'
    
    # load
    data, ds_info = tfds.load(
        dataset_str,
        split=split, shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # package data
    if train_test_split is True:
        data = {'train': data[0], 
                'test': data[1]}
    else:
        data = {'train': data,
                'test': None}
    return data