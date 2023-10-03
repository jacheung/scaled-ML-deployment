import tensorflow_datasets as tfds


def load_tensorflow_dataset(dataset_str: str):
    # load tensorflow dataset
    data, ds_info = tfds.load(
        dataset_str,
        split='all', shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    return data


