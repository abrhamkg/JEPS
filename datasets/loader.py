import torch
from torch.nn.utils.rnn import pad_sequence
import datasets.video_text_datasets


def build_dataset(dataset_name, cfg):
    """
    Instantiates an object from a torch.utils.data.Dataset subclass whose name is given and returns it.

    Parameters
    ----------
    dataset_name : str
        The name of the torch.utils.data.Dataset subclass
    cfg : utils.parser.AttributeDict
        A loaded configuration object to be passed to the object to be instantiated

    Returns
    -------
    The instantiated object.

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> cfg_file = load_config('configs/default.yaml')
    >>> dset = build_dataset('SomethingSomethingV2Dataset', cfg_file)

    """
    dataset_class = getattr(datasets.video_text_datasets, dataset_name)

    return dataset_class(cfg)


def construct_loader(cfg):
    """
    Instantiates a torch.utils.data.DataLoader object using settings specified in a loaded configuration object.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        The loaded configuration object

    Returns
    -------
    A torch.utils.data.DataLoader object

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> cfg_file = load_config('configs/default.yaml')
    >>> loader = construct_loader(cfg_file)

    """
    dataset_name = cfg.TRAIN.DATASET
    batch_size = cfg.TRAIN.BATCH_SIZE
    shuffle = cfg.TRAIN.SHUFFLE
    drop_last = cfg.TRAIN.DROP_LAST

    dataset = build_dataset(dataset_name, cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=something_something_collate,
    )
    return loader


def something_something_collate(batch):
    """
    This function takes a batch of data loaded through the datasets.datasets.SomethingSomethingV2Dataset and pads the
    loaded text and motor command sequence to the same size.

    Parameters
    ----------
    batch : list
        A batch of data automatically passed to this function by a torch.utils.data.DataLoader object.

    Returns
    -------
    A tuple of perceived image, goal image, padded text, padded motor command sequence, lengths of text sequences, and
    lengths of motor command sequences.
    """
    per_img, goal_img, text, commands, lengths_text, lengths_cmd = [list(item) for item in zip(*batch)]

    # Account for the 1 subtracted in the __getitem__ in the dataset class by adding it back here
    max_len_text = max(lengths_text) + 1
    max_len_cmd = max(lengths_cmd) + 1
    if max_len_text > max_len_cmd:
        len_first = len(commands[0])
        # Make sure that the first sequence contains max_len_text number of entries so padded size for
        # the two sequences will be the same
        length_diff = max_len_text - len_first
        commands[0] = torch.cat(commands[0], torch.zeros((length_diff, )).long())

    elif max_len_cmd > max_len_text:
        len_first = len(text[0])
        # Make sure that the first sequence contains max_len_cmd number of entries so padded size for
        # the two sequences will be the same
        length_diff = max_len_cmd - len_first
        text[0] = torch.cat([text[0], torch.zeros((length_diff, ))])

    text = pad_sequence(text, batch_first=True)
    commands = pad_sequence(commands, batch_first=True)

    return torch.stack(per_img), torch.stack(goal_img), text, commands, lengths_text, lengths_cmd


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.parser import load_config
    cfg_file = load_config('configs/default.yaml')

    # dset = build_dataset('SomethingSomethingV2Dataset', cfg_file)

    loader = construct_loader(cfg_file)
    for per_img, goal_img, text, commands, lengths_text, lengths_cmd in loader:
        print(per_img.shape, goal_img.shape, text.shape, commands.shape, lengths_text, lengths_cmd)

    # for i in [24837, 50991, 53527, 88403, 89674, 94590, 131723, 141318]:
    #     dset[i]

