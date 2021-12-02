import torch
import json
import random
import string
from torch.nn.functional import one_hot
import cv2


def generate_motor_command_sequence(num_commands, sos=1, eos=2):
    """
    Generates a random sequence of motor commands. This is useful in cases where the dataset contains only images and
    text. The generated motor command sequence can be used to validate various classes such as torch.utils.data.Dataset
    subclasses and torch.utils.data.DataLoader subclasses while waiting for the actual motor command to be available.

    Parameters
    ----------
    num_commands : int
        The count of motor commands in the dataset
    sos : int
        The integer representing the start-of-sequence symbol
    eos : int
        The integer representing the end-of-sequence symbol

    Returns
    -------
    A motor command sequence of random length and random entries (motor commands). The returned sequence first element is
    the start-of-sequence symbol while the last one is the end-of-sequence symbol.

    Examples
    ----------
    >>> from utils.parser import load_config
    >>> cfg_file = load_config('configs/default.yaml')
    >>> motor_comands = generate_motor_command_sequence(20)
    >>> print(motor_comands)

    """
    length = random.randint(1, 50)
    commands = torch.cat([torch.tensor([sos]).long(), torch.randint(0, num_commands, (length,)), torch.tensor([eos]).long()])
    return commands


def get_bacth(cfg, text_len=200, cmds_len=30):
    """
    Generates dummy data that can be used to test models. The batch size specified in the
    configuration file will be used to determine the batch size of the generated batch.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        The configuration file to use when determine the vocabulary size and number of motor commands
    text_len : int
        The length of the longest text description to put in the batch of data
    cmds_len : int
        The length of the longest motor command sequence to put in the batch of data

    Returns
    -------
    per_image : torch.Tensor
        Perceived image torch.Tensor of shape (batch_size, 3, 224, 224)
    goal_image : torch.Tensor
        Goal image torch.Tensor of shape (batch_size, 3, 224, 224)
    text : torch.Tensor
         Text sequence torch.Tensor of shape (batch_size, max_sequence_length, vocabulary_size)
    command : torch.Tensor
        Motor command sequence torch.Tensor of shape (batch_size, max_sequence_length, num_motor_commands)
    lengths_text : list
        The lengths of the generated text sequences in the batch
    lengths_cmd : list
        The lengths of the generated motor command sequences in the batch

    """

    batch_size = cfg.TRAIN.BATCH_SIZE
    per_image = torch.randn((batch_size, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
    goal_image = torch.randn((batch_size, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))

    vocab_size = cfg.DATASET.VOCABULARY_SIZE
    num_commands = cfg.DATASET.NUM_COMMANDS

    max_len = max(text_len, cmds_len)
    text = torch.randint(0, vocab_size, (batch_size, max_len))

    commands = torch.randint(0, num_commands, (batch_size, max_len))
    commands = one_hot(commands, num_classes=num_commands)

    lengths_text = [random.randint(1, text_len + 1) for i in range(batch_size - 1)] + [text_len]
    lengths_cmd = [random.randint(1, cmds_len + 1) for i in range(batch_size - 1)] + [cmds_len]

    return per_image, goal_image, text, commands.float(), lengths_text, lengths_cmd


def get_encoded_text(cfg, add_sos=True, add_eos=True):
    """
    Loads a JSON file with a specific format containing textual action descriptions, encodes them to integers, and
    returns the word to integer and mapping as well as the integer-encoded-text. The file that will be used is specified
    in the configuration file under the DATASET category as TRAIN_FILE i.e cfg.DATASET.TRAIN_FILE

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        A loaded configuration file
    add_sos : bool
        Whether to prepend the start-of-sequence at the front of the sequence.
        Default is True
    add_eos : bool
        Whether to append the end-of-sequence at the front of the sequence.
        Default is True

    Returns
    -------
    word2int : dict
        A dictionary (hashmap) with the words in the dataset as keys and the assigned integer as value
    all_descriptions : dict
        A dictionary (hashmap) with an id identifying a data point in the dataset as its key and the integer-encoded
        text as its value.

    Expected Format for TRAIN FILE
    ------------------------------
    The JSON file must contain a list whose elements are dictionaries who have two required keys. The required keys are
    'id' and 'label' containing the id of the data point and the text description corresponding to the data point respectively.

    >>> [{"id": "45", "label": "putting wood onto cable", "template": "Putting [something] onto [something]", "placeholders": ["wood", "cable"]}, {"id": "30", "label": "pulling tupperware from right to left", "template": "Pulling [something] from right to left", "placeholders": ["tupperware"]}, {"id": "2", "label": "pretending to pick a pillow up", "template": "Pretending to pick [something] up", "placeholders": ["a pillow"]}, {"id": "9", "label": "putting usb behind mouse", "template": "Putting [something] behind [something]", "placeholders": ["usb", "mouse"]}, {"id": "7", "label": "pushing flashdisk from right to left", "template": "Pushing [something] from right to left", "placeholders": ["flashdisk"]}, {"id": "31", "label": "putting coconut kernel", "template": "Putting [something similar to other things that are already on the table]", "placeholders": ["coconut kernel"]}, {"id": "33", "label": "scooping powder up with spoon", "template": "Scooping [something] up with [something]", "placeholders": ["powder", "spoon"]}, {"id": "49", "label": "lifting up one end of hose, then letting it drop down", "template": "Lifting up one end of [something], then letting it drop down", "placeholders": ["hose"]}]

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> cfg_file = load_config('configs/default.yaml')
    >>> word2int, all_descriptions = get_encoded_text(cfg_file)

    """
    train_filename = cfg.DATASET.TRAIN_FILE

    with open(train_filename) as tf:
        trainset = json.load(tf)

    word2int = dict()
    all_int_descriptions = dict()
    word_id = 3
    for video in trainset:
        video_id = video['id']
        description = video['label']

        # Remove punctuation from description
        description = description.translate(str.maketrans('', '', string.punctuation))

        # split on whitespace to get a list of words
        words = description.split()

        int_description = []
        if add_sos:
            int_description.append(cfg.DATASET.SOS)

        # Add all words in lowercase form to count number of words
        for w in words:
            if w not in word2int:
                word2int[w] = word_id
                int_description.append(word_id)
                word_id += 1
            else:
                int_description.append(word2int[w])
        if add_eos:
            int_description.append(cfg.DATASET.EOS)
        all_int_descriptions[video_id] = int_description

    print(f"{len(word2int)} words were found in Something-Something-v2")
    return word2int, all_int_descriptions


def get_first_last_frames(video_path):
    """
    Opens a video whose path is given, loads the first and last frame and returns them.

    Parameters
    ----------
    video_path : str
        The path to the video from which the frames will be loaded.

    Returns
    -------
    first_frame : numpy.ndarray
        The loaded first frame of the video with shape (height, width, 3)
    last_frame : numpy.ndarray
        The loaded last frame of the video with shape (height, width, 3)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> first, last = get_first_last_frames("data/20bn-something-something-v2/2.webm")
    >>> plt.imshow(first)
    >>> plt.figure()
    >>> plt.imshow(last)
    >>> plt.show()
    """
    vs = cv2.VideoCapture(video_path)
    last_frame_num = vs.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    ret, first_frame = vs.read()

    if not ret:
        raise RuntimeError(f"Unable to read first frame from {video_path}")
    # Seek the last frame
    vs.set(cv2.CAP_PROP_POS_FRAMES, last_frame_num)

    ret, last_frame = vs.read()
    if not ret:
        raise RuntimeError(f"Unable to read last frame from {video_path}")

    vs.release()
    return first_frame, last_frame


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.parser import load_config
    cfg_file = load_config('configs/default.yaml')

    motor_comands = generate_motor_command_sequence(20)
    print("Generated motor command sequence: \n", motor_comands)

    get_encoded_text(cfg_file)

    first, last = get_first_last_frames("data/20bn-something-something-v2/2.webm")

    plt.imshow(first)
    plt.figure()
    plt.imshow(last)

    plt.show()