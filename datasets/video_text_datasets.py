import torch
from torch.utils.data import Dataset
import json
from utils.data_utils import get_encoded_text, get_first_last_frames, generate_motor_command_sequence
import numpy as np
import cv2
from torch.nn.functional import one_hot


class SomethingSomethingV2Dataset(Dataset):
    """
    Fetches data points from the Something-Something V2 dataset using paths found in the configuration file.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        A loaded configuration object
    """

    def __init__(self, cfg):
        # Fetch settings from configuration file
        self.video_folder = cfg.DATASET.VIDEO_FOLDER
        self.img_size = (cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)
        self.num_commands = cfg.DATASET.NUM_COMMANDS
        self.sos = cfg.DATASET.SOS
        self.eos = cfg.DATASET.EOS

        # Get the encoded text for all data points in the dataset
        word2int, all_descriptions = get_encoded_text(cfg)

        # Keep data needed in other methods
        self.X = all_descriptions
        self.video_id_list = list(self.X.keys()) # To allow accessing dictionary elements in an ordered manner.

    def __getitem__(self, index):

        video_id = self.video_id_list[index]
        description = self.X[video_id]

        # Get the perceived and goal images
        video_path = f"{self.video_folder}/{video_id}.webm"
        per_img, goal_img = get_first_last_frames(video_path)

        # Convert the loaded frames from shapes (height, width, 3) to (3, self.im_size, self.im_size)
        per_img = cv2.resize(per_img, self.img_size)
        goal_img = cv2.resize(goal_img, self.img_size)
        per_img = np.moveaxis(per_img, -1, 0)
        goal_img = np.moveaxis(goal_img, -1, 0)

        # Generate an artificial motor command sequence
        motor_commands = generate_motor_command_sequence(self.num_commands, self.sos, self.eos)
        motor_commands = torch.LongTensor(motor_commands)
        motor_commands = one_hot(motor_commands, num_classes=self.num_commands).float()

        # Subtract 1 from lengths since the slice [1:] will be given as input to the model
        return torch.Tensor(per_img), torch.Tensor(goal_img), torch.Tensor(description), motor_commands, len(description) - 1, len(motor_commands) - 1

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    from utils.parser import load_config
    cfg_file = load_config('configs/default.yaml')

    dataset = SomethingSomethingV2Dataset(cfg_file)

    # for i in [24837, 50991, 53527, 88403, 89674, 94590, 131723, 141318]:
    #     print(dataset[i])


