import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class InputHandle(Dataset):
    def __init__(self, input_param):
        self.input_data_type = input_param.get("input_data_type", "float32")
        self.output_data_type = input_param.get("output_data_type", "float32")
        self.img_width = input_param["image_width"]
        self.img_height = input_param["image_height"]
        self.length = input_param["total_length"]
        self.data_path = input_param["data_path"]
        self.type = input_param["type"]

        self.case_list = []
        name_list = os.listdir(self.data_path)
        name_list.sort()
        if self.type == "train":
            for name in name_list:
                case_path = os.path.join(self.data_path, name)
                if os.path.isdir(case_path):
                    files_in_folder = os.listdir(case_path)
                    if len(files_in_folder) == 29:
                        case = []
                        for i in range(29):
                            case.append(
                                self.data_path
                                + "/"
                                + name
                                + "/"
                                + name
                                + "-"
                                + str(i).zfill(2)
                                + ".png"
                            )
                        self.case_list.append(case)
        elif self.type == "test":
            for name in name_list:
                case = []
                for i in range(29):
                    case.append(
                        self.data_path
                        + "/"
                        + name
                        + "/"
                        + name
                        + "-"
                        + str(i).zfill(2)
                        + ".png"
                    )
                self.case_list.append(case)
        else:
            raise ValueError("type of data loader unknown %s" % self.type)

    def load(self, index):
        data = []
        for img_path in self.case_list[index]:
            img = cv2.imread(img_path, 2)
            data.append(np.expand_dims(img, axis=0))
        data = np.concatenate(data, axis=0).astype(self.input_data_type) / 10.0 - 3.0
        assert data.shape[1] <= 1024 and data.shape[2] <= 1024
        return data

    def __getitem__(self, index):
        data = self.load(index)[-self.length :].copy()

        mask = np.ones_like(data)
        mask[data < 0] = 0
        data[data < 0] = 0
        data = np.clip(data, 0, 128)

        vid = np.zeros((self.length, self.img_height, self.img_width, 2))
        vid[..., 0] = data
        vid[..., 1] = mask
        img = dict()
        img["radar_frames"] = vid
        return img

    def __len__(self):
        return len(self.case_list)
