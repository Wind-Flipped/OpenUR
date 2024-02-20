import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchvision import transforms


class Trajectories(Dataset):
    def __init__(self,
                 data_path,
                 seq_len: int,
                 epoch: int,
                 scale=1):
        # plt.xlim(115.5, 117.5)
        # plt.ylim(39.3, 41.3)

        now_path = data_path[epoch % int(len(data_path))]
        self.trajectories = self.init_npy(now_path, 0)
        shuffled_index = np.random.permutation(len(self.trajectories))
        self.uids = self.init_npy(now_path.replace("traj", "uid"), 1)
        
        self.homes = self.init_npy(now_path.replace("traj", "home"), 2)
        self.scale = scale
        if scale == 1:
            self.scales = self.init_npy(now_path.replace("traj", "scale"), 3)
        # print("fuck", self.uids.shape)
        # self.trajectories = self.trajectories[shuffled_index]
        # self.uids = self.uids[shuffled_index]
        # self.homes = self.homes[shuffled_index]

        self.seq_len = seq_len
        self.size = len(self.trajectories)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.scale == 1:
            return self.uids[index], self.homes[index], self.trajectories[index], self.scales[index]
        else:
            return self.uids[index], self.homes[index], self.trajectories[index]

    def init_npy(self, data_path, type=0):
        now_list = np.load(data_path, allow_pickle=True)

        if type == 0:
            temp_list = []
            for traj in now_list:
                # traj = traj.transpose()
                if len(traj.shape) > 1:
                    if (116.2 <= traj[:, 0]).all() and (traj[:, 0] <= 116.55).all() \
                            and (39.75 <= traj[:, 1]).all() and (traj[:, 1] <= 40.1).all():
                        mid1 = (116.55 + 116.2) / 2
                        mid2 = (40.1 + 39.75) / 2
                        len1 = (116.55 - 116.2) / 2
                        len2 = (40.1 - 39.75) / 2
                        traj[:, 0] = (traj[:, 0] - mid1) / len1
                        traj[:, 1] = (traj[:, 1] - mid2) / len2
                        temp_list.append(traj.transpose())
                else:
                    temp_list.append(traj.transpose())
            now_list = temp_list
        elif type == 2:
            temp_list = []
            for home in now_list:
                home[0] = (home[0] - 116.5)
                home[1] = (home[1] - 40.3)
                temp_list.append([home[1], home[0]])
            now_list = temp_list
        elif type == 3:
            temp_list = []
            for scale in now_list:
                temp_list.append([scale, scale])
            now_list = temp_list
        now_data = torch.tensor(np.array(now_list), dtype=float)
        del now_list
        return now_data
