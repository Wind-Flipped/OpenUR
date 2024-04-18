from model.t5 import t5_encode_text
from functools import partial
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Text(Dataset):
    def __init__(self):
        data_path = "files/testFiles/str.npy"
        self.text = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]


class text2vec():
    def __init__(self, t5_name="files/LLM"):
        self.encode_text = partial(t5_encode_text, name=t5_name)
        data_path = "files/testFiles/str.npy"
        self.text = np.load(data_path, allow_pickle=True)


    def show_text(self):
        print(self.text.shape[0])
        print("---------------------")
        print(self.text[:10])

    def encoding(self):
        tensors_list = []
        for i in trange(self.text.shape[0]):
            # (1 * m * 2048)
            text_encode = self.encode_text(self.text[i]).float()

            # 计算需要填充的数量
            max_length = 64
            padding_length = max_length - text_encode.shape[1]
            padding = (0, 0, 0, padding_length)  # 前后不填充，上下填充236个0
            padded_tensor = F.pad(text_encode, pad=padding, mode='constant', value=0)
            # 显存内的Tensor存到内存里
            padded_tensor_cpu = padded_tensor.detach().cpu()
            tensors_list.append(padded_tensor_cpu)
            del text_encode
            del padded_tensor


        concatenated_tensor = torch.cat(tensors_list, dim=0)
        # 打印调整形状后的张量大小
        print(concatenated_tensor.size())  # 应该输出 torch.Size([m, 256, 2048])
        np.save("files/testFiles/t5emb.npy", concatenated_tensor.numpy())

if __name__ == "__main__":
    # text = Text()
    # dataloader = torch.utils.data.DataLoader(text, batch_size=256, shuffle=True)
    test = text2vec()
    test.show_text()
    test.encoding()