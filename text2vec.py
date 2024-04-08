from model.t5 import t5_encode_text
from functools import partial
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import Dataset

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
        text_embeds = torch.tensor([]).cuda(0)
        for i in trange(self.text.shape[0]):
            text_embeds = torch.cat((text_embeds, self.encode_text(self.text[i]).float().unsqueeze(0)), dim=0)
            # print(self.encode_text(self.text[i]).float().shape)
        np.save("files/testFiles/t5emb.npy", text_embeds.cpu().numpy())

if __name__ == "__main__":
    # text = Text()
    # dataloader = torch.utils.data.DataLoader(text, batch_size=256, shuffle=True)
    test = text2vec()
    test.show_text()
    test.encoding()