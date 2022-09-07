
from torch.utils.data import Dataset
import torch
class MyDataset(Dataset):
    def __init__(self, args, MyDataCorpus, path, deviceEng_dict, deviceNum_dict, event_dict, desc_dict):
        super().__init__()
        corpus = MyDataCorpus(args, path, deviceEng_dict, deviceNum_dict, event_dict, desc_dict)
        self.id = corpus.id
        self.deviceEng = corpus.deviceEng
        self.deviceNum = corpus.deviceNum
        self.deviceLen = corpus.deviceLen
        self.event = corpus.event
        self.eventLen = corpus.eventLen
        self.desc = corpus.desc
        self.label = corpus.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        return self.id[idx], torch.tensor(self.deviceEng[idx]), torch.tensor(self.deviceNum[idx]), torch.tensor(self.deviceLen[idx]), \
               torch.tensor(self.event[idx]), self.eventLen[idx], torch.tensor(self.desc[idx]),\
                torch.tensor(self.label[idx], dtype=torch.float32)
