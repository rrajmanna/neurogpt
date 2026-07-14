import torch
from torch.utils.data import Dataset

class NeuralSequenceDataset(Dataset):
    def __init__(self, windows, labels, seq_len=16):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.windows) - self.seq_len - 1

    def __getitem__(self, idx):
        seq = self.windows[idx : idx + self.seq_len]
        target = self.windows[idx + self.seq_len]
        target_label = self.labels[idx + self.seq_len]
        return seq, target, target_label