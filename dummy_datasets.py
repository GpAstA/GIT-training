# dummy_dataset.py
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = [("This is a positive example." if i % 2 == 0 else "This is a negative example.", 1 if i % 2 == 0 else 0) for i in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {"text": self.data[idx][0], "labels": self.data[idx][1]}
