import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ReviewDataset(Dataset):
    ''' 
    класс датасета
    '''
    def __init__(self, data, labels):
        self.data = data  # список списков чисел (векторы)
        self.labels = labels  # список меток

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Преобразуем элементы в тензоры PyTorch
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
