from .cohort import CohortDataset
from torch.utils.data import DataLoader

def build_dataloader(config, df, scaler, is_train, batch_size, shuffle):
    dataset = build_dataset(config, df, scaler, is_train)
    return DataLoader(dataset, batch_size, shuffle, drop_last=False, num_workers=config.num_workers)

def build_dataset(config, df, scaler, is_train):
    return CohortDataset(config, df, scaler, is_train)