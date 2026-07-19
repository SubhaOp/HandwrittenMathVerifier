from src.dataset import HMEDataset

dataset = HMEDataset()

print(dataset.df.iloc[0]["label"])