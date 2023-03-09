import torch
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CohortDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, scaler, is_train=True):
        self.df = df
        self.config = config
        
        self.scaler = scaler
        if is_train:
            self.scaler.fit(self.df[self.config.NUMERIC_FEATURES])
        self.df[self.config.NUMERIC_FEATURES] = self.scaler.transform(self.df[self.config.NUMERIC_FEATURES])

        self.image_transform = A.Compose(
            [A.Resize(*self.config.image_size),
            A.Normalize(mean=(0.4452, 0.4457, 0.4464), std=(0.2592, 0.2596, 0.2600)),
            ToTensorV2()]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        image_path = row.image_path
        image = io.imread(image_path)
        image = self.image_transform(image=image)['image']

        numeric_features = torch.tensor(row[self.config.NUMERIC_FEATURES], dtype=torch.float32)

        categorical_features = []
        cumsum_index = 0
        for one_hot_cols in self.config.ONE_HOT_CAT_FEATURES:
            categorical_features.append(row[one_hot_cols].to_numpy().argmax()+cumsum_index)
            cumsum_index += len(one_hot_cols)
        categorical_features = torch.tensor(categorical_features, dtype=torch.long)

        label = torch.tensor(row.label, dtype=torch.float32)

        return image, numeric_features, categorical_features, label
