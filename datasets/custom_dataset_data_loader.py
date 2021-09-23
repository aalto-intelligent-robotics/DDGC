import torch.utils.data
from datasets.dataset import DatasetFactory


class CustomDatasetDataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.num_threds = opt.n_threads_train
        self.create_dataset()

    def create_dataset(self):
        self.dataset = DatasetFactory.get_by_name(
            self.opt.dataset_mode, self.opt)

    def split_dataset(self, split_size_percentage=0.9):
        dataset_size = len(self.dataset)
        size_of_left_split = round(split_size_percentage*dataset_size)
        size_of_right_split = dataset_size - size_of_left_split
        return torch.utils.data.random_split(
            self.dataset, [size_of_left_split, size_of_right_split])

    def create_dataloader(self, data_loader, shuffle_batches):
        if hasattr(self.dataset, 'collate_fn'):
            self.dataloader = torch.utils.data.DataLoader(
                data_loader,
                batch_size=self.opt.batch_size,
                collate_fn=self.dataset.collate_fn,
                shuffle=shuffle_batches,
                num_workers=int(self.num_threds),
                drop_last=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                data_loader,
                batch_size=self.opt.batch_size,
                shuffle=shuffle_batches,
                num_workers=int(self.num_threds),
                drop_last=True)
        return self.dataloader

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
