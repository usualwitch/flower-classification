import os
import torch
import torchvision


class MyDataSet(object):
    def __init__(self, model_config_object):
        self.transforms = model_config_object.data_transforms
        self.datasets_name = model_config_object.datasets_name
        self.data_dir = model_config_object.data_dir
        self.batch_size = model_config_object.batch_size

    def load_data(self):
        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                              self.transforms[x]) for x in self.datasets_name}
        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=self.batch_size, num_workers=0, shuffle=True)
                        for x in self.datasets_name}
        data_size = {x: len(image_datasets[x]) for x in self.datasets_name}

        return image_datasets, data_loaders, data_size
