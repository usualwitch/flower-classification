import torch
from torchvision import transforms


class ModelConfig(object):
    def __init__(self, resnet_model, model_num):
        # resnet_model = 'resnet{num}' num = 18, 34, 50, 101, 152
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        self.data_dir = 'data/'
        self.datasets_name = ['train', 'valid']  # Manually divided train & validation sets, last 10% as validation.

        self.resnet_model = resnet_model
        self.pretrained_weight = f'best_resnet{model_num}.pth'
        self.model_save_dir = './checkpoints/'
        self.save_best_model_name = f'best_resnet{model_num}.pth'
        self.save_model_name = f'resnet{model_num}.pth'
        self.model_num = model_num

        self.batch_size = 8
        self.max_epoch = 60
        self.lr = 1e-2  # learning rate
        self.lr_decay = 0.1
        self.milestones = [20, 40]
        self.best_accuracy = -1
        self.start_epoch = 1
