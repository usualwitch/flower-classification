import os
import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MyDataSet


class ModelCNN(object):
    def __init__(self, model_config_object):
        self.config = model_config_object
        self.model_num = self.config.model_num
        self.model = self.config.resnet_model
        self.model = self.model.to(self.config.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.milestones,
                                                        gamma=self.config.lr_decay,
                                                        last_epoch=-1)

    def train_valid(self):
        image_datasets, data_loaders, data_size = MyDataSet(self.config).load_data()
        train_loader, valid_loader = data_loaders['train'], data_loaders['valid']

        train_size, valid_size = data_size['train'], data_size['valid']
        print('train_size:%04d, valid_size:%04d\n' % (train_size, valid_size))

        plot_train_loss = []
        plot_train_accuracy = []
        plot_valid_accuracy = []
        plot_valid_loss = []
        for epoch in tqdm(range(self.config.start_epoch, self.config.max_epoch + 1)):
            loss_train, loss_valid, correct_train, correct_valid = 0, 0, 0, 0

            # Train.
            for batch_idx, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(self.config.device)
                target = target.to(self.config.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = (self.criterion(outputs, target)).sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_train += loss.item()
                correct_train += torch.sum(preds == target.data).to(torch.float32)

            # Validate.
            with torch.no_grad():
                for batch_idx, (inputs, target) in enumerate(valid_loader):
                    inputs = inputs.to(self.config.device)
                    target = target.to(self.config.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = (self.criterion(outputs, target)).sum()

                    loss_valid += loss.item()
                    correct_valid += torch.sum(preds == target.data).to(torch.float32)

            # Calculate accuracy.
            train_accuracy = correct_train.data.cpu().numpy() / train_size
            valid_accuracy = correct_valid.data.cpu().numpy() / valid_size

            # Save model.
            plot_train_loss.append(loss_train)
            plot_train_accuracy.append(train_accuracy)
            plot_valid_accuracy.append(valid_accuracy)
            plot_valid_loss.append(loss_valid)
            state = {'state_dict': self.model.state_dict(), 'epoch': epoch, 'train_loss': loss_train,
                     'valid_loss': loss_valid, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                     'class_to_idx': image_datasets['valid'].class_to_idx}
            torch.save(state,
                       os.path.join(self.config.model_save_dir, "epoch_%d_" % epoch + self.config.save_model_name))

            # Save best model.
            if valid_accuracy > self.config.best_accuracy:
                self.config.best_accuracy = valid_accuracy
                torch.save(state, os.path.join(self.config.model_save_dir, self.config.save_best_model_name))
            print('epoch:%04d, train loss:%.4f, valid loss:%.4f, train accuracy:%.4f, valid accuracy:%.4f, best accuracy:%.4f\n'
                  % (epoch, loss_train, loss_valid, train_accuracy, valid_accuracy, self.config.best_accuracy))

            # Adjust learning rate.
            self.scheduler.step()

        # Plot loss/accuracy curve.
        plt.figure()

        plt.subplot(211)
        plt.plot(plot_train_loss, label='train loss')
        plt.plot(plot_valid_loss, label='validation loss')
        plt.ylabel('loss')

        plt.subplot(212)
        plt.plot(plot_train_accuracy, label='train accuracy')
        plt.plot(plot_valid_accuracy, label='validation accuracy')
        plt.ylabel('accuracy')

        plt.savefig(f'model{self.model_num}.png')
