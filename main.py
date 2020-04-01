import os
import torch
import torchvision
from PIL import Image
import csv
from config import ModelConfig
from model import ModelCNN


def workflow(resnet_model, model_num):
    # Build and train model.
    model_config_object = ModelConfig(resnet_model, model_num)
    ResNetModel = ModelCNN(model_config_object)
    ResNetModel.train_valid()
    # Evaluate best model.
    model_config_object = ModelConfig(resnet_model, model_num)
    ResNetModel = ModelCNN(model_config_object)
    best_model = model_config_object.resnet_model
    state = torch.load(os.path.join(model_config_object.model_save_dir, model_config_object.save_best_model_name),
                       map_location=model_config_object.device)
    best_model.load_state_dict(state['state_dict'])
    best_model = best_model.to(model_config_object.device)
    best_model.eval()
    # Write test results to csv.
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Expected"])

        for i in range(432):
            image_name = 'data/test/' + str(i) + '.jpg'
            test_image = Image.open(image_name)
            test_image_transform = model_config_object.data_transforms['test'](test_image)

            outputs = best_model(test_image_transform.unsqueeze(0).to(model_config_object.device))
            _, preds = torch.max(outputs.data, 1)

            class_ = int(preds.data.cpu().numpy())
            class_to_idx = state['class_to_idx']
            category = list(class_to_idx.keys())[list(class_to_idx.values()).index(class_)]

            writer.writerow([str(i), category])


if __name__ == '__main__':
    base_models = [torchvision.models.resnet152(pretrained=True),
                   torchvision.models.resnet50(pretrained=True),
                   torchvision.models.resnet18(pretrained=True)]
    for i, resnet_model in enumerate(base_models):
        workflow(resnet_model, i)
