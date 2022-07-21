'''
@Author     : Ali Mustofa HALOTEC
@Module     : Character Recognition Neural Network
@Created on : 20 Jul 2022
'''
#!/usr/bin/env python3
# Path: src/apps/char_recognition.py

import os
import cv2
import numpy as np
from PIL import Image
from src.utils.utils import download_and_unzip_model
import torch
import torch.nn as nn
from torchvision import transforms

class _NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(_NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, num_classes),
        )
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CharRecognition:

    def __init__(self, root_path:str, model_config:dict) -> None:
        '''
        Load model 
        @params:
            - model_name: str of model name
            - classes: list of classes
        '''
        self.model_name = f'{root_path}/{model_config["filename"]}'
        self.classes    = model_config['classes']
        self.device     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__check_model(root_path, model_config)
        self.model      = self.__load_model()

    def __check_model(self, root_path:str, model_config:dict) -> None:
        if not os.path.isfile(self.model_name):
            download_and_unzip_model(
                root_dir    = root_path,
                name        = model_config['filename'],
                url         = model_config['url'],
                file_size   = model_config['file_size'],
                unzip       = False
            )
        else: print('Load model')

    def __load_model(self) -> nn.Module:
        '''
        Load model from file
        @return:
            - model: nn.Module
        '''
        model = _NeuralNetwork(len(self.classes))
        model.load_state_dict(torch.load(self.model_name, map_location=self.device))
        model.to(self.device)
        return model.eval()

    @staticmethod
    def __image_transform(image) -> torch.Tensor:
        return  transforms.Compose([
                transforms.Resize(size=(31,31)),
                transforms.CenterCrop(size=31),
                transforms.ToTensor(),
                transforms.Grayscale(3),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])(image)

    def recognition(self, image:np.array) -> dict:
        '''
        Recognize character from image
        @params:
            - image: np.array
        @return:
            - result: dict -> {class: recognition, prob: confidence}
        '''
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.__image_transform(image)
        if torch.device('cuda') == self.device:
            image = image.view(1, 3, 31, 31).cuda()
        else:
            image = image.view(1, 3, 31, 31)
        
        with torch.no_grad():
            output = self.model(image)

        output = nn.functional.log_softmax(output, dim=1)
        output = torch.exp(output)
        prob, top_class = torch.topk(output, k=1, dim=1)
        res_class  = self.classes[top_class.cpu().numpy()[0][0]]
        res_prob   = round((prob.cpu().numpy()[0][0]), 2)
        return {
            'text': res_class,
            'conf': res_prob
        }

if __name__ == '__main__':
    char_recog = CharRecognition('./models/text_recognition.ali')
    image = cv2.imread('./images/1_10043.jpg')
    result = char_recog.recognition(image)
    print(result)