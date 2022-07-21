'''
@Author     : Ali Mustofa HALOTEC
@Module     : Character Detection Faster RCNN
@Created on : 19 Jul 2022
'''
#!/usr/bin/env python3
# Path: src/apps/char_detection.py
import os
import cv2
import numpy as np
from PIL import Image
from src.utils.utils import download_and_unzip_model
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CharDetection:

    def __init__(self, root_path:str, model_config:dict) -> None:
        '''
        Load model 
        @params:
            - root_path:str ->  root of path model
            - model_config:dict -> config of model {filename, classes, url, file_size}
        '''
        self.model_name     = f'{root_path}/{model_config["filename"]}'
        self.classes        = model_config['classes']
        self.device         = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__check_model()
        self.model          = self.__load_model()

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

    @staticmethod
    def __image_transform(image) -> torch.Tensor:
        return transforms.Compose([transforms.ToTensor()])(image)

    def __load_model(self) -> torch.nn.Module:
        model = self.__fasterrcnn_resnet50_fpn()
        model.load_state_dict(torch.load(self.model_name, map_location=self.device), False)
        model.to(self.device)
        return model.eval()

    def __fasterrcnn_resnet50_fpn(self)-> torch.nn.Module:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes)+1)
        return model

    @staticmethod
    def __filter_threshold(probs:dict, threshold:float) -> dict:
        num_filtered = (probs['scores']>threshold).float()
        keep        = (num_filtered == torch.tensor(1)).nonzero().flatten()
        final_probs = probs
        final_probs['boxes']    = final_probs['boxes'][keep]
        final_probs['scores']   = final_probs['scores'][keep]
        final_probs['labels']   = final_probs['labels'][keep]
        return final_probs

    @staticmethod
    def __original_boxes(boxes:torch.Tensor, img_size:tuple,resized:int) -> torch.Tensor:
        image_width, image_height = img_size[1], img_size[0]
        boxes = torch.tensor([[
            (x_min/resized)*image_width, (y_min/resized)*image_height, \
            (x_max/resized)*image_width, (y_max/resized)*image_height] \
        for (x_min, y_min, x_max, y_max) in boxes.cpu().numpy()])
        return boxes
    
    @staticmethod
    def __sort_by_boxes(probs:dict) -> dict:
        x_min_list = [i[0] for i in probs['boxes']]
        idx = [x_min_list.index(x) for x in sorted(x_min_list)]
        probs['boxes']    = probs['boxes'][idx]
        probs['scores']   = probs['scores'][idx]
        probs['labels']   = probs['labels'][idx]
        return probs     

    def detect(self, image:np.array, size:int = None, 
            boxes_ori:bool = False, threshold:float = 0.5, sorted:bool = True) -> dict:
        '''
        @params:
            - image: numpy array of image
            - size: int of image resize
            - boxes_ori: bool of original boxes
            - threshold: float of threshold
            - sorted: bool of sorted by boxes
        @return:
            probs: dict of probs -> {
                'boxes' : [x_min, y_min, x_max, y_max], 
                'scores': [float], 
                'labels': [int]
            }
        '''
        im_shape = (image.shape[0], image.shape[1])
        image = cv2.resize(image, (size,size)) if size else image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.__image_transform(image)
        with torch.no_grad():
            probs = self.model([image])[0]
        probs = self.__filter_threshold(probs, threshold)
        if boxes_ori and size:
            probs['boxes'] = self.__original_boxes(probs['boxes'],im_shape, size)
        if sorted:
            probs = self.__sort_by_boxes(probs)
        return {k: v.cpu().numpy() for k, v in probs.items()}


if __name__ == '__main__':
    char_detection = CharDetection('./models/text_detection.ali', ['text'])
    image  = cv2.imread('./images/1.jpg')
    results = char_detection.detect(image, size=244, boxes_ori=True, threshold=0.01)
    print(results)