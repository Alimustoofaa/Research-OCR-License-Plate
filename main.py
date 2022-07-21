import cv2
import glob
from src.configs.models import *
from src.apps.ocr import Ocr

root_model 		= DIRECTORY_MODEL
config_det 		= MODELS['char_detection']
config_recog 	= MODELS['char_recognition']
ocr_model = Ocr(
    root_path_model=root_model,
    detection_config=config_det,
    recognition_config=config_recog)

image = cv2.imread('/Users/alimustofa/Halotec/Datasets/JASAMARGA/REPORT/LPR/old_images/1251RP_1657688195.jpg')
result = ocr_model.ocr(image=image, output_type='advanced')
visualize = ocr_model.visualize_result(image, result)
cv2.imwrite('result.jpg', visualize)
print(''.join([i['text'] for i in result]))