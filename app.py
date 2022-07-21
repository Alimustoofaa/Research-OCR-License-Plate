import cv2
import gradio as gr
from src.configs.models import *
from src.apps.ocr import Ocr

root_model 		= DIRECTORY_MODEL
config_det 		= MODELS['char_detection']
config_recog 	= MODELS['char_recognition']

ocr_model = Ocr(
    root_path_model=root_model,
    detection_config=config_det,
    recognition_config=config_recog)

def process_ocr(image):
    result = ocr_model.ocr(image=image, output_type='advanced')
    visualize = ocr_model.visualize_result(image, result)
    # cv2.imwrite('result.jpg', visualize)
    return visualize, ''.join([i['text'] for i in result])

title = "OCR License Plate Indonesia"
css = ".image-preview {height: auto !important;}"


iface = gr.Interface(
    title   = title,
    fn      = process_ocr, 
    inputs  = [gr.Image()], 
    outputs = [gr.Image(), 'text'],
    css=css
)

iface.launch()