from importlib.resources import path
import cv2
import glob
import json
import base64
import argparse
from pathlib import Path
from src.configs.models import *
from src.apps.ocr import Ocr

root_model 		= DIRECTORY_MODEL
config_det 		= MODELS['char_detection']
config_recog 	= MODELS['char_recognition']

ocr_model = Ocr(
    root_path_model=root_model,
    detection_config=config_det,
    recognition_config=config_recog)

def encode_image2string(image):
	image_list = cv2.imencode('.jpg', image)[1]
	image_bytes = image_list.tobytes()
	image_encoded = base64.b64encode(image_bytes)
	return image_encoded

def generate_dataset(path_image, output_json):
    Path(output_json).mkdir(parents=True, exist_ok=True)

    for i in glob.glob(f'{path_image}*.jpg'):
        filename = i.split('/')[-1]
        image = cv2.imread(i)
        image_encoded = encode_image2string(image).decode('utf-8')
        h,w,_ = image.shape

        labelme = {
            'version': '5.0.1',
            'flags': {},
            'shapes': list(),
            'imagePath': str(),
            'imageData': str(),
            'imageHeight': int(),
            'imageWidth': int(),
        }

        labelme['imagePath'] = i
        labelme['imageData'] = image_encoded
        labelme['imageHeight'] = h
        labelme['imageWidth'] = w

        # shapes 
        det_result = ocr_model.char_detection(image=image)
        for box in det_result['boxes']:
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            image_crop = image[y_min:y_max, x_min:x_max]
            result = ocr_model.char_recognition(image=image_crop)
            text = result['text']

            shape_dict = {
                'label': str(text),
                'points': [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {},
            }
            labelme['shapes'].append(shape_dict)

        print('Generate : ', i.replace(".jpg", ".json"))
        with open(output_json+filename.replace('.jpg', '.json'), 'w', encoding='utf-8') as f:
            json.dump(labelme, f, ensure_ascii=False, indent=4)
            print(f'Done writing {i.replace(".jpg", ".json")}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--image', type=str, default=None, help='Path to image')
    parser.add_argument('--output', type=str, default=None, help='Path to output json file')
    args = parser.parse_args()
    assert args.image is not None, 'Please provide path to image'
    assert args.output is not None, 'Please provide path to output json file'
    generate_dataset(path_image=args.image, output_json=args.output)


