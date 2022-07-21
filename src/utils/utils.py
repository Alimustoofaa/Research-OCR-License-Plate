import os
import cv2
import base64
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile

def download_and_unzip_model(root_dir:str, name:str, 
    url:str, file_size:int, unzip:bool = False):
	'''
	Checking model in model_path
	download model if file not found
	@params:
 		root_dir(str): The root directory of model.
		name(str): The name of model.
		url(str): The url of model.
		file_size(int): The size of model.
		unzip(bool): Unzip the model or not.
	'''
	Path(root_dir).mkdir(parents=True, exist_ok=True)

	# check if model is already or not
	print(f'Downloading {root_dir.split("/")[-1]} model, please wait.')
	response = requests.get(url, stream=True)
	
	progress = tqdm(response.iter_content(1024), 
				f'Downloading model', 
				total=file_size, unit='B', 
				unit_scale=True, unit_divisor=1024)
	save_dir = f'{root_dir}/{name}'
	with open(save_dir, 'wb') as f:
		for data in progress:
			f.write(data)
			progress.update(len(data))
		print(f'Done downloading  {root_dir.split("/")[-1]} model.')

	# unzip model
	if unzip:
		with ZipFile(save_dir, 'r') as zip_obj:
			zip_obj.extractall(root_dir)
			print(f'Done unzip {root_dir.split("/")[-1]} model.')
		os.remove(save_dir)

def encode_image2string(image):
	image_list = cv2.imencode('.jpg', image)[1]
	image_bytes = image_list.tobytes()
	image_encoded = base64.b64encode(image_bytes)
	return image_encoded

def decode_string2image(image_encoded):
	jpg_original = base64.b64decode(image_encoded)
	jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
	image = cv2.imdecode(jpg_as_np, flags=1)
	return image

def resize_image(image, size_percent):
    '''
    Resize an image so that its longest edge equals to the given size.
    Args:
        image(cv2.Image): The input image.
        size_percent(int): The size of longest edge.
    Returns:
        image(cv2.Image): The output image.
    '''
    width   = int(image.shape[1] * size_percent / 100)
    height  = int(image.shape[0] * size_percent / 100)
    dim     = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized