import cv2
import numpy as np

class Ocr:
	def __init__(self, root_path_model:str, 
		detection_config:dict = None, recognition_config:dict = None) -> None:
		self.detection_config	= detection_config
		self.recognition_config = recognition_config
		if detection_config:
			from .char_detection import CharDetection
			self.detection_model = CharDetection(
				root_path=root_path_model, model_config=detection_config)
		if recognition_config:
			from .char_recognition import CharRecognition
			self.recog_model = CharRecognition(
				root_path=root_path_model, model_config=recognition_config)

	def char_detection(self, image:np.array, image_size:int = 244, 
		threshold:float = 0.5, boxes_ori:bool = True, det_sorted:bool = True) -> dict:
		'''
		Detect character from image
		@params:
			- image: np.array -> image to be detected
			- image_size: int -> size of image to be detected
			- threshold: float -> threshold for detection
			- boxes_ori: bool -> if True, return boxes in original image
			- det_sorted: bool -> if True, return boxes in sorted order
		@return:
			- result: {'boxes': np.array, 'confidences': np.array, 'labels': np.array}
		'''
		# assert error if model is not loaded
		assert self.detection_config, 'Model is not loaded'

		result_det = self.detection_model.detect(image, image_size, 
						boxes_ori, threshold, sorted=det_sorted)
		return result_det

	def char_recognition(self, image: np.array) -> dict:
		'''
		Read single character from image
		@params:
			- image: np.array -> image to be read
		@return:
			- result: {'text': str, 'conf': float}
		'''
		# assert error if model is not loaded
		assert self.recognition_config, 'Model is not loaded'

		return self.recog_model.recognition(image)

	def __calculate_confidence(self, result:dict) -> float:
		return round(sum([i['conf'] for i in result])/len(result),2)

	def __marger_text(self, result:dict) -> str:
		return ''.join([i['text'] for i in result])

	def visualize_result(self, image:np.array, results:list) -> np.array:
		'''
		Visualize result of OCR
		@params:
			- image: np.array -> image to be draw
			- results: list -> result of OCR(output type advanced)
		@return:
			- image: np.array -> image with result
		'''
		# Draw boxes
		for box in results:
			x_min, y_min, x_max, y_max = box['box']
			cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			# Draw text
			text = box['text']
			cv2.putText(image, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		return image

	def ocr(self, image:np.array, det_size:int = 244, boxes_ori:bool = True, 
		det_threshold:float=0.5, det_sorted:bool=True, output_type:str='normal') -> None:
		'''
		Read text from image using Text Detection and Recognition
		@params:
			- image: np.array -> image to be read
			- det_size: int -> size of image to be detected
			- boxes_ori: bool -> if True, return boxes in original image
			- det_threshold: float -> threshold for detection
			- det_sorted: bool -> if True, return boxes in sorted order
			- output_type: str -> 'normal' or 'advanced'
		@return:
			- result: result of detection and recognition
				- normal : {'text': str, 'conf': float} 
				- advanced : [{'text': str, 'conf': float, 'box': tuple}]
		'''
		# assert error if output type not in ['normal', 'advanced']
		assert output_type in ['normal', 'advanced'], 'Output type is not valid'
		# Char detection
		res_detection = self.char_detection(image=image, image_size=det_size,
			threshold=det_threshold, boxes_ori=boxes_ori, det_sorted=det_sorted)
		boxes = res_detection['boxes'].astype(int)

		# Char recognition
		print(res_detection)
		result_recognition = list()
		for box in boxes:
			x_min, y_min, x_max, y_max = box
			image_crop = image[y_min:y_max, x_min:x_max]
			res_recognition = self.char_recognition(image_crop)
			if output_type == 'normal':
				result_recognition.append(res_recognition)
			elif output_type == 'advanced':
				result_recognition.append({
					'text': res_recognition['text'], 
					'conf': res_recognition['conf'], 
					'box': box})

		# Output type
		if output_type == 'normal':
			confidence = self.__calculate_confidence(result_recognition)
			text = self.__marger_text(result_recognition)
			result = {'confidence': confidence, 'text': text}
		elif output_type == 'advanced':
			result =result_recognition
		return result

if __name__ == '__main__':
	import os
	import cv2
	import sys
	import glob

	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(SCRIPT_DIR))
	from configs.models import *

	root_model 		= DIRECTORY_MODEL
	config_det 		= MODELS['char_detection']
	config_recog 	= MODELS['char_recognition']

	ocr = Ocr(root_path_model=root_model, 
		detection_config=config_det, recognition_config=config_recog)

	for i in glob.glob('/Users/alimustofa/Halotec/Datasets/JASAMARGA/REPORT/LPR/old_images/A122_1657688221.jpg'):
		image = cv2.imread(i)
	
		result = ocr.ocr(image, output_type='advanced', det_threshold=0.9)
		text_ocr = ''.join([i['text'] for i in result])
		cv2.imwrite(text_ocr+'.jpg', ocr.visualize_result(image, result))
		print(
			''.join([i['text'] for i in result]),
		)