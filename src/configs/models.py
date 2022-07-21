import os
import string

#========================== DIRECTORY Models =====================================
ROOT 					= os.path.normpath(os.path.dirname(__file__))

DIRECTORY_MODEL         = os.path.expanduser('~/.Halotec/Models')

DIRECTORY_LOGGER        = os.path.expanduser('~/.Halotec/logger')

#============================ MODELS ======================================
MODELS = {
	'char_recognition' : {
		'filename'  : 'char_recognition.ali',
        'classes'   :  string.digits+string.ascii_uppercase,
		'url'       : 'https://huggingface.co/spaces/Alimustoofaa/ocr-license-plate-indonesia/resolve/main/saved_model/models.zip',
		'file_size' : 8326131
	},
	'char_detection' : {
		'filename': 'char_detection.ali',
        'classes': ['text'],
		'url' : 'https://github.com/Alimustoofaa/1-PlateDetection/releases/download/plate_detection_v2/plate_detection_v2.pt',
		'file_size' : 14753191
	},
}
