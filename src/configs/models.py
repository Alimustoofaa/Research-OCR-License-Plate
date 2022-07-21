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
        'classes'   : string.digits+string.ascii_uppercase,
		'url'       : 'https://github.com/Alimustoofaa/Research-OCR-License-Plate/releases/download/recognition/text_recognition.ali',
		'file_size' : 592694
	},
	'char_detection' : {
		'filename'	: 'char_detection.ali',
        'classes'	: ['text'],
		'url' 		: 'https://github.com/Alimustoofaa/Research-OCR-License-Plate/releases/download/detection/text_detection.ali',
		'file_size' : 165726259
	},
}
