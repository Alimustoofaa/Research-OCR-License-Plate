import string

LABEL = string.digits+string.ascii_uppercase
label_dict = {idx : label for idx, label in enumerate(LABEL)}
num_classes = len(label_dict)