import tensorflow as tf

# from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import tensorflow as tf
import pytesseract as pt
import os

from PIL import Image
from text_utils import ctc_decoder


import string
string = string.digits+string.ascii_uppercase+string.ascii_lowercase
alphabets = string
blank_index = len(alphabets)


def run_tflite_model(image_path):
    input_data = cv2.imread(image_path)
    input_data = cv2.resize(input_data, (128, 32))
    input_data = input_data[np.newaxis]
    # input_data = np.expand_dims(input_data, 3)
    input_data = input_data.astype('float32') #/255
    # path = f'ocr_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path='TugasAkhir_OCR/ModelCNN-CTC.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

## CAN NOT USE THIS DIFFERENT. GRAYSCALE IMAGE FIRST
# image_path = '/home/ladymerii/project/text-detection-master/img_56.jpg'
# tflite_output = run_tflite_model(image_path)
# pred = tflite_output
# text = ctc_decoder(pred, alphabets)[0]
# print(text)

# extract text from all the images in a folder
# storing the text in a single file

	
def main():
	# path for the folder for getting the raw images
	path ="/home/ladymerii/project/text-detection-master/FolderIMG"

	# link to the file in which output needs to be kept
	fullTempPath ="/home/ladymerii/project/text-detection-master/outputFile.txt"

	# iterating the images inside the folder
	for imageName in os.listdir(path):
		inputPath = os.path.join(path, imageName)
		img = Image.open(inputPath)
		img_ = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
		thresh = cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    	# img_ = cv2.imread(img_)
    	# img_ = cv2.resize(img_, (128, 32))
    	# img_ = img_[np.newaxis]
		text = pt.image_to_string(thresh, config= '--psm 6')
		# saving the text for appending it to the output.txt file
		# a + parameter used for creating the file if not present
		# and if present then append the text content
		file1 = open(fullTempPath, "a+")

		# providing the name of the image
		# file1.write(imageName+"\n")

		# providing the content in the image
		file1.write(text+"\n")
		file1.close()

	## Filtering List of file1 alphanumeric only
	with open(fullTempPath, "r") as f:
		line_list = f.readlines()
		line_list = [item.rstrip() for item in line_list]
		this_if = [' '.join([word for word in line_list.split() if len(word) > 3]) for line_list in line_list]
		this_if2 =  list(filter(None,this_if))
		this_if3 = set(this_if2)
		if_set = [n for n in this_if3]
		if_set2 = if_set.copy()
		if_set2 = [n.replace(' ', '') for n in if_set2]
		result = [n for n in this_if if n.isupper() or n.isdigit()]
		resultText = [n for n in result if not n.isdigit()]
		resultDigit =  [n for n in result if n.isdigit()]
		

	# for printing the output file
	file2 = open('ExtractText.txt', 'r')
	print(file2.read())
	file2.close()		


if __name__ == '__main__':
	main()

