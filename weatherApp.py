import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

model = keras.models.load_model("modelLand.h5")

uploaded_file = st.file_uploader("Choose a file")


def image_prep(image):
	img_size = 100

	new_img = cv2.resize(image, (img_size,img_size))

	return new_img

def image_resize(image):

	image = image.reshape(-1,100,100,3)

	return image

	
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
	# Scene recognition app

	The idea of this app is to predict the general scene of a given image. Which means 
	it will receive an image and output whether it is a set of buildings, a forest, a group of mountains,
	a glacier, a sea or a street. 

	Try it out!





	This app is in very early development, its predictions might not be correct all the times.


	""")




if uploaded_file is not None:
	
	img = np.array(Image.open(uploaded_file))
	st.write("""
		
		## This is your selected picture

		Let see what we are working with...
	""")
	plt.imshow(img)
	st.pyplot()

	img = image_prep(img)
	
	img = image_resize(img)


	prediction = model.predict(img)

	message = ["Buildings", "Forest",
	"Mountains", "Glacier", "Sea", "Street"]

	st.write("""

		### With {}% confidence, I can tell you that this scene is: {}

		""".format(np.max(prediction)*100,message[np.argmax(prediction)]))
	
