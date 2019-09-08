# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:44:50 2019

@author: Vinay Valson
"""

from flask import Flask,render_template,url_for,request,redirect,flash
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image


# import pandas as pd
# import numpy as np 

# #ML packages
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.extenals import joblib


application = app = Flask(__name__)
Bootstrap(app)

@app.route("/query")
def query():
	print(request.query_string)
	return "no query received",200

app.config["IMAGE_UPLOADS"] = 'C:/Users/Vinay Valson/Desktop/syndicate/static/img/uploads'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG","GIF","TIF"]

def allowed_image(filename):
	if not "." in filename:
		return False

	ext = filename.rsplit(".",1)[1]

	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	else:
		return False

def load_model():
	#loading model
	json = open('models/model.json','r')
	model = json.read()
	model = model_from_json(model)
	model.load_weights('models/model.h5')
	return model

def signature(img):
	model = load_model()
	ret,img =cv2.threshold(img,110,255,cv2.THRESH_BINARY)
	#plt.imshow(img[650:1000,1700:2200],cmap='gray')
	#plt.show()
	img = img[650:1000,1700:2200]
	count = 0
	for x in range (0,350):
	    for y in range (0,400):
	        if img[x][y] < 100:
	            count = count + 1
	#print(count)
	if count < 800:
	    flash("Signature missing")


def name(img):
    # plt.imshow(img,cmap='gray')
	# plt.show()
	model = load_model()
	img1 = img[160:340,200:1600]
	img1 = tf.keras.utils.normalize(cv2.resize(img1,(145,14)))
	img1 = np.array(img1).reshape((1, 145, 14,1))
	predict = model.predict_classes(img1)
	if predict[0] == 1:
		flash("Name is tampered")
	img2 = img[320:500,200:1800]
	img2 = tf.keras.utils.normalize(cv2.resize(img2,(145,14)))
	img2 = np.array(img2).reshape((1, 145, 14,1))
	predict = model.predict_classes(img2)
	if predict[0] == 1:
		flash("Amount is tampered")


@app.route('/',methods=["GET" , "POST"])
def index():
	if request.method=="POST":
		if request.files:
			image = request.files["image"]
			if image.filename == "":
				print("image must have a filename")
				return redirect(request.url)
			if not allowed_image(image.filename):
				print("That image extension is not allowed")
				return redirect(request.url)

			check=(image.filename)
			image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
			print("image saved")
			i = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
	
						# img = cv2.imread(i,0)
			# plt.imshow(img, cmap='gray')
			# plt.show()
			img = cv2.imread(i,0)
			img = cv2.resize(img,(2300,1100))
			print(image.filename)
			signature(img)
			name(img)
			return redirect(request.url)
	return render_template('index.html')





if __name__ == '__main__':
	app.secret_key='12345'
	app.run(debug = True)
