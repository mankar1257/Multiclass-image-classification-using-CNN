#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:30:16 2020

@author: vaibhav
""" 



# --------------------------------------Make-Prediction---------------------------



from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 

filename = 'sample_image.png'
# load the image
img = load_img(filename, target_size=(150, 150))


# convert to array
img = img_to_array(img)


# reshape into a single sample with 3 channels
img = img.reshape(1,150, 150, 3)



# load model
model = load_model('model.h5')


# predict the class
result = model.predict_classes(img)
print(result[0])


 