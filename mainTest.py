from PIL import Image
import cv2
from keras import models
from  keras.models import load_model
from PIL import Image
import numpy as np


model =  load_model ('BrainTumor10Epochs.h5')
image = cv2.imread('C:\\Users\\Teja\\OneDrive\\Documents\\Brain_tumor_Image_classification_project\\pred\\pred0.jpg')
#convert the image in to array format 

img = Image.fromarray(image)
img = img.resize((64,64)) #resizatation is done here in 64*64 gride

img = np.array(img)
input_img = np.expand_dims(img,axis=0) #expands the dimensfion of the images
#print(img) #print the image in numpy arrayfromate 
result =(model.predict(input_img) > 0.5).astype("int32") 
##np.argmax(model.predict(x), axis=-1), if your model does multi-class classification (e.g. if it uses a softmax last-layer activation).*
# (model.predict(x) > 0.5).astype("int32"), if your model does binary classification (e.g. if it uses a sigmoid last-layer activation).
print(result) 
#  if the result is 1 the brain is effectved 
#  if its 0 the brain is not effectived 



