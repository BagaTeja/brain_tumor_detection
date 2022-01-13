from re import split

#from numpy.core.einsumfunc import _OptimizeKind
from sklearn.utils import validation
import cv2    #opencv to read the images from the folder
import os
import keras
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.utils import normalize
from keras.utils.np_utils import normalize
from keras.models import Sequential
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D , Activation, Dropout, Flatten,Dense




image_directory = 'dataset/'

no_tumor_image = os.listdir(image_directory + 'no/')
yes_tumor_image = os.listdir(image_directory + 'yes/')
INPUT_SIZE = 64
dataset= []
lable = []

#print(no_tumor_image)

#simple check to read the 'jpg files only '
##path = 'no0.jpg'
#print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_image):
    if (image_name.split('.')[1]=='jpg'):
        image = cv2 . imread(image_directory+'no/'+ image_name)
        image = Image.fromarray(image,'RGB') 
               #Image.fromarray() function helps to get back the image from converted numpy array. We get back the pixels also same after converting back and forth. Hence, this is very much efficient
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        #By using numpy.array() function which takes an image as the argument and converts to NumPy array
        lable.append(0)


for i , image_name in enumerate(yes_tumor_image):
    if (image_name.split('.')[1]=='jpg'):
        image = cv2 . imread(image_directory+'yes/'+ image_name)
        image = Image.fromarray(image,'RGB')
        image =image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        lable.append(1)

#print(len(dataset)) #3000
#print(len(lable))#3000

dataset = np.array(dataset)
lable = np.array(lable)

#training of the module
#train_test_split this split the dataset into 80% of training and 20% for testing out 
 
x_train , x_test, y_train , y_test = train_test_split(dataset,lable , test_size=0.2 , random_state =0)
# in train_test__split 1. arrays , 2. textsize - means the number of byte need for the testing of the module 


#print(x_train.shape)
#print(x_test.shape)
# result is (2400, INPUT_SIZE ,INPUT_SIZE , 3)
#in order of (80%of images, INPUT_SIZE*INPUT_SIZE rid , colourcode 3 (rgb))

#OBJECT DETECTION BY TENSORFLOW

x_train= normalize(x_train, axis=1)
x_test= normalize(x_test, axis=1)


y_train = to_categorical(y_train , num_classes = 2)
y_test = to_categorical(y_test , num_classes = 2)


# MODEL BUILDING 
#binnary classification 

model = Sequential()

model.add (Conv2D(32 , (3,3), input_shape =(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add (Conv2D(32 , (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))



model.add (Conv2D(64 , (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('Softmax'))


# BINARY functon of CROSSENTROPY = 1 , activation function is sigmoid
##if your model does binary classification (e.g. if it uses a sigmoid last-layer activation).

##if your model does multi-class classification (e.g. if it uses a softmax last-layer activation)
#binary fuction of categorical cross Entryopy = 2 , activation fuction is Softmax

model.compile(loss ='categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

# feeding the model

model.fit(x_train,y_train, batch_size = 16 , verbose = 1 , epochs = 10,
validation_data =(x_test,y_test), shuffle= False)

model.save('BrainTumor10EpochsCategorical.h5') 

#the result is we traing the model and we get result as pictural frames of the images
 
