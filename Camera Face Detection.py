

import cv2 as cv
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import ctypes 
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


def new_user(name):
    
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def save_pic(event,x,y,flags,param):
         if event == cv.EVENT_RBUTTONDBLCLK:
                if not os.path.exists("DataSet\\"+name):
                    os.makedirs("DataSet\\"+name)
                image_name = "DataSet\\"+name+"\\"+id_generator()+".jpg"
                cv.imwrite(image_name,gray_frame)
            
    cap = cv.VideoCapture(0)
    ctypes.windll.user32.MessageBoxW(0, "double left click in the frame to take a picture", "worinning", 1)
    cv.namedWindow("frame")
    cv.setMouseCallback("frame",save_pic)
    while True:
        ret , frame = cap.read()
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        name = name
        cv.imshow("frame",gray_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


new_user("Bahaa_with_mask")



def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = np.random.randint(0,255)
            else:
                output[i][j] = image[i][j]
    return output



def data_augmentation(image,kernal):
    if kernal == 0:
        img = cv.GaussianBlur(image,(11,11),0)
    elif kernal == 1:
        img = cv.rotate(image,cv.ROTATE_90_CLOCKWISE)
    elif kernal == 2:
        img = cv.flip(image,-1)
    elif kernal == 3:
        img = sp_noise(image,0.05)
    else:
        return image
    return img
        



def read_data():
    users = os.listdir('DataSet')
    data = []
    for user in users:
        for img in os.listdir('DataSet\\'+user):
            for aug in range(4):
                img_cp = cv.resize(data_augmentation(cv.imread("DataSet\\"+user+"\\"+img,0),aug),(128,128)).flatten()
                li = list(img_cp)
                li.append(user)
                data.append(li)
    return data




import matplotlib.pyplot as plt
plt.imshow(np.array(read_data()[0][:-1]).reshape((128,128)),cmap="gray")


# Part 1 Create model Classification

classifier = Sequential()



# Step 1 Convolution layer
classifier.add(Convolution2D(32,6,6,input_shape=(640, 480, 3),activation='relu')) 




# Step 2 Max Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))



classifier.add(Flatten())



# Add Images To ANN Layer 
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))





classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# Add Images To ANN Layer 
classifier.add(Dense(output_dim=256,activation='relu'))
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))


# Part 2 data preprocessing import data 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'DataSet',
        target_size=(640, 480),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'DataSet',
        target_size=(640, 480),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=200,
        epochs=10,
        validation_data=test_set,
        validation_steps=100)



cap = cv.VideoCapture(0)

while True:
    ret , frame = cap.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow("frame",gray_frame)
    cv.waitKey(1)
    if 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()





