# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:22:01 2018

@author: AI Enigma
"""
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
#import keras.preprocessing.image as image

def prepare_images():
    with open('all/fer2013/fer2013/fer2013.csv') as f:
        content = f.readlines()
        lines = np.array(content)
        
    num_of_instances = lines.size
    count = 0 
    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        usage = usage[:-1]
        val = img.split(" ")
        pixels = np.array(val , "int32")
        pixels = pixels.reshape((48,48))
        img = Image.fromarray(np.uint8(pixels))
        img = img.convert('L')
        path = "dataset/"+str(usage)+"/"+str(emotion)+"/"+str(count)+".jpg"
        img.save(path)
        count+=1
        print(count)
     
def build_model():
    
    model = Sequential()
 
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
     
    model.add(Flatten())
     
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
     
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__=="__main__":
    
    #prepare_images()
    
    classifier = build_model()
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set=train_datagen.flow_from_directory('dataset/Training',
                                               target_size=(48,48),
                                               batch_size=128,
                                               class_mode='categorical')
    
    test_set=test_datagen.flow_from_directory('dataset/PublicTest',
                                               target_size=(48,48),
                                               batch_size=128,
                                               class_mode='categorical')
    
    classifier.fit_generator(training_set,
                         steps_per_epoch=28709/128, 
                         epochs=50,
                         validation_data=test_set,
                         validation_steps=3589/128 ) 

    classifier.save('face_model.h5')
    
#    img = image.load_img("test2.jpg",target_size=(48, 48))
# 
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis = 0)
# 
#    x /= 255
# 
#    custom = classifier.predict_classes(x)
    